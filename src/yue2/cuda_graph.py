"""AR decode graphs for one request, CFG, or opt-in independent batches.

Default CFG callers combine logits and pass one shared token to ``step``.
``independent_batch=True`` additionally permits one token per row and arbitrary
nonempty batches. Prefixes retain independent RoPE positions and cache slots.
Triton attention is imported lazily only for ``attention_backend='triton'`` or
``YUE2_AR_ATTENTION=triton``; auto/default behavior otherwise stays unchanged.

Additional independently opt-in controls: ``YUE2_AR_LINEAR=triton`` selects
FP32-accumulating decode GEMV (batch <=8), ``YUE2_AR_NORM=triton`` selects
rounding-preserving RMSNorm, and ``YUE2_AR_FUSE_PROJECTIONS=1`` concatenates
QKV and gate/up weights. Explicit constructor selections override environment.
These primitives do not alter prefill or other model users. GEMV/attention
are not bitwise/sample-identical to vendor kernels; defaults stay torch/SDPA.
"""
from __future__ import annotations
from numbers import Integral

import os

import torch
import torch.nn.functional as F


class _PrefixCache:
    """A single branch view used only by the original eager HF prefill."""
    def __init__(self, keys, values, branch):
        self.key_cache = [value[branch:branch+1].transpose(1, 2) for value in keys]
        self.value_cache = [value[branch:branch+1].transpose(1, 2) for value in values]
        self.seen = 0

    def get_seq_length(self, layer_idx=0):
        return self.seen

    def update(self, key, value, layer_idx, cache_kwargs=None):
        end = self.seen + key.shape[2]
        if end > self.key_cache[layer_idx].shape[2]:
            raise ValueError("Prefix exceeds preallocated KV capacity")
        self.key_cache[layer_idx][:, :, self.seen:end].copy_(key)
        self.value_cache[layer_idx][:, :, self.seen:end].copy_(value)
        if layer_idx == len(self.key_cache) - 1:
            self.seen = end
        return self.key_cache[layer_idx][:, :, :end], self.value_cache[layer_idx][:, :, :end]


class GraphAR:
    """Fixed-capacity decode with independent positions for every row.

    By default only one request or two shared-token CFG branches are allowed.
    Set ``independent_batch=True`` for native multi-request batches; ``step``
    then accepts int32/int64 tensors of shape [B] or [B,1]. GPU token IDs are
    assumed validated by the caller, as with the legacy sampled scalar.
    ``prefill()`` predicts the first token. At most ``max_tokens-1`` calls to
    ``step(token)`` predict the remaining tokens. Returned graph logits share
    output storage and remain valid until the next step. ``capture=False`` is
    an eager diagnostic path for CPU/tiny-model correctness checks.
    ``linear_backend``/``norm_backend`` accept auto, torch, or triton. Auto
    consults the corresponding environment variable and otherwise means torch.
    ``fuse_projections=None`` consults its environment flag (otherwise False).
    """
    def __init__(self, model, prefixes, max_tokens, *, capture=True, attention_backend="auto", fuse_projections=None, independent_batch=False, linear_backend="auto", norm_backend="auto"):
        if model.training:
            raise ValueError("GraphAR requires model.eval()")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        prefixes = [list(prefix) for prefix in prefixes]
        if not prefixes or (not independent_batch and len(prefixes) not in {1, 2}):
            raise ValueError("GraphAR supports one request or exactly two CFG branches")
        config = model.config
        for prefix in prefixes:
            if not prefix or any(isinstance(token, bool) or not isinstance(token, Integral) or
                                 not 0 <= token < config.vocab_size for token in prefix):
                raise ValueError("Prefixes require valid integer token IDs")
            if len(prefix) + max_tokens > config.max_position_embeddings:
                raise ValueError("Prefix plus generation budget exceeds model context; no length was shortened")
        weight = model.model.embed_tokens.weight
        self.model, self.device, self.dtype = model, weight.device, weight.dtype
        linears = [module for layer in model.model.layers for module in
                   (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                    layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)]
        if getattr(model, "_yue2_fp8_originals", None) or any(
                not isinstance(module, torch.nn.Linear) or module.weight.dtype != self.dtype for module in linears):
            raise ValueError("GraphAR only supports unquantized standard AR Linear layers; use eager for FP8")
        if capture and self.device.type != "cuda":
            raise ValueError("CUDA graphs require a CUDA model; capture=False is for diagnostics")
        self.prefixes = [[int(token) for token in prefix] for prefix in prefixes]
        self.max_tokens, self.branches = int(max_tokens), len(prefixes)
        self.independent_batch = independent_batch
        self.capacity = max(map(len, prefixes)) + self.max_tokens
        self.capture, self.graph, self.output = capture, None, None
        if attention_backend == "auto":
            attention_backend = os.environ.get("YUE2_AR_ATTENTION", "auto")
        if attention_backend not in {"auto", "flash", "cudnn", "sdpa", "triton"}:
            raise ValueError("attention_backend must be auto, flash, cudnn, sdpa, or triton")
        if attention_backend == "triton":
            if self.device.type != "cuda" or self.dtype not in {torch.bfloat16, torch.float16}:
                raise ValueError("Triton attention requires CUDA/HIP BF16 or FP16")
            from .ar_triton import decode_attention
            self._triton_attention = decode_attention
        fused = self.device.type == "cuda" and self.dtype in {torch.bfloat16, torch.float16} and config.head_dim % 8 == 0
        flash = fused and config.head_dim <= 256 and hasattr(torch.ops.aten, "_flash_attention_forward") and (
            "seqused_k" in str(torch.ops.aten._flash_attention_forward.default._schema))
        # Torch 2.10 is pinned by the package. Its native variable-length FA
        # accepts GPU effective lengths; the public masked SDPA can select a
        # much slower math kernel. Keep a cuDNN/public-SDPA fallback explicit.
        # ROCm exposes AMD devices through torch.cuda but its private
        # _flash_attention_forward rejects the seqused_k contract used here.
        # Keep CUDA graphs, but select portable public SDPA on HIP.
        if attention_backend == "auto":
            if torch.version.hip is not None:
                attention_backend = "sdpa"
            else:
                attention_backend = "flash" if flash else "cudnn" if fused and torch.backends.cudnn.is_available() else "sdpa"
        if attention_backend == "flash" and not flash:
            raise ValueError("Pinned PyTorch variable-length CUDA FlashAttention is unavailable")
        if attention_backend == "cudnn" and not (fused and torch.backends.cudnn.is_available()):
            raise ValueError("cuDNN attention requires a supported CUDA dtype/head dimension")
        self.attention_backend = attention_backend
        if fuse_projections is None:
            setting = os.environ.get("YUE2_AR_FUSE_PROJECTIONS", "0").lower()
            if setting not in {"0", "1", "false", "true"}:
                raise ValueError("YUE2_AR_FUSE_PROJECTIONS must be 0/1/false/true")
            fuse_projections = setting in {"1", "true"}
        self.linear_backend = os.environ.get("YUE2_AR_LINEAR", "torch") if linear_backend == "auto" else linear_backend
        self.norm_backend = os.environ.get("YUE2_AR_NORM", "torch") if norm_backend == "auto" else norm_backend
        if self.linear_backend not in {"torch", "triton"} or self.norm_backend not in {"torch", "triton"}:
            raise ValueError("linear_backend and norm_backend must be auto, torch or triton")
        if "triton" in {self.linear_backend, self.norm_backend}:
            if self.device.type != "cuda" or self.dtype not in {torch.bfloat16, torch.float16}:
                raise ValueError("Triton AR primitives require CUDA/HIP BF16 or FP16")
            from .ar_ops import linear, rms_norm
            self._triton_linear, self._triton_norm = linear, rms_norm
        if self.linear_backend == "triton" and (self.branches > 8 or any(module.bias is not None for module in linears) or model.lm_head.bias is not None):
            raise ValueError("Triton AR GEMV requires batch <=8 and bias-free linears")
        self.ready, self.closed, self.steps = False, False, 0
        # Sequence-major layout makes the packed FA view contiguous without
        # copying all cached keys at each decode step.
        shape = (self.branches, self.capacity, config.num_key_value_heads, config.head_dim)
        self.keys = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in model.model.layers]
        self.values = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in model.model.layers]
        self.positions = torch.tensor([len(prefix) for prefix in prefixes], dtype=torch.long, device=self.device)
        self.initial_positions = self.positions.clone()
        self.key_positions = torch.arange(self.capacity, dtype=torch.long, device=self.device)
        self.cu_q = torch.arange(self.branches + 1, dtype=torch.int32, device=self.device)
        self.cu_k = self.cu_q * self.capacity
        self.tokens = torch.tensor([[prefix[-1]] for prefix in prefixes], dtype=torch.long, device=self.device)
        self.kv_bytes = 2 * len(self.keys) * self.keys[0].numel() * self.keys[0].element_size()
        self.fused_weights = []
        if fuse_projections:
            if any(module.bias is not None for module in linears):
                raise ValueError("Fused projections require the checkpoint's bias-free AR linears")
            with torch.no_grad():
                for layer in model.model.layers:
                    self.fused_weights.append((torch.cat([layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight,
                                                         layer.self_attn.v_proj.weight], dim=0),
                                               torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0)))
        self.fused_weight_bytes = sum(value.numel() * value.element_size() for pair in self.fused_weights for value in pair)

    def _linear(self, x, module):
        if self.linear_backend == "triton":
            return self._project(x, module.weight)
        return module(x)

    def _project(self, x, weight):
        if self.linear_backend == "triton":
            # Fixed measured gfx1151 decode choices; no runtime autotuning.
            n, k = weight.shape
            bm, nw = (1, 4) if k > 2048 or (n <= 4096 and self.branches == 1) else (4, 8)
            return self._triton_linear(x, weight, block_m=bm, num_warps=nw)
        return F.linear(x, weight)

    def _norm(self, x, module):
        if self.norm_backend == "triton":
            return self._triton_norm(x, module.weight, module.eps)
        return module(x)

    @torch.inference_mode()
    def _decode(self):
        backbone = self.model.model
        cos, sin = backbone.rotary_emb(self.positions[:, None])
        x = backbone.embed_tokens(self.tokens)
        visible = None
        if self.attention_backend not in {"flash", "triton"}:
            visible = (self.key_positions[None, :] <= self.positions[:, None])[:, None, None, :]
        used_lengths = (self.positions + 1).to(torch.int32)
        config = self.model.config
        slots = self.positions[:, None, None, None].expand(
            self.branches, 1, config.num_key_value_heads, config.head_dim)
        for index, (layer, keys, values) in enumerate(zip(backbone.layers, self.keys, self.values)):
            normalized = self._norm(x, layer.input_layernorm)
            if self.fused_weights or self.linear_backend == "triton" or self.norm_backend == "triton":
                from .modeling_yue2 import _apply_rotary
                sizes = (config.num_attention_heads * config.head_dim,
                         config.num_key_value_heads * config.head_dim, config.num_key_value_heads * config.head_dim)
                if self.fused_weights:
                    q, k, v = self._project(normalized, self.fused_weights[index][0]).split(sizes, dim=-1)
                else:
                    q, k, v = (self._linear(normalized, module) for module in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj))
                q = self._norm(q.view(self.branches, 1, config.num_attention_heads, config.head_dim), layer.self_attn.q_norm)
                k = self._norm(k.view(self.branches, 1, config.num_key_value_heads, config.head_dim), layer.self_attn.k_norm)
                v = v.view(self.branches, 1, config.num_key_value_heads, config.head_dim)
                q = _apply_rotary(q, cos.unsqueeze(2), sin.unsqueeze(2))
                k = _apply_rotary(k, cos.unsqueeze(2), sin.unsqueeze(2))
            else:
                q, k, v = layer.self_attn.project_qkv(normalized, cos, sin)
            keys.scatter_(1, slots, k)
            values.scatter_(1, slots, v)
            # All allocated slots are present, but only each branch's completed
            # prefix and the current token are visible. Future slots never leak.
            if self.attention_backend == "flash":
                # seqused_k is respected by the 3D packed/varlen entrypoint.
                # The 4D fixed-batch entrypoint ignores it in torch 2.10, so do
                # not replace this call with an apparently equivalent 4D call.
                h = torch.ops.aten._flash_attention_forward(
                    q[:, 0], keys.view(-1, config.num_key_value_heads, config.head_dim),
                    values.view(-1, config.num_key_value_heads, config.head_dim),
                    self.cu_q, self.cu_k, 1, self.capacity, 0.0, False, False,
                    seqused_k=used_lengths)[0][:, None]
            elif self.attention_backend == "triton":
                h = self._triton_attention(q, keys, values, used_lengths)
            elif self.attention_backend == "cudnn":
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    h = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                            attn_mask=visible, is_causal=False,
                            enable_gqa=config.num_attention_heads != config.num_key_value_heads).transpose(1, 2)
            else:
                h = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                            attn_mask=visible, is_causal=False,
                            enable_gqa=config.num_attention_heads != config.num_key_value_heads).transpose(1, 2)
            x = x + self._linear(h.reshape(self.branches, 1, -1), layer.self_attn.o_proj)
            normalized = self._norm(x, layer.post_attention_layernorm)
            if self.fused_weights:
                gate, up = self._project(normalized, self.fused_weights[index][1]).chunk(2, dim=-1)
                x = x + self._linear(F.silu(gate) * up, layer.mlp.down_proj)
            elif self.linear_backend == "triton":
                gate = self._linear(normalized, layer.mlp.gate_proj)
                up = self._linear(normalized, layer.mlp.up_proj)
                x = x + self._linear(F.silu(gate) * up, layer.mlp.down_proj)
            else:
                x = x + layer.mlp(normalized)
        output = self._linear(self._norm(x, backbone.norm), self.model.lm_head)[:, 0]
        self.positions.add_(1)
        return output

    @torch.inference_mode()
    def _capture(self):
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            warmup = torch.cuda.Stream(device=self.device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    self.positions.copy_(self.initial_positions)
                    self._decode()
                self.positions.copy_(self.initial_positions)
            current.wait_stream(warmup)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self._decode()
            # Warmup/capture wrote one future slot. The first real step
            # overwrites that slot in every layer before making it visible.
            self.positions.copy_(self.initial_positions)

    @torch.inference_mode()
    def prefill(self):
        if self.closed or self.ready:
            raise RuntimeError("prefill must be called exactly once on an open GraphAR")
        logits = []
        for branch, prefix in enumerate(self.prefixes):
            cache = _PrefixCache(self.keys, self.values, branch)
            result = self.model(torch.tensor([prefix], dtype=torch.long, device=self.device),
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits.append(result.logits[:, -1])
        result = torch.cat(logits, dim=0)
        if self.capture and self.max_tokens > 1:
            self._capture()
        self.ready = True
        return result

    @torch.inference_mode()
    def step(self, token):
        if self.closed or not self.ready:
            raise RuntimeError("Call prefill before step and do not use a closed GraphAR")
        if self.steps >= self.max_tokens - 1:
            raise ValueError("Requested generation budget is exhausted")
        if isinstance(token, Integral) and not isinstance(token, bool):
            if not 0 <= int(token) < self.model.config.vocab_size:
                raise ValueError("Token is outside the model vocabulary")
            self.tokens.fill_(int(token))
        elif isinstance(token, torch.Tensor) and token.numel() == 1 and token.dtype in {torch.int32, torch.int64}:
            # Sampling already produced/validated this scalar. Copying the
            # tensor avoids a second GPU-to-CPU synchronization in the caller.
            if token.device.type == "cpu" and not 0 <= token.item() < self.model.config.vocab_size:
                raise ValueError("Token is outside the model vocabulary")
            self.tokens.copy_(token.reshape(1, 1).expand(self.branches, 1))
        elif (self.independent_batch and isinstance(token, torch.Tensor)
              and token.shape in {(self.branches,), (self.branches, 1)}
              and token.dtype in {torch.int32, torch.int64}):
            if token.device.type == "cpu" and ((token < 0).any() or (token >= self.model.config.vocab_size).any()):
                raise ValueError("Token is outside the model vocabulary")
            self.tokens.copy_(token.reshape(self.branches, 1))
        else:
            raise ValueError("step needs one shared integer token, including for CFG; per-row tokens require independent_batch=True")
        if self.graph is not None:
            self.graph.replay()
            result = self.output
        else:
            result = self._decode()
        self.steps += 1
        return result

    def close(self):
        self.graph = self.output = None
        self.keys.clear()
        self.values.clear()
        self.fused_weights.clear()
        self.tokens = self.positions = self.initial_positions = self.key_positions = None
        self.cu_q = self.cu_k = None
        self.closed = True
