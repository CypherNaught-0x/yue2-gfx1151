"""Opt-in, independent request AR batching; existing single/CFG paths unchanged.

Use ``plan_batch(pipe, requests)`` then ``generate_semantic_batch(pipe, plans)``.
These return native SymbolicPlan/SemanticResult objects for existing synthesis,
decode and artifact saving. Only unquantized torch and guidance exactly 1 are
supported. Off requests must explicitly set cfg_scale=1 (their default is 1.01).

Static batches prefill each distinct prefix separately, then share a decode
launch. Finished rows remain allocated and receive a harmless in-vocabulary
filler; they never sample, consume RNG, invoke callbacks or extend history again.
The graph's conservative shared budget must fit *every* prefix; no truncation,
reordering, row replacement or automatic CFG fallback is performed.

RNG state is local to each request and reset independently in each stage, as in
native sampling. BF16 matrix arithmetic can differ with batch shape: identical
random streams do NOT promise identical sampled songs across B1/B2/B4.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import time
from typing import Callable, Sequence

import torch

from .protocol import (ABC_END, MUSIC_END, CODEC_OFFSET, CONTEXT, Sampling,
                       SongRequest, token_prefixes, resolve_sampling)
from .sampling import distribution, synchronize


@dataclass(frozen=True)
class TokenRequest:
    id: str
    prefix: Sequence[int]
    sampling: Sampling
    seed: int
    phase: str
    cfg_scale: float = 1.0
    legacy_off: bool = False


@dataclass
class TokenResult:
    id: str
    tokens: list[int]
    timing: dict
    truncated: bool


@dataclass
class TokenBatchResult:
    results: list[TokenResult]
    timing: dict


def _validate_requests(requests, model):
    if not requests:
        raise ValueError("Batch must contain at least one request")
    if any(not isinstance(r, TokenRequest) for r in requests):
        raise TypeError("Expected TokenRequest rows")
    if any(not isinstance(r.id, str) or not r.id for r in requests):
        raise ValueError("Request IDs must be nonempty strings")
    if len({r.id for r in requests}) != len(requests):
        raise ValueError("Request IDs must be unique within a batch")
    for r in requests:
        if r.cfg_scale != 1.0:
            raise ValueError("Independent batching supports cfg_scale=1 only")
        if r.phase not in {"abc", "semantic"}:
            raise ValueError("Phase must be abc or semantic")
        if not isinstance(r.sampling, Sampling):
            raise TypeError("Each row needs a validated Sampling object")
        if type(r.seed) is not int or not 0 <= r.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not r.prefix or any(isinstance(t, bool) or not isinstance(t, Integral)
                               or not 0 <= t < model.config.vocab_size for t in r.prefix):
            raise ValueError("Prefixes require valid integer token IDs")
    budget = max(r.sampling.max_tokens for r in requests)
    context = min(CONTEXT, model.config.max_position_embeddings)
    if any(len(r.prefix) + budget > context for r in requests):
        raise ValueError("Shared batch budget plus a prefix exceeds context; split the batch, no implicit truncation")
    return budget


@torch.inference_mode()
def generate_token_batch(model, requests: Sequence[TokenRequest], *,
                         attention_backend="triton", capture=True,
                         cancelled: Callable[[], bool] | None = None,
                         on_token: Callable[[str, str, int], None] | None = None):
    """Generate raw vocabulary tokens; EOS excluded, matching generate_tokens.

    Cancellation aborts the entire batch with InterruptedError; no partial
    successful result is fabricated. on_token(id, phase, token) includes EOS.
    Per-row completion/TTFT and batch wall timings include graph setup, while
    decode_seconds excludes allocation/prefill/capture. Callbacks are timed.
    ``capture=False, attention_backend='sdpa'`` is a CPU diagnostic option.
    """
    from .cuda_graph import GraphAR

    requests = list(requests)
    budget = _validate_requests(requests, model)
    device = next(model.parameters()).device
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before batch prefill")
    # Never seed a global generator; row order/completion cannot advance a peer.
    rng_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    generators = [torch.Generator(device=rng_device).manual_seed(r.seed) for r in requests]
    histories = [[] for _ in requests]
    active = [True] * len(requests)
    eos = [False] * len(requests)
    first = [None] * len(requests)
    completed = [None] * len(requests)
    counts = [0] * len(requests)
    graph = None
    synchronize(device)
    start = time.perf_counter()
    try:
        graph = GraphAR(model, [r.prefix for r in requests], budget,
                        independent_batch=True, capture=capture,
                        attention_backend=attention_backend)
        synchronize(device)
        allocation_seconds = time.perf_counter() - start
        logits = graph.prefill()
        synchronize(device)
        setup_seconds = time.perf_counter() - start
        decode_start = time.perf_counter()
        for step in range(budget):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during batch generation")
            # EOD is not safe for tiny diagnostic vocabularies; 0 always is.
            next_ids = torch.zeros(len(requests), dtype=torch.long, device=device)
            for row, r in enumerate(requests):
                if not active[row]:
                    continue
                scores = distribution(logits[row:row + 1], r.sampling,
                                      histories[row], step, r.phase, r.legacy_off)
                if r.sampling.temperature == 0:
                    next_id = scores.argmax(-1, keepdim=True)
                else:
                    probabilities = scores.softmax(-1)
                    if device.type == "mps":
                        next_id = torch.multinomial(probabilities.cpu(), 1,
                                                   generator=generators[row]).to(device)
                    else:
                        next_id = torch.multinomial(probabilities, 1, generator=generators[row])
                token = int(next_id.item())
                next_ids[row].copy_(next_id.reshape(()))
                counts[row] += 1
                if first[row] is None:
                    first[row] = time.perf_counter() - start
                if on_token is not None:
                    on_token(r.id, r.phase, token)
                end = ABC_END if r.phase == "abc" else MUSIC_END
                eos[row] = token == end
                if not eos[row]:
                    histories[row].append(token)
                if eos[row] or counts[row] == r.sampling.max_tokens:
                    active[row] = False
                    completed[row] = time.perf_counter() - start
            if not any(active):
                break
            # All rows share launch geometry, NOT tokens, histories or RNG.
            # An inactive row's cache advances within the conservatively checked
            # capacity; independent attention prevents its filler leaking out.
            logits = graph.step(next_ids)
        synchronize(device)
        end_time = time.perf_counter()
        seconds = end_time - start
        decode_seconds = end_time - decode_start
        timing = {"seconds": seconds, "setup_seconds": setup_seconds,
                  "allocation_seconds": allocation_seconds,
                  "prefill_capture_seconds": setup_seconds - allocation_seconds,
                  "decode_seconds": decode_seconds,
                  "output_tokens": sum(counts), "content_tokens": sum(map(len, histories)),
                  "output_tps": sum(counts) / seconds,
                  "decode_output_tps": sum(counts) / decode_seconds,
                  "batch_size": len(requests), "kv_bytes": graph.kv_bytes,
                  "capacity": graph.capacity, "graph_steps": graph.steps,
                  "attention": graph.attention_backend,
                  "execution": "independent_cuda_graph" if capture else "independent_eager"}
        results = []
        for row, r in enumerate(requests):
            row_timing = {"seconds": completed[row], "ttft_seconds": first[row],
                          "prefill_seconds": setup_seconds, "output_tokens": counts[row],
                          "content_tokens": len(histories[row]), "prefix_tokens": len(r.prefix),
                          "output_tps": counts[row] / completed[row], "cfg_branches": 1,
                          "stop_reason": "eos" if eos[row] else "budget",
                          "execution": timing["execution"], "attention": graph.attention_backend,
                          "batch": dict(timing)}
            results.append(TokenResult(r.id, histories[row], row_timing, not eos[row]))
        return TokenBatchResult(results, timing)
    finally:
        if graph is not None:
            graph.close()


def _check_pipeline(pipe, requests):
    if pipe.backend != "torch" or pipe.quantization != "none":
        raise ValueError("Independent batching requires backend='torch', quantization='none'")
    if not requests or any(not isinstance(r, SongRequest) for r in requests):
        raise ValueError("Expected a nonempty sequence of SongRequest")
    if len({r.id for r in requests}) != len(requests):
        raise ValueError("Song request IDs must be unique")
    if any(r.guidance != 1.0 for r in requests):
        raise ValueError("Independent batching supports cfg_scale=1 only; off defaults to 1.01, set it explicitly")


def _samplings(value, count, default):
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError("Per-row sampling length must match the batch")
        return [resolve_sampling(v, default) for v in value]
    return [resolve_sampling(value, default) for _ in range(count)]


@torch.inference_mode()
def plan_batch(pipe, requests: Sequence[SongRequest], *, abc_sampling=None,
               cancelled=None, on_token=None, attention_backend="triton"):
    """Plan distinct songs together; off/provided-ABC rows bypass AR planning.

    Sampling may be one override or a list aligned to input rows. Outputs retain
    input order and exact native ABC token IDs, never decode/re-encode them.
    """
    from .pipeline import SymbolicPlan

    requests = list(requests)
    _check_pipeline(pipe, requests)
    samplings = _samplings(abc_sampling, len(requests), pipe.generation_config.abc)
    plans = [None] * len(requests)
    pending, indices = [], []
    for row, r in enumerate(requests):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before batch planning")
        if r.cot == "off":
            plans[row] = SymbolicPlan(r, None, [], token_prefixes(r, pipe.tokenizer))
        elif r.abc is not None:
            ids = pipe.tokenizer.encode(r.abc)
            plans[row] = SymbolicPlan(r, r.abc, ids, token_prefixes(r, pipe.tokenizer, ids),
                                      {"seconds": 0., "output_tokens": 0, "external_prefix_tokens": len(ids)})
        else:
            indices.append(row)
            pending.append(TokenRequest(r.id, token_prefixes(r, pipe.tokenizer),
                                        samplings[row], r.seed, "abc"))
    if pending:
        generated = generate_token_batch(pipe._load_model(), pending, cancelled=cancelled,
                                         on_token=on_token, attention_backend=attention_backend)
        for row, result in zip(indices, generated.results):
            r = requests[row]
            plans[row] = SymbolicPlan(r, pipe.tokenizer.decode(result.tokens), result.tokens,
                                      token_prefixes(r, pipe.tokenizer, result.tokens),
                                      result.timing, result.truncated)
    return plans


@torch.inference_mode()
def generate_semantic_batch(pipe, plans, *, sampling=None, cancelled=None,
                            on_token=None, attention_backend="triton"):
    """Return native SemanticResult rows (zero-based codec IDs) for synthesis."""
    from .pipeline import SymbolicPlan, SemanticResult

    plans = list(plans)
    if any(not isinstance(p, SymbolicPlan) for p in plans):
        raise TypeError("Pass SymbolicPlan objects returned by plan_batch/pipe.plan")
    _check_pipeline(pipe, [p.request for p in plans])
    samplings = _samplings(sampling, len(plans), pipe.generation_config.semantic)
    pending = []
    for plan, settings in zip(plans, samplings):
        r = plan.request
        if token_prefixes(r, pipe.tokenizer, plan.abc_ids) != plan.prefix:
            raise ValueError("Plan prefix disagrees with request/exact ABC IDs")
        pending.append(TokenRequest(r.id, plan.prefix, settings, r.seed, "semantic",
                                    cfg_scale=r.guidance, legacy_off=r.cot == "off"))
    generated = generate_token_batch(pipe._load_model(), pending, cancelled=cancelled,
                                     on_token=on_token, attention_backend=attention_backend)
    return [SemanticResult(plan, [t - CODEC_OFFSET for t in result.tokens],
                           result.timing, result.truncated)
            for plan, result in zip(plans, generated.results)]
