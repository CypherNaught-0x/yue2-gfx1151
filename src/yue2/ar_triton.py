"""Opt-in, graph-safe split-K single-token GQA for sequence-major caches.

GPU lengths include the current token. No host length read or full-cache copies.
Future positions are masked on load (including NaN/Inf-filled cache storage).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partials(Q, K, V, L, P, Z, CAP: tl.constexpr, HQ: tl.constexpr,
              HK: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
              QS0: tl.constexpr, QS2: tl.constexpr, BLOCK: tl.constexpr):
    bh = tl.program_id(0)
    s = tl.program_id(1)
    b, h = bh // HQ, bh % HQ
    length = tl.load(L + b)
    n = s * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    # Empty partitions still initialize their scratch to neutral values.
    if s * BLOCK < length:
        q = tl.load(Q + b * QS0 + h * QS2 + d).to(tl.float32)
        offsets = ((b * CAP + n[:, None]) * HK + h // (HQ // HK)) * D + d[None, :]
        valid = (n < length) & (n < CAP)
        k = tl.load(K + offsets, mask=valid[:, None], other=0).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * (D ** -0.5)
        scores = tl.where(valid, scores, float('-inf'))
        m = tl.max(scores, 0)
        w = tl.exp(scores - m)
        denom = tl.sum(w, 0)
        v = tl.load(V + offsets, mask=valid[:, None], other=0).to(tl.float32)
        out = tl.sum(w[:, None] * v, 0) / denom
        lse = m + tl.log(denom)
    else:
        out = tl.full((D,), 0, tl.float32)
        lse = float('-inf')
    tl.store(P + (bh * SPLITS + s) * D + d, out)
    tl.store(Z + bh * SPLITS + s, lse)


@triton.jit
def _combine(P, Z, O, D: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr):
    bh = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    z = tl.load(Z + bh * SPLITS + s, s < SPLITS, float('-inf'))
    w = tl.exp(z - tl.max(z, 0))
    p = tl.load(P + (bh * SPLITS + s[:, None]) * D + d[None, :],
                s[:, None] < SPLITS, 0)
    out = tl.sum(p * w[:, None], 0) / tl.sum(w, 0)
    tl.store(O + bh * D + d, out)


@triton.jit
def _partials_pair(Q, K, V, L, P, Z, CAP: tl.constexpr, HQ: tl.constexpr,
                   HK: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
                   QS0: tl.constexpr, QS2: tl.constexpr, BLOCK: tl.constexpr):
    # Exactly two query heads share each KV head in the real YuE checkpoint.
    # Consume K for both scores before loading V to avoid two live KV tiles.
    bh = tl.program_id(0)
    s = tl.program_id(1)
    b, h = bh // HK, bh % HK
    length = tl.load(L + b)
    n = s * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    if s * BLOCK < length:
        q0 = tl.load(Q + b * QS0 + (h * 2) * QS2 + d).to(tl.float32)
        q1 = tl.load(Q + b * QS0 + (h * 2 + 1) * QS2 + d).to(tl.float32)
        offsets = ((b * CAP + n[:, None]) * HK + h) * D + d[None, :]
        valid = (n < length) & (n < CAP)
        k = tl.load(K + offsets, mask=valid[:, None], other=0).to(tl.float32)
        a0 = tl.where(valid, tl.sum(k * q0[None, :], 1) * (D ** -0.5), float('-inf'))
        a1 = tl.where(valid, tl.sum(k * q1[None, :], 1) * (D ** -0.5), float('-inf'))
        m0, m1 = tl.max(a0, 0), tl.max(a1, 0)
        w0, w1 = tl.exp(a0 - m0), tl.exp(a1 - m1)
        z0, z1 = tl.sum(w0, 0), tl.sum(w1, 0)
        v = tl.load(V + offsets, mask=valid[:, None], other=0).to(tl.float32)
        o0, o1 = tl.sum(w0[:, None] * v, 0) / z0, tl.sum(w1[:, None] * v, 0) / z1
        l0, l1 = m0 + tl.log(z0), m1 + tl.log(z1)
    else:
        o0, o1 = tl.full((D,), 0, tl.float32), tl.full((D,), 0, tl.float32)
        l0, l1 = float('-inf'), float('-inf')
    tl.store(P + ((b * HQ + h * 2) * SPLITS + s) * D + d, o0)
    tl.store(P + ((b * HQ + h * 2 + 1) * SPLITS + s) * D + d, o1)
    tl.store(Z + (b * HQ + h * 2) * SPLITS + s, l0)
    tl.store(Z + (b * HQ + h * 2 + 1) * SPLITS + s, l1)


def decode_attention(q, k, v, lengths, *, block=256, paired=False, num_warps=8):
    """Q[B,1,Hq,D], KV[B,capacity,Hkv,D], lengths[B] -> Q-shaped BF16/FP16.

    Preconditions: contiguous KV, 1 <= lengths <= capacity, D power-of-two,
    Hq divisible by Hkv. Length validity is enforced by GraphAR's budget.
    """
    b, one, hq, d = q.shape
    if one != 1 or k.shape != v.shape or k.shape[0] != b or k.shape[-1] != d:
        raise ValueError('single-token Q and matching sequence-major KV required')
    if not k.is_contiguous() or not v.is_contiguous() or q.stride(-1) != 1:
        raise ValueError('contiguous KV and head dimension required')
    if d & (d - 1) or hq % k.shape[2] or lengths.numel() != b:
        raise ValueError('power-of-two head dimension, GQA ratio and batch lengths required')
    cap, hk = k.shape[1:3]
    splits = triton.cdiv(cap, block)
    p = torch.empty((b * hq, splits, d), dtype=torch.float32, device=q.device)
    z = torch.empty((b * hq, splits), dtype=torch.float32, device=q.device)
    out = torch.empty((b, 1, hq, d), dtype=q.dtype, device=q.device)
    if paired:
        if hq != 2 * hk:
            raise ValueError('Paired attention requires exactly two query heads per KV head')
        _partials_pair[(b * hk, splits)](q, k, v, lengths, p, z, cap, hq, hk, d,
                                        splits, q.stride(0), q.stride(2), block, num_warps=num_warps)
    else:
        _partials[(b * hq, splits)](q, k, v, lengths, p, z, cap, hq, hk, d,
                                   splits, q.stride(0), q.stride(2), block, num_warps=num_warps)
    _combine[(b * hq,)](p, z, out, d, splits, triton.next_power_of_2(splits), num_warps=4)
    return out
