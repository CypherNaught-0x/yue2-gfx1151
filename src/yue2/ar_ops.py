"""Experimental opt-in graph-safe AR decode primitives; never patch modules.

RMSNorm preserves YuE's intermediate low-precision roundings. GEMV accumulates
in FP32 and is intentionally not sample-identical to vendor GEMM.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
          BM:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    k=tl.arange(0,BK)
    w=tl.load(W+m[:,None]*K+k[None,:],(m[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
    for b in tl.static_range(B):
        x=tl.load(X+b*K+k,k<K,0).to(tl.float32)
        y=tl.sum(w*x[None,:],1)
        tl.store(Y+b*N+m,y,m<N)


def linear(x,weight,*,block_m=4,num_warps=4):
    if not x.is_contiguous() or not weight.is_contiguous() or x.shape[-1]!=weight.shape[1]:
        raise ValueError('GEMV requires contiguous matching input/weight')
    b=x.numel()//x.shape[-1];n,k=weight.shape
    if b>8:raise ValueError('Experimental AR GEMV is limited to batch <=8')
    out=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    _gemv[(triton.cdiv(n,block_m),)](x,weight,out,n,k,b,block_m,triton.next_power_of_2(k),num_warps=num_warps,enable_fp_fusion=False)
    return out


@triton.jit
def _rms(X,W,Y,N:tl.constexpr,S:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0);j=tl.arange(0,BLOCK)
    x=tl.load(X+r*S+j,j<N,0).to(tl.float32)
    w=tl.load(W+j,j<N,0).to(tl.float32)
    scale=tl.rsqrt(tl.sum(x*x,0)/N+EPS).to(Y.dtype.element_ty).to(tl.float32)
    normalized=(x*scale).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+r*N+j,normalized*w,j<N)


def rms_norm(x,weight,eps):
    # GraphAR Q/K are contiguous within head but may have a fused-QKV row gap.
    n=x.shape[-1]
    if x.stride(-1)!=1 or (x.ndim>2 and x.numel()//n>1 and not x.is_contiguous()):
        x=x.contiguous()
    out=torch.empty_like(x,memory_format=torch.contiguous_format)
    _rms[(x.numel()//n,)](x,weight,out,n,x.stride(-2),eps,triton.next_power_of_2(n),num_warps=4,enable_fp_fusion=False)
    return out
