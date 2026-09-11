"""Optional FP32 inference-only SnakeBeta fusion (CUDA/HIP, lazy import).

Keep IEEE division and libdevice sin, and disable FMA contraction. This kernel
removes large activation intermediates without changing the convolution path.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _snake_kernel(X, A, B, Y, N:tl.constexpr, T:tl.constexpr, C:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=i<N
    channel=(i//T)%C
    x=tl.load(X+i,mask,0)
    alpha=tl.load(A+channel)
    beta=tl.load(B+channel)
    reciprocal=tl.div_rn(1.,beta+1.e-9)
    sine=libdevice.sin(x*alpha)
    y=x+reciprocal*(sine*sine)
    tl.store(Y+i,y,mask)

def fused_snake_beta(x, alpha, beta):
    if x.device.type!='cuda' or x.dtype!=torch.float32 or x.ndim!=3:
        raise ValueError('Fused SnakeBeta requires CUDA/HIP FP32 [B,C,T]')
    x=x.contiguous()
    y=torch.empty_like(x)
    _snake_kernel[(triton.cdiv(x.numel(),512),)](x,alpha,beta,y,x.numel(),x.shape[-1],x.shape[1],512,enable_fp_fusion=False)
    return y
