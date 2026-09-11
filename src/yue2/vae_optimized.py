"""Opt-in reusable listening decoder; never changes global precision settings.

Reuse one instance across requests to retain MIOpen's in-process shape cache.
Fused activations retain original parameters/state_dict and fall back to stock
SnakeBeta on CPU, non-FP32 inputs, training, or autograd-enabled calls.
"""
import os
import torch

def configure_miopen_fast():
    """Opt into FAST solver selection BEFORE first HIP/CUDA initialization.

    This is process-wide and must be called by the application, never silently
    by a model constructor. Determinism/TF32 flags are not changed. A fresh
    process is required for reliable comparison with another find mode.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError('Configure MIOpen before CUDA/HIP initialization; start a fresh process')
    os.environ['MIOPEN_FIND_MODE']='FAST'

def enable_fused_snake(model, enabled=True):
    from .modeling_vae import SnakeBeta
    implementation = None
    if enabled:
        from .vae_kernels import fused_snake_beta  # Fail immediately if unavailable.
        implementation = fused_snake_beta
    count=0
    for child in model.decoder.modules():
        if isinstance(child,SnakeBeta):
            child._fused_snake_impl = implementation
            count+=1
    return count

class ReusableVAEDecoder:
    """FP32 exact-context tiled decoder with optional fused SnakeBeta kernels.

    No synthetic zero padding: true song endpoints remain true endpoints.
    Call ``warmup(latent)`` with representative real lengths if startup latency
    can be amortized; it is an explicit real decode, not a free operation.
    Model parameters must not be mutated concurrently with decoding.
    """
    def __init__(self, model, *, core_frames=1024, halo_frames=16, fused=True):
        if model.training:
            raise ValueError('Reusable decoder requires model.eval()')
        if isinstance(core_frames,bool) or not isinstance(core_frames,int) or core_frames<1:
            raise ValueError('core_frames must be a positive integer')
        if isinstance(halo_frames,bool) or not isinstance(halo_frames,int) or halo_frames<model.required_halo(core_frames):
            raise ValueError('halo_frames is below decoder receptive-field requirement')
        self.model=model
        self.core_frames,self.halo_frames=core_frames,halo_frames
        self.fused_layers=enable_fused_snake(model,fused)

    @classmethod
    def from_pretrained(cls,path,*,device='cuda',**kwargs):
        from .modeling_vae import YuE2VAE
        return cls(YuE2VAE.from_pretrained(path,decoder_only=True,device=device),**kwargs)

    @torch.inference_mode()
    def decode(self,latent,*,output_device='cpu',on_progress=None):
        return self.model.decode_tiled(latent,core_frames=self.core_frames,
            halo_frames=self.halo_frames,output_device=output_device,on_progress=on_progress)

    @torch.inference_mode()
    def warmup(self,latent):
        audio=self.decode(latent)
        return tuple(audio.shape)
