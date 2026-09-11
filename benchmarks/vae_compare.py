#!/usr/bin/env python3
"""Optional local-checkpoint/saved-latent FP32 tiled VAE comparison (no downloads)."""
import argparse
import math
import os
from pathlib import Path
import statistics
import time

from _common import Report, import_error, load_gpu, metrics, parser, positive_int


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def core_list(value):
    try:
        result = [positive_int(item) for item in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive core sizes") from exc
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("core sizes must be unique")
    return result


def load_latent(np, torch, args):
    array = np.load(args.latent, allow_pickle=False)
    if array.dtype.kind != "f":
        raise ValueError("latent must be a floating-point .npy array")
    expected_dims = {"BCT": 3, "BTC": 3, "CT": 2, "TC": 2}
    if array.ndim != expected_dims[args.latent_layout]:
        raise ValueError("latent shape disagrees with --latent-layout; layout is never guessed")
    if args.latent_layout == "BTC":
        array = array.transpose(0, 2, 1)
    elif args.latent_layout == "CT":
        array = array[None]
    elif args.latent_layout == "TC":
        array = array.T[None]
    if args.frames is not None:
        array = array[..., :args.frames]
    return torch.from_numpy(np.array(array, dtype=np.float32, copy=True, order="C"))


def waveform_metrics(torch, actual, reference, core, ratio):
    result = metrics(torch, actual, reference)
    delta = actual.float() - reference.float()
    windows = [delta[..., max(0, boundary - 960):boundary + 960].reshape(-1)
               for boundary in range(core * ratio, actual.shape[-1], core * ratio)]
    if windows:
        boundary_delta = torch.cat(windows)
        result.update(boundary_max_abs=boundary_delta.abs().max().item(),
                      boundary_rms_error=boundary_delta.square().mean().sqrt().item())
    spectral = {}
    for nfft in (512, 2048):
        if actual.shape[-1] <= nfft // 2:
            spectral[str(nfft)] = None
            continue
        window = torch.hann_window(nfft)
        a = torch.stft(actual.reshape(-1, actual.shape[-1]), nfft,
                       hop_length=nfft // 4, window=window, return_complex=True).abs()
        b = torch.stft(reference.reshape(-1, reference.shape[-1]), nfft,
                       hop_length=nfft // 4, window=window, return_complex=True).abs()
        denominator = torch.linalg.vector_norm(b).item()
        spectral[str(nfft)] = (torch.linalg.vector_norm(a - b).item() / denominator
                               if denominator > 0 else None)
    result["spectral_relative_l2"] = spectral
    return result


def main():
    cli = parser(__doc__, seed=2026, repeats=1)
    cli.set_defaults(warmup=1)
    cli.add_argument("--vae", type=Path, required=True, help="local VAE safetensors/config directory; no network download")
    cli.add_argument("--latent", type=Path, required=True, help="user-supplied floating-point .npy latent; never copied into this repository")
    cli.add_argument("--latent-layout", choices=("BCT", "BTC", "CT", "TC"), default="BCT")
    cli.add_argument("--frames", type=positive_int, help="explicitly limit input frames; omitted means all frames")
    cli.add_argument("--reference", type=Path, help="optional saved FP32 [B,C,T_audio] .npy; otherwise compute stock reference in this process")
    cli.add_argument("--reference-core", type=positive_int, default=1024)
    cli.add_argument("--cores", type=core_list, default=[1024], help="candidate core frame sizes, e.g. 1024,64,128,256,512")
    cli.add_argument("--halo", type=positive_int, default=16)
    cli.add_argument("--atol", type=nonnegative_float, required=True, help="caller-selected waveform absolute acceptance tolerance (not a published quality guarantee)")
    cli.add_argument("--rtol", type=nonnegative_float, required=True, help="caller-selected waveform relative acceptance tolerance")
    cli.add_argument("--miopen-fast", action="store_true", help="set MIOPEN_FIND_MODE=FAST before GPU initialization; use a separate process to compare modes")
    cli.add_argument("--nondeterministic", action="store_true", help="allow nondeterministic convolution algorithms; default is deterministic")
    args = cli.parse_args()
    if not args.vae.is_dir() or not args.latent.is_file():
        cli.error("--vae must be a local directory and --latent an existing .npy file")
    if args.reference is not None and not args.reference.is_file():
        cli.error("--reference must be an existing .npy file")
    if args.miopen_fast:
        os.environ["MIOPEN_FIND_MODE"] = "FAST"
    torch = load_gpu(args, cli, bf16=False)
    try:
        import numpy as np
        from yue2.modeling_vae import YuE2VAE
        from yue2.vae_optimized import ReusableVAEDecoder, enable_fused_snake
    except ImportError as exc:
        import_error(cli, exc)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = not args.nondeterministic
    with Report(args, torch, "vae_saved_latent", cores=args.cores, halo=args.halo,
                reference_core=args.reference_core, atol=args.atol, rtol=args.rtol,
                deterministic=not args.nondeterministic, cudnn_benchmark=False,
                saved_reference=args.reference is not None, latent_layout=args.latent_layout,
                timing="synchronized wall clock including CPU output copy") as report, torch.inference_mode():
        latent = load_latent(np, torch, args)
        if not bool(latent.isfinite().all()) or latent.numel() == 0:
            raise ValueError("latent must be nonempty and finite")
        model = YuE2VAE.from_pretrained(str(args.vae), decoder_only=True, device="cuda",
                                       torch_dtype=torch.float32, local_files_only=True)
        if latent.ndim != 3 or latent.shape[1] != model.config.latent_dim:
            raise ValueError("latent channel count disagrees with the VAE config")
        expected_shape = [latent.shape[0], model.config.audio_channels,
                          model.natural_output_length(latent.shape[-1])]
        report.data["latent_shape"] = list(latent.shape)
        report.data["output_shape"] = expected_shape
        reference = None
        if args.reference is not None:
            array = np.load(args.reference, allow_pickle=False)
            if array.dtype != np.float32 or list(array.shape) != expected_shape:
                raise ValueError("reference must be FP32 and exactly match the expected [B,C,T_audio] shape")
            reference = torch.from_numpy(np.array(array, copy=True))
            if not bool(reference.isfinite().all()):
                raise ValueError("reference contains nonfinite values")
        ratio = model.config.downsampling_ratio

        def measured(label, core, fn):
            nonlocal reference
            row = dict(backend=label, core=core, calls=[], passed=False)
            report.add(**row)
            saved = report.data["rows"][-1]
            output = None
            for repeat in range(args.warmup + args.repeats):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                output = fn()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                saved["calls"].append(dict(seconds=seconds, warmup=repeat < args.warmup,
                                           peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30))
                report.save()
                assert list(output.shape) == expected_shape
                assert bool(output.isfinite().all()), "nonfinite decoded audio"
                if reference is None:
                    reference = output.clone()
                torch.testing.assert_close(output, reference, atol=args.atol, rtol=args.rtol)
            saved.update(passed=True, cold_s=saved["calls"][0]["seconds"],
                         warm_median_s=statistics.median(call["seconds"] for call in saved["calls"] if not call["warmup"]),
                         numerical=waveform_metrics(torch, output, reference, core, ratio))
            report.save()

        enable_fused_snake(model, False)
        measured("stock", args.reference_core,
                 lambda: model.decode_tiled(latent, core_frames=args.reference_core,
                                            halo_frames=args.halo, output_device="cpu"))
        for core in args.cores:
            runner = ReusableVAEDecoder(model, core_frames=core, halo_frames=args.halo, fused=True)
            if runner.fused_layers < 1:
                raise AssertionError("no SnakeBeta layers enabled: wrong VAE configuration for this comparison")
            measured("reusable_fused_snake", core, lambda: runner.decode(latent, output_device="cpu"))
        assert len(report.data["rows"]) == 1 + len(args.cores)


if __name__ == "__main__":
    main()
