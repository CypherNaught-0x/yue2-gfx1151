"""Standard-library-only CLI/report helpers; GPU imports happen after opt-in."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import NoReturn


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser(description, *, seed, repeats=30):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--run-gpu", action="store_true", help="explicitly authorize GPU allocation, JIT compilation and graph replay")
    result.add_argument("--output", type=Path, help="optional JSON report (atomically replaced after each completed case); otherwise stdout only")
    result.add_argument("--seed", type=int, default=seed)
    result.add_argument("--repeats", type=positive_int, default=repeats, help="timed warm graph replays/calls, excluding warmup")
    result.add_argument("--warmup", type=positive_int, default=3)
    result.add_argument("--threads", type=positive_int, default=1, help="PyTorch CPU threads")
    result.add_argument("--device-index", type=int, default=0, help="CUDA/HIP device ordinal")
    return result


def load_gpu(args, cli: argparse.ArgumentParser, *, bf16=True):
    if not args.run_gpu:
        cli.error("GPU execution is opt-in: add --run-gpu when the device is idle. --help needs no GPU or PyTorch.")
    if args.device_index < 0:
        cli.error("--device-index must be nonnegative")
    try:
        import torch
        import triton  # noqa: F401
    except (ImportError, OSError) as exc:
        cli.error(f"PyTorch/Triton runtime unavailable: {exc}. Use the project's matched ROCm/PyTorch environment; do not replace its Torch wheel with a generic wheel.")
    if not torch.cuda.is_available():
        cli.error("No CUDA/HIP GPU is available. Use a compatible GPU PyTorch build; on ROCm expose /dev/kfd and /dev/dri with render/video permissions. CPU fallback would not test these kernels.")
    if args.device_index >= torch.cuda.device_count():
        cli.error("--device-index is outside the visible GPU range")
    torch.cuda.set_device(args.device_index)
    if bf16 and not torch.cuda.is_bf16_supported():
        cli.error("These AR shapes require BF16 support on the selected device.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def import_error(cli: argparse.ArgumentParser, exc) -> NoReturn:
    cli.error(f"Cannot import the installed YuE2 optimization modules: {exc}. Install this checkout into the active matched GPU environment (for example, pip install --no-deps -e .). No source-path injection is used.")


def metrics(torch, actual, reference):
    if actual.shape != reference.shape:
        raise AssertionError(f"shape mismatch: {tuple(actual.shape)} != {tuple(reference.shape)}")
    value, ref = actual.detach().float(), reference.detach().float()
    delta = value - ref
    return {
        "finite": bool(value.isfinite().all().item() and ref.isfinite().all().item()),
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rms_error": delta.square().mean().sqrt().item(),
        "reference_rms": ref.square().mean().sqrt().item(),
        "output_rms": value.square().mean().sqrt().item(),
        "equal_values": bool(torch.equal(actual, reference)),
    }


def capture(torch, fn, warmup):
    """Compile/initialize on a side stream before capture; keep output alive."""
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    current.wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def graph_ms(torch, fn, args):
    graph, output = capture(torch, fn, args.warmup)
    for _ in range(5):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeats):
        graph.replay()
    end.record()
    end.synchronize()
    # Keep captured output allocated through the final event.
    assert output is not None
    return start.elapsed_time(end) / args.repeats


class Report:
    def __init__(self, args, torch, benchmark, **settings):
        self.path = args.output
        try:
            package_version = importlib.metadata.version("yue2")
        except importlib.metadata.PackageNotFoundError:
            package_version = "distribution metadata unavailable"
        props = torch.cuda.get_device_properties(args.device_index)
        self.data = {
            "benchmark": benchmark,
            "complete": False,
            "passed": False,
            "runtime": {
                "torch": str(torch.__version__), "hip": torch.version.hip,
                "cuda": torch.version.cuda, "yue2": package_version,
                "device": torch.cuda.get_device_name(args.device_index),
                "architecture": getattr(props, "gcnArchName", None),
                "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE", "unset"),
            },
            "settings": dict(seed=args.seed, repeats=args.repeats, warmup=args.warmup,
                             threads=args.threads, tf32=False, **settings),
            "rows": [],
        }

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp", delete=False) as handle:
                name = handle.name
                json.dump(self.data, handle, indent=2, allow_nan=False)
                handle.write("\n")
            os.replace(name, self.path)
        finally:
            if name is not None and os.path.exists(name):
                os.unlink(name)

    def add(self, **row):
        self.data["rows"].append(row)
        self.save()
        print(json.dumps(row, allow_nan=False), file=sys.stderr, flush=True)

    def __enter__(self):
        self.save()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self.data["error"] = f"{exc_type.__name__}: {exc}"
        else:
            self.data.update(complete=True, passed=True)
        self.save()
        print(json.dumps(self.data, indent=2, allow_nan=False), flush=True)
        return False
