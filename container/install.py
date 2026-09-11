"""Install the application without resolving or replacing the base's GPU stack."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import re
import subprocess
import sys
from pathlib import Path

# YuE2 0.1.6 upstream pyproject.toml, excluding its incompatible PyPI Torch pin.
EXPECTED = {
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.7.0",
    "tiktoken": "0.12.0",
    "numpy": "2.2.6",
    "soundfile": "0.13.1",
    "accelerate": "1.13.0",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def protected_stack() -> dict:
    result = {}
    for dist in metadata.distributions():
        name = normalize(dist.metadata["Name"])
        if name in {"torch", "torchaudio", "torchvision", "triton", "pytorch-triton-rocm"} or name.startswith(("rocm", "amd-")):
            record = dist.read_text("RECORD") or ""
            result[name] = (dist.version, hashlib.sha256(record.encode()).hexdigest())
    return result


def main() -> None:
    pins = {}
    for line in Path("requirements-runtime.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            raise SystemExit(f"Runtime requirements must be plain exact pins: {line!r}")
        name, version = match.groups()
        name = normalize(name)
        if name in pins:
            raise SystemExit(f"Duplicate runtime requirement: {name}")
        pins[name] = version
    if pins != EXPECTED:
        raise SystemExit("Runtime requirements differ from reviewed upstream non-Torch pins; review container/install.py before rebuilding")

    import torch

    if not torch.version.hip:
        raise SystemExit("BASE_IMAGE must already provide a coherent ROCm Torch build, not CPU/CUDA Torch")
    before = protected_stack()
    print(f"Preserving Torch {torch.__version__}, HIP {torch.version.hip}", flush=True)
    missing = []
    for name, version in EXPECTED.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            missing.append(f"{name}=={version}")
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", *missing], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "."], check=True)
    if before != protected_stack():
        raise SystemExit("Refusing image: protected GPU distribution metadata changed during install")
    # CPU-only import check. Missing transitive dependencies are a base-image
    # prerequisite failure; never repair them using an unconstrained resolver.
    for module in ("transformers", "huggingface_hub", "safetensors", "tiktoken", "numpy", "soundfile", "accelerate"):
        importlib.import_module(module)
    print("Application installed; protected GPU packages unchanged; CPU imports passed", flush=True)


if __name__ == "__main__":
    main()
