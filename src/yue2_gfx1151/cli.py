"""Dependency-light, explicit-path entry point. GPU imports follow validation."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

MODEL_REVISION = "1a96eca688d6ae5d7f0feb88573fec89920fcd19"
VAE_REVISION = "95535e72a97bc0f09b8ada125d26b4009428c0e8"
OPTIMIZED_ENV = {"MIOPEN_FIND_MODE": "FAST", "YUE2_AR_ATTENTION": "triton",
                 "YUE2_AR_LINEAR": "triton", "YUE2_AR_NORM": "triton",
                 "YUE2_AR_FUSE_PROJECTIONS": "1"}
CAMPAIGN_FILES = frozenset({'campaign.json', 'summary.json', 'status.json', 'ar-stages.json'})


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("IDs/artifact names must be simple ASCII basenames (no path traversal)")
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_root(value=None):
    if value:
        root = Path(value).expanduser().resolve()
        if (root / "src/yue2/__init__.py").is_file():
            root /= "src"
        if not (root / "yue2/__init__.py").is_file():
            raise ValueError("--source must contain yue2/ or src/yue2/")
        return root
    return Path(__file__).resolve().parents[1]


def load_requests(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = [data] if isinstance(data, dict) else data
    if not isinstance(data, list) or not 1 <= len(data) <= 4:
        raise ValueError("Expected one request or a list of 1–4 requests")
    allowed = {"id", "style", "lyrics", "cot", "seed", "abc", "cfg_scale"}
    ids = []
    for row in data:
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("Unknown request fields; inline ABC only (no external file references)")
        ids.append(safe_name(row.get("id")))
        if row['id'] in CAMPAIGN_FILES:
            raise ValueError("Request ID collides with campaign metadata")
        if not isinstance(row.get("style"), str) or not isinstance(row.get("lyrics"), str):
            raise ValueError("Each request requires string style and lyrics")
        if row.get("cot", "full") not in {"full", "melody", "off"}:
            raise ValueError("cot must be full, melody or off")
        if row.get("cfg_scale", 1) not in (None, 1, 1.0):
            raise ValueError("Validated optimized batching supports CFG=1 only")
        # Upstream cot=off defaults to CFG1.01, which this batching port disallows.
        row["cfg_scale"] = 1.0
        if "seed" in row and (type(row["seed"]) is not int or not 0 <= row['seed'] < 2**63):
            raise ValueError("seed must be an integer in [0, 2**63)")
        if row.get("abc") is not None and (not isinstance(row["abc"], str) or not row['abc'].strip() or row.get('cot') == 'off'):
            raise ValueError("abc requires nonempty text and cot=melody/full")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate request IDs")
    return data


def validate(args):
    args.data = load_requests(args.request)
    args.source = str(source_root(args.source))
    for name in ("model", "vae"):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError(f"--{name} must be an existing local model directory with config.json")
        setattr(args, name, str(path))
    if (Path(args.model) / 'pipeline.json').exists():
        raise ValueError('--model must be a direct model snapshot, not a redirecting pipeline export')
    output = Path(args.output).expanduser().resolve()
    inputs = [Path(args.model), Path(args.vae), Path(args.source), Path(args.request).resolve()]
    if args.generation_config:
        inputs.append(Path(args.generation_config).expanduser().resolve())
    inputs.append(Path(args.gpu_lock).expanduser().resolve())
    if output == Path(output.anchor) or any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError("Output must not overlap model, VAE, source, request, generation config or GPU lock paths")
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not args.resume:
        raise FileExistsError("Output must be fresh; --resume only for matching checkpoints")
    args.output = str(output)
    if args.smoke and args.generation_config:
        raise ValueError("Do not combine --smoke and --generation-config")
    if args.generation_config:
        if not isinstance(json.loads(Path(args.generation_config).read_text()), dict):
            raise ValueError("Generation config must be a JSON object")
    if args.threads < 1 or not math.isfinite(args.budget) or args.budget <= 0:
        raise ValueError("Threads and memory budget must be positive")
    return {"dry_run": True, "requests": len(args.data), "ids": [r["id"] for r in args.data],
            "model": args.model, "vae": args.vae, "source": args.source, "output": args.output,
            "model_weights_verified": False, "gpu_exercised": False, "environment": OPTIMIZED_ENV}


def check_source_modules(source):
    """Never silently mix a requested checkout with cached Python modules."""
    package = (Path(source) / 'yue2').resolve()
    for name, module in tuple(sys.modules.items()):
        if name == 'yue2' or name.startswith('yue2.'):
            file = getattr(module, '__file__', None)
            if file is None or not Path(file).resolve().is_relative_to(package):
                raise ValueError('Loaded yue2 modules disagree with --source; use a fresh process')


@contextlib.contextmanager
def gpu_lock(path):
    import fcntl
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def download(args):
    from huggingface_hub import snapshot_download
    paths = [Path(args.model).expanduser().resolve(), Path(args.vae).expanduser().resolve()]
    if paths[0] == paths[1] or paths[0] in paths[1].parents or paths[1] in paths[0].parents:
        raise ValueError("Model and VAE download directories must not overlap")
    for repo, revision, path in zip(("m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"), (MODEL_REVISION, VAE_REVISION), paths):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Fresh download directory required: {path}")
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(path))
        print(json.dumps({"repository": repo, "revision": revision, "directory": str(path)}))
    return 0


def parser():
    p = argparse.ArgumentParser(description="Experimental YuE2 gfx1151 port (CC BY-NC 4.0)")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("generate", help="Explicit local models; optimized independent AR batch")
    for name in ("model", "vae", "request", "output"):
        q.add_argument("--" + name, required=True)
    q.add_argument("--source", help="Optional source tree containing yue2/ or src/yue2/")
    for name in ("dry-run", "smoke", "resume", "sequential"):
        q.add_argument("--" + name, action="store_true")
    q.add_argument("--generation-config")
    q.add_argument("--threads", type=int, default=8)
    q.add_argument("--budget", type=float, default=48, help="Model memory budget GiB, not a measured peak")
    q.add_argument("--gpu-lock", default=os.environ.get("YUE2_GPU_LOCK", str(Path.home()/".cache/yue2-gfx1151/gpu.lock")))
    d = sub.add_parser("download-models", help="Download public snapshots at immutable revisions")
    d.add_argument("--model", required=True); d.add_argument("--vae", required=True)
    v = sub.add_parser("verify", help="CPU full artifact/hash/audio validation")
    v.add_argument("output"); v.add_argument("--expected", required=True, type=int)
    v.add_argument("--allow-truncated", action="store_true")
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    try:
        if args.command == "download-models":
            return download(args)
        if args.command == "verify":
            from .verification import verify
            print(json.dumps(verify(args.output, args.expected, args.allow_truncated), indent=2))
            return 0
        report = validate(args)
        if args.dry_run:
            print(json.dumps(report, indent=2)); return 0
        os.environ.update(OPTIMIZED_ENV)
        check_source_modules(args.source)
        sys.path.insert(0, args.source)
        with gpu_lock(args.gpu_lock):
            from .campaign import run
            run(args)
            from .verification import verify
            print(json.dumps(verify(args.output, len(args.data), args.smoke), indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        p.exit(2, f"error: {exc}\n")
