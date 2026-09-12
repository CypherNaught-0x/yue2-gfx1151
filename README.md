# YuE2 for AMD gfx1151

Optimized, non-commercial YuE2 inference for AMD Strix Halo (`gfx1151`). This
repository packages YuE2 0.1.6 with ROCm-focused AR and VAE speedups, independent
batching for up to four songs, resumable generation, and strict artifact checks.

> [!IMPORTANT]
> This is an unofficial compatibility and performance port, not upstream AMD
> support. It requires a coherent gfx1151 ROCm/PyTorch/Triton environment and is
> licensed for non-commercial use under CC BY-NC 4.0.

## Performance at a glance

Measured locally on a Radeon 8060S with Torch
`2.13.0a0+rocm7.13.0a20260422` and HIP `7.13.26154`:

| Workload | Generated audio | Generation time | Real-time factor |
| --- | ---: | ---: | ---: |
| One full song | 224.759 s | 352.408 s | 1.568× |
| Four-song batch | 840.315 s total | 891.344 s | 1.061× aggregate |

The four-song result is close to real-time aggregate throughput. Against the
earlier local deployment, the final batch campaign measured about **16.1× higher
duration-normalized throughput**. This is not a same-waveform benchmark: optimized
AR kernels can change sampled tokens, song length, and output audio even with the
same seed.

Component measurements explain most of the gain:

- **VAE:** a 197.399-second latent decoded in 6.661 seconds on the warm optimized
  path. The matched stock/optimized waveform comparison had RMS error `2.326e-7`.
- **AR decode:** approximately 46 / 84 / 138 aggregate tokens/s at batch sizes
  1 / 2 / 4 in repeated real-prefix tests.
- **Independent batching:** a short four-request AR comparison took 4.207 seconds
  batched versus 14.809 seconds sequentially (3.52×).

NAR and VAE currently run sequentially per request. NAR remains the largest
remaining bottleneck; the native 32-step midpoint solver is intentionally kept
for quality stability.

## How the speedups work

The optimized launcher enables five coordinated changes:

1. **Length-aware Triton GQA attention** reads each row's effective KV length and
   excludes unused future-cache entries without a dense visibility mask.
2. **FP32-accumulating Triton GEMV** accelerates token-at-a-time AR linear layers
   while preserving BF16 output constraints.
3. **Rounding-preserving RMSNorm and fused projections** reduce AR memory traffic
   without changing the model's expected intermediate BF16 rounding pattern.
4. **Independent AR batching** gives each request its own RNG, history, position,
   EOS state, and token budget. Batch sizes 1–4 are supported at CFG=1.
5. **Optimized VAE decoding** selects MIOpen FAST before HIP initialization, reuses
   the loaded decoder, tiles with a 1024-frame core and 16-frame halo, and uses a
   fused FP32 SnakeBeta kernel.

At a high level, generation is:

```text
request(s) -> planning/prefill -> batched AR decode -> sequential NAR
           -> sequential optimized VAE -> FLAC + manifests -> verification
```

The CLI sets `MIOPEN_FIND_MODE=FAST`, `YUE2_AR_ATTENTION=triton`,
`YUE2_AR_LINEAR=triton`, `YUE2_AR_NORM=triton`, and
`YUE2_AR_FUSE_PROJECTIONS=1` before model initialization. `--sequential` disables
multi-request AR batching for comparison or troubleshooting.

## Requirements

- Linux on an AMD `gfx1151` GPU with working `/dev/kfd` access
- Python 3.10+
- A matched ROCm, PyTorch, and Triton stack with BF16 support
- Approximately 48 GiB of model-loading budget by default
- Local YuE2-3B and YuE2-VAE snapshots

Do **not** mix host ROCm libraries with a different PyTorch ROCm userspace. Device
enumeration alone is not enough: validate a real BF16 GPU operation before loading
the model. The project intentionally has no PyPI Torch dependency so an installer
cannot silently replace a working gfx1151 build.

## Native setup

Start inside a coherent, already validated ROCm Python environment:

```bash
git clone https://github.com/CypherNaught-0x/yue2-gfx1151.git
cd yue2-gfx1151

# Install the reviewed direct runtime pins without resolving/replacing Torch.
python -m pip install --no-deps -r requirements-runtime.txt
python -m pip install --no-deps .
yue2-gfx1151 --help
```

The direct pins are not a universal environment lockfile; your base environment
must provide their compatible transitive dependencies. Avoid `pip install
.[runtime]` unless you have explicitly protected the existing ROCm Torch stack.

Download immutable model snapshots to explicit directories:

```bash
yue2-gfx1151 download-models \
  --model "$HOME/models/YuE2-3B" \
  --vae "$HOME/models/YuE2-Vae"
```

Pinned revisions:

- `m-a-p/YuE2-3B@1a96eca688d6ae5d7f0feb88573fec89920fcd19`
- `m-a-p/YuE2-Vae@95535e72a97bc0f09b8ada125d26b4009428c0e8`

Weights remain separate downloads under their own terms. The launcher records
model/config/tokenizer identities in each campaign manifest.

## Generate and verify

Run a dry-run first, then use a fresh output directory for generation:

```bash
yue2-gfx1151 generate \
  --model "$HOME/models/YuE2-3B" \
  --vae "$HOME/models/YuE2-Vae" \
  --request examples/request.json \
  --output "$HOME/yue2-results/example" \
  --dry-run

# Remove --dry-run for a complete generation.
# Use --smoke only for a deliberately truncated ~8-second pipeline check.
```

The request file can contain one object or a list of up to four objects. Each
request has a unique `id`, `style`, `lyrics`, and optional `cot`, `seed`, `abc`,
and `cfg_scale`; the optimized path supports CFG=1 only.

```bash
yue2-gfx1151 verify "$HOME/yue2-results/example" --expected 1
```

Verification checks campaign identity, stage and artifact hashes, complete FLAC
decoding, 48 kHz stereo format, finite samples, and non-silence. It is a technical
gate, not a listening-quality judgment. Smoke outputs require
`--allow-truncated` when verified separately.

### Operational behavior

- Output directories must be fresh unless `--resume` is used.
- Resume requires matching requests, source, model/config/tokenizer identities,
  runtime, execution mode, budgets, and stage hashes.
- `--gpu-lock PATH` serializes this CLI's jobs; use the same lock for every process
  sharing the physical GPU.
- `--generation-config FILE` accepts an upstream `GenerationConfig` JSON. Changed
  token budgets or ODE steps are no longer comparable with the measurements above.
- `--budget` is a loading budget, not a measured peak-memory guarantee.

## Container setup

The container layer is bring-your-own-base because no public image is claimed to
reproduce the validated local ROCm stack. Supply a fully qualified, preferably
digest-pinned base that already contains matched gfx1151 ROCm, PyTorch, Triton, and
runtime dependencies:

```bash
BASE_IMAGE='registry.example/validated-gfx1151@sha256:REPLACE' \
  CONTAINER_ENGINE=podman scripts/build-container.sh

mkdir -p "$HOME/yue2-results/container-example"
scripts/run-container.sh \
  --model "$HOME/models/YuE2-3B" \
  --vae "$HOME/models/YuE2-Vae" \
  --request examples/request.json \
  --output "$HOME/yue2-results/container-example" \
  --dry-run
```

The build preserves the base GPU packages and installs this project with
`--no-deps`. The runtime mounts only explicit inputs and output, disables network
access, and exposes `/dev/kfd` and `/dev/dri` only for real GPU generation. See
[docs/container.md](docs/container.md) for the full base-image contract, Podman and
Docker details, and hardware acceptance procedure.

## Validation and benchmarks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m unittest discover -s container -p 'test_*.py' -v
python -m compileall -q src benchmarks tests
```

CPU CI covers CLI validation, unsafe paths, pinned-download behavior, provenance,
resume/verification rules, and container command construction. Opt-in GPU kernel
correctness tests and microbenchmarks are documented in
[benchmarks/README.md](benchmarks/README.md); they require `--run-gpu` and exclusive
GPU access.

The packaged local image also completed the generic example to natural EOS:
86.679 seconds of verified 48 kHz stereo audio in a 138.494-second generation
attempt. Machine-readable details are in
[docs/packaged-validation.json](docs/packaged-validation.json). This validates the
tested local base, not an arbitrary public ROCm image.

## Scope and provenance

The vendored source is based on YuE2 0.1.6 at upstream commit
`8e06871aa2e704d87ffb9bc71b5f5420f6813724`. No model weights, audio, recordings,
or personal requests are included. Optimized AR numerics can alter sampling, so
same-seed output identity and subjective musical equivalence are not claimed.

## License

The YuE2 derivative is **CC BY-NC 4.0** and non-commercial. See [LICENSE](LICENSE),
[NOTICE](NOTICE.md), [MODEL_LICENSE](MODEL_LICENSE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Retain attribution and identify
modifications. Upstream: YuE by HKUST / M-A-P.