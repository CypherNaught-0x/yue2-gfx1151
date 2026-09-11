# YuE2 gfx1151 experimental optimization port

Portable, non-commercial YuE2 inference for AMD gfx1151 (Strix Halo).
Vendors YuE2 0.1.6 at upstream commit
`8e06871aa2e704d87ffb9bc71b5f5420f6813724`, with the measured
`release-candidate-2` kernel changes. **Unofficial; not upstream AMD support.**

## What changes

- Length-aware Triton GQA decode attention with future-cache exclusion.
- FP32-accumulating AR GEMV, rounding-preserving RMSNorm, fused projections.
- Independent AR batches of 1–4 requests at CFG=1; independent RNG/history/EOS.
- MIOpen FAST solver selection before HIP initialization, reusable FP32 VAE,
  fused SnakeBeta, core 1024 / halo 16, deterministic decoding and TF32 off.
- Native NAR midpoint solver remains **32 steps** by default, without quantization.
  AR batches; NAR and VAE remain sequential within the process.

AR numerical differences can change sampling, score length and song duration.
**Same seed does not mean identical samples or equivalent musical quality.**
No audio, weights, recordings or personal requests are redistributed here.

## Measured performance (historical local campaign)

Hardware: AMD Radeon 8060S / gfx1151. Selected coherent runtime:
Torch `2.13.0a0+rocm7.13.0a20260422`, HIP `7.13.26154`.
These are previous measurements of the vendored kernel candidate, **not measurements
of a freshly built public image or this portable adapter**.

| Configuration | Generated audio | Wall time incl. setup/artifacts | Wall / audio |
|---|---:|---:|---:|
| Historical baseline, three songs | Different earlier outputs | 45.8–74.7 min/song | 17.097× aggregate |
| Triton attention + optimized VAE, four full songs | 807.475 s | 1161.479 s | 1.438× |
| Full Triton AR + optimized VAE, one full song | 224.759 s | 352.408 s | 1.568× |
| Full Triton AR + optimized VAE, four full songs | 840.315 s | 891.344 s | 1.061× |

The final historical duration-normalized aggregate improvement is approximately
16.12×, **not a same-waveform controlled speedup or fourfold latency reduction**.
The four-request candidates retained requests/seeds and native budgets/32-step
solver, but generated different durations. Private inputs are not included, so
these historical song numbers cannot be independently reproduced from this repo.
Use your own requests and the public synthetic benchmarks for new measurements.

| Controlled component evidence | Result | Qualification |
|---|---|---|
| 10 s VAE, stock warm → FAST warm → FAST/fused warm | 2.905 → 0.408 → 0.329 s | Same saved latent; solver/fusion comparison |
| 197.399 s VAE, FAST stock / fused warm | 8.354 / 6.661 s | Fused cold 14.395 s; old historical decode 912.933 s |
| Full VAE fused vs new stock | RMS error 2.326e-7; max 1.827e-5 | Peak allocation 5.574 GiB for this VAE test only |
| Real-prefix AR throughput, batch 1 / 2 / 4 | ~46 / 84 / 138 aggregate tokens/s | Short repeated decode tests, not full-pipeline rate |
| AR GEMV argmax agreement | 93.75–96.875% | Not sample-identical to compared SDPA stack |
| Short distinct-request AR batch / sequential | 4.207 / 14.809 s (3.52×) | Attention-only test; not full-song wall |

NAR accounted for 64.74% of final four-song wall time. An experimental hipBLASLt
NAR path saved 9.02% on one full solve but changed latent/waveform relative RMS
by 3.53% / 4.42%; **not enabled here**. Full-pipeline peak GPU allocation was not
collected. Modern ROCm is selected for coherent operation, not credited with the
kernel speedup. Listening quality and note-by-note fidelity remain unverified.

## Install without replacing ROCm Torch

Linux, Python 3.10+, matched gfx1151 ROCm Torch/Triton and working `/dev/kfd` are
required for generation. The package deliberately does not depend on PyPI Torch.
On a validated GPU environment:

```bash
python -m pip install --no-deps .
# Inspect/install the exact non-Torch pins in requirements-runtime.txt;
# do not let an unconstrained resolver replace your coherent ROCm stack.
yue2-gfx1151 --help
```

CLI parsing, dry-run and CPU source tests use only the standard library.
Runtime dependency pins are in `requirements-runtime.txt`. Transitive dependencies,
Torch/Triton and ROCm come from your selected coherent environment; this is not a
universal lockfile. `pip install .[runtime]` may resolve Torch through accelerate:
**do not use it without protecting the existing GPU stack**.

## Download explicit pinned snapshots

Install `huggingface-hub==0.36.2` in an appropriate download environment:

```bash
yue2-gfx1151 download-models --model "$HOME/models/YuE2-3B" --vae "$HOME/models/YuE2-Vae"
```

Requires fresh separate output directories. Pins:
- `m-a-p/YuE2-3B@1a96eca688d6ae5d7f0feb88573fec89920fcd19`
- `m-a-p/YuE2-Vae@95535e72a97bc0f09b8ada125d26b4009428c0e8`

Weights are separate downloads under their own terms. Loading uses the reviewed
vendored implementation, not arbitrary downloaded Python code. Local model hashes
and configuration identities are recorded for generation/resume.

## Generate

```bash
yue2-gfx1151 generate \
  --model "$HOME/models/YuE2-3B" --vae "$HOME/models/YuE2-Vae" \
  --request examples/request.json --output "$HOME/yue2-results/example" --dry-run
# Remove --dry-run for a complete native generation.
# Add --smoke only for a deliberately truncated ~8-second technical test.
```

Request JSON is one object or a list of 1–4 objects, with unique simple `id`,
`style`, `lyrics`, optional `cot`, `seed`, inline `abc`, and CFG=1 only.
The generic instrumental example contains no private song or lyrics. Supplied ABC
can guide generation but does not guarantee exact score fidelity or duration.
`--source DIR` optionally selects a tree containing `yue2/` or `src/yue2/`;
defaults to the installed vendored package, not a frozen host path.

`--sequential` disables independent AR batching. `--generation-config FILE` accepts
upstream GenerationConfig JSON; explicit changes to budgets/ODE steps invalidate
comparisons with the default measurements. `--budget` defaults to 48 GiB, which is
a loading budget, not a measured peak or guarantee of fit. Requests/output/model
paths must not overlap. Output must be fresh unless `--resume` is supplied.

Resume requires matching requests, source/launcher hashes, model/config/tokenizer
hashes, runtime, execution mode and budgets. Stage hashes are verified before
reuse. Do not edit checkpoints. All jobs on the same physical GPU should share
`--gpu-lock PATH` (or `YUE2_GPU_LOCK`); the default serializes this user's direct
CLI invocations, not unrelated applications or separately isolated containers.

```bash
yue2-gfx1151 verify "$HOME/yue2-results/example" --expected 1
```

Generation runs this full artifact/hash/48kHz stereo/non-silence validation too.
Signal checks are not listening acceptance. Truncated smoke needs
`verify ... --allow-truncated` and must not be presented as a full song.
Generated manifests contain the request and local identities; keep outputs private.

## Optional container

See [container setup and public-base caveats](docs/container.md).
There is **no public default image claimed to reproduce the measured runtime**.
`BASE_IMAGE` is mandatory; no unpublished local image is used as a hidden default.

```bash
BASE_IMAGE='registry.example/your-validated-rocm-image@sha256:REPLACE' \
  scripts/build-container.sh
mkdir -p "$HOME/yue2-results/container-example"
scripts/run-container.sh --model "$HOME/models/YuE2-3B" \
  --vae "$HOME/models/YuE2-Vae" --request examples/request.json \
  --output "$HOME/yue2-results/container-example" --dry-run
```

Replace the placeholder with an actual coherent base you have validated.
Build preserves the base GPU distributions, resets inherited application paths,
and packages the source. Run mounts only explicit inputs/output, disables networking,
and exposes no GPU devices for dry-run. GPU job serialization across containers is
an operator responsibility; wrap the script in your shared host `flock`.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m unittest discover -s container -p 'test_*.py' -v
```

CPU CI covers CLI validation, unsafe paths, pinned downloads using a mock,
source provenance, dependency-free help and container argument construction.
[GPU benchmarks](benchmarks/README.md) adapt the actual measured AR/VAE kernel
shapes, include numerical gates and require explicit `--run-gpu`. Run them only
with exclusive access; no GPU tests run automatically on generic GitHub runners.
CPU passing does **not** establish kernel correctness on a new ROCm stack.

## Packaged-image verification

The reviewed installed package passed 38 CPU tests, 9 container-wrapper tests,
and GitHub CI on Python 3.10/3.12. A separate GPU run in the rebuilt image produced
the generic example to natural EOS: **86.679 seconds of 48 kHz stereo audio**,
with both truncation flags false and campaign identity, artifact hashes and full
FLAC decoding verified. Its recorded generation attempt was **138.494 seconds**
(not total container startup/weight-hashing wall time). See
[machine-readable evidence](docs/packaged-validation.json) and
[container acceptance details](docs/container.md#reviewed-image-hardware-check).

This validates the operator-local coherent base, **not a fresh publicly
downloadable base**. Subjective listening is not claimed.

## License and attribution

**CC BY-NC 4.0** for the YuE2 derivative; non-commercial only. See [LICENSE](LICENSE),
[NOTICE](NOTICE.md), [MODEL_LICENSE](MODEL_LICENSE), and
[third-party notices](THIRD_PARTY_NOTICES.md). Upstream: YuE by HKUST / M-A-P.
Retain attribution and identify modifications. MIT third-party portions keep their
original terms. No rights to input recordings or lyrics are implied.
