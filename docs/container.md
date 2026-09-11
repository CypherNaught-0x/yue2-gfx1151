# Optional container deployment

This is a **bring-your-own-base** compatibility layer, not a published or independently GPU-certified image. Native installation remains available; a container is optional. YuE2 code/model terms are **CC-BY-NC-4.0**; preserve the repository's attribution, model license, and third-party notices. Base operating-system and dependency licenses remain separate. Do not push an image derived from a private base until you have audited its inherited layers, licenses, files, and metadata; deleting files in a derived layer does not remove them from the base.

## Choose a coherent base explicitly

`BASE_IMAGE` has **no default**. It must provide:

- A Linux Python environment (`python` on `PATH`), pip, setuptools >=77, and wheel.
- A matched gfx1151-capable ROCm/TheRock **and ROCm PyTorch** stack, including its native libraries and Python transitive dependencies. Keep the base's library paths; do not bind-mount host ROCm libraries into it.
- The transitive dependencies of the seven pins in `requirements-runtime.txt`. The build installs only missing/mismatched direct non-Torch pins, with `--no-deps --only-binary=:all:`; it deliberately does not resolve an arbitrary dependency graph.

The package is installed with `pip install --no-deps --no-build-isolation .`. The build rejects CPU/CUDA-only Torch, verifies the reviewed non-Torch pins, checks protected GPU distribution versions/RECORD fingerprints before and after installation, imports direct dependencies without a GPU, and runs package help. It never installs upstream's `torch==2.10.0`, vLLM, Triton, or a different ROCm stack. Fix missing dependencies in your separately maintained coherent base rather than removing these guards.

Use a fully qualified public registry reference, preferably pinned by digest, or an explicitly supplied locally available image ID. The scripts do not silently pull a guessed short name:

```bash
# Set this to a real base you have independently selected and validated.
export BASE_IMAGE='registry.example.org/your-team/coherent-gfx1151@sha256:REPLACE_WITH_REAL_DIGEST'
export IMAGE='localhost/yue2-gfx1151:latest'
CONTAINER_ENGINE=podman scripts/build-container.sh
# Docker is also supported: CONTAINER_ENGINE=docker scripts/build-container.sh
```

The registry/digest above is a placeholder, not a published image. Build from the repository using the script, which creates a temporary context containing only distribution source, licenses, runtime pins, and container helpers. It does not send the parent workspace, model weights, requests, generated audio, caches, or credentials. The temporary context is removed on exit.

### Public candidates researched, not YuE2-qualified

- [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) documents official matched ROCm/PyTorch package installation. The [GPU support matrix](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md) lists gfx1151 Linux build/sanity/release readiness, but architecture support is not validation of this application's image. Nightlies remain subject to regressions; the current multi-arch instructions differ from older 7.13-era packaging.
- [kyuz0's gfx1151 vLLM toolbox](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) advertises `docker.io/kyuz0/vllm-therock-gfx1151:latest` as its verified-working vLLM channel. The public tag's manifest was successfully inspected with `docker manifest inspect docker.io/kyuz0/vllm-therock-gfx1151:latest`. **No public layers were pulled, no public image was built on, and no YuE2 test was run on that tag.** A vLLM serving claim is not a YuE2 claim.
- The toolbox's [Fedora/TheRock Dockerfile](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes/blob/main/Dockerfile) and its current README describe different build variants: the README identifies a newer Ubuntu image, while the Fedora file uses a gfx1151 TheRock package index. Do not infer that a mutable tag has the old Fedora/7.13 stack from its historical name. Select a digest and inspect the actual image.

No public **YuE2-tested** base was established by this research, so none is promoted to a default. An existing operator-local ROCm 7.13-era base was used for the packaging checks and the subsequent hardware validation below; it is not represented as publicly downloadable.

## CPU-only CLI check

No device flags, mounts, or network access are needed for help:

```bash
podman run --rm --network=none localhost/yue2-gfx1151:latest --help
podman run --rm --network=none localhost/yue2-gfx1151:latest generate --help
```

The image overrides inherited `ENTRYPOINT`, `PYTHONPATH`, `WORKDIR`, and cache location. It runs `python -m yue2_gfx1151` from `/app`, importing installed `site-packages`, not an inherited application checkout. The base's Python/ROCm environment remains intact. The default command is `--help`, never generation.

## Explicit, narrow runtime mounts

Download complete model and VAE snapshots separately, following the main README. Hugging Face snapshot symlinks must resolve **inside** each mounted directory; materialize snapshots if they refer to a cache outside it. Prepare a request JSON and a dedicated empty writable output directory. No downloading or credentials are provided by this wrapper.

```bash
mkdir -p /data/yue2-runs/example
IMAGE=localhost/yue2-gfx1151:latest scripts/run-container.sh \
  --model /data/models/YuE2-3B \
  --vae /data/models/YuE2-Vae \
  --request /data/requests/example.json \
  --output /data/yue2-runs/example \
  --dry-run
```

All four host paths are mandatory. Inputs are mounted read-only at `/inputs/model`, `/inputs/vae`, and `/inputs/request.json`; only the selected output directory is writable at `/output`. Optional `--source /path/to/checkout` mounts only that explicitly selected source directory at `/inputs/source` and forwards `--source`; omit it to use packaged code. No parent workspace is mounted. Root/home mounts and overlapping input/output mounts are rejected. Mount paths containing commas, double quotes, or newlines are rejected rather than ambiguously parsed; spaces are supported.

`--dry-run` forwards the CLI's validation/planning mode and **does not expose GPU devices**. It is not CPU music generation and does not establish model or GPU correctness. Remove `--dry-run` only when you intend to launch generation on a free GPU:

- Linux ROCm needs `/dev/kfd`, `/dev/dri`, and correct host device permissions.
- Podman uses `--userns=keep-id --group-add keep-groups`; use a runtime such as `crun` that supports supplementary-group preservation.
- Docker uses the caller's UID/GID and numeric supplementary groups. Rootless Docker device access depends on your host/runtime configuration.
- GPU generation uses `--ipc=host`. SELinux labeling is disabled for these narrowly selected mounts; this avoids recursively relabeling model stores but reduces that isolation boundary. Capabilities are dropped. The wrapper does not use `--privileged`, mount host libraries, or publish ports.
- Network access is disabled; Hugging Face/Transformers offline mode is enabled. Caches are ephemeral under `/tmp/yue2-cache`.

Only Podman execution is checked locally; Docker command construction is supported but not an assertion of tested Docker-daemon or GPU access. Do not run concurrent GPU jobs without checking the operator's workload schedule.

## Acceptance boundary

The helper contract tests are CPU-only and use recording fake container engines:

```bash
python -m unittest discover -s container -p 'test_*.py' -v
bash -n scripts/build-container.sh scripts/run-container.sh
```

Nine helper tests and 38 CPU tests passed after launcher hardening. GitHub CI also passed on Python 3.10 and 3.12. A no-device, network-disabled run of an operator-local base confirmed Torch `2.13.0a0+rocm7.13.0a20260422`, HIP `7.13.26154`, and all seven upstream non-Torch pins. A derived image was then built successfully from that explicitly supplied local base: protected GPU package metadata was unchanged, runtime imports passed, and installed CLI help passed with packages loaded from site-packages. These observations concern the local build, **not** the researched public tag or a fresh public-base rebuild.

Container packaging checks are separate from hardware acceptance. Promoting a new base still requires, outside the CPU-only build: exact image provenance; Torch/HIP versions; a real gfx1151 kernel; model integrity checks; complete end-to-end generation; and verified audio artifacts. The packaging lane itself exposed no GPU. A separate rebuilt-image hardware run subsequently passed, as recorded below.

## Reviewed-image hardware check

The installed package at code commit `a08a41ea952fe5df3115bf785434094566e56ee4`
was rebuilt as local image ID
`9f4156c071d2da38ede863a8080caa9f72fc102f65df8dba6635bbc51edecb66`.
Using the pinned model snapshots and the unmodified `examples/request.json`,
with no source override or parent-workspace mount:

- `--smoke` exercised planning, semantic AR, native NAR and fused VAE, producing
  7.999 seconds of audio. Smoke intentionally hit its tiny token limits; this
  was a pipeline test, not a complete-song claim.
- A separate fresh output without `--smoke` produced
  86.679 seconds, both ABC and semantic stopping at EOS. Campaign identity,
  request binding, hashes and full 48 kHz stereo FLAC decoding passed.
- The full generation attempt recorded 138.494 seconds, including 40.512 seconds
  NAR and 10.523 seconds VAE. This excludes container startup and earlier weight
  hashing; it is not a cold launch wall-time benchmark.
- Torch `2.13.0a0+rocm7.13.0a20260422`, HIP `7.13.26154`.
- Finite non-silent decoded audio: RMS 0.130427, peak 0.884410.

The raw numerical record is [packaged-validation.json](packaged-validation.json).
The existing source/kernel benchmarks remain separate from this interface test.
Subjective music quality and equivalence on another machine are **not** established.
The validated base is local-only; a fresh public-base rebuild remains an explicit
reproducibility limitation, not a hidden download step.

To repeat the interface sequence with your independently validated base, use
fresh directories for smoke and full generation:

```bash
# First run the dry-run above. Then, with exclusive GPU access:
IMAGE=localhost/yue2-gfx1151:latest scripts/run-container.sh \
  --model /data/models/YuE2-3B --vae /data/models/YuE2-Vae \
  --request examples/request.json --output /data/yue2-runs/smoke --smoke
IMAGE=localhost/yue2-gfx1151:latest scripts/run-container.sh \
  --model /data/models/YuE2-3B --vae /data/models/YuE2-Vae \
  --request examples/request.json --output /data/yue2-runs/full
```

Choosing this launcher opts into its optimized AR kernels, which can alter
samples. It does not change your separate upstream deployment. These script
paths work with either explicitly selected engine;
only the Podman hardware path was actually exercised here.
