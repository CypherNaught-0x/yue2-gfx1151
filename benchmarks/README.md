# YuE2 kernel correctness and microbenchmarks

These are portable, opt-in reproductions of the **synthetic kernel cases** used
in the local YuE2 ROCm optimization experiments. They import the **installed
`yue2` package**; they do not inject a checkout path, download checkpoints, or
contain weights, prompts, recordings, latent arrays, historical logs, or measured
result files. The only checkpoint-dependent program is `vae_compare.py`, which
requires explicit local inputs.

**Status of this public adaptation:** Python compilation and dependency-free
`--help` checks were run. GPU execution was deliberately not performed while a
separate generation occupied the device. No speed or numerical result is claimed
for these adapted programs until you run them. Preserved experiment shapes and
acceptance thresholds are not a substitute for rerunning the tests.

## Environment and opt-in

Use this project's matched PyTorch/ROCm/Triton environment and install this
checkout there, e.g. `python -m pip install --no-deps -e .`. Do not replace an
existing gfx1151 Torch build with a generic PyPI wheel just to run a benchmark.
On ROCm, PyTorch uses the `cuda` API. The runtime needs GPU device permissions
and, inside a container, access to `/dev/kfd` and `/dev/dri`.

Every executable script parses arguments **before** importing Torch, Triton or
YuE2. Every GPU path requires `--run-gpu`; missing dependencies, invisible GPUs
and unsupported BF16 devices fail with an actionable error instead of silently
falling back to CPU. `--device-index` selects a visible device. Run one benchmark
at a time on an idle GPU. Compilation, shape initialization, and large graph
pools can consume substantial memory and compete with generation.

From the repository root (no Torch installation needed):

```sh
python benchmarks/ar_micro.py --help
python benchmarks/ar_ops_micro.py --help
python benchmarks/ar_correctness.py --help
python benchmarks/vae_verify.py --help
python benchmarks/vae_compare.py --help
python -m compileall -q benchmarks
```

For an initial, explicitly authorized GPU check:

```sh
python benchmarks/ar_micro.py --run-gpu --quick --output results/attention-smoke.json
python benchmarks/ar_ops_micro.py --run-gpu --quick --output results/ops-smoke.json
python benchmarks/ar_correctness.py --run-gpu --output results/ar-graph.json
python benchmarks/vae_verify.py --run-gpu --output results/snake.json
```

Full synthetic sweeps / alternative graph integration:

```sh
python benchmarks/ar_micro.py --run-gpu --paired --output results/attention.json
python benchmarks/ar_ops_micro.py --run-gpu --include-vocab --output results/ops.json
python benchmarks/ar_ops_micro.py --run-gpu --numerics-only --output results/ops-numerics.json
python benchmarks/ar_correctness.py --run-gpu --linear triton --norm triton --fused --output results/ar-all-opt-ins.json
```

`--quick` is a deliberately smaller smoke subset, **not** a replication of the
full sweep. `--include-vocab` separately enables the large vocabulary matrix;
the BF16 weights plus temporary FP32 reference conversion require multi-GiB
peak memory. Omit it on memory-constrained systems. No model checkpoint is used
for any of the synthetic cases.

## Cases and provenance

The source baseline for the optimization work was upstream YuE commit
`8e06871aa2e704d87ffb9bc71b5f5420f6813724`; the optimized kernels are the
release-candidate-2 `yue2` modules carried by this project, not a claim that these
custom kernels exist in that upstream commit. Adaptation inputs were the
experiment harnesses listed below. Only their code, dimensions, seeds, and
numerical comparison rules were used; their measured outputs and private inputs
were not copied.

| Public program | Original harness / preserved cases |
| --- | --- |
| `ar_micro.py` | `ar_micro.py`: seed 123; BF16 Q `[B,1,16,128]`, KV `[B,8192,8,128]`; batches 1/2/4; target used lengths 1/127/256/257/2048/8192; row `i` has `max(1,L-37*i)` tokens; split blocks 128/256/512. These are the real AR decode head/cache dimensions, populated with random tensors. |
| `ar_ops_micro.py` | `ar_ops_micro.py`: seed 2026; BF16 input `[B,1,K]`; batches 1/4; weight `[N,K]` shapes `(4096,2048)` fused QKV, `(12288,2048)` fused gate/up, `(2048,6144)` down projection, and opt-in `(184704,2048)` vocabulary projection. Weights are random BF16 values scaled by 0.02. Launch pairs `(block_m,num_warps)` are `(1,4)/(4,4)/(4,8)/(8,4)/(8,8)/(16,8)`. RMS shapes use `(N,heads)=(128,16)/(2048,1)`, epsilon `1e-6`. |
| `ar_correctness.py` | `ar_correctness.py`: seed 146; random two-layer model, hidden 256, intermediate 512, query/KV heads 4/2, head dimension 64, vocabulary 256, context 1024. Batches 1/2/4; prefixes of length `128-23*i`; seven decode steps per row against independent eager caches, plus the legacy shared-token CFG contract. |
| `vae_verify.py` | `vae_verify.py`: FP32 SnakeBeta `[2,16,1001]`, random input scaled by 3, tolerance `1e-6`. Adds seed 2026, noncontiguous inputs, modest nondefault log-alpha/log-beta values, and explicit throwing-sentinel fallback checks. |
| `vae_compare.py` | `vae_controlled.py` / `vae_spike.py`: FP32 tiled decode, default core 1024 and halo 16; optional original core sweep 1024/64/128/256/512; waveform, 960-sample boundary-neighborhood and 512/2048-point magnitude-STFT diagnostics. No latent supplied or implicitly selected. |

The real-checkpoint `ar_ops_numerics.py` originally intercepted teacher-forced
model activations from a saved song. That private input route is **not** copied.
Its FP32-error, BF16-rounding-mismatch and half-ULP-envelope diagnostics are
instead integrated into the synthetic `ar_ops_micro.py`; `--numerics-only`
disables graph timing. This does not establish full-checkpoint activation parity.

Additions relative to the original small scripts:

- Attention verifies **both eager and graph** exclusion of future NaNs, as well
  as graph replay with GPU used lengths changed to one and restored. A length-one
  reference explicitly repeats each KV head for its two query heads. The graph
  must consume live length tensors rather than capture-time host constants.
- The poisoned-cache oracle is computed from clean finite KV first. A boolean
  SDPA mask is not a reliable NaN oracle: an unused `NaN` can still contaminate a
  fallback matmul. The timed SDPA baseline is now explicitly **SDPA MATH**, not
  auto-selected and not labelled FlashAttention. This deliberate baseline
  pinning means timing is not automatically comparable with an older auto-SDPA
  run on a different runtime.
- `--paired` also tests the optional paired-query kernel; the unpaired kernel
  remains the default. RMS cases include row-gap storage, exercising fused-QKV
  view handling. SnakeBeta tests parameter/state preservation and CPU, training,
  and autograd fallback without routing those calls into an inference-only kernel.

## What “correct” means here

| Case | Acceptance rule | Not implied |
| --- | --- | --- |
| Attention | `torch.testing.assert_close(atol=0.002, rtol=0.02)` against BF16 public SDPA MATH, including poisoned caches and changed graph lengths | Bitwise attention identity, identical sampled songs, or end-to-end throughput |
| AR linear | Against vendor FP32 `F.linear`: `atol=0.003, rtol=0.008`, **and** absolute error `<= 0.501 * local_BF16_ULP + 1e-4`; finite outputs required | Identical vendor BF16 rounding or an exact-real dot-product reference |
| RMSNorm | `atol=0, rtol=0` against the original expression with its BF16 intermediate roundings | A tolerance-based approximation; also, `torch.equal` does not distinguish signed-zero bit patterns |
| Tiny AR graph | Prefill `atol=rtol=0`; decode `atol=0.008, rtol=0.04`; exact integer token/position assertions | Full-sized model, long-context, generation-quality or sampling parity |
| SnakeBeta | FP32 `atol=rtol=1e-6`; unchanged parameter values and state keys; explicit fallback assertions | Universal bitwise agreement of transcendental functions or whole-VAE agreement |
| Saved-latent VAE | Exact shape and finite output; `--atol` / `--rtol` are required caller-selected waveform limits, checked on **every** decode | A historically validated universal waveform tolerance, spectral quality gate or listening test |

AR linear reports max/mean FP32 error for both candidate and vendor BF16 output,
maximum excess over half a local ULP, and each BF16 rounding-mismatch fraction.
The local ULP is the larger spacing to the neighboring BF16 values around the
rounded FP32 result. The extra `1e-4` is the original cancellation/reduction-order
allowance, **not** proof of mathematically correct rounding. Different GPU
reductions may fail a preserved gate; investigate and report that failure rather
than widening the tolerance until it passes. FP32 reference matmul runs with
TF32 disabled.

“Exact context” in tiled VAE decoding means retaining the complete convolution
receptive field without crossfading or synthetic endpoint padding. It does not
mean convolution algorithms, fused sine evaluation, or final waveforms must be
bitwise identical on every GPU/runtime.

## Optional saved-latent VAE comparison

Supply your own local VAE export and `.npy` latent. No remote model IDs, pickle
loading, layout guessing, or waveform export is performed. Choose acceptance
limits appropriate to your experiment; the example values below are illustrative
user policy, not a measured guarantee:

```sh
python benchmarks/vae_compare.py --run-gpu \
  --vae "$VAE_DIR" --latent "$LATENT_NPY" --latent-layout TC \
  --frames 250 --cores 1024,64,128,256,512 \
  --atol 1e-5 --rtol 1e-5 --output results/vae-comparison.json
```

`--latent-layout` is explicit (`BCT`, `BTC`, `CT`, `TC`; default `BCT`). Channel
count must match the loaded config. The original short comparison used 250
latent frames (10 seconds at 25 frames/second); this script only truncates when
`--frames` is supplied. Without it, the entire supplied latent is decoded.
A final waveform can be slightly shorter than nominal frame-rate duration;
expected length is obtained from `model.natural_output_length`, not guessed.

By default the stock decoder at `--reference-core 1024` supplies the reference,
then the reusable fused decoder is tested at each requested core. A supplied
`--reference` must be finite FP32 `.npy` with exactly matching `[B,C,T_audio]`;
even the current stock decoder is checked against it. A saved reference is only
meaningful if you independently match the model, input latent, frame crop, source
revision, precision and runtime policy. The script does not authenticate its
provenance. Reports do not embed the checkpoint/latent paths or array contents.

Convolution benchmark mode is off, determinism is requested by default, and
TF32 is disabled. `--nondeterministic` is explicit. `--miopen-fast` sets
`MIOPEN_FIND_MODE=FAST` **before** GPU initialization; compare solver modes in
separate processes and record the runtime. This script does not claim the
nondeterministic path becomes deterministic by passing a single comparison.

## Timing and reports

Synthetic microbenchmarks warm up/compile on a side stream before graph capture,
perform five untimed replays, then use GPU events for `--repeats` warm replays.
Attention defaults to 40 measured replays; AR primitives and SnakeBeta to 30.
These are average **captured-operation** milliseconds, excluding compilation,
capture, model loading, and Python/eager launch overhead. They are not full-song
benchmarks. The AR integration script does no timing (`--warmup` / `--repeats`
are common CLI options but unused there).

VAE timings are synchronized wall-clock seconds including the output copy to
CPU. Its default is one warmup and one timed call per configuration; reports
retain all calls, the first/cold call, warm median, and peak allocated GPU memory.
The stock-first run can warm shared convolution caches for the candidate; these
are warm-path comparisons, not unbiased cold-start comparisons. STFT and error
analysis happen outside timed regions. Spectral ratios are diagnostic only and
are `null` if the waveform is too short or the reference has zero spectral norm.
Boundary metrics are absent if no internal tile boundary exists.

A final JSON report goes to stdout. Progress rows go to stderr; rows marked
`passed: false` can be pending checks, so consult the final report. `--output`
optionally writes an atomic incremental report; an interrupted/failed run must
not be treated as successful unless `complete` and `passed` are both true.
Runtime Torch/HIP/CUDA, installed package version, GPU name/architecture, seeds,
settings, and numerical metrics are recorded. Save your checkout revision and
container digest alongside the report if comparing runs. Do not commit private
inputs or reports containing information you do not intend to publish.
