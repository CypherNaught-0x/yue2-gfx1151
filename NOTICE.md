# Attribution and scope

This is an unofficial derivative of YuE2 (YuE by HKUST / M-A-P):
https://github.com/multimodal-art-projection/YuE

Upstream reference: `8e06871aa2e704d87ffb9bc71b5f5420f6813724` (YuE2 0.1.6).
`src/yue2/` vendors the minimal Python inference package and retains upstream
license and third-party notices. Modifications add ROCm public-SDPA compatibility,
Triton AR attention/GEMV/RMSNorm and projection fusion, isolated independent-row
AR batching, fused SnakeBeta and a reusable deterministic FP32 VAE decoder.
`docs/source-provenance.json` lists per-file hashes and upstream differences.
The portable launcher, benchmark adapters and container tooling are new additions.

The YuE2 derivative code is CC BY-NC 4.0, **not commercially licensed**.
Retain attribution, link the license, and indicate modifications on redistribution.
Third-party MIT portions retain their original notices and terms; see
THIRD_PARTY_NOTICES.md and licenses/. Weights are not included and have their own
MODEL_LICENSE terms. Generated outputs should be attributed as YuE2-generated
according to upstream guidance. No rights in input recordings or lyrics are
conveyed. This repository includes no recordings, checkpoints or user requests.
