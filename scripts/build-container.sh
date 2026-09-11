#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf '%s\n' 'Usage: BASE_IMAGE=<qualified-image-or-local-id> [IMAGE=localhost/yue2-gfx1151:latest] [CONTAINER_ENGINE=podman|docker] scripts/build-container.sh' 'No base is assumed. Choose and validate a matched gfx1151 ROCm/PyTorch runtime first.'
    exit 0
fi
[[ $# == 0 ]] || { printf 'Unexpected arguments; use --help\n' >&2; exit 2; }
: "${BASE_IMAGE:?Set BASE_IMAGE explicitly; no public YuE2-tested base is assumed}"
image=${IMAGE:-localhost/yue2-gfx1151:latest}
engine=${CONTAINER_ENGINE:-}
if [[ -z $engine ]]; then
    if command -v podman >/dev/null 2>&1; then engine=podman; else engine=docker; fi
fi
[[ $engine == podman || $engine == docker ]] || { printf 'CONTAINER_ENGINE must be podman or docker\n' >&2; exit 2; }
command -v "$engine" >/dev/null
# Disallow ambiguous registry short names; explicit local image IDs are allowed.
qualified_image() {
    [[ $1 =~ ^(sha256:)?[[:xdigit:]]{64}$ ]] ||
    [[ $1 == */* && ( ${1%%/*} == *.* || ${1%%/*} == *:* || ${1%%/*} == localhost ) ]]
}
qualified_image "$BASE_IMAGE" || { printf 'BASE_IMAGE must be fully qualified (registry/repository:tag or @digest), or an explicit local image ID\n' >&2; exit 2; }
qualified_image "$image" || { printf 'IMAGE must be fully qualified, e.g. localhost/yue2-gfx1151:latest\n' >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
context=$(mktemp -d "${TMPDIR:-/tmp}/yue2-build.XXXXXXXX")
trap 'rm -rf -- "$context"' EXIT
# Never send weights, requests, caches, outputs, credentials, or the parent
# workspace. Copy only distribution inputs, excluding local Python caches.
for item in pyproject.toml README.md LICENSE MODEL_LICENSE NOTICE.md THIRD_PARTY_NOTICES.md requirements-runtime.txt src licenses container; do
    [[ -e $root/$item ]] || { printf 'Missing package input: %s\n' "$item" >&2; exit 2; }
done
tar -C "$root" --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
    -cf - pyproject.toml README.md LICENSE MODEL_LICENSE NOTICE.md THIRD_PARTY_NOTICES.md \
    requirements-runtime.txt src licenses container | tar -C "$context" -xf -
"$engine" build --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    -f "$context/container/Containerfile" -t "$image" "$context"
"$engine" image inspect "$image" --format '{{.Id}}'
printf 'Built %s. Build checks do not validate GPU inference.\n' "$image"
