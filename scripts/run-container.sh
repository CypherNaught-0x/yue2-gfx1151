#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: scripts/run-container.sh --model DIR --vae DIR --request FILE --output DIR [--source DIR] [--dry-run] [--smoke]' \
        'Environment: IMAGE=localhost/yue2-gfx1151:latest; CONTAINER_ENGINE=podman|docker' \
        'All four host paths are required. --dry-run performs CLI validation without GPU devices.' \
        'Normal generation exposes /dev/kfd and /dev/dri. No host workspace or ROCm libraries are mounted.'
}
fail() { printf '%s\n' "$*" >&2; exit 2; }
model= vae= request= output= source= dry_run=0 smoke=0
while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=1; shift ;;
        --smoke) smoke=1; shift ;;
        --model|--vae|--request|--output|--source)
            (( $# >= 2 )) && [[ -n $2 && $2 != --* ]] || fail "Missing value for $1"
            key=${1#--}
            [[ -z ${!key} ]] || fail "Duplicate option: $1"
            printf -v "$key" '%s' "$2"
            shift 2 ;;
        *) fail "Unknown argument: $1 (use --help)" ;;
    esac
done
[[ -n $model && -n $vae && -n $request && -n $output ]] || fail 'Explicit --model, --vae, --request and --output paths are required'
[[ -d $model && -d $vae ]] || fail 'Model and VAE must be existing directories'
[[ -f $request ]] || fail 'Request must be an existing file'
[[ -z $source || -d $source ]] || fail 'Source must be an existing source directory'
# Existing output avoids implicit host filesystem creation and typo surprises.
[[ -d $output && -w $output ]] || fail 'Output must be an existing writable directory; create it explicitly first'
model=$(realpath -- "$model"); vae=$(realpath -- "$vae")
request=$(realpath -- "$request"); output=$(realpath -- "$output")
[[ -z $source ]] || source=$(realpath -- "$source")
for path in "$model" "$vae" "$request" "$output" "$source"; do
    [[ $path != *','* && $path != *$'\n'* && $path != *'"'* ]] || fail 'Mount paths cannot contain commas, double quotes, or newlines'
done
[[ $output != / && $output != "$HOME" ]] || fail 'Refusing broad output mount'
for input in "$model" "$vae" "$request" "$source"; do
    [[ -z $input ]] && continue
    [[ $input != / && $input != "$HOME" ]] || fail 'Refusing broad input mount'
    [[ $output != "$input" && $output != "$input/"* && $input != "$output/"* ]] || fail 'Output and input paths must not overlap'
done
engine=${CONTAINER_ENGINE:-}
if [[ -z $engine ]]; then
    if command -v podman >/dev/null 2>&1; then engine=podman; else engine=docker; fi
fi
[[ $engine == podman || $engine == docker ]] || fail 'CONTAINER_ENGINE must be podman or docker'
command -v "$engine" >/dev/null
image=${IMAGE:-localhost/yue2-gfx1151:latest}
[[ $image == */* && ( ${image%%/*} == *.* || ${image%%/*} == *:* || ${image%%/*} == localhost ) ]] || fail 'IMAGE must be a fully qualified reference'
args=(run --rm --network=none --security-opt=label=disable --cap-drop=ALL
    --workdir /app --env PYTHONPATH= --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/yue2-cache --env HF_HOME=/tmp/yue2-cache/huggingface
    --mount "type=bind,src=$model,dst=/inputs/model,readonly"
    --mount "type=bind,src=$vae,dst=/inputs/vae,readonly"
    --mount "type=bind,src=$request,dst=/inputs/request.json,readonly"
    --mount "type=bind,src=$output,dst=/output")
if [[ $engine == podman ]]; then
    args+=(--userns=keep-id --user "$(id -u):$(id -g)")
else
    args+=(--user "$(id -u):$(id -g)")
fi
if (( ! dry_run )); then
    [[ -e /dev/kfd && -d /dev/dri ]] || fail 'ROCm devices /dev/kfd and /dev/dri are required for generation'
    args+=(--device /dev/kfd --device /dev/dri --ipc=host)
    if [[ $engine == podman ]]; then
        args+=(--group-add keep-groups)
    else
        # Docker has no keep-groups: pass the caller's numeric supplementary IDs.
        for gid in $(id -G); do args+=(--group-add "$gid"); done
    fi
fi
cmd=(generate --model /inputs/model --vae /inputs/vae --request /inputs/request.json --output /output)
if [[ -n $source ]]; then
    args+=(--mount "type=bind,src=$source,dst=/inputs/source,readonly")
    cmd+=(--source /inputs/source)
fi
(( ! smoke )) || cmd+=(--smoke)
(( ! dry_run )) || cmd+=(--dry-run)
exec "$engine" "${args[@]}" "$image" "${cmd[@]}"
