#!/usr/bin/env bash
set -euo pipefail

dataset_root=""
synthesis_session=""
interval_seconds=3600
once=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root) dataset_root="${2:-}"; shift 2 ;;
        --synthesis-session) synthesis_session="${2:-}"; shift 2 ;;
        --interval-seconds) interval_seconds="${2:-}"; shift 2 ;;
        --once) once=1; shift ;;
        *) printf '{"event":"synthetic_monitor_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ -n "${dataset_root}" && -n "${synthesis_session}" ]] || {
    printf '%s\n' '{"event":"synthetic_monitor_failed","error":"dataset_root_and_synthesis_session_are_required"}' >&2
    exit 64
}
[[ "${interval_seconds}" =~ ^[1-9][0-9]*$ ]] || {
    printf '%s\n' '{"event":"synthetic_monitor_failed","error":"interval_must_be_positive"}' >&2
    exit 64
}

count_images() {
    local split="$1"
    if [[ ! -d "${dataset_root}/${split}/images" ]]; then
        printf '0\n'
        return
    fi
    find "${dataset_root}/${split}/images" -type f | wc -l
}

while :; do
    train_count="$(count_images train)"
    validation_count="$(count_images validation)"
    test_count="$(count_images test)"
    running=false
    if tmux has-session -t "${synthesis_session}" 2>/dev/null; then
        running=true
    fi
    complete=false
    if [[ -f "${dataset_root}/train/manifest.jsonl" \
       && -f "${dataset_root}/validation/manifest.jsonl" \
       && -f "${dataset_root}/test/manifest.jsonl" \
       && "${running}" == false ]]; then
        complete=true
    fi
    printf '{"event":"synthetic_layout_monitor","at":"%s","dataset_root":"%s","synthesis_session":"%s","running":%s,"train_images":%s,"validation_images":%s,"test_images":%s,"ready_for_audit":%s}\n' \
        "$(date --iso-8601=seconds)" "${dataset_root}" "${synthesis_session}" \
        "${running}" "${train_count}" "${validation_count}" "${test_count}" "${complete}"
    if [[ "${complete}" == true || "${once}" == 1 ]]; then
        break
    fi
    sleep "${interval_seconds}"
done
