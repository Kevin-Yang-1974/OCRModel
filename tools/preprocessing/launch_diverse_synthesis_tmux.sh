#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

session=""
content_root=""
output_root=""
seed=20260817
train_pages=8000
validation_pages=1000
test_pages=1000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --content-root) content_root="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --train-pages-per-tier) train_pages="$2"; shift 2 ;;
        --validation-pages-per-tier) validation_pages="$2"; shift 2 ;;
        --test-pages-per-tier) test_pages="$2"; shift 2 ;;
        *) printf '{"event":"diverse_synthesis_launch_failed","error":"unknown argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '%s\n' '{"event":"diverse_synthesis_launch_failed","error":"invalid tmux session"}' >&2
    exit 64
}
for value in "${seed}" "${train_pages}" "${validation_pages}" "${test_pages}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || {
        printf '%s\n' '{"event":"diverse_synthesis_launch_failed","error":"seed/page counts must be integers"}' >&2
        exit 64
    }
done
[[ "${train_pages}" -gt 0 && "${validation_pages}" -gt 0 && "${test_pages}" -gt 0 ]] || {
    printf '%s\n' '{"event":"diverse_synthesis_launch_failed","error":"page counts must be positive"}' >&2
    exit 64
}

python_bin="${OCR_WORKSPACE}/envs/layout-synthesis/bin/python"
content_manifest="${content_root}/content.jsonl"
config="${ocrmodel_root}/config/synthetic_layout.ancient_photo_diverse_v1.json"
log_root="${GOT_LAYOUT_DATA}/_diverse_synthesis_logs"
log_path="${log_root}/${session}.log"

command -v tmux >/dev/null 2>&1 || {
    printf '%s\n' '{"event":"diverse_synthesis_launch_failed","error":"tmux is unavailable"}' >&2
    exit 69
}
[[ -x "${python_bin}" && -f "${content_manifest}" && -f "${config}" ]] || {
    printf '%s\n' '{"event":"diverse_synthesis_launch_failed","error":"python, content manifest, or config is missing"}' >&2
    exit 66
}
[[ "${output_root}" == "${GOT_LAYOUT_DATA}/"* && "${output_root}" != "${GOT_LAYOUT_DATA}/" ]] || {
    printf '%s\n' '{"event":"diverse_synthesis_launch_failed","error":"output must be a child of GOT_LAYOUT_DATA"}' >&2
    exit 64
}
[[ ! -e "${output_root}" ]] || {
    printf '{"event":"diverse_synthesis_launch_failed","error":"output already exists","output":"%s"}\n' "${output_root}" >&2
    exit 73
}
if tmux has-session -t "${session}" 2>/dev/null; then
    printf '{"event":"diverse_synthesis_launch_failed","error":"tmux session already exists","session":"%s"}\n' "${session}" >&2
    exit 73
fi

mkdir -p "${log_root}"
args=(
    "${python_bin}" "${script_dir}/prepare_diverse_synthetic_layout.py"
    --content-manifest "${content_manifest}"
    --content-root "${content_root}"
    --output-root "${output_root}"
    --config "${config}"
    --seed "${seed}"
    --train-pages-per-tier "${train_pages}"
    --validation-pages-per-tier "${validation_pages}"
    --test-pages-per-tier "${test_pages}"
    --chromium-executable /usr/bin/google-chrome
    --progress-every 1000
)
printf -v launch_command '%q ' "${args[@]}"
printf -v quoted_root '%q' "${ocrmodel_root}"
printf -v quoted_log '%q' "${log_path}"
tmux new-session -d -s "${session}" \
    "cd ${quoted_root} && export CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PLAYWRIGHT_BROWSERS_PATH=${OCR_WORKSPACE}/cache/ms-playwright && exec ${launch_command}>${quoted_log} 2>&1"

sleep 5
if ! tmux has-session -t "${session}" 2>/dev/null; then
    tail -n 20 "${log_path}" >&2 || true
    printf '{"event":"diverse_synthesis_launch_failed","error":"tmux process exited during startup","session":"%s","log":"%s"}\n' \
        "${session}" "${log_path}" >&2
    exit 1
fi
pane_pid="$(tmux display-message -p -t "${session}:0.0" '#{pane_pid}')"
printf '{"event":"diverse_synthesis_started","session":"%s","pane_pid":%s,"output":"%s","log":"%s","total_pages":%s,"cuda_visible_devices":""}\n' \
    "${session}" "${pane_pid}" "${output_root}" "${log_path}" \
    "$((3 * (train_pages + validation_pages + test_pages)))"
