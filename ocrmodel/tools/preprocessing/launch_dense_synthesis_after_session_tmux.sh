#!/usr/bin/env bash
# Wait for an existing immutable synthesis session, then launch a new dataset root.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

session="layout_s3s4_dense_wait_20260827_v4"
wait_session="layout_s3s4_formal_20260826_v1"
synthesis_session="layout_s3s4_dense_20260827_v4"
content_root="${GOT_LAYOUT_DATA}/formal_pdf_short_seed20260812/source"
output_root="${GOT_LAYOUT_DATA}/ancient_photo_diverse_formal_s3s4_dense_20260827_v4"
smoke_root="${GOT_LAYOUT_DATA}/dense_synthesis_renderer_smoke_20260827_v3"
interval_seconds=3600
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --wait-session) wait_session="$2"; shift 2 ;;
        --synthesis-session) synthesis_session="$2"; shift 2 ;;
        --content-root) content_root="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --smoke-root) smoke_root="$2"; shift 2 ;;
        --interval-seconds) interval_seconds="$2"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        *) printf '{"event":"dense_synthesis_waiter_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

for value in "$session" "$wait_session" "$synthesis_session"; do
    [[ "$value" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf '%s\n' '{"event":"dense_synthesis_waiter_failed","error":"invalid_session"}' >&2; exit 64; }
done
[[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '{"event":"dense_synthesis_waiter_failed","error":"invalid_interval"}' >&2; exit 64; }
[[ "$output_root" == "${GOT_LAYOUT_DATA}/"* && "$output_root" != "${GOT_LAYOUT_DATA}/" ]] || {
    printf '%s\n' '{"event":"dense_synthesis_waiter_failed","error":"invalid_output_root"}' >&2; exit 64;
}
[[ "$smoke_root" == "${GOT_LAYOUT_DATA}/"* && "$smoke_root" != "${GOT_LAYOUT_DATA}/" ]] || {
    printf '%s\n' '{"event":"dense_synthesis_waiter_failed","error":"invalid_smoke_root"}' >&2; exit 64;
}

run_renderer_smoke() {
    [[ ! -e "$smoke_root" ]] || {
        printf '{"event":"dense_synthesis_renderer_smoke_failed","error":"smoke_output_already_exists","output":"%s"}\n' "$smoke_root" >&2
        return 1
    }
    local python_bin="${OCR_WORKSPACE}/envs/layout-synthesis/bin/python"
    "$python_bin" "${script_dir}/generate_synthetic_layout.py" \
        --content-manifest "${content_root}/content.jsonl" \
        --content-root "$content_root" --output-dir "$smoke_root" --split train \
        --tier s3-ancient-hard --tier s4-mixed --num-pages 64 --seed 20260827 \
        --config "${ocrmodel_root}/config/synthetic_layout.ancient_photo_diverse_v1.json" \
        --chromium-executable /usr/bin/google-chrome --progress-every 32
    "$python_bin" "${script_dir}/audit_synthetic_layout.py" \
        --manifest "${smoke_root}/manifest.jsonl" --summary-json "${smoke_root}/audit_summary.json"
    "$python_bin" - "${smoke_root}/audit_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
buckets = summary.get("region_count_bucket_counts", {})
if summary.get("status") != "ok" or summary.get("page_count") != 128:
    raise SystemExit("dense renderer smoke audit did not complete 128 valid pages")
if not any(buckets.get(bucket, 0) for bucket in ("33-64", "65-128", ">128")):
    raise SystemExit("dense renderer smoke emitted no high-region pages")
print(json.dumps({"event": "dense_synthesis_renderer_smoke_passed", "pages": 128, "buckets": buckets}, separators=(",", ":")))
PY
}

run_waiter() {
    while tmux has-session -t "$wait_session" 2>/dev/null; do
        printf '{"event":"dense_synthesis_waiting_for_prior_session","at":"%s","session":"%s","interval_seconds":%s}\n' \
            "$(date --iso-8601=seconds)" "$wait_session" "$interval_seconds"
        sleep "$interval_seconds"
    done
    run_renderer_smoke
    [[ ! -e "$output_root" ]] || {
        printf '{"event":"dense_synthesis_waiter_failed","error":"output_already_exists","output":"%s"}\n' "$output_root" >&2
        return 1
    }
    exec bash "${script_dir}/launch_diverse_synthesis_tmux.sh" \
        --session "$synthesis_session" \
        --content-root "$content_root" \
        --output-root "$output_root" \
        --seed 20260827 \
        --train-pages-per-tier 12500 \
        --validation-pages-per-tier 2000 \
        --test-pages-per-tier 2000 \
        --min-train-pages-total 10000 \
        --target-train-region-exposures 1000000 \
        --tier s3-ancient-hard --tier s4-mixed
}

if (( session_inner == 1 )); then
    run_waiter
    exit
fi

command -v tmux >/dev/null 2>&1 || { printf '%s\n' '{"event":"dense_synthesis_waiter_failed","error":"tmux_unavailable"}' >&2; exit 69; }
tmux has-session -t "$session" 2>/dev/null && {
    printf '{"event":"dense_synthesis_waiter_failed","error":"tmux_session_exists","session":"%s"}\n' "$session" >&2; exit 73;
}
script_path="$(realpath "${BASH_SOURCE[0]}")"
log_root="${GOT_LAYOUT_DATA}/_diverse_synthesis_logs"
log_path="${log_root}/${session}.log"
mkdir -p "$log_root"
tmux new-session -d -s "$session" \
    "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --wait-session '${wait_session}' --synthesis-session '${synthesis_session}' --content-root '${content_root}' --output-root '${output_root}' --smoke-root '${smoke_root}' --interval-seconds '${interval_seconds}' >'${log_path}' 2>&1"
sleep 3
tmux has-session -t "$session" 2>/dev/null || { tail -n 20 "$log_path" >&2 || true; exit 1; }
printf '{"event":"dense_synthesis_waiter_armed","session":"%s","wait_session":"%s","output":"%s","log":"%s"}\n' \
    "$session" "$wait_session" "$output_root" "$log_path"
