#!/usr/bin/env bash
# Serial q32 comparison: current GLMOCR geometry versus official GLMOCR
# content-only baseline, both on a100-yky and the same five physical GPUs.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
session="${GLMOCR_Q32_COMPARE_SESSION:-glmocr_dunhuang_local_gazetteer_q32_compare_260913_v4}"
current_run_id="${GLMOCR_Q32_CURRENT_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_geometry_2k_a100_260913_v4}"
baseline_run_id="${GLMOCR_Q32_BASELINE_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_official_content_only_2k_a100_260913_v4}"
smoke_current_run_id="${GLMOCR_Q32_SMOKE_CURRENT_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_geometry_smoke_gate_260913_v4}"
smoke_baseline_run_id="${GLMOCR_Q32_SMOKE_BASELINE_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_official_content_only_smoke_gate_260913_v4}"
runner="${code}/tools/training/run_dunhuang_local_gazetteer_glmocr_a100.sh"
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) foreground=1; shift ;;
        *) printf '{"event":"glmocr_q32_compare_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "${current_run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${baseline_run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '{"event":"glmocr_q32_compare_failed","error":"invalid_run_id"}\n' >&2
    exit 64
}
[[ "${current_run_id}" != "${baseline_run_id}" ]] || {
    printf '{"event":"glmocr_q32_compare_failed","error":"run_ids_must_differ"}\n' >&2
    exit 64
}

compare_log="${remote_root}/runs/${session}.log"
compare_status="${remote_root}/runs/${session}.status.json"
compare_summary="${remote_root}/runs/${session}.summary.json"

if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || {
        printf '{"event":"glmocr_q32_compare_failed","error":"tmux_missing"}\n' >&2
        exit 69
    }
    tmux has-session -t "${session}" 2>/dev/null && {
        printf '{"event":"glmocr_q32_compare_failed","error":"session_already_exists","session":"%s"}\n' "${session}" >&2
        exit 73
    }
    mkdir -p "${remote_root}/runs"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    command_line="$(printf '%q ' bash "${script_path}" --foreground)"
    tmux new-session -d -s "${session}" "cd $(printf '%q' "${code}") && exec ${command_line} >$(printf '%q' "${compare_log}") 2>&1"
    printf '{"event":"glmocr_q32_compare_armed","session":"%s","log":"%s","status":"%s","summary":"%s","current_run_id":"%s","baseline_run_id":"%s","serial":true}\n' \
        "${session}" "${compare_log}" "${compare_status}" "${compare_summary}" "${current_run_id}" "${baseline_run_id}"
    exit 0
fi

[[ -f "${runner}" ]] || {
    printf '{"event":"glmocr_q32_compare_failed","error":"runner_missing","path":"%s"}\n' "${runner}" >&2
    exit 66
}
mkdir -p "${remote_root}/runs" "${remote_root}/training_runs"
for run_id in \
    "${smoke_current_run_id}" "${smoke_current_run_id}_smoke" \
    "${smoke_baseline_run_id}" "${smoke_baseline_run_id}_smoke" \
    "${current_run_id}" "${current_run_id}_smoke" \
    "${baseline_run_id}" "${baseline_run_id}_smoke"; do
    [[ ! -e "${remote_root}/training_runs/${run_id}" ]] || {
        printf '{"event":"glmocr_q32_compare_failed","error":"run_already_exists","run_id":"%s"}\n' "${run_id}" >&2
        exit 74
    }
done

write_status() {
    printf '{"status":"%s","phase":"%s","run_id":"%s","mode":"%s","updated_at":"%s","current_run_id":"%s","baseline_run_id":"%s","serial":true}\n' \
        "$1" "$2" "$3" "$4" "$(date -u +%FT%TZ)" "${current_run_id}" "${baseline_run_id}" > "${compare_status}"
}

run_one() {
    local label="$1"
    local mode="$2"
    local run_id="$3"
    local stop_after_smoke="$4"
    local log_path="${remote_root}/runs/${run_id}.pipeline.log"
    write_status running "${label}" "${run_id}" "${mode}"
    set +e
    GLMOCR_A100_STOP_AFTER_SMOKE="${stop_after_smoke}" \
        bash "${runner}" "${run_id}" "${mode}" > "${log_path}" 2>&1
    local rc=$?
    set -e
    if (( rc == 0 )); then
        write_status complete "${label}_complete" "${run_id}" "${mode}"
    else
        write_status failed "${label}_failed" "${run_id}" "${mode}"
    fi
    printf '{"label":"%s","mode":"%s","run_id":"%s","return_code":%s,"log":"%s"}\n' \
        "${label}" "${mode}" "${run_id}" "${rc}" "${log_path}"
    return "${rc}"
}

smoke_current_rc=0
run_one smoke_current geometry "${smoke_current_run_id}" 1 || smoke_current_rc=$?
smoke_baseline_rc=0
run_one smoke_official_content_only content_only "${smoke_baseline_run_id}" 1 || smoke_baseline_rc=$?

if (( smoke_current_rc != 0 || smoke_baseline_rc != 0 )); then
    printf '{"status":"failed","phase":"smoke_gate_failed","smoke_current":{"run_id":"%s","return_code":%s},"smoke_baseline":{"run_id":"%s","return_code":%s},"formal_started":false,"test_used_for_selection":false}\n' \
        "${smoke_current_run_id}" "${smoke_current_rc}" "${smoke_baseline_run_id}" "${smoke_baseline_rc}" \
        > "${compare_summary}"
    write_status failed smoke_gate_failed "-" "-"
    cat "${compare_summary}"
    exit 1
fi

current_rc=0
run_one current geometry "${current_run_id}" 0 || current_rc=$?
baseline_rc=0
run_one official_content_only content_only "${baseline_run_id}" 0 || baseline_rc=$?

overall_status="complete"
if (( current_rc != 0 || baseline_rc != 0 )); then
    overall_status="failed"
fi
printf '{"status":"%s","smoke_gate":{"status":"complete","current":{"run_id":"%s","mode":"geometry","return_code":%s},"baseline":{"run_id":"%s","mode":"content_only","return_code":%s}},"formal_started":true,"current":{"run_id":"%s","mode":"geometry","return_code":%s,"summary":"%s","test_summary":"%s"},"baseline":{"run_id":"%s","mode":"content_only","return_code":%s,"summary":"%s","test_summary":"%s"},"dataset":"dunhuang_local_gazetteer_q32_v1","seed":42,"num_queries":32,"max_steps":2000,"checkpoint_steps":[500,1000,1500,2000],"test_used_for_selection":false,"serial":true}\n' \
    "${overall_status}" \
    "${smoke_current_run_id}" "${smoke_current_rc}" "${smoke_baseline_run_id}" "${smoke_baseline_rc}" \
    "${current_run_id}" "${current_rc}" "${remote_root}/training_runs/${current_run_id}/seed42/summary.json" "${remote_root}/training_runs/${current_run_id}/seed42/locked-test/locked_test_summary.json" \
    "${baseline_run_id}" "${baseline_rc}" "${remote_root}/training_runs/${baseline_run_id}/seed42/summary.json" "${remote_root}/training_runs/${baseline_run_id}/seed42/locked-test/locked_test_summary.json" \
    > "${compare_summary}"
write_status "${overall_status}" complete "-" "-"
cat "${compare_summary}"
(( current_rc == 0 && baseline_rc == 0 ))
