#!/usr/bin/env bash
# Serial selection-locked test for the q32 geometry/content-only comparison.
# This is the only stage that reads the test manifest.
set -Eeuo pipefail

root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code="${GLMOCR_A100_CODE_ROOT:-${root}/code/ocrmodel}"
python="${GLMOCR_A100_ENV:-${root}/envs/glmocr_a100_py311_cu128}/bin/python"
dataset="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
session="${GLMOCR_Q32_TEST_SESSION:-glmocr_q32_locked_test_260913_v1}"
current_run_id="${GLMOCR_Q32_CURRENT_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_geometry_2k_a100_260913_v4}"
baseline_run_id="${GLMOCR_Q32_BASELINE_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_official_content_only_2k_a100_260913_v4}"
seed="${GLMOCR_A100_SEED:-42}"
gpu_ids="${GLMOCR_A100_GPU_IDS:-0,1,2,3,4}"
num_queries="${GLMOCR_A100_NUM_QUERIES:-32}"
max_eval_new_tokens="${GLMOCR_A100_MAX_EVAL_NEW_TOKENS:-1536}"
dataset_label="dunhuang_local_gazetteer_q32_v1"
protocol_label="glm_ocr_dunhuang_local_gazetteer_group_isolated_v1"
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) foreground=1; shift ;;
        *) exit 64 ;;
    esac
done

[[ "${current_run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${baseline_run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ && "${seed}" =~ ^[0-9]+$ ]] || exit 64
[[ "${num_queries}" =~ ^[1-9][0-9]*$ && "${max_eval_new_tokens}" =~ ^[1-9][0-9]*$ ]] || exit 64

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
gpu_count="${#gpu_array[@]}"
(( gpu_count == 5 )) || {
    printf '{"event":"glmocr_q32_locked_test_failed","error":"five_gpus_required","gpu_ids":"%s"}\n' "${gpu_ids}" >&2
    exit 64
}
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ && -z "${seen_gpu[${gpu}]+present}" ]] || exit 64
    seen_gpu[${gpu}]=1
done

run_root="${root}/training_runs"
train_manifest="${dataset}/train/manifest.jsonl"
validation_manifest="${dataset}/validation/manifest.jsonl"
test_manifest="${dataset}/test/manifest.jsonl"
status_file="${root}/runs/${session}.status.json"
summary_file="${root}/runs/${session}.summary.json"
current_phase="preflight"

write_status() {
    printf '{"status":"%s","phase":"%s","run_id":"%s","mode":"%s","updated_at":"%s","current_run_id":"%s","baseline_run_id":"%s","gpu_ids":"%s","serial":true}\n' \
        "$1" "$2" "$3" "$4" "$(date -u +%FT%TZ)" \
        "${current_run_id}" "${baseline_run_id}" "${gpu_ids}" > "${status_file}"
}

on_error() {
    local rc=$?
    write_status failed "${current_phase}_failed" "-" "-" || true
    exit "${rc}"
}
trap on_error ERR

validate_run_inputs() {
    local run_id="$1"
    local mode="$2"
    local group="${run_root}/${run_id}"
    local run_dir="${group}/seed${seed}"
    local selection_file="${group}/selection.json"
    [[ -d "${run_dir}" && -f "${run_dir}/COMPLETED" && -f "${run_dir}/metadata.json" && -f "${run_dir}/summary.json" ]] || exit 66
    [[ -f "${selection_file}" ]] || exit 66
    [[ ! -e "${run_dir}/locked-test" && ! -e "${run_dir}/locked-test-shards" ]] || exit 74
    "${python}" - "${run_dir}/metadata.json" "${run_dir}/summary.json" "${selection_file}" "${mode}" "${num_queries}" <<'PY'
import json
import sys
metadata = json.loads(open(sys.argv[1], encoding="utf-8").read())
summary = json.loads(open(sys.argv[2], encoding="utf-8").read())
selection = json.loads(open(sys.argv[3], encoding="utf-8").read())
if metadata.get("status") != "complete" or summary.get("status") != "complete":
    raise SystemExit("training run is not complete")
if metadata.get("mode") != sys.argv[4] or int(metadata.get("num_queries", -1)) != int(sys.argv[5]):
    raise SystemExit("training mode or query count mismatch")
if metadata.get("test_manifest_read") is not False or summary.get("test_manifest_read") is not False:
    raise SystemExit("training protocol is not test-free")
if metadata.get("test_used_for_selection") is not False or summary.get("test_used_for_selection") is not False:
    raise SystemExit("training selection is not test-free")
if selection.get("status") != "complete" or selection.get("test_used_for_selection") is not False:
    raise SystemExit("validation selection is not complete and locked")
if not isinstance(selection.get("selected_step"), int) or selection["selected_step"] <= 0:
    raise SystemExit("validation selection has no valid checkpoint")
PY
}

prepare_test_protocol() {
    local run_id="$1"
    local protocol_file="${root}/protocols/${run_id}.test_locked.json"
    mkdir -p "${root}/protocols" "${run_root}/${run_id}/logs"
    if [[ ! -f "${protocol_file}" ]]; then
        "${python}" "${code}/tools/audit_mthv2_manifest.py" \
            --train-manifest "${train_manifest}" \
            --validation-manifest "${validation_manifest}" \
            --test-manifest "${test_manifest}" \
            --num-queries "${num_queries}" \
            --dataset-label "${dataset_label}" \
            --protocol-label "${protocol_label}" \
            --allow-count-mismatch \
            --output "${protocol_file}" \
            > "${run_root}/${run_id}/logs/test-protocol-audit.log" 2>&1
    fi
    "${python}" - "${protocol_file}" "${train_manifest}" "${validation_manifest}" "${test_manifest}" <<'PY' >/dev/null
import hashlib
import json
import sys
from pathlib import Path
protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
paths = {"train": Path(sys.argv[2]), "validation": Path(sys.argv[3]), "test": Path(sys.argv[4])}
if protocol.get("dataset") != "dunhuang_local_gazetteer_q32_v1":
    raise SystemExit("wrong dataset label")
if protocol.get("protocol") != "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1":
    raise SystemExit("wrong protocol label")
if protocol.get("num_queries") != 32 or protocol.get("test_manifest_read") is not True:
    raise SystemExit("invalid test protocol boundary")
if protocol.get("test_used_for_selection") is not False:
    raise SystemExit("test protocol is not selection-locked")
if protocol.get("split_pages") != {"train": 240, "validation": 80, "test": 59}:
    raise SystemExit("unexpected q32 split counts")
for split, path in paths.items():
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = ((protocol.get("manifest_stats") or {}).get(split) or {}).get("manifest_sha256")
    if digest != recorded:
        raise SystemExit(f"{split} manifest fingerprint changed")
PY
    printf '%s\n' "${protocol_file}"
}

verify_test_summary() {
    local run_id="$1"
    local mode="$2"
    local protocol_file="$3"
    local run_dir="${run_root}/${run_id}/seed${seed}"
    local summary_path="${run_dir}/locked-test/locked_test_summary.json"
    local selection_path="${run_root}/${run_id}/selection.json"
    local expected_pages
    expected_pages="$("${python}" - "${protocol_file}" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["split_pages"]["test"])
PY
)"
    [[ -f "${summary_path}" && -f "${run_dir}/locked-test/LOCKED_TEST_COMPLETED" ]] || exit 1
    "${python}" - "${summary_path}" "${selection_path}" "${mode}" "${seed}" "${num_queries}" "${gpu_count}" "${expected_pages}" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selection = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if summary.get("status") != "complete" or summary.get("split") != "test":
    raise SystemExit("locked test summary is incomplete")
if summary.get("mode") != sys.argv[3] or summary.get("seed") != int(sys.argv[4]):
    raise SystemExit("locked test mode or seed mismatch")
if summary.get("num_queries") != int(sys.argv[5]) or summary.get("test_pages") != int(sys.argv[7]):
    raise SystemExit("locked test pages or query count mismatch")
if summary.get("test_pages_total") != int(sys.argv[7]) or summary.get("test_shard_count") != int(sys.argv[6]):
    raise SystemExit("locked test shard coverage mismatch")
if summary.get("test_shard_gpu_ids") != "0,1,2,3,4":
    raise SystemExit("locked test did not use the five requested GPUs")
if summary.get("test_used_for_selection") is not False:
    raise SystemExit("locked test is not selection-locked")
if summary.get("selected_step") != selection.get("selected_step"):
    raise SystemExit("test step differs from validation selection")
if summary.get("decoder_adaptation") != "lora" or summary.get("decoder_lora_loaded") is not True:
    raise SystemExit("decoder LoRA was not loaded")
metrics = summary.get("metrics") or {}
if metrics.get("test_used_for_selection") is not False or metrics.get("pages") != int(sys.argv[7]):
    raise SystemExit("test metrics do not prove complete coverage")
print(json.dumps({"event": "glmocr_q32_locked_test_verified", "selected_step": summary["selected_step"], "test_pages": summary["test_pages"], "metrics": metrics}, ensure_ascii=False, separators=(",", ":")))
PY
}

run_one() {
    local label="$1"
    local mode="$2"
    local run_id="$3"
    local protocol_file="$4"
    local log_path="${root}/runs/${run_id}.locked-test.pipeline.log"
    write_status running "${label}_test" "${run_id}" "${mode}"
    set +e
    bash "${code}/tools/training/run_glmocr_mthv2_locked_test.sh" \
        --run-id "${run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" \
        --mode "${mode}" --num-queries "${num_queries}" \
        --max-eval-new-tokens "${max_eval_new_tokens}" \
        --dataset-root "${dataset}" --protocol-file "${protocol_file}" \
        --model-dir "${model_dir}" --foreground \
        > "${log_path}" 2>&1
    local rc=$?
    set -e
    if (( rc != 0 )); then
        write_status failed "${label}_test_failed" "${run_id}" "${mode}"
        printf '{"event":"glmocr_q32_locked_test_failed","label":"%s","run_id":"%s","return_code":%s,"log":"%s"}\n' \
            "${label}" "${run_id}" "${rc}" "${log_path}" >&2
        return "${rc}"
    fi
    verify_test_summary "${run_id}" "${mode}" "${protocol_file}"
}

if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || exit 69
    tmux has-session -t "${session}" 2>/dev/null && exit 73
    mkdir -p "${root}/runs"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    command_line="$(printf '%q ' env \
        GLMOCR_A100_ROOT="${root}" \
        GLMOCR_A100_CODE_ROOT="${code}" \
        GLMOCR_A100_ENV="$(dirname "$(dirname "${python}")")" \
        GLMOCR_A100_MTHV2_ROOT="${dataset}" \
        GLMOCR_A100_MODEL="${model_dir}" \
        GLMOCR_Q32_TEST_SESSION="${session}" \
        GLMOCR_Q32_CURRENT_RUN_ID="${current_run_id}" \
        GLMOCR_Q32_BASELINE_RUN_ID="${baseline_run_id}" \
        GLMOCR_A100_SEED="${seed}" \
        GLMOCR_A100_GPU_IDS="${gpu_ids}" \
        GLMOCR_A100_NUM_QUERIES="${num_queries}" \
        GLMOCR_A100_MAX_EVAL_NEW_TOKENS="${max_eval_new_tokens}" \
        bash "${script_path}" --foreground)"
    tmux new-session -d -s "${session}" \
        "cd $(printf '%q' "${code}") && exec ${command_line} >$(printf '%q' "${root}/runs/${session}.log") 2>&1"
    printf '{"event":"glmocr_q32_locked_test_armed","session":"%s","gpu_ids":"%s","current_run_id":"%s","baseline_run_id":"%s","status":"%s","summary":"%s","serial":true}\n' \
        "${session}" "${gpu_ids}" "${current_run_id}" "${baseline_run_id}" "${status_file}" "${summary_file}"
    exit 0
fi

mkdir -p "${root}/runs" "${root}/protocols"
[[ -x "${python}" && -f "${code}/tools/audit_mthv2_manifest.py" && -f "${code}/tools/training/run_glmocr_mthv2_locked_test.sh" ]] || exit 66
[[ -f "${train_manifest}" && -f "${validation_manifest}" && -f "${test_manifest}" ]] || exit 66

validate_run_inputs "${current_run_id}" geometry
validate_run_inputs "${baseline_run_id}" content_only
current_test_protocol="$(prepare_test_protocol "${current_run_id}")"
baseline_test_protocol="$(prepare_test_protocol "${baseline_run_id}")"

current_phase="geometry_test"
run_one geometry geometry "${current_run_id}" "${current_test_protocol}"
current_phase="baseline_test"
run_one official_content_only content_only "${baseline_run_id}" "${baseline_test_protocol}"

"${python}" - "${summary_file}" "${current_run_id}" "${baseline_run_id}" "${current_test_protocol}" "${baseline_test_protocol}" <<'PY'
import json
import sys
from pathlib import Path
output = Path(sys.argv[1])
run_ids = sys.argv[2:4]
protocols = sys.argv[4:6]
root = output.parents[1]
payload = {
    "status": "complete",
    "session": output.stem.replace(".summary", ""),
    "dataset": "dunhuang_local_gazetteer_q32_v1",
    "gpu_ids": "0,1,2,3,4",
    "gpu_count": 5,
    "num_queries": 32,
    "test_used_for_selection": False,
    "serial": True,
}
for run_id, protocol in zip(run_ids, protocols):
    path = root / "training_runs" / run_id / "seed42" / "locked-test" / "locked_test_summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    payload["geometry" if value.get("mode") == "geometry" else "official_content_only"] = {
        "run_id": run_id,
        "mode": value.get("mode"),
        "selected_step": value.get("selected_step"),
        "protocol": protocol,
        "test_summary": str(path),
        "metrics": value.get("metrics") or {},
        "test_used_for_selection": value.get("test_used_for_selection"),
    }
if set(payload) & {"geometry", "official_content_only"} != {"geometry", "official_content_only"}:
    raise SystemExit("final comparison summary is missing one test arm")
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event": "glmocr_q32_locked_test_complete", "summary": str(output), "test_used_for_selection": False}, separators=(",", ":")))
PY
current_phase="complete"
write_status complete complete "-" "-"
cat "${summary_file}"
