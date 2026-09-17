#!/usr/bin/env bash
# Paired 256-step Q32 fine-tuning from the completed box-equalized MTHv2
# checkpoint. Attention and geometry are serial arms; each validates step
# 256 and then performs selection-locked test.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
wrapper="${code_root}/tools/training/run_dunhuang_local_gazetteer_glmocr_a100.sh"
python="${env_dir}/bin/python"

source_run_id="${GLMOCR_DUNHUANG_GATE005_SOURCE_RUN_ID:-glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1}"
attention_run_id="${GLMOCR_DUNHUANG_GATE005_ATTENTION_RUN_ID:-glmocr_dunhuang_local_q32_boxeq820_gate005_lr1e5_attention_256_frommthv2_260916_v1}"
geometry_run_id="${GLMOCR_DUNHUANG_GATE005_GEOMETRY_RUN_ID:-glmocr_dunhuang_local_q32_boxeq820_gate005_lr1e5_geometry_256_frommthv2_260916_v1}"
session="${GLMOCR_DUNHUANG_GATE005_SESSION:-glmocr_dunhuang_local_q32_gate005_attn_geom_260916_v1}"
seed="${GLMOCR_DUNHUANG_GATE005_SEED:-42}"
num_queries="${GLMOCR_DUNHUANG_GATE005_NUM_QUERIES:-32}"
max_steps="${GLMOCR_DUNHUANG_GATE005_STEPS:-256}"
validation_interval="${GLMOCR_DUNHUANG_GATE005_VALIDATION_INTERVAL:-256}"
learning_rate="${GLMOCR_DUNHUANG_GATE005_LR:-1e-5}"
decoder_learning_rate="${GLMOCR_DUNHUANG_GATE005_DECODER_LR:-2e-6}"
warmup_steps="${GLMOCR_DUNHUANG_GATE005_WARMUP_STEPS:-216}"
auxiliary_weight="${GLMOCR_DUNHUANG_GATE005_AUXILIARY_WEIGHT:-0.4}"
layout_loss_profile="${GLMOCR_DUNHUANG_GATE005_LAYOUT_LOSS_PROFILE:-history_box_equalized_v2}"
residual_scale="${GLMOCR_DUNHUANG_GATE005_RESIDUAL_SCALE:-0.005}"
gpu_ids="${GLMOCR_DUNHUANG_GATE005_GPU_IDS:-0,1,2,3,4}"
gpu_utilization_limit="${GLMOCR_DUNHUANG_GATE005_GPU_UTILIZATION_LIMIT:-50}"
max_eval_new_tokens="${GLMOCR_DUNHUANG_GATE005_MAX_EVAL_NEW_TOKENS:-1536}"
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) foreground=1; shift ;;
        *)
            printf '{"event":"glmocr_dunhuang_gate005_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2
            exit 64
            ;;
    esac
done

[[ "${seed}" =~ ^[0-9]+$ && "${num_queries}" == "32" ]] || exit 64
[[ "${max_steps}" =~ ^[1-9][0-9]*$ && "${validation_interval}" == "${max_steps}" ]] || exit 64
[[ "${warmup_steps}" =~ ^[0-9]+$ && "${warmup_steps}" -lt "${max_steps}" ]] || exit 64
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 64
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ && "${max_eval_new_tokens}" =~ ^[1-9][0-9]*$ ]] || exit 64
for value in "${source_run_id}" "${attention_run_id}" "${geometry_run_id}" "${session}"; do
    [[ "${value}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
done
[[ "${attention_run_id}" != "${geometry_run_id}" ]] || exit 64

workspace_runs="${remote_root}/runs"
source_checkpoint="${remote_root}/training_runs/${source_run_id}/seed${seed}/checkpoint-3000"
status_file="${workspace_runs}/${session}.status.json"
summary_file="${workspace_runs}/${session}.summary.json"
current_phase="preflight"
current_run_id="-"

write_status() {
    local status="$1"
    local phase="$2"
    local active_run_id="${3:--}"
    printf '{"status":"%s","phase":"%s","run_id":"%s","session":"%s","source_run_id":"%s","attention_run_id":"%s","geometry_run_id":"%s","seed":%s,"num_queries":%s,"max_steps":%s,"validation_interval":%s,"learning_rate":%s,"decoder_learning_rate":%s,"warmup_steps":%s,"auxiliary_weight":%s,"layout_loss_profile":"%s","residual_scale":%s,"gpu_ids":"%s","test_used_for_selection":false,"updated_at":"%s"}\n' \
        "${status}" "${phase}" "${active_run_id}" "${session}" "${source_run_id}" \
        "${attention_run_id}" "${geometry_run_id}" "${seed}" "${num_queries}" \
        "${max_steps}" "${validation_interval}" "${learning_rate}" \
        "${decoder_learning_rate}" "${warmup_steps}" "${auxiliary_weight}" \
        "${layout_loss_profile}" "${residual_scale}" "${gpu_ids}" "$(date -u +%FT%TZ)" > "${status_file}"
}

on_error() {
    local rc=$?
    write_status failed "${current_phase}_failed" "${current_run_id}" || true
    exit "${rc}"
}
trap on_error ERR

preflight() {
    [[ -x "${python}" && -f "${wrapper}" ]] || exit 66
    [[ -f "${source_checkpoint}/adapter.safetensors" && -f "${source_checkpoint}/decoder_lora.safetensors" ]] || exit 66
    [[ ! -e "${status_file}" && ! -e "${summary_file}" ]] || exit 74
    for run_id in "${attention_run_id}" "${geometry_run_id}"; do
        [[ ! -e "${remote_root}/training_runs/${run_id}" && ! -e "${remote_root}/training_runs/${run_id}_smoke" ]] || {
            printf '{"event":"glmocr_dunhuang_gate005_failed","error":"run_already_exists","run_id":"%s"}\n' "${run_id}" >&2
            exit 74
        }
    done
    "${python}" - "${source_checkpoint}" "${seed}" "${num_queries}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
seed = int(sys.argv[2])
queries = int(sys.argv[3])
metadata_path = checkpoint.parent / "metadata.json"
if not metadata_path.is_file():
    raise SystemExit("source checkpoint metadata is missing")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if metadata.get("status") != "complete":
    raise SystemExit("source run is not complete")
if int(metadata.get("seed", -1)) != seed or int(metadata.get("num_queries", -1)) != queries:
    raise SystemExit("source seed or query count mismatch")
if metadata.get("test_manifest_read") is not False or metadata.get("test_used_for_selection") is not False:
    raise SystemExit("source run does not prove a test-free training boundary")
print(json.dumps({"event": "glmocr_dunhuang_gate005_source_ok", "checkpoint": str(checkpoint)}, separators=(",", ":")))
PY
    mkdir -p "${workspace_runs}"
}

run_arm() {
    local mode="$1"
    local run_id="$2"
    local arm_log="${workspace_runs}/${run_id}.orchestrator.log"
    current_phase="${mode}_pipeline"
    current_run_id="${run_id}"
    write_status running "${current_phase}" "${run_id}"
    export GLMOCR_A100_ROOT="${remote_root}"
    export GLMOCR_A100_CODE_ROOT="${code_root}"
    export GLMOCR_A100_MODEL="${model_dir}"
    export GLMOCR_A100_INIT_CHECKPOINT_DIR="${source_checkpoint}"
    export GLMOCR_A100_INIT_CHECKPOINT_OVERRIDE_RESIDUAL_SCALE="${residual_scale}"
    export GLMOCR_A100_INIT_CHECKPOINT_ALLOW_MODE_MISMATCH=1
    export GLMOCR_A100_INITIAL_RESIDUAL_SCALE="${residual_scale}"
    export GLMOCR_A100_LAYOUT_LOSS_PROFILE="${layout_loss_profile}"
    export GLMOCR_A100_AUXILIARY_WEIGHT="${auxiliary_weight}"
    export GLMOCR_A100_MAX_STEPS="${max_steps}"
    export GLMOCR_A100_LR_SCHEDULE_STEPS="${max_steps}"
    export GLMOCR_A100_VALIDATION_INTERVAL="${validation_interval}"
    export GLMOCR_A100_LR="${learning_rate}"
    export GLMOCR_A100_DECODER_LR="${decoder_learning_rate}"
    export GLMOCR_A100_WARMUP_STEPS="${warmup_steps}"
    export GLMOCR_A100_MIN_LR_RATIO=0.1
    export GLMOCR_A100_NUM_QUERIES="${num_queries}"
    export GLMOCR_A100_SEED="${seed}"
    export GLMOCR_A100_GPU_IDS="${gpu_ids}"
    export GLMOCR_A100_GPU_UTILIZATION_LIMIT="${gpu_utilization_limit}"
    export GLMOCR_A100_MAX_EVAL_NEW_TOKENS="${max_eval_new_tokens}"
    export GLMOCR_A100_GENERATION_MODE=loop_recovery
    export GLMOCR_A100_GLOBAL_STEP_OFFSET=0
    bash "${wrapper}" "${run_id}" "${mode}" > "${arm_log}" 2>&1

    current_phase="${mode}_verify"
    write_status running "${current_phase}" "${run_id}"
    "${python}" - "${remote_root}/training_runs/${run_id}/seed${seed}" "${run_id}" "${mode}" "${source_checkpoint}" "${residual_scale}" "${max_steps}" "${auxiliary_weight}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
run_id = sys.argv[2]
mode = sys.argv[3]
source_checkpoint = sys.argv[4]
residual_scale = float(sys.argv[5])
max_steps = int(sys.argv[6])
auxiliary_weight = float(sys.argv[7])
metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
test = json.loads((run_dir / "locked-test" / "locked_test_summary.json").read_text(encoding="utf-8"))
if metadata.get("status") != "complete" or summary.get("status") != "complete":
    raise SystemExit("fine-tuning arm is incomplete")
if metadata.get("mode") != mode or summary.get("mode") != mode:
    raise SystemExit("fine-tuning mode mismatch")
if metadata.get("test_manifest_read") is not False or summary.get("test_manifest_read") is not False:
    raise SystemExit("fine-tuning read test before locked test")
if selection.get("status") != "complete" or selection.get("selected_step") != max_steps:
    raise SystemExit("validation did not select the only requested checkpoint")
if selection.get("test_used_for_selection") is not False:
    raise SystemExit("validation selection used test")
if test.get("status") != "complete" or test.get("selected_step") != max_steps:
    raise SystemExit("locked test is incomplete or selected the wrong step")
if test.get("test_used_for_selection") is not False:
    raise SystemExit("locked test is not selection-locked")
objective = (summary.get("training") or {}).get("loss_objective") or {}
if not math.isclose(float(objective.get("layout_weight", -1.0)), auxiliary_weight):
    raise SystemExit("unexpected outer layout loss weight")
checkpoint = run_dir / f"checkpoint-{max_steps}"
for name in ("adapter.safetensors", "decoder_lora.safetensors"):
    if not (checkpoint / name).is_file():
        raise SystemExit(f"missing {name} in final checkpoint")
metrics = test.get("metrics") or {}
print(json.dumps({"event": "glmocr_dunhuang_gate005_arm_ok", "run_id": run_id, "mode": mode,
                  "source_checkpoint": source_checkpoint, "residual_scale": residual_scale,
                  "selected_step": max_steps, "metrics": {
                      key: metrics.get(key) for key in (
                          "layout_box_iou", "layout_box_mae", "cer", "exact_page_rate",
                          "generation_eos_hit_rate", "generation_limit_hit_rate",
                          "repeated_cycle_page_rate", "loop_detected_page_rate",
                      ) if key in metrics}}, separators=(",", ":")))
PY
}

write_final_summary() {
    current_phase="complete"
    current_run_id="-"
    write_status complete complete "-"
    "${python}" - "${summary_file}" "${remote_root}" "${attention_run_id}" "${geometry_run_id}" "${source_run_id}" "${seed}" "${num_queries}" "${max_steps}" "${learning_rate}" "${decoder_learning_rate}" "${auxiliary_weight}" "${layout_loss_profile}" "${residual_scale}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
root = Path(sys.argv[2])
attention_id, geometry_id, source_id = sys.argv[3:6]
seed, queries, steps = (int(value) for value in sys.argv[6:9])
learning_rate, decoder_learning_rate = (float(value) for value in sys.argv[9:11])
auxiliary_weight = float(sys.argv[11])
layout_loss_profile = sys.argv[12]
residual_scale = float(sys.argv[13])

def arm(run_id):
    run_dir = root / "training_runs" / run_id / f"seed{seed}"
    selection_path = run_dir / "selection.json"
    test_path = run_dir / "locked-test" / "locked_test_summary.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "mode": test.get("mode"),
        "run_dir": str(run_dir),
        "selected_step": selection.get("selected_step"),
        "selection": str(selection_path),
        "test_summary": str(test_path),
        "metrics": test.get("metrics") or {},
        "test_used_for_selection": test.get("test_used_for_selection"),
    }

payload = {
    "status": "complete",
    "source_run_id": source_id,
    "source_checkpoint": str(root / "training_runs" / source_id / f"seed{seed}" / "checkpoint-3000"),
    "seed": seed,
    "num_queries": queries,
    "max_steps": steps,
    "learning_rate": learning_rate,
    "decoder_learning_rate": decoder_learning_rate,
    "auxiliary_weight": auxiliary_weight,
    "layout_loss_profile": layout_loss_profile,
    "residual_scale": residual_scale,
    "validation_steps": [steps],
    "test_used_for_selection": False,
    "attention": arm(attention_id),
    "geometry": arm(geometry_id),
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({"event": "glmocr_dunhuang_gate005_complete", "summary": str(out),
                  "attention": payload["attention"]["metrics"],
                  "geometry": payload["geometry"]["metrics"]}, separators=(",", ":")))
PY
}

run_inner() {
    preflight
    run_arm attention "${attention_run_id}"
    run_arm geometry "${geometry_run_id}"
    write_final_summary
    trap - ERR
}

if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || {
        printf '{"event":"glmocr_dunhuang_gate005_failed","error":"tmux_missing"}\n' >&2
        exit 69
    }
    preflight
    tmux has-session -t "${session}" 2>/dev/null && {
        printf '{"event":"glmocr_dunhuang_gate005_failed","error":"session_already_exists","session":"%s"}\n' "${session}" >&2
        exit 73
    }
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    mkdir -p "${workspace_runs}"
    command_line="$(printf '%q ' bash "${script_path}" --foreground)"
    tmux new-session -d -s "${session}" \
        "cd $(printf '%q' "${code_root}") && exec ${command_line} >$(printf '%q' "${workspace_runs}/${session}.log") 2>&1"
    printf '{"event":"glmocr_dunhuang_gate005_armed","session":"%s","source_run_id":"%s","attention_run_id":"%s","geometry_run_id":"%s","status":"%s","log":"%s"}\n' \
        "${session}" "${source_run_id}" "${attention_run_id}" "${geometry_run_id}" "${status_file}" "${workspace_runs}/${session}.log"
else
    run_inner
fi
