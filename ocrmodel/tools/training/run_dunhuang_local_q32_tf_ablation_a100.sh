#!/usr/bin/env bash
# Three-mode q32 teacher-forcing ablation on a100-yky.
#
# The modes are trained serially on the same five physical A100s so that the
# effective batch, backend, and resource conditions are identical.  Training
# and the smoke gate use a protocol without the test manifest.  After all
# three 256-step runs finish, each fixed final checkpoint is evaluated once on
# the held-out test split; no validation selection is performed in this
# diagnostic.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dataset_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"
gpu_ids="${GLMOCR_Q32_TF_GPU_IDS:-0,1,2,3,4}"
seed="${GLMOCR_Q32_TF_SEED:-42}"
num_queries="${GLMOCR_Q32_TF_NUM_QUERIES:-32}"
max_steps="${GLMOCR_Q32_TF_MAX_STEPS:-256}"
lr_schedule_steps="${GLMOCR_Q32_TF_LR_SCHEDULE_STEPS:-256}"
warmup_steps="${GLMOCR_Q32_TF_WARMUP_STEPS:-216}"
learning_rate="${GLMOCR_Q32_TF_LR:-2.5e-5}"
decoder_learning_rate="${GLMOCR_Q32_TF_DECODER_LR:-5e-6}"
min_lr_ratio="${GLMOCR_Q32_TF_MIN_LR_RATIO:-0.1}"
initial_residual_scale="${GLMOCR_Q32_TF_INITIAL_RESIDUAL_SCALE:-0.005}"
max_eval_new_tokens="${GLMOCR_Q32_TF_MAX_EVAL_NEW_TOKENS:-1536}"
generation_mode="${GLMOCR_Q32_TF_GENERATION_MODE:-plain}"
gpu_utilization_limit="${GLMOCR_Q32_TF_GPU_UTILIZATION_LIMIT:-101}"
session="${GLMOCR_Q32_TF_SESSION:-glmocr_q32_tf_ablation_gate005_256_a100_260915_v1}"

content_run_id="${GLMOCR_Q32_TF_CONTENT_RUN_ID:-glmocr_q32_tf_content_only_gate005_256_a100_260915_v1}"
attention_run_id="${GLMOCR_Q32_TF_ATTENTION_RUN_ID:-glmocr_q32_tf_attention_gate005_256_a100_260915_v1}"
geometry_run_id="${GLMOCR_Q32_TF_GEOMETRY_RUN_ID:-glmocr_q32_tf_geometry_gate005_256_a100_260915_v1}"
run_ids=("${content_run_id}" "${attention_run_id}" "${geometry_run_id}")
modes=("content_only" "attention" "geometry")
auxiliary_weights=("0.0" "0.4" "0.4")

[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ && "${seed}" =~ ^[0-9]+$ ]] || exit 64
[[ "${num_queries}" =~ ^[1-9][0-9]*$ && "${max_steps}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ "${lr_schedule_steps}" =~ ^[1-9][0-9]*$ && "${warmup_steps}" =~ ^[0-9]+$ ]] || exit 64
(( lr_schedule_steps <= max_steps && warmup_steps < max_steps )) || exit 64
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 64
[[ "${generation_mode}" == "plain" ]] || exit 64
[[ "${initial_residual_scale}" == "0.005" ]] || exit 64
[[ "${#run_ids[@]}" == "3" ]] || exit 64
for run_id in "${run_ids[@]}"; do
    [[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
done

python="${env_dir}/bin/python"
ddp_launcher="${code_root}/tools/training/run_glmocr_mthv2_ddp.sh"
locked_test_launcher="${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh"
manifest_auditor="${code_root}/tools/audit_mthv2_manifest.py"
dataset_label="dunhuang_local_gazetteer_q32_v1"
protocol_label="glm_ocr_dunhuang_local_gazetteer_group_isolated_v1"
workspace_runs="${remote_root}/runs"
bundle_status="${workspace_runs}/${session}.status.json"
bundle_summary="${workspace_runs}/${session}.summary.json"

write_status() {
    local status="$1"
    local phase="$2"
    local mode="${3:--}"
    local run_id="${4:--}"
    mkdir -p "${workspace_runs}"
    printf '{"status":"%s","phase":"%s","mode":"%s","run_id":"%s","session":"%s","dataset":"%s","seed":%s,"num_queries":%s,"max_steps":%s,"initial_residual_scale":%s,"test_used_for_selection":false,"updated_at":"%s"}\n' \
        "${status}" "${phase}" "${mode}" "${run_id}" "${session}" "${dataset_label}" \
        "${seed}" "${num_queries}" "${max_steps}" "${initial_residual_scale}" "$(date -u +%FT%TZ)" \
        > "${bundle_status}"
}

current_phase="preflight"
on_error() {
    local rc=$?
    write_status failed "${current_phase}" "${current_mode:--}" "${current_run_id:--}"
    exit "${rc}"
}
trap on_error ERR

preflight() {
    [[ -x "${python}" && -x "${env_dir}/bin/torchrun" ]] || exit 66
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || exit 66
    [[ -f "${ddp_launcher}" && -f "${locked_test_launcher}" && -f "${manifest_auditor}" ]] || exit 66
    [[ -f "${dataset_root}/train/manifest.jsonl" ]] || exit 66
    [[ -f "${dataset_root}/validation/manifest.jsonl" ]] || exit 66
    [[ -f "${dataset_root}/test/manifest.jsonl" ]] || exit 66
    for index in "${!run_ids[@]}"; do
        local run_id="${run_ids[${index}]}"
        [[ ! -e "${remote_root}/training_runs/${run_id}" ]] || exit 74
        [[ ! -e "${remote_root}/training_runs/${run_id}_smoke" ]] || exit 74
    done
    [[ ! -e "${bundle_status}" && ! -e "${bundle_summary}" ]] || exit 74
    mkdir -p "${remote_root}/training_runs" "${remote_root}/protocols" "${workspace_runs}"
}

common_ddp_args() {
    local mode="$1"
    local auxiliary_weight="$2"
    local run_id="$3"
    local protocol_file="$4"
    printf '%s\n' \
        --foreground --run-id "${run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit "${gpu_utilization_limit}" --remote-root "${remote_root}" \
        --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}" \
        --dataset-root "${dataset_root}" --protocol-file "${protocol_file}" \
        --dataset-label "${dataset_label}" --protocol-label "${protocol_label}" \
        --allow-count-mismatch --mode "${mode}" --num-queries "${num_queries}" \
        --learning-rate "${learning_rate}" --decoder-adaptation lora \
        --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
        --decoder-learning-rate "${decoder_learning_rate}" --min-lr-ratio "${min_lr_ratio}" \
        --initial-residual-scale "${initial_residual_scale}" --gate-freeze-steps 0 \
        --auxiliary-weight "${auxiliary_weight}" --auxiliary-weight-start "${auxiliary_weight}" \
        --auxiliary-ramp-steps 0 --layout-loss-profile full \
        --generation-mode "${generation_mode}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --log-steps 16
}

validate_smoke() {
    local mode="$1"
    local auxiliary_weight="$2"
    local run_id="$3"
    "${python}" - "${remote_root}/training_runs/${run_id}/smoke/seed${seed}/smoke_summary.json" \
        "${mode}" "${auxiliary_weight}" "${initial_residual_scale}" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "complete" or summary.get("checkpoint_reload") is not True:
    raise SystemExit("teacher-forcing smoke did not complete with checkpoint reload")
if summary.get("test_used_for_selection") is not False:
    raise SystemExit("teacher-forcing smoke is not test-free")
if (summary.get("natural_loop_config") or {}).get("enabled") is not False:
    raise SystemExit("natural-loop objective unexpectedly enabled")
objective = (summary.get("training") or {}).get("loss_objective") or {}
if objective.get("formula") != "L_official + auxiliary_weight * L_layout":
    raise SystemExit(f"unexpected objective: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), float(sys.argv[3])):
    raise SystemExit("smoke layout weight mismatch")
print(json.dumps({"event":"glmocr_q32_tf_smoke_ok","mode":sys.argv[2]}, separators=(",", ":")))
PY
}

run_smoke() {
    local mode="$1"
    local auxiliary_weight="$2"
    local run_id="$3"
    local smoke_id="${run_id}_smoke"
    local protocol_file="${remote_root}/protocols/${smoke_id}.train_validation_no_test.json"
    local log_path="${workspace_runs}/${smoke_id}.pipeline.log"
    current_mode="${mode}"
    current_run_id="${smoke_id}"
    current_phase="${mode}_smoke"
    write_status running "${current_phase}" "${mode}" "${smoke_id}"
    mapfile -t args < <(common_ddp_args "${mode}" "${auxiliary_weight}" "${smoke_id}" "${protocol_file}")
    args+=(--max-steps 8 --lr-schedule-steps 8 --warmup-steps 0 --validation-interval 9 --without-test --smoke)
    bash "${ddp_launcher}" "${args[@]}" > "${log_path}" 2>&1
    validate_smoke "${mode}" "${auxiliary_weight}" "${smoke_id}"
}

validate_training() {
    local mode="$1"
    local auxiliary_weight="$2"
    local run_id="$3"
    "${python}" - "${remote_root}/training_runs/${run_id}/seed${seed}/summary.json" \
        "${remote_root}/training_runs/${run_id}/status/seed${seed}.json" \
        "${remote_root}/training_runs/${run_id}/seed${seed}/selection.json" \
        "${mode}" "${auxiliary_weight}" "${max_steps}" "${lr_schedule_steps}" \
        "${warmup_steps}" "${initial_residual_scale}" "${generation_mode}" "${max_eval_new_tokens}" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_status = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
selection = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
mode, expected_weight = sys.argv[4], float(sys.argv[5])
expected_steps = int(sys.argv[6])
if summary.get("status") != "complete" or run_status.get("status") != "complete":
    raise SystemExit("training run is not complete")
if summary.get("test_manifest_read") is not False or run_status.get("test_manifest_read") is not False:
    raise SystemExit("training protocol is not test-free")
if run_status.get("max_steps") != expected_steps or run_status.get("lr_schedule_steps") != int(sys.argv[7]):
    raise SystemExit("training schedule mismatch")
if run_status.get("warmup_steps") != int(sys.argv[8]):
    raise SystemExit("training warmup mismatch")
if not math.isclose(float(run_status.get("initial_residual_scale", -1.0)), float(sys.argv[9])):
    raise SystemExit("initial residual scale mismatch")
if run_status.get("max_eval_new_tokens") != int(sys.argv[11]):
    raise SystemExit("generation metadata mismatch")
if run_status.get("decoder_adaptation") != "lora":
    raise SystemExit("decoder LoRA was not enabled")
training = summary.get("training") or {}
if [int(step) for step in training.get("checkpoint_steps", [])] != [expected_steps]:
    raise SystemExit(f"unexpected checkpoint steps: {training.get('checkpoint_steps')}")
if any(item.get("checkpoint_finite") is not True for item in training.get("checkpoint_health", [])):
    raise SystemExit("checkpoint health is not finite")
objective = training.get("loss_objective") or {}
if objective.get("formula") != "L_official + auxiliary_weight * L_layout":
    raise SystemExit(f"unexpected teacher-forcing objective: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), expected_weight):
    raise SystemExit("training layout weight mismatch")
if objective.get("extra_terms") != []:
    raise SystemExit("unexpected auxiliary objective terms")
if selection.get("status") != "complete" or selection.get("selected_step") != expected_steps:
    raise SystemExit("fixed final selection is not checkpoint-256")
if selection.get("selection_performed") is not False or selection.get("test_used_for_selection") is not False:
    raise SystemExit("training was not fixed-final and test-free")
print(json.dumps({"event":"glmocr_q32_tf_training_ok","mode":mode,"checkpoint":expected_steps}, separators=(",", ":")))
PY
}

run_training() {
    local mode="$1"
    local auxiliary_weight="$2"
    local run_id="$3"
    local protocol_file="${remote_root}/protocols/${run_id}.train_validation_no_test.json"
    local log_path="${workspace_runs}/${run_id}.pipeline.log"
    current_mode="${mode}"
    current_run_id="${run_id}"
    current_phase="${mode}_training"
    write_status running "${current_phase}" "${mode}" "${run_id}"
    mapfile -t args < <(common_ddp_args "${mode}" "${auxiliary_weight}" "${run_id}" "${protocol_file}")
    args+=(--max-steps "${max_steps}" --lr-schedule-steps "${lr_schedule_steps}" --warmup-steps "${warmup_steps}" \
        --validation-interval "$((max_steps + 1))" --no-validation --without-test)
    bash "${ddp_launcher}" "${args[@]}" > "${log_path}" 2>&1
    validate_training "${mode}" "${auxiliary_weight}" "${run_id}"
}

run_test() {
    local mode="$1"
    local run_id="$2"
    local test_protocol="${remote_root}/protocols/${run_id}.test_direct.json"
    local log_path="${workspace_runs}/${run_id}.test.pipeline.log"
    current_mode="${mode}"
    current_run_id="${run_id}"
    current_phase="${mode}_test"
    write_status running "${current_phase}" "${mode}" "${run_id}"
    if [[ ! -f "${test_protocol}" ]]; then
        "${python}" "${manifest_auditor}" \
            --train-manifest "${dataset_root}/train/manifest.jsonl" \
            --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
            --test-manifest "${dataset_root}/test/manifest.jsonl" \
            --num-queries "${num_queries}" --dataset-label "${dataset_label}" \
            --protocol-label "${protocol_label}" --allow-count-mismatch \
            --output "${test_protocol}" > "${workspace_runs}/${run_id}.test-protocol.log" 2>&1
    fi
    bash "${locked_test_launcher}" --foreground --run-id "${run_id}" --seed "${seed}" \
        --gpu-ids "${gpu_ids}" --mode "${mode}" --num-queries "${num_queries}" \
        --max-eval-new-tokens "${max_eval_new_tokens}" --gpu-utilization-limit "${gpu_utilization_limit}" \
        --protocol-file "${test_protocol}" \
        --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" \
        --model-dir "${model_dir}" --dataset-root "${dataset_root}" > "${log_path}" 2>&1
}

preflight
current_phase="smoke_gate"
write_status running "${current_phase}"
for index in "${!modes[@]}"; do
    run_smoke "${modes[${index}]}" "${auxiliary_weights[${index}]}" "${run_ids[${index}]}"
done

current_phase="training"
write_status running "${current_phase}"
for index in "${!modes[@]}"; do
    run_training "${modes[${index}]}" "${auxiliary_weights[${index}]}" "${run_ids[${index}]}"
done

current_phase="direct_test"
write_status running "${current_phase}"
for index in "${!modes[@]}"; do
    run_test "${modes[${index}]}" "${run_ids[${index}]}"
done

current_phase="summarize"
write_status running "${current_phase}"
"${python}" - "${bundle_summary}" "${remote_root}" "${seed}" "${session}" \
    "${content_run_id}" "${attention_run_id}" "${geometry_run_id}" <<'PY'
import json
import sys
from pathlib import Path

output, root, seed, session = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
entries = []
for mode, run_id in zip(("content_only", "attention", "geometry"), sys.argv[5:]):
    run_dir = root / "training_runs" / run_id / f"seed{seed}"
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    test = json.loads((run_dir / "locked-test" / "locked_test_summary.json").read_text(encoding="utf-8"))
    entries.append({
        "mode": mode,
        "run_id": run_id,
        "checkpoint": int(test["selected_step"]),
        "training_summary": str(run_dir / "summary.json"),
        "test_summary": str(run_dir / "locked-test" / "locked_test_summary.json"),
        "metrics": test.get("metrics", {}),
        "test_used_for_selection": test.get("test_used_for_selection"),
        "teacher_forcing": (training.get("training") or {}).get("loss_objective", {}),
    })
payload = {
    "status": "complete",
    "session": session,
    "experiment": "q32_teacher_forcing_ablation",
    "modes": entries,
    "seed": seed,
    "num_queries": 32,
    "max_steps": 256,
    "checkpoint": 256,
    "initial_residual_scale": 0.005,
    "test_protocol": "direct_test_fixed_final_checkpoint_no_validation_selection",
    "test_used_for_selection": False,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event":"glmocr_q32_tf_ablation_complete","modes":[item["mode"] for item in entries],"checkpoint":256,"test_used_for_selection":False}, separators=(",", ":")))
PY
write_status complete complete
