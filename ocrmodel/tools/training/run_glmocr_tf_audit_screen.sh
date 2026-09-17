#!/usr/bin/env bash
# Minimal A0/A1 teacher-forcing audit screen.
#
# The two runs share one stratified 64-page validation manifest, seed 42, and
# the same five-GPU DDP protocol.  A1 only adds bounded scheduled sampling;
# there is no free-running or autoregressive-region experiment here.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dataset_root="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
screen_id="${GLMOCR_TF_AUDIT_SCREEN_ID:-glmocr_loop_escape_audit_260910_v1}"
gpu_ids="${GLMOCR_TF_AUDIT_GPU_IDS:-0,1,2,3,4}"
subset_pages="${GLMOCR_TF_AUDIT_PAGES:-64}"
seed=42

export GLMOCR_DDP_TIMEOUT_SECONDS="${GLMOCR_TF_AUDIT_DDP_TIMEOUT_SECONDS:-3600}"

[[ "${screen_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo '{"error":"invalid_screen_id"}' >&2; exit 64; }
[[ "${gpu_ids}" == "0,1,2,3,4" ]] || { echo '{"error":"tf_audit_requires_five_gpus"}' >&2; exit 64; }
[[ "${subset_pages}" =~ ^[1-9][0-9]*$ ]] || { echo '{"error":"invalid_subset_pages"}' >&2; exit 64; }

python="${env_dir}/bin/python"
launcher="${code_root}/tools/training/run_glmocr_mthv2_ddp.sh"
subset_tool="${code_root}/tools/subset_mthv2_manifest.py"
train_manifest="${dataset_root}/train/manifest.jsonl"
validation_manifest="${dataset_root}/validation/manifest.jsonl"
screen_root="${remote_root}/training_runs/${screen_id}"
subset_manifest="${screen_root}/validation64_seed42.jsonl"
protocol_file="${screen_root}/protocol_validation64_no_test.json"

[[ -x "${python}" && -f "${launcher}" && -f "${subset_tool}" ]] || {
    echo '{"error":"tf_audit_source_or_environment_missing"}' >&2
    exit 66
}
[[ ! -e "${screen_root}" ]] || {
    printf '{"error":"tf_audit_output_exists","path":"%s"}\n' "${screen_root}" >&2
    exit 74
}
mkdir -p "${screen_root}"

"${python}" "${subset_tool}" \
    --input "${validation_manifest}" \
    --output "${subset_manifest}" \
    --count "${subset_pages}" \
    --seed "${seed}" \
    --stratify-regions > "${screen_root}/subset.json"

base_args=(
    --foreground
    --gpu-ids "${gpu_ids}"
    --seed "${seed}"
    --max-steps 256
    --lr-schedule-steps 256
    --warmup-steps 32
    --validation-interval 256
    # Do not spend the screen budget on validation-0; the agreed protocol
    # starts training immediately and only audits the 128/256 checkpoints.
    --diagnostic-steps 128,256
    --decoder-adaptation lora
    --decoder-lora-rank 8
    --decoder-lora-alpha 8
    --decoder-learning-rate 1e-6
    --validation-manifest "${subset_manifest}"
    --protocol-file "${protocol_file}"
    --without-test
    --remote-root "${remote_root}"
    --code-root "${code_root}"
    --env-dir "${env_dir}"
    --model-dir "${model_dir}"
    --dataset-root "${dataset_root}"
)

bash "${launcher}" "${base_args[@]}" \
    --run-id "${screen_id}_A0" \
    --experiment-label "tf_audit_A0_teacher_forcing" \
    --session "${screen_id}_A0" \
    --text-repeat-suppression \
    --continuation-escape \
    --repeat-cycle-penalty 1.0 \
    --repeat-force-eos-steps 0

bash "${launcher}" "${base_args[@]}" \
    --run-id "${screen_id}_A1" \
    --experiment-label "tf_audit_A1_scheduled_sampling" \
    --session "${screen_id}_A1" \
    --text-repeat-suppression \
    --text-ul-weight 0.1 \
    --text-eos-loss-weight 0.05 \
    --repeat-recent-window 96 \
    --repeat-min-cycle-length 8 \
    --repeat-max-cycle-length 32 \
    --repeat-cycle-repeats 3 \
    --repeat-cycle-penalty 1.0 \
    --repeat-force-eos-steps 0 \
    --continuation-escape \
    --loop-escape-training \
    --continuation-head \
    --loop-escape-cycle-length 8 \
    --loop-escape-horizon 8 \
    --loop-escape-ramp-steps 128 \
    --loop-escape-weight 0.1 \
    --loop-escape-margin 0.5 \
    --loop-escape-margin-weight 0.05 \
    --loop-continue-weight 0.05 \
    --continuation-head-hidden-size 32 \
    --continuation-head-weight 0.05 \
    --continuation-head-learning-rate 5e-4 \
    --scheduled-sampling \
    --scheduled-sampling-warmup-steps 64 \
    --scheduled-sampling-ramp-steps 64 \
    --scheduled-sampling-max-probability 0.1

"${python}" - "${screen_root}" "${subset_manifest}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
subset = Path(sys.argv[2])
runs = []
for label in ("A0", "A1"):
    summary_path = root.parent / f"{root.name}_{label}" / "seed42" / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing completed summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = payload.get("validation") or {}
    training = payload.get("training") or {}
    runs.append({
        "label": label,
        "run_dir": str(summary_path.parent),
        "selected_step": json.loads(
            (summary_path.parent / "selection.json").read_text(encoding="utf-8")
        ).get("selected_step"),
        "validation_cer": validation.get("cer"),
        "teacher_forced_validation_loss": validation.get("teacher_forced_validation_loss"),
        "teacher_forced_ocr_loss": validation.get("teacher_forced_ocr_loss"),
        "insertions": validation.get("insertions"),
        "deletions": validation.get("deletions"),
        "substitutions": validation.get("substitutions"),
        "repeated_cycle_page_rate": validation.get("repeated_cycle_page_rate"),
        "generation_limit_hit_rate": validation.get("generation_limit_hit_rate"),
        "generation_eos_hit_rate": validation.get("generation_eos_hit_rate"),
        "loop_detected_page_rate": validation.get("loop_detected_page_rate"),
        "loop_escape_success_rate": validation.get("loop_escape_success_rate"),
        "loop_post_eos_rate": validation.get("loop_post_eos_rate"),
        "loop_early_eos_rate": validation.get("loop_early_eos_rate"),
        "training_mean_ocr_loss": training.get("mean_ocr_loss"),
        "training_mean_token_weighted_ocr_loss": training.get("mean_token_weighted_ocr_loss"),
        "training_mean_mixed_prefix_loss": training.get("mean_mixed_prefix_loss"),
        "training_mean_ocr_objective_loss": training.get("mean_ocr_objective_loss"),
        "scheduled_sampling": training.get("scheduled_sampling"),
        "loop_escape": training.get("loop_escape"),
        "continuation_head": training.get("continuation_head"),
        "mean_loop_escape_loss": training.get("mean_loop_escape_loss"),
        "mean_loop_margin_loss": training.get("mean_loop_margin_loss"),
        "mean_loop_continue_loss": training.get("mean_loop_continue_loss"),
        "mean_continuation_head_loss": training.get("mean_continuation_head_loss"),
        "prompt_target_audit": validation.get("prompt_target_audit"),
    })
summary = {
    "status": "complete",
    "screen_id": root.name,
    "seed": 42,
    "validation_pages": sum(1 for line in subset.read_text(encoding="utf-8").splitlines() if line.strip()),
    "test_manifest_read": False,
    "test_used_for_selection": False,
    "runs": runs,
}
(root / "tf_audit_screen_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
PY
