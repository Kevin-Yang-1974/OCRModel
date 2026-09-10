#!/usr/bin/env bash
# Minimal validation-only A0/A1/B1 screen.
#
# The caller must run this on a prepared A100 host.  It creates one fixed,
# region-density-stratified 64-page validation manifest and then executes the
# three runs serially on the same five GPUs.  No test manifest is opened.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dataset_root="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
screen_id="${GLMOCR_FAST_SCREEN_ID:-glmocr_fast_screen_v1}"
gpu_ids="${GLMOCR_FAST_SCREEN_GPU_IDS:-0,1,2,3,4}"
subset_pages="${GLMOCR_FAST_SCREEN_PAGES:-64}"
seed=42

# The 64-page whole-page validation is rank-0-only and can spend more than
# ten minutes in generation before the other DDP ranks rendezvous again.
# Keep the short-screen protocol bounded, but do not let the default 600 s
# process-group watchdog kill a valid validation pass.
export GLMOCR_DDP_TIMEOUT_SECONDS="${GLMOCR_FAST_SCREEN_DDP_TIMEOUT_SECONDS:-3600}"

[[ "${screen_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo '{"error":"invalid_screen_id"}' >&2; exit 64; }
[[ "${gpu_ids}" == "0,1,2,3,4" ]] || { echo '{"error":"fast_screen_requires_five_gpus"}' >&2; exit 64; }
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
    echo '{"error":"fast_screen_source_or_environment_missing"}' >&2
    exit 66
}
[[ ! -e "${screen_root}" ]] || {
    printf '{"error":"fast_screen_output_exists","path":"%s"}\n' "${screen_root}" >&2
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
    --lr-schedule-steps 128
    --warmup-steps 32
    --validation-interval 128
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
    --experiment-label "fast_screen_A0" \
    --session "${screen_id}_A0"

bash "${launcher}" "${base_args[@]}" \
    --run-id "${screen_id}_A1" \
    --experiment-label "fast_screen_A1" \
    --session "${screen_id}_A1" \
    --text-repeat-suppression \
    --text-ul-weight 0.1 \
    --text-eos-loss-weight 0.05 \
    --repeat-recent-window 96 \
    --repeat-min-cycle-length 8 \
    --repeat-max-cycle-length 32 \
    --repeat-cycle-repeats 3 \
    --repeat-cycle-penalty 2.0 \
    --repeat-force-eos-steps 16

bash "${launcher}" "${base_args[@]}" \
    --run-id "${screen_id}_B1" \
    --experiment-label "fast_screen_B1" \
    --session "${screen_id}_B1" \
    --region-autoregressive \
    --region-decoder-hidden-size 256 \
    --region-decoder-layers 2 \
    --region-decoder-num-heads 8 \
    --region-pointer-mask \
    --region-spatial-penalty 4.0 \
    --region-spatial-iou-threshold 0.8 \
    --diagnostic-steps 128,256,512 \
    --max-steps 512 \
    --lr-schedule-steps 128

"${python}" - "${screen_root}" "${subset_manifest}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
subset = Path(sys.argv[2])
runs = []
for label in ("A0", "A1", "B1"):
    # The launcher keeps each run directly under training_runs, while root
    # contains the fixed subset/protocol and the aggregate screen summary.
    run = root.parent / f"{root.name}_{label}" / "summary.json"
    if not run.is_file():
        raise SystemExit(f"missing completed summary: {run}")
    payload = json.loads(run.read_text(encoding="utf-8"))
    validation = payload.get("validation") or {}
    runs.append({
        "label": label,
        "run_dir": str(run.parent),
        "selected_step": (json.loads((run.parent / "selection.json").read_text(encoding="utf-8"))).get("selected_step"),
        "validation_cer": validation.get("cer"),
        "insertions": validation.get("insertions"),
        "deletions": validation.get("deletions"),
        "substitutions": validation.get("substitutions"),
        "repeated_cycle_page_rate": validation.get("repeated_cycle_page_rate"),
        "generation_limit_hit_rate": validation.get("generation_limit_hit_rate"),
        "region_pointer_reuse_rate": validation.get("region_pointer_reuse_rate"),
        "region_spatial_duplicate_rate": validation.get("region_spatial_duplicate_rate"),
        "region_eos_hit_rate": validation.get("region_eos_hit_rate"),
        "region_recall": validation.get("region_recall"),
        "region_bbox_ap50": validation.get("region_bbox_ap50"),
        "region_bbox_recall50": validation.get("region_bbox_recall50"),
        "region_reading_order_accuracy": validation.get("region_reading_order_accuracy"),
        "stratified": validation.get("stratified"),
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
(root / "fast_screen_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
PY
