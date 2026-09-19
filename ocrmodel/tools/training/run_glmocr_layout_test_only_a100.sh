#!/usr/bin/env bash
# Select by validation layout IoU, then run the locked MTHv2 test.
# The final report intentionally exposes layout metrics only.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
mthv2_root="${GLMOCR_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
sparse_root="${GLMOCR_MTHV2_SPARSE_Q32_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24}"
gpu_ids="${GLMOCR_LAYOUT_TEST_GPU_IDS:-0,1,2,3,4}"
validation_gpu_ids="${GLMOCR_LAYOUT_VALIDATION_GPU_IDS:-0,1,2}"
gpu_utilization_limit="${GLMOCR_LAYOUT_TEST_GPU_UTILIZATION_LIMIT:-50}"
run_id="${GLMOCR_LAYOUT_TEST_RUN_ID:-glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou10x_mlp_refine_3000_from_boxeq820_20260917_v1}"
session="${GLMOCR_LAYOUT_TEST_SESSION:-glmocr_layout_test_iou10x_20260917_retry1}"
seed="${GLMOCR_LAYOUT_TEST_SEED:-42}"
num_queries="${GLMOCR_LAYOUT_TEST_NUM_QUERIES:-32}"
steps="${GLMOCR_LAYOUT_TEST_STEPS:-1000,2000,3000}"
layout_loss_profile="${GLMOCR_LAYOUT_TEST_LOSS_PROFILE:-iou_consistent_giou10x}"
max_eval_new_tokens="${GLMOCR_LAYOUT_TEST_MAX_EVAL_NEW_TOKENS:-1536}"
foreground=0
validation_only=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) run_id="$2"; shift 2 ;;
        --session) session="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --validation-gpu-ids) validation_gpu_ids="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --layout-loss-profile) layout_loss_profile="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        --validation-only) validation_only=1; shift ;;
        *) printf '{"event":"glmocr_layout_test_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

protocol_file="${remote_root}/protocols/${run_id}.train_validation_no_test.json"
group_root="${remote_root}/training_runs/${run_id}"
run_dir="${group_root}/seed${seed}"
validation_root="${run_dir}/layout-validation-${session}"

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ "${seed}" =~ ^[0-9]+$ && "${num_queries}" == "32" ]] || exit 64
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ && "${validation_gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 64
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ && "${max_eval_new_tokens}" =~ ^[1-9][0-9]*$ ]] || exit 64

python="${env_dir}/bin/python"
group_root="${remote_root}/training_runs/${run_id}"
run_dir="${group_root}/seed${seed}"
validation_root="${run_dir}/layout-validation-${session}"
protocol_file="${remote_root}/protocols/${run_id}.train_validation_no_test.json"
test_protocol_file="${remote_root}/protocols/${run_id}.layout_test_locked.json"
workspace_runs="${remote_root}/runs"
status_file="${workspace_runs}/${session}.status.json"
summary_file="${workspace_runs}/${run_id}.${session}.summary.json"

write_status() {
    local status="$1"
    local phase="$2"
    printf '{"status":"%s","phase":"%s","run_id":"%s","session":"%s","seed":%s,"steps":"%s","layout_loss_profile":"%s","test_used_for_selection":false,"updated_at":"%s"}\n' \
        "${status}" "${phase}" "${run_id}" "${session}" "${seed}" "${steps}" \
        "${layout_loss_profile}" "$(date -u +%FT%TZ)" > "${status_file}"
}

on_error() {
    local rc=$?
    write_status failed "${current_phase:-unknown}_failed" || true
    exit "${rc}"
}
trap on_error ERR

admit_gpu_set() {
    local requested="$1"
    declare -A observed=()
    while IFS=',' read -r gpu utilization; do
        gpu="${gpu//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${gpu}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || exit 69
        observed["${gpu}"]="${utilization}"
    done < <(nvidia-smi -i "${requested}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    IFS=',' read -r -a requested_gpus <<< "${requested}"
    for gpu in "${requested_gpus[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || exit 69
        (( observed[${gpu}] < gpu_utilization_limit )) || {
            printf '{"event":"glmocr_layout_test_failed","error":"gpu_admission_failed","gpu":%s,"utilization":%s,"limit":%s}\n' \
                "${gpu}" "${observed[${gpu}]}" "${gpu_utilization_limit}" >&2
            exit 75
        }
    done
    printf '{"event":"glmocr_layout_test_gpu_admission_ok","gpu_ids":"%s"}\n' "${requested}"
}

cuda_library_path() {
    local torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
    local machine_arch
    machine_arch="$(uname -m)"
    local target_arch
    case "${machine_arch}" in
        x86_64) target_arch="x86_64-linux" ;;
        aarch64|arm64) target_arch="aarch64-linux" ;;
        *) target_arch="${machine_arch}-linux" ;;
    esac
    local result="/usr/local/cuda/targets/${target_arch}/lib:${torch_lib}"
    for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
        local component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
        [[ -d "${component_lib}" ]] && result="${result}:${component_lib}"
    done
    printf '%s' "${result}"
}

preflight() {
    [[ -x "${python}" ]] || exit 66
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || exit 66
    for path in \
        "${code_root}/src/layout_ocr/train_screen.py" \
        "${code_root}/tools/select_layout_iou_checkpoint.py" \
        "${code_root}/tools/select_low_density_mthv2.py" \
        "${code_root}/tools/audit_mthv2_manifest.py" \
        "${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh"; do
        [[ -f "${path}" ]] || exit 66
    done
    [[ -f "${sparse_root}/train/manifest.jsonl" && -f "${sparse_root}/validation/manifest.jsonl" ]] || exit 66
    [[ -f "${protocol_file}" ]] || exit 66
    [[ -f "${run_dir}/metadata.json" ]] || exit 66
    if (( validation_only == 0 )); then
        [[ -f "${run_dir}/summary.json" && -f "${run_dir}/COMPLETED" ]] || exit 66
        [[ ! -e "${run_dir}/selection.json" && ! -e "${group_root}/selection.json" ]] || exit 74
    fi
    [[ ! -e "${validation_root}" && ! -e "${run_dir}/locked-test" && ! -e "${run_dir}/locked-test-shards" ]] || exit 74
    command -v nvidia-smi >/dev/null 2>&1 || exit 69
    mkdir -p "${workspace_runs}"
    admit_gpu_set "${validation_gpu_ids}"
}

run_validation() {
    current_phase="layout_validation"
    write_status running "${current_phase}"
    mkdir -p "${validation_root}/logs"
    IFS=',' read -r -a eval_gpus <<< "${validation_gpu_ids}"
    IFS=',' read -r -a eval_steps <<< "${steps}"
    (( ${#eval_gpus[@]} >= ${#eval_steps[@]} )) || exit 64
    local libs
    libs="$(cuda_library_path)"
    local -a pids=()
    local index
    for index in "${!eval_steps[@]}"; do
        local step="${eval_steps[${index}]}"
        local gpu="${eval_gpus[${index}]}"
        local eval_dir="${validation_root}/step-${step}"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export TMPDIR="${run_dir}/tmp/layout-validation-${step}"
            export HF_HOME="${TMPDIR}/huggingface"
            export TRANSFORMERS_CACHE="${HF_HOME}"
            export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
            export LD_LIBRARY_PATH="${libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
            mkdir -p "${TMPDIR}" "${HF_HOME}"
            cd "${code_root}"
            "${python}" -m layout_ocr.train_screen \
                --mode geometry --model-path "${model_dir}" \
                --train-manifest "${sparse_root}/train/manifest.jsonl" \
                --validation-manifest "${sparse_root}/validation/manifest.jsonl" \
                --protocol-file "${protocol_file}" --output-dir "${eval_dir}" \
                --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 \
                --num-queries "${num_queries}" --seed "${seed}" \
                --experiment-label "${run_id}_layout_validation_step${step}" \
                --decoder-adaptation frozen --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
                --decoder-learning-rate 5e-6 --learning-rate 2.5e-5 --min-lr-ratio 0.1 \
                --initial-residual-scale 0.0 --auxiliary-weight 1.0 --auxiliary-weight-start 1.0 \
                --max-grad-norm 1.0 --max-pixels 1003520 --max-eval-new-tokens "${max_eval_new_tokens}" \
                --validation-interval 2 --log-steps 16 --adapter-precision fp32 \
                --layout-loss-profile "${layout_loss_profile}" --query-assignment hungarian \
                --processor-mode fast --generation-mode plain --layout-only \
                --box-head-mlp --box-head-hidden 0 --query-refine-layers 1 \
                --eval-checkpoint-dir "${run_dir}/checkpoint-${step}" --eval-only
        ) > "${validation_root}/logs/step-${step}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || exit 1
}

write_validation_only_summary() {
    current_phase="complete"
    "${python}" - "${summary_file}" "${run_id}" "${session}" "${validation_root}" "${validation_gpu_ids}" "${steps}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_id, session, root, gpu_ids, steps = sys.argv[2:]
root_path = Path(root)
gpus = [int(value) for value in gpu_ids.split(",")]
step_values = [int(value) for value in steps.split(",")]
checkpoints = []
for step, gpu in zip(step_values, gpus):
    directory = root_path / f"step-{step}"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    predictions = directory / "validation_predictions.jsonl"
    checkpoints.append({
        "step": step,
        "gpu": gpu,
        "status": summary.get("status"),
        "pages": summary.get("pages"),
        "regions": summary.get("regions"),
        "layout_box_iou": summary.get("layout_box_iou"),
        "layout_box_mae": summary.get("layout_box_mae"),
        "direction_accuracy": summary.get("direction_accuracy"),
        "prediction_lines": sum(1 for _ in predictions.open(encoding="utf-8")),
    })
payload = {
    "status": "complete",
    "run_id": run_id,
    "session": session,
    "purpose": "layout_validation_only",
    "validation_root": str(root_path),
    "gpu_assignment": {str(step): gpu for step, gpu in zip(step_values, gpus)},
    "test_used_for_selection": False,
    "checkpoints": checkpoints,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
(root_path / "COMPLETED").write_text("complete\n", encoding="utf-8", newline="\n")
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
    write_status complete complete
}

run_locked_test() {
    current_phase="layout_locked_test"
    write_status running "${current_phase}"
    "${python}" "${code_root}/tools/select_layout_iou_checkpoint.py" \
        --run-dir "${run_dir}" --validation-root "${validation_root}" \
        --steps "${steps}" --dataset-label "MTHv2_sparse24_q32_layout_iou_consistent_giou10x" \
        --expected-world-size 5 \
        > "${workspace_runs}/${run_id}.${session}.layout-selection.log" 2>&1
    "${python}" "${code_root}/tools/select_low_density_mthv2.py" \
        --input-root "${mthv2_root}" --output-root "${sparse_root}" \
        --max-regions 24 --splits test \
        > "${workspace_runs}/${run_id}.${session}.test-manifest.log" 2>&1
    "${python}" "${code_root}/tools/audit_mthv2_manifest.py" \
        --train-manifest "${sparse_root}/train/manifest.jsonl" \
        --validation-manifest "${sparse_root}/validation/manifest.jsonl" \
        --test-manifest "${sparse_root}/test/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch \
        --dataset-label "MTHv2_sparse24_q32_layout_iou_consistent_giou10x" \
        --protocol-label "glm_ocr_mthv2_sparse24_q32_layout_iou_consistent_giou10x_locked_test_v1" \
        --output "${test_protocol_file}" \
        > "${workspace_runs}/${run_id}.${session}.test-protocol.log" 2>&1
    bash "${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh" --foreground \
        --run-id "${run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit "${gpu_utilization_limit}" --mode geometry \
        --num-queries "${num_queries}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" \
        --model-dir "${model_dir}" --dataset-root "${sparse_root}" \
        --protocol-file "${test_protocol_file}" \
        > "${workspace_runs}/${run_id}.${session}.layout-test.pipeline.log" 2>&1
}

write_summary() {
    current_phase="complete"
    write_status complete complete
    "${python}" - "${summary_file}" "${run_dir}" "${run_id}" "${validation_root}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
run_id = sys.argv[3]
validation_root = Path(sys.argv[4])
selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
test = json.loads((run_dir / "locked-test" / "locked_test_summary.json").read_text(encoding="utf-8"))
test_metrics = test.get("metrics") or {}
payload = {
    "status": "complete",
    "run_id": run_id,
    "split": "test",
    "purpose": "layout_detection_only",
    "selected_step": selection.get("selected_step"),
    "selection_metric": selection.get("selection_metric"),
    "validation_root": str(validation_root),
    "test_summary": str(run_dir / "locked-test" / "locked_test_summary.json"),
    "test_pages": test.get("test_pages"),
    "test_used_for_selection": test.get("test_used_for_selection"),
    "layout_metrics": {
        key: test_metrics.get(key)
        for key in (
            "layout_box_iou",
            "layout_box_mae",
            "layout_box_iou_regions",
            "layout_direction_accuracy",
            "layout_direction_regions",
            "mean_annotated_queries",
            "mean_unannotated_queries",
            "region_recall",
            "region_bbox_ap50",
            "region_bbox_precision50",
            "region_bbox_recall50",
            "region_reading_order_accuracy",
        )
    },
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
}

run_inner() {
    preflight
    run_validation
    if (( validation_only == 1 )); then
        write_validation_only_summary
        return 0
    fi
    run_locked_test
    write_summary
}

if (( foreground == 0 )); then
    preflight
    command -v tmux >/dev/null 2>&1 || exit 69
    tmux has-session -t "${session}" 2>/dev/null && exit 73
    mkdir -p "${workspace_runs}"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    child_args=(
        bash "${script_path}" --foreground
        --run-id "${run_id}" --session "${session}" --seed "${seed}"
        --validation-gpu-ids "${validation_gpu_ids}" --steps "${steps}"
        --layout-loss-profile "${layout_loss_profile}"
    )
    (( validation_only == 1 )) && child_args+=(--validation-only)
    command_line="$(printf '%q ' "${child_args[@]}")"
    tmux new-session -d -s "${session}" \
        "cd $(printf '%q' "${code_root}") && exec ${command_line} >$(printf '%q' "${workspace_runs}/${session}.log") 2>&1"
    printf '{"event":"glmocr_layout_test_armed","session":"%s","run_id":"%s","gpu_ids":"%s","validation_gpu_ids":"%s","status":"%s","log":"%s"}\n' \
        "${session}" "${run_id}" "${gpu_ids}" "${validation_gpu_ids}" "${status_file}" "${workspace_runs}/${session}.log"
else
    run_inner
fi
