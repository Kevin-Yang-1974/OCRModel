#!/usr/bin/env bash
# Continue validation-only evaluation for an existing q32 training run.
set -Eeuo pipefail

root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code="${GLMOCR_A100_CODE_ROOT:-${root}/code/ocrmodel}"
python="${GLMOCR_A100_ENV:-${root}/envs/glmocr_a100_py311_cu128}/bin/python"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
dataset="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
run_id="${1:-}"
mode="${2:-}"
seed="${GLMOCR_A100_SEED:-42}"
gpu_ids="${GLMOCR_A100_GPU_IDS:-0,1,2,3,4}"
max_steps="${GLMOCR_A100_MAX_STEPS:-2000}"
validation_interval="${GLMOCR_A100_VALIDATION_INTERVAL:-500}"
num_queries="${GLMOCR_A100_NUM_QUERIES:-32}"
max_eval_new_tokens="${GLMOCR_A100_MAX_EVAL_NEW_TOKENS:-1536}"
generation_mode="${GLMOCR_A100_GENERATION_MODE:-loop_recovery}"
decoder_learning_rate="${GLMOCR_A100_DECODER_LR:-5e-6}"
dataset_label="dunhuang_local_gazetteer_q32_v1"
steps=(500 1000 1500 2000)

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
case "${mode}" in
    geometry) auxiliary_weight="0.4" ;;
    content_only) auxiliary_weight="0.0" ;;
    *) exit 64 ;;
esac

group="${root}/training_runs/${run_id}"
run_dir="${group}/seed${seed}"
validation_root="${run_dir}/parallel-validation"
protocol_file="${root}/protocols/${run_id}.train_validation_no_test.json"
train_manifest="${dataset}/train/manifest.jsonl"
validation_manifest="${dataset}/validation/manifest.jsonl"
[[ -d "${run_dir}" && -f "${run_dir}/COMPLETED" && -f "${run_dir}/metadata.json" && -f "${run_dir}/summary.json" ]] || exit 66
[[ -f "${protocol_file}" && -f "${train_manifest}" && -f "${validation_manifest}" ]] || exit 66
[[ ! -f "${run_dir}/selection.json" ]] || exit 74

"${python}" - "${run_dir}/metadata.json" "${run_dir}/summary.json" "${mode}" "${num_queries}" <<'PY'
import json
import sys
metadata = json.load(open(sys.argv[1], encoding="utf-8"))
summary = json.load(open(sys.argv[2], encoding="utf-8"))
if metadata.get("status") != "complete" or summary.get("status") != "complete":
    raise SystemExit("training run is not complete")
if metadata.get("mode") != sys.argv[3] or metadata.get("num_queries") != int(sys.argv[4]):
    raise SystemExit("training protocol does not match validation request")
if metadata.get("test_manifest_read") is not False or summary.get("test_manifest_read") is not False:
    raise SystemExit("training protocol reads test")
if metadata.get("test_used_for_selection") is not False or summary.get("test_used_for_selection") is not False:
    raise SystemExit("training protocol uses test for selection")
PY

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
(( ${#gpu_array[@]} >= 4 )) || exit 64
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ && -z "${seen_gpu[${gpu}]+present}" ]] || exit 64
    seen_gpu[${gpu}]=1
done

cuda_libraries="${root}/envs/glmocr_a100_py311_cu128/lib/python3.11/site-packages/torch/lib"
cuda_arch="$(uname -m)"
[[ "${cuda_arch}" != "arm64" ]] || cuda_arch="aarch64"
system_cuda="/usr/local/cuda/targets/${cuda_arch}-linux/lib"
[[ ! -d "${system_cuda}" ]] || cuda_libraries="${system_cuda}:${cuda_libraries}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    [[ ! -d "${lib}" ]] || cuda_libraries="${cuda_libraries}:${lib}"
done
export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${code}/src:${code}${PYTHONPATH:+:${PYTHONPATH}}"

admit_validation_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || exit 69
    declare -A observed=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${observed_id}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || exit 69
        observed[${observed_id}]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for index in 0 1 2 3; do
        gpu="${gpu_array[${index}]}"
        [[ -n "${observed[${gpu}]+present}" ]] || exit 69
        (( observed[${gpu}] < 50 )) || {
            printf '{"event":"glmocr_q32_validation_only_failed","error":"gpu_admission_failed","gpu":"%s","utilization":%s}\n' "${gpu}" "${observed[${gpu}]}" >&2
            exit 75
        }
    done
    printf '{"event":"glmocr_q32_validation_gpu_admission_ok","gpu_ids":"%s"}\n' "${gpu_ids}"
}

mkdir -p "${validation_root}" "${group}/logs" "${group}/tmp"
pending=()
declare -A pending_step=()
for step in "${steps[@]}"; do
    eval_dir="${validation_root}/step-${step}"
    if [[ -e "${eval_dir}" ]]; then
        [[ -f "${eval_dir}/COMPLETED" && -f "${eval_dir}/summary.json" && -f "${eval_dir}/metadata.json" ]] || exit 74
    else
        pending+=("${step}")
        pending_step[${step}]=1
    fi
done

if (( ${#pending[@]} > 0 )); then
    admit_validation_gpus
    pids=()
    for index in 0 1 2 3; do
        step="${steps[${index}]}"
        [[ -n "${pending_step[${step}]+present}" ]] || continue
        eval_dir="${validation_root}/step-${step}"
        gpu="${gpu_array[${index}]}"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export TMPDIR="${group}/tmp/validation-${step}"
            export HF_HOME="${TMPDIR}/huggingface"
            export TRANSFORMERS_CACHE="${HF_HOME}"
            mkdir -p "${TMPDIR}" "${HF_HOME}"
            cd "${code}"
            exec "${python}" -m layout_ocr.train_screen \
                --mode "${mode}" --model-path "${model_dir}" \
                --train-manifest "${train_manifest}" --validation-manifest "${validation_manifest}" \
                --protocol-file "${protocol_file}" --output-dir "${eval_dir}" \
                --per-device-batch-size 1 --gradient-accumulation-steps 1 \
                --max-steps "${max_steps}" --num-queries "${num_queries}" --seed "${seed}" \
                --experiment-label "q32_validation_only_${mode}_step${step}" \
                --learning-rate 2.5e-5 --decoder-adaptation lora \
                --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
                --decoder-learning-rate "${decoder_learning_rate}" \
                --warmup-steps 0 --lr-schedule-steps "${max_steps}" --min-lr-ratio 0.1 \
                --residual-scale-cap 0.03 --initial-residual-scale 0 \
                --auxiliary-weight "${auxiliary_weight}" --auxiliary-weight-start "${auxiliary_weight}" \
                --auxiliary-ramp-steps 0 --gate-freeze-steps 0 --max-grad-norm 1.0 \
                --max-pixels 1003520 --max-eval-new-tokens "${max_eval_new_tokens}" \
                --validation-interval "${validation_interval}" --log-steps 16 \
                --adapter-precision fp32 --layout-loss-profile full --query-assignment hungarian \
                --processor-mode fast --generation-mode "${generation_mode}" \
                --eval-checkpoint-dir "${run_dir}/checkpoint-${step}" --eval-only
        ) > "${group}/logs/validation-only.step${step}.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || exit 1
fi

"${python}" "${code}/tools/summarize_glmocr_parallel_validation.py" \
    --run-dir "${run_dir}" --validation-root "${validation_root}" \
    --group-root "${group}" --steps "$(IFS=,; echo "${steps[*]}")" \
    --expected-world-size 5 --expected-layout-weight "${auxiliary_weight}" \
    --dataset-label "${dataset_label}" \
    > "${group}/logs/validation-only-summary.log" 2>&1

"${python}" - "${run_dir}/selection.json" "${run_dir}/parallel_validation_summary.json" <<'PY'
import json
import sys
selection = json.load(open(sys.argv[1], encoding="utf-8"))
parallel = json.load(open(sys.argv[2], encoding="utf-8"))
if selection.get("status") != "complete" or parallel.get("status") != "complete":
    raise SystemExit("validation selection is not complete")
if selection.get("test_used_for_selection") is not False or parallel.get("test_used_for_selection") is not False:
    raise SystemExit("validation selection used test")
print(json.dumps({
    "event": "glmocr_q32_validation_only_complete",
    "selected_step": selection.get("selected_step"),
    "selected_validation_cer": parallel.get("selected_validation_cer"),
    "candidate_count": len(selection.get("candidates") or []),
    "test_used_for_selection": False,
}, separators=(",", ":")))
PY
