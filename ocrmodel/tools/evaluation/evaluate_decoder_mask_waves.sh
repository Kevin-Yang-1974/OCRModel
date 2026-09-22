#!/usr/bin/env bash
# Prompt-only re-scoring of every (arm, step) checkpoint of a decoder-mask screen.
#
# Why this exists: the Gate C training loop records only *aggregate* validation
# metrics, so a paired page-level bootstrap has nothing to resample.  This driver
# re-scores the saved checkpoints with evaluate_decoder_mask.py -- which writes
# per-page predictions -- and dispatches one job per allowlisted GPU in waves, so
# all arms at a matched step cost one wave instead of four serial runs.
#
# It never opens the MTHv2 test manifest: --manifest is the validation subset.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"

screen_id="glmocr_decoder_mask_screen_v1"
arms=(B0 B1 B2 B3)
steps=(512 1024)
gpu_ids="0,1,2,3"
output_root=""
max_new_tokens=512
max_pixels=1003520
seed=42
gpu_utilization_limit=50

while [[ $# -gt 0 ]]; do
    case "$1" in
        --screen-id) screen_id="$2"; shift 2 ;;
        --arms) IFS=',' read -r -a arms <<< "$2"; shift 2 ;;
        --steps) IFS=',' read -r -a steps <<< "$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --max-new-tokens) max_new_tokens="$2"; shift 2 ;;
        --max-pixels) max_pixels="$2"; shift 2 ;;
        *) printf '{"event":"eval_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

screen_root="${remote_root}/training_runs/${screen_id}"
split_root="${screen_root}/split"
validation_manifest="${split_root}/validation64_screen_seed${seed}.jsonl"
train_manifest="${split_root}/train128_screen_seed${seed}.jsonl"
[[ -n "${output_root}" ]] || output_root="${screen_root}/eval"

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
python="${env_dir}/bin/python"
evaluate_cli="${code_root}/tools/evaluation/evaluate_decoder_mask.py"

torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
machine_arch="$(uname -m)"
case "${machine_arch}" in
    aarch64|arm64) cuda_target_arch="aarch64" ;;
    *) cuda_target_arch="${machine_arch}" ;;
esac
cuda_library_path="${torch_lib}"
system_cuda_library="/usr/local/cuda/targets/${cuda_target_arch}-linux/lib"
if [[ -d "${system_cuda_library}" ]]; then
    cuda_library_path="${system_cuda_library}:${cuda_library_path}"
fi
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    if [[ -d "${component_lib}" ]]; then
        cuda_library_path="${cuda_library_path}:${component_lib}"
    fi
done

[[ -x "${python}" && -f "${evaluate_cli}" ]] || {
    printf '{"event":"eval_failed","error":"missing_source"}\n' >&2; exit 66
}
[[ -f "${validation_manifest}" ]] || {
    printf '{"event":"eval_failed","error":"missing_validation_split","path":"%s"}\n' "${validation_manifest}" >&2; exit 66
}

# AGENTS.md rule 9: only the allowlisted cards are queried, and all must be
# strictly below the limit before anything starts.
declare -A observed=()
while IFS=',' read -r observed_id utilization; do
    observed_id="${observed_id//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    observed[${observed_id}]="${utilization}"
done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
for gpu in "${gpu_array[@]}"; do
    [[ -n "${observed[${gpu}]+present}" ]] || {
        printf '{"event":"eval_failed","error":"requested_gpu_not_reported","gpu":"%s"}\n' "${gpu}" >&2; exit 69
    }
    (( observed[${gpu}] < gpu_utilization_limit )) || {
        printf '{"event":"eval_failed","error":"gpu_admission_failed","gpu":"%s","utilization":%s}\n' "${gpu}" "${observed[${gpu}]}" >&2; exit 75
    }
done

mkdir -p "${output_root}"
export PYTHONPATH="${code_root}/src:${code_root}"
export LD_LIBRARY_PATH="${cuda_library_path}"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${code_root}"

run_one() {
    local arm="$1" step="$2" card="$3"
    local checkpoint="${screen_root}/arms/${arm}/step-${step}"
    local out="${output_root}/${arm}_step-${step}"
    if [[ ! -d "${checkpoint}" ]]; then
        printf '{"event":"eval_failed","error":"missing_checkpoint","arm":"%s","step":%s,"dir":"%s"}\n' "${arm}" "${step}" "${checkpoint}" >&2
        return 1
    fi
    if [[ -e "${out}" ]]; then
        printf '{"event":"eval_skip","reason":"output_exists","out":"%s"}\n' "${out}"
        return 0
    fi
    local tmp="${output_root}/tmp/${arm}_step-${step}"
    mkdir -p "${tmp}"
    CUDA_VISIBLE_DEVICES="${card}" TMPDIR="${tmp}" HF_HOME="${tmp}/huggingface" \
        "${python}" "${evaluate_cli}" \
        --model-path "${model_dir}" \
        --checkpoint-dir "${checkpoint}" \
        --manifest "${validation_manifest}" \
        --train-manifest "${train_manifest}" \
        --output-dir "${out}" \
        --max-new-tokens "${max_new_tokens}" \
        --max-pixels "${max_pixels}" \
        --processor-mode slow > "${out}.log" 2>&1
    printf '{"event":"eval_complete","arm":"%s","step":%s,"gpu":"%s","out":"%s"}\n' "${arm}" "${step}" "${card}" "${out}"
}

jobs=()
for step in "${steps[@]}"; do
    for arm in "${arms[@]}"; do
        jobs+=("${arm}:${step}")
    done
done

index=0
failures=0
while (( index < ${#jobs[@]} )); do
    pids=()
    slot=0
    while (( slot < ${#gpu_array[@]} && index < ${#jobs[@]} )); do
        IFS=':' read -r job_arm job_step <<< "${jobs[${index}]}"
        run_one "${job_arm}" "${job_step}" "${gpu_array[${slot}]}" &
        pids+=("$!")
        index=$((index + 1))
        slot=$((slot + 1))
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || failures=$((failures + 1))
    done
done

if (( failures > 0 )); then
    printf '{"event":"eval_failed","error":"wave_failures","count":%s}\n' "${failures}" >&2
    exit 1
fi
printf '{"event":"eval_complete","screen_id":"%s","jobs":%s,"output_root":"%s","test_used_for_selection":false}\n' \
    "${screen_id}" "${#jobs[@]}" "${output_root}"
