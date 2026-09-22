#!/usr/bin/env bash
# Bias-strength sweep for the GT-mask oracle.
#
# The zero-shot oracle at bias_max=2.0 is *worse* than B0 (CER 0.297 vs 0.287)
# with a pure over-generation/insertion failure mode, and the line oracle is
# worse still (0.356).  That shows the routing *information* is real (correct
# mask 0.297 < trained dense mask 0.37-0.58) but the additive-bias *mechanism*
# at beta=2.0 cannot turn it into a gain.  This sweep lowers bias_max to find
# whether any strength makes the GT-mask injection neutral or beneficial vs B0.
#
# Five beta values on five idle cards, reusing the exact 64-page validation
# manifest of the prior screen.  ``--oracle-mode token`` plus ``--checkpoint-dir``
# is the model-vs-mask-shape isolation screen; the defaults preserve the prior
# zero-shot window sweep.
#
# This reads evaluation ground truth.  Diagnostic only: never enters checkpoint
# selection and is never reported as a deployable result.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
sparse_root="${GLMOCR_MTHV2_SPARSE_Q32_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24}"

screen_id="glmocr_decoder_mask_bias_sweep_v1"
gpu_ids="0,1,2,3,4"
bias_values="0.1 0.25 0.5 1.0 1.5"
gpu_utilization_limit=50
max_pixels=4000000
max_eval_new_tokens=1536
oracle_mode="window"
window_size="3 5"
layout_mode="geometry"
checkpoint_dir=""
validation_manifest="${GLMOCR_VALIDATION_MANIFEST:-${sparse_root}/validation/manifest.char.jsonl}"
train_manifest="${GLMOCR_TRAIN_MANIFEST:-${sparse_root}/train/manifest.char.jsonl}"
processor_mode="fast"
raster_mode="hard"
pointer_mode="synced"
split_layer=0
seed=42
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --screen-id) screen_id="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --bias-values) bias_values="$2"; shift 2 ;;
        --oracle-mode) oracle_mode="$2"; shift 2 ;;
        --layout-mode) layout_mode="$2"; shift 2 ;;
        --checkpoint-dir) checkpoint_dir="$2"; shift 2 ;;
        --validation-manifest) validation_manifest="$2"; shift 2 ;;
        --train-manifest) train_manifest="$2"; shift 2 ;;
        --processor-mode) processor_mode="$2"; shift 2 ;;
        --raster-mode) raster_mode="$2"; shift 2 ;;
        --pointer-mode) pointer_mode="$2"; shift 2 ;;
        --split-layer) split_layer="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        --remote-root) remote_root="$2"; shift 2 ;;
        --code-root) code_root="$2"; shift 2 ;;
        --env-dir) env_dir="$2"; shift 2 ;;
        --model-dir) model_dir="$2"; shift 2 ;;
        *) printf '{"event":"bias_sweep_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
read -r -a bias_array <<< "${bias_values}"
[[ "${#gpu_array[@]}" -eq "${#bias_array[@]}" ]] || {
    printf '{"event":"bias_sweep_failed","error":"gpu_bias_count_mismatch","gpus":%s,"bias":%s}\n' \
        "${#gpu_array[@]}" "${#bias_array[@]}" >&2; exit 64
}

python="${env_dir}/bin/python3"
oracle_cli="${code_root}/tools/oracle_decoder_mask_generate.py"

screen_root="${remote_root}/training_runs/${screen_id}"
oracle_root="${screen_root}/oracle"
launcher_log="${remote_root}/runs/${screen_id}.launcher.log"

setup_environment() {
    local cuda_libraries="${env_dir}/lib/python3.11/site-packages/torch/lib"
    local cuda_arch
    cuda_arch="$(uname -m)"
    [[ "${cuda_arch}" != "arm64" ]] || cuda_arch="aarch64"
    local system_cuda="/usr/local/cuda/targets/${cuda_arch}-linux/lib"
    [[ ! -d "${system_cuda}" ]] || cuda_libraries="${system_cuda}:${cuda_libraries}"
    local component lib
    for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
        lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
        [[ ! -d "${lib}" ]] || cuda_libraries="${cuda_libraries}:${lib}"
    done
    export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
}

write_status() {
    mkdir -p "${screen_root}/status"
    printf '{"status":"%s","screen_id":"%s","gpu_ids":"%s"}\n' \
        "$1" "${screen_id}" "${gpu_ids}" > "${screen_root}/status/screen.json"
}

run_oracle() {
    local bias="$1" gpu="$2"
    local out_dir="${oracle_root}/beta_${bias}"
    local tmp="${screen_root}/tmp/oracle_beta_${bias}"
    local checkpoint_args=""
    if [[ -n "${checkpoint_dir}" ]]; then
        checkpoint_args="--checkpoint-dir ${checkpoint_dir} --layout-mode ${layout_mode}"
    fi
    mkdir -p "${out_dir}" "${tmp}"
    printf '{"event":"bias_sweep_oracle_launched","bias":%s,"gpu":"%s"}\n' "${bias}" "${gpu}"
    TMPDIR="${tmp}" HF_HOME="${tmp}/huggingface" CUDA_VISIBLE_DEVICES="${gpu}" \
        setsid nohup bash -c "exec ${python} ${oracle_cli} \
        --model-path ${model_dir} \
        --manifest ${validation_manifest} \
        --train-manifest ${train_manifest} \
        --output-dir ${out_dir} \
        --oracle-mode ${oracle_mode} --window-size ${window_size} \
        ${checkpoint_args} \
        --bias-max ${bias} \
        --max-pixels ${max_pixels} \
        --max-new-tokens ${max_eval_new_tokens} \
        --processor-mode ${processor_mode} \
        --raster-mode ${raster_mode} \
        --pointer-mode ${pointer_mode} \
        --split-layer ${split_layer} \
        --seed ${seed}" > "${out_dir}/run.log" 2>&1 < /dev/null &
    disown 2>/dev/null || true
}

run_inner() {
    trap 'rc=$?; write_status failed; exit "$rc"' ERR
    [[ -x "${python}" && -f "${oracle_cli}" ]] || {
        printf '{"event":"bias_sweep_failed","error":"missing_source"}\n' >&2; exit 66
    }
    [[ -f "${validation_manifest}" && -f "${train_manifest}" ]] || {
        printf '{"event":"bias_sweep_failed","error":"missing_manifest"}\n' >&2; exit 66
    }
    if [[ -n "${checkpoint_dir}" ]]; then
        [[ -f "${checkpoint_dir}/adapter.safetensors" && -f "${checkpoint_dir}/decoder_lora.safetensors" ]] || {
            printf '{"event":"bias_sweep_failed","error":"missing_checkpoint","checkpoint_dir":"%s"}\n' \
                "${checkpoint_dir}" >&2; exit 66; }
    fi
    # A100 admission: every target card must be under the utilization limit, or
    # the whole sweep exits without touching any GPU.
    declare -A observed=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
        observed[${observed_id}]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for gpu in "${gpu_array[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || {
            printf '{"event":"bias_sweep_failed","error":"gpu_not_reported","gpu":"%s"}\n' "${gpu}" >&2; exit 69; }
        (( observed[${gpu}] < gpu_utilization_limit )) || {
            printf '{"event":"bias_sweep_failed","error":"gpu_admission_failed","gpu":"%s","utilization":%s}\n' \
                "${gpu}" "${observed[${gpu}]}" >&2; exit 75; }
    done
    printf '{"event":"bias_sweep_gpu_admission_ok","gpu_ids":"%s","limit":%s}\n' "${gpu_ids}" "${gpu_utilization_limit}"

    setup_environment
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    mkdir -p "${screen_root}/logs" "${screen_root}/status" "${oracle_root}" "${remote_root}/runs"
    write_status running
    cd "${code_root}"

    for i in "${!bias_array[@]}"; do
        run_oracle "${bias_array[$i]}" "${gpu_array[$i]}"
    done

    write_status launched
    trap - ERR
    printf '{"event":"bias_sweep_launched","screen_id":"%s","bias_values":"%s","oracle_mode":"%s","checkpoint_dir":"%s","validation_manifest":"%s","processor_mode":"%s","raster_mode":"%s","pointer_mode":"%s","split_layer":%s,"test_used_for_selection":false}\n' \
        "${screen_id}" "${bias_values}" "${oracle_mode}" "${checkpoint_dir}" "${validation_manifest}" "${processor_mode}" "${raster_mode}" "${pointer_mode}" "${split_layer}"
}

if (( foreground == 1 )); then
    run_inner
else
    mkdir -p "${remote_root}/runs"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    command_line="$(printf '%q ' bash "${script_path}" --foreground --screen-id "${screen_id}" --gpu-ids "${gpu_ids}" --bias-values "${bias_values}" --oracle-mode "${oracle_mode}" --layout-mode "${layout_mode}" --checkpoint-dir "${checkpoint_dir}" --validation-manifest "${validation_manifest}" --train-manifest "${train_manifest}" --processor-mode "${processor_mode}" --raster-mode "${raster_mode}" --pointer-mode "${pointer_mode}" --split-layer "${split_layer}" --seed "${seed}" --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}")"
    setsid nohup bash -c "exec ${command_line}" > "${launcher_log}" 2>&1 < /dev/null &
    disown 2>/dev/null || true
    printf '{"event":"bias_sweep_armed","screen_id":"%s","gpu_ids":"%s","bias_values":"%s","oracle_mode":"%s","checkpoint_dir":"%s","validation_manifest":"%s","processor_mode":"%s","raster_mode":"%s","pointer_mode":"%s","split_layer":%s,"log":"%s","test_used_for_selection":false}\n' \
        "${screen_id}" "${gpu_ids}" "${bias_values}" "${oracle_mode}" "${checkpoint_dir}" "${validation_manifest}" "${processor_mode}" "${raster_mode}" "${pointer_mode}" "${split_layer}" "${launcher_log}"
fi
