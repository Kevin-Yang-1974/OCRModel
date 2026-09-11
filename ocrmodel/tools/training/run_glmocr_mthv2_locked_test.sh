#!/usr/bin/env bash
# Evaluate one completed seed at the aggregate validation-selected step.
# This script is intentionally separate from the DDP training launcher so the
# test split cannot be read by training or checkpoint selection.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dataset_root="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
protocol_file="${GLMOCR_A100_MTHV2_PROTOCOL:-${remote_root}/protocols/mthv2_full_2159_240_800_v1.json}"
run_id="glmocr_mthv2_full_ddp_v1"
seed=42
gpu_id=0
gpu_ids=""
session=""
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) run_id="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --gpu-id) gpu_id="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --session) session="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        --remote-root) remote_root="$2"; shift 2 ;;
        --code-root) code_root="$2"; shift 2 ;;
        --env-dir) env_dir="$2"; shift 2 ;;
        --model-dir) model_dir="$2"; shift 2 ;;
        --dataset-root) dataset_root="$2"; shift 2 ;;
        --protocol-file) protocol_file="$2"; shift 2 ;;
        *) printf '{"event":"glmocr_locked_test_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ -z "${gpu_ids}" ]] && gpu_ids="${gpu_id}"
[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${seed}" =~ ^[0-9]+$ && "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    printf '{"event":"glmocr_locked_test_failed","error":"invalid_run_seed_or_gpu_ids"}\n' >&2
    exit 64
}
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
gpu_count="${#gpu_array[@]}"
declare -A seen_gpus=()
for gpu in "${gpu_array[@]}"; do
    [[ -z "${seen_gpus[${gpu}]+x}" ]] || {
        printf '{"event":"glmocr_locked_test_failed","error":"duplicate_gpu_id","gpu":%s}\n' "${gpu}" >&2
        exit 64
    }
    seen_gpus["${gpu}"]=1
done
gpu_id="${gpu_array[0]}"
[[ -z "${session}" ]] && session="${run_id}_test_seed${seed}"
[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '{"event":"glmocr_locked_test_failed","error":"invalid_session"}\n' >&2
    exit 64
}

python="${env_dir}/bin/python"
group_root="${remote_root}/training_runs/${run_id}"
run_dir="${group_root}/seed${seed}"
selection_file="${group_root}/selection.json"
train_manifest="${dataset_root}/train/manifest.jsonl"
test_manifest="${dataset_root}/test/manifest.jsonl"
launcher_log="${remote_root}/runs/${run_id}.locked-test.seed${seed}.launcher.log"
test_log="${group_root}/logs/seed${seed}.locked-test.log"
torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
cuda_library_path="/usr/local/cuda/targets/x86_64-linux/lib:${torch_lib}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    if [[ -d "${component_lib}" ]]; then
        cuda_library_path="${cuda_library_path}:${component_lib}"
    fi
done

preflight_paths() {
    [[ -x "${python}" ]] || {
        printf '{"event":"glmocr_locked_test_failed","error":"missing_python"}\n' >&2
        exit 66
    }
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || {
        printf '{"event":"glmocr_locked_test_failed","error":"missing_model"}\n' >&2
        exit 66
    }
    [[ -f "${code_root}/tools/evaluate_glmocr_locked_test.py" ]] || {
        printf '{"event":"glmocr_locked_test_failed","error":"missing_evaluator"}\n' >&2
        exit 66
    }
    [[ -f "${train_manifest}" && -f "${test_manifest}" && -f "${protocol_file}" ]] || {
        printf '{"event":"glmocr_locked_test_failed","error":"missing_test_inputs"}\n' >&2
        exit 66
    }
    [[ -f "${run_dir}/summary.json" && -f "${run_dir}/selection.json" && -f "${run_dir}/COMPLETED" ]] || {
        printf '{"event":"glmocr_locked_test_failed","error":"training_run_not_complete"}\n' >&2
        exit 66
    }
    [[ -f "${selection_file}" ]] || {
        printf '{"event":"glmocr_locked_test_failed","error":"aggregate_selection_missing","selection_file":"%s"}\n' "${selection_file}" >&2
        exit 66
    }
    [[ ! -e "${run_dir}/locked-test" ]] || {
        printf '{"event":"glmocr_locked_test_failed","error":"locked_test_output_already_exists"}\n' >&2
        exit 74
    }
    [[ ! -e "${run_dir}/locked-test-shards" ]] || {
        printf '{"event":"glmocr_locked_test_failed","error":"locked_test_shards_already_exists"}\n' >&2
        exit 74
    }
}

query_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf '{"event":"glmocr_locked_test_failed","error":"nvidia_smi_missing"}\n' >&2
        exit 69
    }
    utilization_json=""
    for gpu in "${gpu_array[@]}"; do
        row="$(nvidia-smi -i "${gpu}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)"
        IFS=',' read -r observed_id utilization <<< "${row}"
        observed_id="${observed_id//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${observed_id}" == "${gpu}" && "${utilization}" =~ ^[0-9]+$ ]] || {
            printf '{"event":"glmocr_locked_test_failed","error":"cannot_parse_gpu_utilization","gpu":%s}\n' "${gpu}" >&2
            exit 69
        }
        (( utilization < 50 )) || {
            printf '{"event":"glmocr_locked_test_failed","error":"gpu_admission_failed","gpu":%s,"utilization":%s}\n' "${gpu}" "${utilization}" >&2
            exit 75
        }
        utilization_json+="${gpu}:${utilization},"
    done
    utilization_json="${utilization_json%,}"
    printf '{"event":"glmocr_locked_test_gpu_admission_ok","gpu_ids":"%s","utilization":"%s"}\n' "${gpu_ids}" "${utilization_json}"
}

run_inner() {
    preflight_paths
    query_gpus
    mkdir -p "${group_root}/logs" "${group_root}/tmp" "${remote_root}/runs"
    if (( gpu_count == 1 )); then
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export TMPDIR="${group_root}/tmp/test-seed${seed}"
        export HF_HOME="${TMPDIR}/huggingface"
        export TRANSFORMERS_CACHE="${HF_HOME}"
        export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
        export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        mkdir -p "${TMPDIR}" "${HF_HOME}"
        cd "${code_root}"
        "${python}" -m tools.evaluate_glmocr_locked_test \
            --model-path "${model_dir}" \
            --train-manifest "${train_manifest}" \
            --test-manifest "${test_manifest}" \
            --protocol-file "${protocol_file}" \
            --run-dir "${run_dir}" \
            --selection-file "${selection_file}" \
            --seed "${seed}" \
            > "${test_log}" 2>&1
    else
        shard_root="${run_dir}/locked-test-shards"
        mkdir -p "${shard_root}"
        shard_pids=()
        for shard_index in "${!gpu_array[@]}"; do
            gpu="${gpu_array[${shard_index}]}"
            shard_dir="${shard_root}/shard${shard_index}"
            shard_tmp="${group_root}/tmp/test-seed${seed}-shard${shard_index}"
            shard_log="${group_root}/logs/seed${seed}.locked-test.shard${shard_index}.log"
            (
                export CUDA_VISIBLE_DEVICES="${gpu}"
                export TMPDIR="${shard_tmp}"
                export HF_HOME="${TMPDIR}/huggingface"
                export TRANSFORMERS_CACHE="${HF_HOME}"
                export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
                export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
                export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
                export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
                export CUBLAS_WORKSPACE_CONFIG=:4096:8
                export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
                mkdir -p "${TMPDIR}" "${HF_HOME}"
                cd "${code_root}"
                exec "${python}" -m tools.evaluate_glmocr_locked_test \
                    --model-path "${model_dir}" \
                    --train-manifest "${train_manifest}" \
                    --test-manifest "${test_manifest}" \
                    --protocol-file "${protocol_file}" \
                    --run-dir "${run_dir}" \
                    --output-dir "${shard_dir}" \
                    --selection-file "${selection_file}" \
                    --seed "${seed}" \
                    --test-shard-index "${shard_index}" \
                    --test-shard-count "${gpu_count}"
            ) > "${shard_log}" 2>&1 &
            shard_pids+=("$!")
        done
        failed=0
        for pid in "${shard_pids[@]}"; do
            if ! wait "${pid}"; then
                failed=1
            fi
        done
        (( failed == 0 )) || {
            printf '{"event":"glmocr_locked_test_failed","error":"test_shard_failed","gpu_ids":"%s"}\n' "${gpu_ids}" >&2
            exit 1
        }
        export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
        export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export HF_HOME="${group_root}/tmp/merge-test-seed${seed}/huggingface"
        export TRANSFORMERS_CACHE="${HF_HOME}"
        export TMPDIR="${group_root}/tmp/merge-test-seed${seed}"
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
        mkdir -p "${TMPDIR}" "${HF_HOME}"
        cd "${code_root}"
        "${python}" -m tools.merge_glmocr_locked_test \
            --run-dir "${run_dir}" \
            --shards-dir "${shard_root}" \
            --output-dir "${run_dir}/locked-test" \
            --train-manifest "${train_manifest}" \
            --test-manifest "${test_manifest}" \
            --protocol-file "${protocol_file}" \
            --selection-file "${selection_file}" \
            --seed "${seed}" \
            --gpu-ids "${gpu_ids}" \
            > "${test_log}" 2>&1
    fi
    printf '{"event":"glmocr_locked_test_complete","run_id":"%s","seed":%s,"test_used_for_selection":false,"summary":"%s"}\n' \
        "${run_id}" "${seed}" "${run_dir}/locked-test/locked_test_summary.json"
}

preflight_paths
if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || {
        printf '{"event":"glmocr_locked_test_failed","error":"tmux_missing"}\n' >&2
        exit 69
    }
    tmux has-session -t "${session}" 2>/dev/null && {
        printf '{"event":"glmocr_locked_test_failed","error":"session_already_exists","session":"%s"}\n' "${session}" >&2
        exit 73
    }
    mkdir -p "${remote_root}/runs"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    command_line="$(printf '%q ' bash "${script_path}" --foreground --run-id "${run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}" --dataset-root "${dataset_root}" --protocol-file "${protocol_file}")"
    tmux new-session -d -s "${session}" "cd $(printf '%q' "${code_root}") && exec ${command_line} >$(printf '%q' "${launcher_log}") 2>&1"
    printf '{"event":"glmocr_locked_test_armed","session":"%s","run_id":"%s","seed":%s,"gpu_ids":"%s","test_used_for_selection":false,"log":"%s"}\n' \
        "${session}" "${run_id}" "${seed}" "${gpu_ids}" "${launcher_log}"
else
    run_inner
fi
