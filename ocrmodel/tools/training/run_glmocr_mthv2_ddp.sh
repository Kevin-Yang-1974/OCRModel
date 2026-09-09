#!/usr/bin/env bash
# Launch one true five-process DDP seed on the complete official MTHv2 split.
# This intentionally replaces the old independent single-GPU comparison only
# for the full-data geometry protocol; the legacy launcher remains separate.
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
gpu_ids="0,1,2,3,4"
gpu_utilization_limit=50
max_steps=3456
lr_schedule_steps=3456
learning_rate=5e-5
warmup_steps=216
min_lr_ratio=0.1
gradient_accumulation_steps=1
initial_residual_scale=0.0
gate_freeze_steps=0
auxiliary_weight=0.2
auxiliary_weight_start=""
auxiliary_ramp_steps=0
diagnostic_steps=""
layout_loss_profile="full"
validation_interval=432
max_eval_new_tokens=1536
log_steps=16
use_validity_head=0
initial_valid_probability=0.05
skip_selection=0
without_test=0
session=""
foreground=0
smoke=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) run_id="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --gpu-utilization-limit) gpu_utilization_limit="$2"; shift 2 ;;
        --max-steps) max_steps="$2"; shift 2 ;;
        --lr-schedule-steps) lr_schedule_steps="$2"; shift 2 ;;
        --learning-rate) learning_rate="$2"; shift 2 ;;
        --warmup-steps) warmup_steps="$2"; shift 2 ;;
        --min-lr-ratio) min_lr_ratio="$2"; shift 2 ;;
        --gradient-accumulation-steps) gradient_accumulation_steps="$2"; shift 2 ;;
        --initial-residual-scale) initial_residual_scale="$2"; shift 2 ;;
        --gate-freeze-steps) gate_freeze_steps="$2"; shift 2 ;;
        --auxiliary-weight) auxiliary_weight="$2"; shift 2 ;;
        --auxiliary-weight-start) auxiliary_weight_start="$2"; shift 2 ;;
        --auxiliary-ramp-steps) auxiliary_ramp_steps="$2"; shift 2 ;;
        --diagnostic-steps) diagnostic_steps="$2"; shift 2 ;;
        --layout-loss-profile) layout_loss_profile="$2"; shift 2 ;;
        --validation-interval) validation_interval="$2"; shift 2 ;;
        --max-eval-new-tokens) max_eval_new_tokens="$2"; shift 2 ;;
        --log-steps) log_steps="$2"; shift 2 ;;
        --use-validity-head) use_validity_head=1; shift ;;
        --initial-valid-probability) initial_valid_probability="$2"; shift 2 ;;
        --skip-selection) skip_selection=1; shift ;;
        --without-test) without_test=1; shift ;;
        --session) session="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        --smoke) smoke=1; shift ;;
        --remote-root) remote_root="$2"; shift 2 ;;
        --code-root) code_root="$2"; shift 2 ;;
        --env-dir) env_dir="$2"; shift 2 ;;
        --model-dir) model_dir="$2"; shift 2 ;;
        --dataset-root) dataset_root="$2"; shift 2 ;;
        --protocol-file) protocol_file="$2"; shift 2 ;;
        *) printf '{"event":"glmocr_mthv2_ddp_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${seed}" =~ ^[0-9]+$ ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_run_or_seed"}\n' >&2
    exit 64
}
[[ "${max_steps}" =~ ^[1-9][0-9]*$ && "${lr_schedule_steps}" =~ ^[1-9][0-9]*$ ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_step_configuration"}\n' >&2
    exit 64
}
[[ "${warmup_steps}" =~ ^[0-9]+$ && "${gradient_accumulation_steps}" =~ ^[1-9][0-9]*$ ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_optimization_configuration"}\n' >&2
    exit 64
}
[[ "${min_lr_ratio}" =~ ^(0|0\.[0-9]+|1(\.0)?)$ ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_min_lr_ratio"}\n' >&2
    exit 64
}
[[ "${gate_freeze_steps}" =~ ^[0-9]+$ && "${auxiliary_ramp_steps}" =~ ^[0-9]+$ ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_schedule_configuration"}\n' >&2
    exit 64
}
[[ "${validation_interval}" =~ ^[1-9][0-9]*$ && "${max_eval_new_tokens}" =~ ^[1-9][0-9]*$ ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_evaluation_configuration"}\n' >&2
    exit 64
}
[[ "${log_steps}" =~ ^[1-9][0-9]*$ ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_log_steps"}\n' >&2
    exit 64
}
[[ "${initial_valid_probability}" =~ ^0\.[0-9]+$ || "${initial_valid_probability}" == "1.0" ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_initial_valid_probability"}\n' >&2
    exit 64
}
case "${layout_loss_profile}" in
    full|ocr_only|no_assignment|no_assignment_validity|no_geometry) ;;
    *) printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_layout_loss_profile"}\n' >&2; exit 64 ;;
esac
(( warmup_steps < lr_schedule_steps && warmup_steps < max_steps )) || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"warmup_must_be_shorter_than_horizon"}\n' >&2
    exit 64
}
(( gate_freeze_steps <= max_steps && auxiliary_ramp_steps <= max_steps )) || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"schedule_exceeds_max_steps"}\n' >&2
    exit 64
}
if [[ -z "${auxiliary_weight_start}" ]]; then
    auxiliary_weight_start="${auxiliary_weight}"
fi
[[ "${initial_residual_scale}" =~ ^-?(0|0\.[0-9]+|1\.0)$ ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_initial_residual_scale"}\n' >&2
    exit 64
}
(( lr_schedule_steps <= max_steps )) || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"lr_schedule_exceeds_max_steps"}\n' >&2
    exit 64
}
[[ -z "${session}" ]] && session="${run_id}_seed${seed}"
[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_session"}\n' >&2
    exit 64
}

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
(( ${#gpu_array[@]} == 5 )) || {
    printf '{"event":"glmocr_mthv2_ddp_failed","error":"exactly_five_gpus_required","gpu_ids":"%s"}\n' "${gpu_ids}" >&2
    exit 64
}
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"invalid_gpu_id","gpu":"%s"}\n' "${gpu}" >&2
        exit 64
    }
    [[ -z "${seen_gpu[${gpu}]+present}" ]] || {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"duplicate_gpu_id","gpu":"%s"}\n' "${gpu}" >&2
        exit 64
    }
    seen_gpu[${gpu}]=1
done

python="${env_dir}/bin/python"
torchrun="${env_dir}/bin/torchrun"
audit_tool="${code_root}/tools/audit_mthv2_manifest.py"
train_manifest="${dataset_root}/train/manifest.jsonl"
validation_manifest="${dataset_root}/validation/manifest.jsonl"
test_manifest="${dataset_root}/test/manifest.jsonl"
group_root="${remote_root}/training_runs/${run_id}"
run_dir="${group_root}/seed${seed}"
launcher_log="${remote_root}/runs/${run_id}.seed${seed}.launcher.log"
run_log="${group_root}/logs/seed${seed}.ddp.log"
smoke_dir="${group_root}/smoke/seed${seed}"

torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
cuda_library_path="/usr/local/cuda/targets/x86_64-linux/lib:${torch_lib}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    if [[ -d "${component_lib}" ]]; then
        cuda_library_path="${cuda_library_path}:${component_lib}"
    fi
done

preflight_paths() {
    [[ -x "${python}" && -x "${torchrun}" ]] || {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"missing_python_or_torchrun"}\n' >&2
        exit 66
    }
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"missing_model"}\n' >&2
        exit 66
    }
    [[ -f "${code_root}/src/layout_ocr/train_screen.py" && -f "${audit_tool}" ]] || {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"missing_glmocr_source"}\n' >&2
        exit 66
    }
    [[ -f "${train_manifest}" && -f "${validation_manifest}" ]] || {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"missing_mthv2_manifest"}\n' >&2
        exit 66
    }
    if (( without_test == 0 )) && [[ ! -f "${test_manifest}" ]]; then
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"missing_test_manifest"}\n' >&2
        exit 66
    fi
    [[ ! -e "${run_dir}" ]] || {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"output_already_exists","run_dir":"%s"}\n' "${run_dir}" >&2
        exit 74
    }
}

query_gpu_utilization() {
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"nvidia_smi_missing"}\n' >&2
        exit 69
    }
    declare -A observed=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${observed_id}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || {
            printf '{"event":"glmocr_mthv2_ddp_failed","error":"cannot_parse_gpu_utilization"}\n' >&2
            exit 69
        }
        observed[${observed_id}]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for gpu in "${gpu_array[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || {
            printf '{"event":"glmocr_mthv2_ddp_failed","error":"requested_gpu_not_reported","gpu":"%s"}\n' "${gpu}" >&2
            exit 69
        }
        (( observed[${gpu}] < gpu_utilization_limit )) || {
            printf '{"event":"glmocr_mthv2_ddp_failed","error":"gpu_admission_failed","gpu":"%s","utilization":%s,"limit":%s}\n' "${gpu}" "${observed[${gpu}]}" "${gpu_utilization_limit}" >&2
            exit 75
        }
    done
    printf '{"event":"glmocr_mthv2_gpu_admission_ok","gpu_ids":"%s","utilization":{' "${gpu_ids}"
    local first=1
    for gpu in "${gpu_array[@]}"; do
        (( first == 1 )) || printf ','
        printf '"%s":%s' "${gpu}" "${observed[${gpu}]}"
        first=0
    done
    printf '}}\n'
}

prepare_protocol() {
    mkdir -p "$(dirname -- "${protocol_file}")"
    audit_args=(
        --train-manifest "${train_manifest}"
        --validation-manifest "${validation_manifest}"
        --num-queries 512
    )
    if (( without_test == 0 )); then
        audit_args+=(--test-manifest "${test_manifest}")
    else
        audit_args+=(--without-test)
    fi
    if [[ ! -f "${protocol_file}" ]]; then
        "${python}" "${audit_tool}" "${audit_args[@]}" \
            --output "${protocol_file}" >/dev/null
    else
        "${python}" "${audit_tool}" "${audit_args[@]}" >/dev/null
    fi
    if (( without_test == 1 )); then
        command -v jq >/dev/null 2>&1 || {
            printf '{"event":"glmocr_mthv2_ddp_failed","error":"jq_required_for_test_protocol_guard"}\n' >&2
            exit 66
        }
        jq -e '.test_manifest_read == false' "${protocol_file}" >/dev/null 2>&1 || {
            printf '{"event":"glmocr_mthv2_ddp_failed","error":"protocol_reads_test"}\n' >&2
            exit 66
        }
    fi
}

write_status() {
    local status="$1"
    local skip_selection_json=false
    local test_manifest_read_json=true
    local use_validity_head_json=false
    local stop_reason_json=null
    local validation_stop_only_json=false
    (( skip_selection == 1 )) && skip_selection_json=true
    (( without_test == 1 )) && test_manifest_read_json=false
    (( use_validity_head == 1 )) && use_validity_head_json=true
    if [[ "${status}" == "stopped_by_user" ]]; then
        stop_reason_json='"user_requested_validation_stop"'
        validation_stop_only_json=true
    fi
    mkdir -p "${group_root}/status"
    printf '{"status":"%s","run_id":"%s","seed":%s,"world_size":5,"global_batch_size":5,"effective_global_batch_size":%s,"gradient_accumulation_steps":%s,"max_steps":%s,"lr_schedule_steps":%s,"learning_rate":%s,"warmup_steps":%s,"min_lr_ratio":%s,"initial_residual_scale":%s,"gate_freeze_steps":%s,"auxiliary_weight_start":%s,"auxiliary_weight":%s,"auxiliary_ramp_steps":%s,"layout_loss_profile":"%s","use_validity_head":%s,"initial_valid_probability":%s,"validation_interval":%s,"log_steps":%s,"max_eval_new_tokens":%s,"skip_selection":%s,"test_manifest_read":%s,"test_used_for_selection":false,"stop_reason":%s,"validation_stop_only":%s}\n' \
        "${status}" "${run_id}" "${seed}" "$((5 * gradient_accumulation_steps))" "${gradient_accumulation_steps}" "${max_steps}" "${lr_schedule_steps}" "${learning_rate}" "${warmup_steps}" "${min_lr_ratio}" "${initial_residual_scale}" "${gate_freeze_steps}" "${auxiliary_weight_start}" "${auxiliary_weight}" "${auxiliary_ramp_steps}" "${layout_loss_profile}" "${use_validity_head_json}" "${initial_valid_probability}" "${validation_interval}" "${log_steps}" "${max_eval_new_tokens}" "${skip_selection_json}" "${test_manifest_read_json}" "${stop_reason_json}" "${validation_stop_only_json}" > "${group_root}/status/seed${seed}.json"
}

run_inner() {
    trap 'rc=$?; write_status failed; exit "$rc"' ERR
    preflight_paths
    prepare_protocol
    query_gpu_utilization
    mkdir -p "${group_root}/logs" "${group_root}/status" "${group_root}/tmp" "${remote_root}/runs"
    write_status running
    local output_dir="${run_dir}"
    if (( smoke == 1 )); then
        [[ ! -e "${smoke_dir}" ]] || {
            printf '{"event":"glmocr_mthv2_ddp_failed","error":"smoke_output_already_exists","output_dir":"%s"}\n' "${smoke_dir}" >&2
            exit 74
        }
        output_dir="${smoke_dir}"
    fi
    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    export TMPDIR="${group_root}/tmp/seed${seed}"
    export HF_HOME="${TMPDIR}/huggingface"
    export TRANSFORMERS_CACHE="${HF_HOME}"
    export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export NCCL_P2P_DISABLE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    mkdir -p "${TMPDIR}" "${HF_HOME}"
    cd "${code_root}"
    if (( smoke == 1 )); then
        export GLMOCR_DDP_TIMEOUT_SECONDS=120
        smoke_args=(
            --model-path "${model_dir}"
            --train-manifest "${train_manifest}"
            --output-dir "${output_dir}"
            --num-queries 512 --seed "${seed}"
            --layout-loss-profile "${layout_loss_profile}"
            --residual-scale-cap 0.03
            --initial-residual-scale "${initial_residual_scale}"
            --initial-valid-probability "${initial_valid_probability}"
        )
        (( use_validity_head == 1 )) && smoke_args+=(--use-validity-head)
        "${torchrun}" --standalone --nnodes=1 --nproc_per_node=5 \
            -m tools.smoke_glmocr_ddp \
            "${smoke_args[@]}" \
            > "${group_root}/logs/seed${seed}.smoke.log" 2>&1
        write_status complete
        trap - ERR
        printf '{"event":"glmocr_mthv2_ddp_smoke_complete","run_id":"%s","seed":%s,"output_dir":"%s","test_used_for_selection":false}\n' \
            "${run_id}" "${seed}" "${output_dir}"
        return 0
    fi
    extra_args=()
    if [[ -n "${diagnostic_steps}" ]]; then
        extra_args+=(--diagnostic-steps "${diagnostic_steps}")
    fi
    if (( skip_selection == 1 )); then
        extra_args+=(--skip-selection)
    fi
    validity_args=(--initial-valid-probability "${initial_valid_probability}")
    (( use_validity_head == 1 )) && validity_args+=(--use-validity-head)
    ddp_rc=0
    if "${torchrun}" --standalone --nnodes=1 --nproc_per_node=5 \
        -m layout_ocr.train_screen \
        --distributed-strategy ddp \
        --mode geometry \
        --model-path "${model_dir}" \
        --train-manifest "${train_manifest}" \
        --validation-manifest "${validation_manifest}" \
        --protocol-file "${protocol_file}" \
        --output-dir "${output_dir}" \
        --per-device-batch-size 1 \
        --gradient-accumulation-steps "${gradient_accumulation_steps}" \
        --max-steps "${max_steps}" \
        --num-queries 512 \
        --seed "${seed}" \
        --learning-rate "${learning_rate}" \
        --warmup-steps "${warmup_steps}" \
        --lr-schedule-steps "${lr_schedule_steps}" \
        --min-lr-ratio "${min_lr_ratio}" \
        --residual-scale-cap 0.03 \
        --initial-residual-scale "${initial_residual_scale}" \
        --gate-freeze-steps "${gate_freeze_steps}" \
        --auxiliary-weight "${auxiliary_weight}" \
        --auxiliary-weight-start "${auxiliary_weight_start}" \
        --auxiliary-ramp-steps "${auxiliary_ramp_steps}" \
        --max-grad-norm 1.0 \
        --max-pixels 1003520 \
        --max-eval-new-tokens "${max_eval_new_tokens}" \
        --validation-interval "${validation_interval}" \
        --log-steps "${log_steps}" \
        --adapter-precision fp32 \
        --layout-loss-profile "${layout_loss_profile}" \
        --query-assignment hungarian \
        --processor-mode fast \
        "${validity_args[@]}" \
        "${extra_args[@]}" \
        > "${run_log}" 2>&1; then
        ddp_rc=0
    else
        ddp_rc=$?
    fi
    if (( ddp_rc == 130 || ddp_rc == 143 )); then
        trap - ERR
        write_status stopped_by_user
        printf '{"event":"glmocr_mthv2_ddp_stopped","status":"stopped_by_user","run_id":"%s","seed":%s,"validation_stop_only":true,"test_used_for_selection":false}\n' "${run_id}" "${seed}"
        exit "${ddp_rc}"
    fi
    (( ddp_rc == 0 )) || return "${ddp_rc}"
    if (( skip_selection == 1 )); then
        [[ -f "${run_dir}/summary.json" && -f "${run_dir}/COMPLETED" ]] || {
            write_status failed
            printf '{"event":"glmocr_mthv2_ddp_failed","error":"run_completed_without_summary","run_dir":"%s"}\n' "${run_dir}" >&2
            exit 1
        }
    elif [[ ! -f "${run_dir}/summary.json" || ! -f "${run_dir}/selection.json" || ! -f "${run_dir}/COMPLETED" ]]; then
        write_status failed
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"run_completed_without_summary","run_dir":"%s"}\n' "${run_dir}" >&2
        exit 1
    fi
    write_status complete
    trap - ERR
    "${python}" - "${run_dir}/summary.json" "${skip_selection}" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
skip_selection = bool(int(sys.argv[2]))
selection = None
if not skip_selection:
    selection = json.loads(open(sys.argv[1].replace("summary.json", "selection.json"), encoding="utf-8").read())
print(json.dumps({
    "event": "glmocr_mthv2_ddp_complete",
    "status": payload.get("status"),
    "seed": payload.get("seed"),
    "final_step": payload.get("validation", {}).get("step"),
    "selected_step": selection.get("selected_step") if selection else None,
    "selection_performed": not skip_selection,
    "test_used_for_selection": payload.get("test_used_for_selection"),
}, ensure_ascii=False, separators=(",", ":")))
PY
}

preflight_paths
if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"tmux_missing"}\n' >&2
        exit 69
    }
    tmux has-session -t "${session}" 2>/dev/null && {
        printf '{"event":"glmocr_mthv2_ddp_failed","error":"session_already_exists","session":"%s"}\n' "${session}" >&2
        exit 73
    }
    mkdir -p "${remote_root}/runs"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    child_args=(
        bash "${script_path}" --foreground --run-id "${run_id}" --seed "${seed}"
        --gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}"
        --max-steps "${max_steps}" --lr-schedule-steps "${lr_schedule_steps}"
        --learning-rate "${learning_rate}" --warmup-steps "${warmup_steps}"
        --min-lr-ratio "${min_lr_ratio}"
        --gradient-accumulation-steps "${gradient_accumulation_steps}"
        --initial-residual-scale "${initial_residual_scale}"
        --gate-freeze-steps "${gate_freeze_steps}"
        --auxiliary-weight "${auxiliary_weight}"
        --auxiliary-weight-start "${auxiliary_weight_start}"
        --auxiliary-ramp-steps "${auxiliary_ramp_steps}"
        --diagnostic-steps "${diagnostic_steps}"
        --layout-loss-profile "${layout_loss_profile}"
        --validation-interval "${validation_interval}"
        --max-eval-new-tokens "${max_eval_new_tokens}"
        --log-steps "${log_steps}"
        --initial-valid-probability "${initial_valid_probability}"
        --remote-root "${remote_root}" --code-root "${code_root}"
        --env-dir "${env_dir}" --model-dir "${model_dir}"
        --dataset-root "${dataset_root}" --protocol-file "${protocol_file}"
    )
    (( use_validity_head == 1 )) && child_args+=(--use-validity-head)
    (( skip_selection == 1 )) && child_args+=(--skip-selection)
    (( without_test == 1 )) && child_args+=(--without-test)
    command_line="$(printf '%q ' "${child_args[@]}")"
    tmux new-session -d -s "${session}" "cd $(printf '%q' "${code_root}") && exec ${command_line} >$(printf '%q' "${launcher_log}") 2>&1"
    printf '{"event":"glmocr_mthv2_ddp_armed","session":"%s","run_id":"%s","seed":%s,"gpu_ids":"%s","steps":%s,"lr_schedule_steps":%s,"learning_rate":%s,"warmup_steps":%s,"initial_residual_scale":%s,"gate_freeze_steps":%s,"auxiliary_weight_start":%s,"auxiliary_weight":%s,"auxiliary_ramp_steps":%s,"gradient_accumulation_steps":%s,"global_batch_size":5,"effective_global_batch_size":%s,"test_used_for_selection":false,"log":"%s"}\n' \
        "${session}" "${run_id}" "${seed}" "${gpu_ids}" "${max_steps}" "${lr_schedule_steps}" "${learning_rate}" "${warmup_steps}" "${initial_residual_scale}" "${gate_freeze_steps}" "${auxiliary_weight_start}" "${auxiliary_weight}" "${auxiliary_ramp_steps}" "${gradient_accumulation_steps}" "$((5 * gradient_accumulation_steps))" "${launcher_log}"
else
    run_inner
fi
