#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="${GOT_PROJECT_ROOT:-$(cd -- "${script_dir}/.." && pwd -P)}"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../../.." && pwd -P)}"
workspace="${OCR_WORKSPACE:-$(cd -- "${ocrmodel_root}/.." && pwd -P)}"
runner="${GOT_RUNNER:-${ocrmodel_root}/tools/environment/run_got2.sh}"
source_model="${GOT_SOURCE_MODEL:-${workspace}/models/GOT-OCR2_0}"
data_root="${ANCIENTDOC_ROOT:-${workspace}/datasets/AncientDoc}"
audit_report="${ANCIENTDOC_AUDIT:-${workspace}/audits/ancientdoc/audit.json}"
runs_root="${GOT_TRAINING_RUNS:-${workspace}/runs/training/GOT}"
run_mode="${ANCIENTDOC_RUN_MODE:-smoke}"
train_scope="${ANCIENTDOC_TRAIN_SCOPE:-decoder_projector}"
run_id="${GOT_RUN_ID:-ancientdoc_${run_mode}_$(date +%Y%m%d_%H%M%S)}"
epochs="${ANCIENTDOC_EPOCHS:-6}"
stream_log="${GOT_STREAM_LOG:-0}"
run_root="${runs_root}/${run_id}"
model_output="${run_root}/model"
scripts_dir="${project_root}/scripts"
env_dir="${workspace}/envs/got2"

case "${run_mode}" in
    smoke)
        train_splits="1,2,3,4"
        record_selection="longest"
        max_train_records=1
        mode_args=(
            --gradient_accumulation_steps 1
            --save_strategy no
            --warmup_ratio 0
            --lr_scheduler_type constant
            --dataloader_num_workers 0
            --per_device_train_batch_size 1
            --max_steps 1
            --learning_rate 1e-4
        )
        ;;
    reference)
        train_splits="1,2,3,4"
        record_selection="all"
        max_train_records=0
        mode_args=(
            --gradient_accumulation_steps 4
            --save_strategy steps
            --save_steps 1000
            --save_total_limit 2
            --warmup_ratio 0.001
            --lr_scheduler_type cosine
            --dataloader_num_workers 4
            --per_device_train_batch_size 1
            --num_train_epochs "${epochs}"
            --learning_rate 2e-5
        )
        ;;
    *)
        echo "Unsupported ANCIENTDOC_RUN_MODE: ${run_mode}" >&2
        exit 64
        ;;
esac

if [[ ! "${epochs}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid ANCIENTDOC_EPOCHS: ${epochs}" >&2
    exit 64
fi

for required_file in \
    "${source_model}/model.safetensors" \
    "${data_root}/label_for_got.json" \
    "${data_root}/label_for_got_split1.json" \
    "${data_root}/label_for_got_split2.json" \
    "${data_root}/label_for_got_split3.json" \
    "${data_root}/label_for_got_split4.json" \
    "${data_root}/label_for_got_split5.json" \
    "${audit_report}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required AncientDoc training asset is missing: ${required_file}" >&2
        exit 66
    fi
done
if [[ ! -x "${env_dir}/bin/python" || ! -x "${env_dir}/bin/deepspeed" ]]; then
    echo "GOT training environment is incomplete: ${env_dir}" >&2
    exit 66
fi
if [[ ! -x "${runner}" ]]; then
    echo "GOT runner is missing or not executable: ${runner}" >&2
    exit 66
fi
if [[ "${stream_log}" != "0" && "${stream_log}" != "1" ]]; then
    echo "Invalid GOT_STREAM_LOG: ${stream_log}" >&2
    exit 64
fi

master_port="${GOT_MASTER_PORT:-}"
if [[ -z "${master_port}" ]]; then
    master_port="$("${env_dir}/bin/python" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
fi
if [[ ! "${master_port}" =~ ^[0-9]+$ ]] || ((master_port < 1024 || master_port > 65535)); then
    echo "Invalid GOT_MASTER_PORT: ${master_port}" >&2
    exit 64
fi

mkdir -p "${runs_root}"
exec 9>"${runs_root}/.ancientdoc_train.lock"
if ! flock -n 9; then
    echo "GOT_ANCIENTDOC_TRAIN_ALREADY_RUNNING" >&2
    exit 73
fi
if [[ -e "${run_root}" ]]; then
    echo "Run output already exists: ${run_root}" >&2
    exit 74
fi

gpu_apps="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
if [[ -n "${gpu_apps//[[:space:]]/}" ]]; then
    echo "GPU0_BUSY" >&2
    printf '%s\n' "${gpu_apps}" >&2
    exit 75
fi

mkdir -p "${run_root}/metadata" "${model_output}"
exec 3>&1
printf 'GOT_ANCIENTDOC_%s_STARTED run_root=%s log=%s\n' "${run_mode^^}" "${run_root}" "${run_root}/train.log" >&3
if [[ "${stream_log}" == "1" ]]; then
    exec > >(tee -a "${run_root}/train.log") 2>&1
else
    exec >> "${run_root}/train.log" 2>&1
fi
status_file="${run_root}/metadata/status.txt"
finished=0
finish_run() {
    exit_code="$?"
    {
        echo "finished_at=$(date --iso-8601=seconds)"
        echo "exit_code=${exit_code}"
        echo "completed=${finished}"
    } >> "${status_file}"
    nvidia-smi -i 0 > "${run_root}/metadata/gpu_after.txt" 2>&1 || true
    if ((exit_code != 0)); then
        printf 'GOT_ANCIENTDOC_%s_FAILED exit_code=%s log=%s\n' "${run_mode^^}" "${exit_code}" "${run_root}/train.log" >&3
    fi
}
trap finish_run EXIT

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "run_root=${run_root}"
    echo "run_mode=${run_mode}"
    echo "project_root=${project_root}"
    echo "source_model=${source_model}"
    echo "ancientdoc_root=${data_root}"
    echo "audit_report=${audit_report}"
    echo "physical_gpu=0"
    echo "input_level=page_reference_compatibility"
    echo "train_splits=${train_splits}"
    echo "eval_split=5_not_used_for_training"
    echo "train_scope=${train_scope}"
    echo "num_train_epochs=${epochs}"
    echo "record_selection=${record_selection}"
    echo "max_train_records=${max_train_records}"
    echo "model_max_length=8192"
    echo "master_port=${master_port}"
    echo "stream_log=${stream_log}"
} > "${status_file}"
sha256sum "${source_model}/model.safetensors" > "${run_root}/metadata/source_model.sha256"
sha256sum "${audit_report}" > "${run_root}/metadata/audit.sha256"
sha256sum \
    "${data_root}/label_for_got_split1.json" \
    "${data_root}/label_for_got_split2.json" \
    "${data_root}/label_for_got_split3.json" \
    "${data_root}/label_for_got_split4.json" \
    > "${run_root}/metadata/train_labels.sha256"
nvidia-smi -i 0 > "${run_root}/metadata/gpu_before.txt"

gpu_apps="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
if [[ -n "${gpu_apps//[[:space:]]/}" ]]; then
    echo "GPU0_BECAME_BUSY_BEFORE_MODEL_START" >&2
    printf '%s\n' "${gpu_apps}" >&2
    exit 75
fi
echo "GPU0_CONFIRMED_FREE_BEFORE_MODEL_START"

export OCR_WORKSPACE="${workspace}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
python_lib="${env_dir}/lib/python3.10/site-packages"
export HF_HOME="${workspace}/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="${python_lib}/torch/lib:${python_lib}/nvidia/cuda_runtime/lib:${python_lib}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

cd "${project_root}"
"${env_dir}/bin/deepspeed" \
    --master_port "${master_port}" \
    "${scripts_dir}/train_GOT_ancientdoc.py" \
    --deepspeed "${project_root}/zero_config/zero2.json" \
    --model_name_or_path "${source_model}" \
    --ancientdoc_root "${data_root}" \
    --audit_report "${audit_report}" \
    --train_splits "${train_splits}" \
    --train_scope "${train_scope}" \
    --record_selection "${record_selection}" \
    --max_train_records "${max_train_records}" \
    --datasets ancientdoc-page \
    --use_im_start_end True \
    --bf16 True \
    --fp16 False \
    --evaluation_strategy no \
    --weight_decay 0 \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --report_to none \
    --seed 42 \
    --data_seed 42 \
    --output_dir "${model_output}" \
    "${mode_args[@]}"

"${runner}" "${scripts_dir}/verify_linelevel_checkpoint.py" \
    --source-model "${source_model}" \
    --trained-model "${model_output}" \
    --metrics-name ancientdoc_training_metrics.json \
    --expected-train-scope "${train_scope}" \
    --output "${run_root}/checkpoint_verification.json"

touch "${run_root}/GOT_ANCIENTDOC_${run_mode^^}_FINISHED"
finished=1
echo "GOT_ANCIENTDOC_${run_mode^^}_OK"
echo "RUN_ROOT=${run_root}"
printf 'GOT_ANCIENTDOC_%s_OK run_root=%s\n' "${run_mode^^}" "${run_root}" >&3
