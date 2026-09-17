#!/usr/bin/env bash
# Continue the MTHv2 sparse-Q32 layout-only branch from checkpoint-3000.
# Validation selects the best continuation checkpoint before the locked test.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
mthv2_root="${GLMOCR_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
sparse_root="${GLMOCR_MTHV2_SPARSE_Q32_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24}"
gpu_ids="${GLMOCR_SPARSE_Q32_GPU_IDS:-0,1,2,3,4}"
validation_gpu_ids="${GLMOCR_SPARSE_Q32_VALIDATION_GPU_IDS:-0,1,2,3}"
gpu_utilization_limit="${GLMOCR_SPARSE_Q32_GPU_UTILIZATION_LIMIT:-50}"
seed="${GLMOCR_SPARSE_Q32_SEED:-42}"
num_queries="${GLMOCR_SPARSE_Q32_NUM_QUERIES:-32}"
max_regions="${GLMOCR_SPARSE_Q32_MAX_REGIONS:-24}"
steps="${GLMOCR_SPARSE_Q32_CONTINUATION_STEPS:-3000}"
validation_interval="${GLMOCR_SPARSE_Q32_CONTINUATION_VALIDATION_INTERVAL:-1000}"
learning_rate="${GLMOCR_SPARSE_Q32_CONTINUATION_LR:-2.5e-5}"
decoder_learning_rate="${GLMOCR_SPARSE_Q32_CONTINUATION_DECODER_LR:-5e-6}"
warmup_steps="${GLMOCR_SPARSE_Q32_CONTINUATION_WARMUP_STEPS:-216}"
layout_loss_profile="${GLMOCR_SPARSE_Q32_CONTINUATION_LAYOUT_LOSS_PROFILE:-history_box_equalized_v2}"
max_eval_new_tokens="${GLMOCR_SPARSE_Q32_MAX_EVAL_NEW_TOKENS:-1536}"
source_run_id="${GLMOCR_SPARSE_Q32_CONTINUATION_SOURCE_RUN_ID:-glmocr_mthv2_sparse24_q32_layout_boxeq58_3000_a100_260916_v1}"
run_id="${GLMOCR_SPARSE_Q32_CONTINUATION_RUN_ID:-glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1}"
session="${GLMOCR_SPARSE_Q32_CONTINUATION_SESSION:-glmocr_sparse_q32_boxeq820_cont_260916_v1}"
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) foreground=1; shift ;;
        *) printf '{"event":"glmocr_sparse_q32_continuation_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "${seed}" =~ ^[0-9]+$ && "${num_queries}" == "32" && "${max_regions}" == "24" ]] || exit 64
[[ "${steps}" =~ ^[1-9][0-9]*$ && "${validation_interval}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ "${validation_interval}" -le "${steps}" ]] || exit 64
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ && "${validation_gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 64
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ && "${max_eval_new_tokens}" =~ ^[1-9][0-9]*$ ]] || exit 64
for value in "${source_run_id}" "${run_id}"; do
    [[ "${value}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
done
[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64

python="${env_dir}/bin/python"
ddp_launcher="${code_root}/tools/training/run_glmocr_mthv2_ddp.sh"
locked_test_launcher="${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh"
manifest_auditor="${code_root}/tools/audit_mthv2_manifest.py"
density_selector="${code_root}/tools/select_low_density_mthv2.py"
iou_selector="${code_root}/tools/select_layout_iou_checkpoint.py"
workspace_runs="${remote_root}/runs"
group_root="${remote_root}/training_runs/${run_id}"
run_dir="${group_root}/seed${seed}"
source_run_dir="${remote_root}/training_runs/${source_run_id}/seed${seed}"
source_checkpoint="${source_run_dir}/checkpoint-3000"
status_file="${workspace_runs}/${session}.status.json"
summary_file="${workspace_runs}/${session}.summary.json"
protocol_file="${remote_root}/protocols/${run_id}.train_validation_no_test.json"
test_protocol_file="${remote_root}/protocols/${run_id}.test_locked.json"

write_status() {
    local status="$1"
    local phase="$2"
    local active_run_id="${3:--}"
    printf '{"status":"%s","phase":"%s","run_id":"%s","session":"%s","source_run_id":"%s","dataset":"MTHv2_sparse24_q32","seed":%s,"num_queries":%s,"max_regions":%s,"continuation_steps":%s,"layout_loss_profile":"%s","learning_rate":%s,"decoder_learning_rate":%s,"test_used_for_selection":false,"updated_at":"%s"}\n' \
        "${status}" "${phase}" "${active_run_id}" "${session}" "${source_run_id}" \
        "${seed}" "${num_queries}" "${max_regions}" "${steps}" "${layout_loss_profile}" "${learning_rate}" "${decoder_learning_rate}" "$(date -u +%FT%TZ)" > "${status_file}"
}

checkpoint_steps_csv() {
    local checkpoint_step="${validation_interval}"
    local -a checkpoint_steps=()
    while (( checkpoint_step < steps )); do
        checkpoint_steps+=("${checkpoint_step}")
        checkpoint_step=$((checkpoint_step + validation_interval))
    done
    checkpoint_steps+=("${steps}")
    local saved_ifs="${IFS}"
    IFS=','
    printf '%s' "${checkpoint_steps[*]}"
    IFS="${saved_ifs}"
}

on_error() {
    local rc=$?
    write_status failed "${current_phase:-unknown}_failed" "${current_run_id:--}" || true
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
            printf '{"event":"glmocr_sparse_q32_continuation_failed","error":"gpu_admission_failed","gpu":%s,"utilization":%s,"limit":%s}\n' \
                "${gpu}" "${observed[${gpu}]}" "${gpu_utilization_limit}" >&2
            exit 75
        }
    done
    printf '{"event":"glmocr_sparse_q32_continuation_gpu_admission_ok","gpu_ids":"%s"}\n' "${requested}"
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
    [[ -x "${python}" && -x "${env_dir}/bin/torchrun" ]] || exit 66
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || exit 66
    for path in "${ddp_launcher}" "${locked_test_launcher}" "${manifest_auditor}" "${density_selector}" "${iou_selector}"; do
        [[ -f "${path}" ]] || exit 66
    done
    [[ -f "${sparse_root}/train/manifest.jsonl" && -f "${sparse_root}/validation/manifest.jsonl" ]] || exit 66
    [[ -f "${source_run_dir}/COMPLETED" ]] || exit 66
    [[ -f "${source_checkpoint}/adapter.safetensors" && -f "${source_checkpoint}/decoder_lora.safetensors" ]] || exit 66
    [[ ! -e "${group_root}" && ! -e "${group_root}_smoke" ]] || exit 74
    [[ ! -e "${status_file}" && ! -e "${summary_file}" ]] || exit 74
    command -v nvidia-smi >/dev/null 2>&1 || exit 69
    mkdir -p "${workspace_runs}" "${remote_root}/training_runs" "${remote_root}/protocols"
    admit_gpu_set "${gpu_ids}"
}

write_protocol() {
    current_phase="prepare_protocol"
    current_run_id="${run_id}"
    write_status running "${current_phase}" "${current_run_id}"
    "${python}" "${manifest_auditor}" \
        --train-manifest "${sparse_root}/train/manifest.jsonl" \
        --validation-manifest "${sparse_root}/validation/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch --without-test \
        --dataset-label "MTHv2_sparse24_q32" \
        --protocol-label "glm_ocr_mthv2_sparse24_q32_layout_continuation_v1" \
        --output "${protocol_file}" \
        > "${workspace_runs}/${run_id}.train-protocol.log" 2>&1
}

ddp_args() {
    local active_run_id="$1"
    printf '%s\n' \
        --foreground --run-id "${active_run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit "${gpu_utilization_limit}" --remote-root "${remote_root}" \
        --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}" \
        --dataset-root "${sparse_root}" --protocol-file "${protocol_file}" \
        --allow-count-mismatch --mode geometry --num-queries "${num_queries}" \
        --max-steps "${steps}" --lr-schedule-steps "${steps}" --warmup-steps "${warmup_steps}" \
        --learning-rate "${learning_rate}" --decoder-adaptation lora \
        --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
        --decoder-learning-rate "${decoder_learning_rate}" --min-lr-ratio 0.1 \
        --initial-residual-scale 0.0 --gate-freeze-steps 0 \
        --auxiliary-weight 1.0 --auxiliary-weight-start 1.0 --auxiliary-ramp-steps 0 \
        --layout-loss-profile "${layout_loss_profile}" --validation-interval "${validation_interval}" \
        --generation-mode plain --max-eval-new-tokens "${max_eval_new_tokens}" --log-steps 16 \
        --defer-validation --without-test --layout-only
}

run_training() {
    current_phase="continuation_smoke"
    current_run_id="${run_id}_smoke"
    write_status running "${current_phase}" "${current_run_id}"
    mapfile -t smoke_args < <(ddp_args "${current_run_id}")
    smoke_args+=(--max-steps 8 --lr-schedule-steps 8 --warmup-steps 0 --validation-interval 9 --smoke \
        --init-checkpoint-dir "${source_checkpoint}")
    bash "${ddp_launcher}" "${smoke_args[@]}" > "${workspace_runs}/${current_run_id}.pipeline.log" 2>&1
    "${python}" - "${remote_root}/training_runs/${current_run_id}/smoke/seed${seed}/smoke_summary.json" <<'PY'
import json,sys
summary=json.load(open(sys.argv[1],encoding='utf-8'))
objective=((summary.get('training') or {}).get('loss_objective') or {})
if summary.get('status')!='complete' or summary.get('checkpoint_reload') is not True:
    raise SystemExit('continuation smoke did not complete')
if objective.get('formula')!='auxiliary_weight * L_layout' or objective.get('layout_only') is not True:
    raise SystemExit(f'unexpected continuation objective: {objective}')
print(json.dumps({'event':'glmocr_sparse_q32_continuation_smoke_ok'},separators=(',',':')))
PY

    current_phase="continuation_training"
    current_run_id="${run_id}"
    write_status running "${current_phase}" "${current_run_id}"
    mapfile -t train_args < <(ddp_args "${run_id}")
    train_args+=(--init-checkpoint-dir "${source_checkpoint}")
    bash "${ddp_launcher}" "${train_args[@]}" > "${workspace_runs}/${run_id}.pipeline.log" 2>&1
    local expected_steps
    expected_steps="$(checkpoint_steps_csv)"
    "${python}" - "${run_dir}/summary.json" "${run_dir}/metadata.json" "${expected_steps}" <<'PY'
import json,sys
summary=json.load(open(sys.argv[1],encoding='utf-8'))
metadata=json.load(open(sys.argv[2],encoding='utf-8'))
steps=[int(x) for x in (summary.get('training') or {}).get('checkpoint_steps',[])]
expected=[int(x) for x in sys.argv[3].split(',') if x]
if summary.get('status')!='complete' or metadata.get('status')!='complete': raise SystemExit('continuation training incomplete')
if summary.get('test_manifest_read') is not False or metadata.get('test_manifest_read') is not False: raise SystemExit('continuation training read test')
if (summary.get('training') or {}).get('layout_only') is not True: raise SystemExit('continuation is not layout-only')
if steps != expected: raise SystemExit(f'unexpected continuation checkpoints: {steps}, expected {expected}')
print(json.dumps({'event':'glmocr_sparse_q32_continuation_training_ok','checkpoint_steps':steps},separators=(',',':')))
PY
}

run_validation() {
    current_phase="continuation_validation"
    current_run_id="${run_id}"
    write_status running "${current_phase}" "${current_run_id}"
    local validation_root="${run_dir}/continuation-validation"
    local validation_log_dir="${validation_root}/logs"
    mkdir -p "${validation_root}" "${validation_log_dir}"
    admit_gpu_set "${validation_gpu_ids}"
    IFS=',' read -r -a eval_gpus <<< "${validation_gpu_ids}"
    local steps_csv
    steps_csv="$(checkpoint_steps_csv)"
    local -a eval_steps=()
    IFS=',' read -r -a eval_steps <<< "${steps_csv}"
    (( ${#eval_gpus[@]} >= ${#eval_steps[@]} )) || exit 64
    local libs
    libs="$(cuda_library_path)"
    local -a pids=()
    for index in "${!eval_steps[@]}"; do
        local step="${eval_steps[${index}]}"
        local gpu="${eval_gpus[${index}]}"
        local eval_dir="${validation_root}/step-${step}"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export TMPDIR="${run_dir}/tmp/continuation-validation-${step}"
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
                --experiment-label "${run_id}_validation_step${step}" \
                --learning-rate "${learning_rate}" --decoder-adaptation lora \
                --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
                --decoder-learning-rate "${decoder_learning_rate}" --min-lr-ratio 0.1 \
                --initial-residual-scale 0.0 --auxiliary-weight 1.0 \
                --auxiliary-weight-start 1.0 --max-grad-norm 1.0 \
                --max-pixels 1003520 --max-eval-new-tokens "${max_eval_new_tokens}" \
                --validation-interval 2 --log-steps 16 --adapter-precision fp32 \
                --layout-loss-profile "${layout_loss_profile}" --query-assignment hungarian \
                --processor-mode fast --generation-mode plain --layout-only \
                --eval-checkpoint-dir "${run_dir}/checkpoint-${step}" --eval-only
        ) > "${validation_log_dir}/step-${step}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || exit 1
}

run_locked_test() {
    current_phase="continuation_test"
    current_run_id="${run_id}"
    write_status running "${current_phase}" "${run_id}"
    local steps_csv
    steps_csv="$(checkpoint_steps_csv)"
    "${python}" "${iou_selector}" \
        --run-dir "${run_dir}" --validation-root "${run_dir}/continuation-validation" \
        --steps "${steps_csv}" --dataset-label "MTHv2_sparse24_q32_continuation" \
        --expected-world-size 5 \
        > "${workspace_runs}/${run_id}.layout-selection.log" 2>&1
    "${python}" "${density_selector}" \
        --input-root "${mthv2_root}" --output-root "${sparse_root}" \
        --max-regions "${max_regions}" --splits test \
        > "${workspace_runs}/${run_id}.test-selection.log" 2>&1
    "${python}" "${manifest_auditor}" \
        --train-manifest "${sparse_root}/train/manifest.jsonl" \
        --validation-manifest "${sparse_root}/validation/manifest.jsonl" \
        --test-manifest "${sparse_root}/test/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch \
        --dataset-label "MTHv2_sparse24_q32_continuation" \
        --protocol-label "glm_ocr_mthv2_sparse24_q32_continuation_locked_test_v1" \
        --output "${test_protocol_file}" \
        > "${workspace_runs}/${run_id}.test-protocol.log" 2>&1
    bash "${locked_test_launcher}" --foreground --run-id "${run_id}" --seed "${seed}" \
        --gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}" \
        --mode geometry --num-queries "${num_queries}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" \
        --model-dir "${model_dir}" --dataset-root "${sparse_root}" \
        --protocol-file "${test_protocol_file}" \
        > "${workspace_runs}/${run_id}.locked-test.pipeline.log" 2>&1
}

write_final_summary() {
    current_phase="complete"
    current_run_id="${run_id}"
    write_status complete complete "${run_id}"
    "${python}" - "${summary_file}" "${run_id}" "${source_run_id}" "${steps}" "${layout_loss_profile}" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
run_id=sys.argv[2]
source_id=sys.argv[3]
continuation_steps=int(sys.argv[4])
layout_loss_profile=sys.argv[5]
root=Path('/data3/yky/yangky_ocr_models/glm_ocr_layout_ot')
run_dir=root/'training_runs'/run_id/'seed42'
selection=json.loads((run_dir/'selection.json').read_text(encoding='utf-8'))
test=json.loads((run_dir/'locked-test/locked_test_summary.json').read_text(encoding='utf-8'))
selected=int(selection['selected_step'])
payload={'status':'complete','session':out.stem.replace('.summary',''),'seed':42,'num_queries':32,'max_regions':24,
 'continuation_steps':continuation_steps,'layout_loss_profile':layout_loss_profile,'source_run_id':source_id,'source_checkpoint':str(root/'training_runs'/source_id/'seed42/checkpoint-3000'),
 'gpu_ids':'0,1,2,3,4','test_used_for_selection':False,
 'run_id':run_id,'run_dir':str(run_dir),'selection':str(run_dir/'selection.json'),
 'selected_step':selected,'selected_validation':selection.get('selected_validation'),
 'adapter_checkpoint':str(run_dir/f'checkpoint-{selected}/adapter.safetensors'),
 'decoder_lora_checkpoint':str(run_dir/f'checkpoint-{selected}/decoder_lora.safetensors'),
 'test_summary':str(run_dir/'locked-test/locked_test_summary.json'),
 'metrics':test.get('metrics') or {},'test_selected_step':test.get('selected_step'),
 'test_used_for_selection':test.get('test_used_for_selection')}
out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'event':'glmocr_sparse_q32_continuation_complete','selected_step':selected,
 'layout_box_iou':payload['metrics'].get('layout_box_iou'),'layout_box_mae':payload['metrics'].get('layout_box_mae')},separators=(',',':')))
PY
    cat "${summary_file}"
}

run_inner() {
    preflight
    write_protocol
    run_training
    run_validation
    run_locked_test
    write_final_summary
}

if (( foreground == 0 )); then
    preflight
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    mkdir -p "${workspace_runs}"
    command_line="$(printf '%q ' bash "${script_path}" --foreground)"
    tmux new-session -d -s "${session}" \
        "cd $(printf '%q' "${code_root}") && exec ${command_line} >$(printf '%q' "${workspace_runs}/${session}.log") 2>&1"
    printf '{"event":"glmocr_sparse_q32_continuation_armed","session":"%s","run_id":"%s","source_run_id":"%s","gpu_ids":"%s","status":"%s","log":"%s"}\n' \
        "${session}" "${run_id}" "${source_run_id}" "${gpu_ids}" "${status_file}" "${workspace_runs}/${session}.log"
else
    run_inner
fi
