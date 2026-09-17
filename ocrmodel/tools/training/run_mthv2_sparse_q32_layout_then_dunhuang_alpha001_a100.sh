#!/usr/bin/env bash
# Q32 sparse-layout pretraining on MTHv2, followed by a gate=0.01
# Dunhuang/local-gazetteer fine-tune. MTHv2 test is materialized only after
# layout-only training and validation checkpoint selection are complete.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
mthv2_root="${GLMOCR_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
sparse_root="${GLMOCR_MTHV2_SPARSE_Q32_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24}"
dunhuang_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"
gpu_ids="${GLMOCR_SPARSE_Q32_GPU_IDS:-0,1,2,3,4}"
validation_gpu_ids="${GLMOCR_SPARSE_Q32_VALIDATION_GPU_IDS:-0,1,2}"
gpu_utilization_limit="${GLMOCR_SPARSE_Q32_GPU_UTILIZATION_LIMIT:-50}"
seed="${GLMOCR_SPARSE_Q32_SEED:-42}"
num_queries="${GLMOCR_SPARSE_Q32_NUM_QUERIES:-32}"
max_regions="${GLMOCR_SPARSE_Q32_MAX_REGIONS:-24}"
phase1_steps="${GLMOCR_SPARSE_Q32_PHASE1_STEPS:-3000}"
phase1_validation_interval="${GLMOCR_SPARSE_Q32_PHASE1_VALIDATION_INTERVAL:-1000}"
phase2_steps="${GLMOCR_SPARSE_Q32_PHASE2_STEPS:-256}"
phase2_gate="${GLMOCR_SPARSE_Q32_PHASE2_GATE:-0.01}"
max_eval_new_tokens="${GLMOCR_SPARSE_Q32_MAX_EVAL_NEW_TOKENS:-1536}"
learning_rate="${GLMOCR_SPARSE_Q32_LR:-2.5e-5}"
decoder_learning_rate="${GLMOCR_SPARSE_Q32_DECODER_LR:-5e-6}"
min_lr_ratio="${GLMOCR_SPARSE_Q32_MIN_LR_RATIO:-0.1}"
layout_loss_profile="${GLMOCR_SPARSE_Q32_LAYOUT_LOSS_PROFILE:-full}"
phase1_run_id="${GLMOCR_SPARSE_Q32_PHASE1_RUN_ID:-glmocr_mthv2_sparse24_q32_layout_only_3000_a100_260915_v1}"
reuse_phase1_run_id="${GLMOCR_SPARSE_Q32_REUSE_PHASE1_RUN_ID:-}"
phase2_run_id="${GLMOCR_SPARSE_Q32_PHASE2_RUN_ID:-glmocr_dunhuang_local_q32_alpha001_from_sparse_layout_256_a100_260915_v1}"
session="${GLMOCR_SPARSE_Q32_SESSION:-glmocr_sparse_q32_layout_alpha001_260915_v1}"
foreground=0

if [[ -n "${reuse_phase1_run_id}" ]]; then
    phase1_run_id="${reuse_phase1_run_id}"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) foreground=1; shift ;;
        *) printf '{"event":"glmocr_sparse_q32_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "${seed}" =~ ^[0-9]+$ && "${num_queries}" == "32" && "${max_regions}" == "24" ]] || exit 64
[[ "${phase1_steps}" =~ ^[1-9][0-9]*$ && "${phase1_validation_interval}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ "${phase2_steps}" == "256" && "${phase2_gate}" == "0.01" ]] || exit 64
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ && "${validation_gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 64
(( phase1_validation_interval <= phase1_steps )) || exit 64
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ && "${max_eval_new_tokens}" =~ ^[1-9][0-9]*$ ]] || exit 64
for run_id in "${phase1_run_id}" "${phase2_run_id}"; do
    [[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
done
[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64

python="${env_dir}/bin/python"
ddp_launcher="${code_root}/tools/training/run_glmocr_mthv2_ddp.sh"
locked_test_launcher="${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh"
manifest_auditor="${code_root}/tools/audit_mthv2_manifest.py"
density_selector="${code_root}/tools/select_low_density_mthv2.py"
iou_selector="${code_root}/tools/select_layout_iou_checkpoint.py"
workspace_runs="${remote_root}/runs"
status_file="${workspace_runs}/${session}.status.json"
summary_file="${workspace_runs}/${session}.summary.json"
phase1_protocol="${remote_root}/protocols/${phase1_run_id}.train_validation_no_test.json"
phase2_protocol="${remote_root}/protocols/${phase2_run_id}.train_validation_no_test.json"
phase1_test_protocol="${remote_root}/protocols/${phase1_run_id}.test_locked.json"
phase2_test_protocol="${remote_root}/protocols/${phase2_run_id}.test_locked.json"

write_status() {
    local status="$1"
    local phase="$2"
    local run_id="${3:--}"
    printf '{"status":"%s","phase":"%s","run_id":"%s","session":"%s","dataset":"MTHv2_sparse24_then_dunhuang_local_q32","seed":%s,"num_queries":%s,"phase1_steps":%s,"phase2_steps":%s,"gpu_ids":"%s","test_used_for_selection":false,"updated_at":"%s"}\n' \
        "${status}" "${phase}" "${run_id}" "${session}" "${seed}" "${num_queries}" \
        "${phase1_steps}" "${phase2_steps}" "${gpu_ids}" "$(date -u +%FT%TZ)" > "${status_file}"
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
            printf '{"event":"glmocr_sparse_q32_failed","error":"gpu_admission_failed","gpu":%s,"utilization":%s,"limit":%s}\n' \
                "${gpu}" "${observed[${gpu}]}" "${gpu_utilization_limit}" >&2
            exit 75
        }
    done
    printf '{"event":"glmocr_sparse_q32_gpu_admission_ok","gpu_ids":"%s"}\n' "${requested}"
}

preflight() {
    [[ -x "${python}" && -x "${env_dir}/bin/torchrun" ]] || exit 66
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || exit 66
    for path in "${ddp_launcher}" "${locked_test_launcher}" "${manifest_auditor}" "${density_selector}" "${iou_selector}"; do
        [[ -f "${path}" ]] || exit 66
    done
    [[ -f "${mthv2_root}/train/manifest.jsonl" && -f "${mthv2_root}/validation/manifest.jsonl" ]] || exit 66
    [[ -f "${dunhuang_root}/train/manifest.jsonl" && -f "${dunhuang_root}/validation/manifest.jsonl" && -f "${dunhuang_root}/test/manifest.jsonl" ]] || exit 66
    if [[ -n "${reuse_phase1_run_id}" ]]; then
        local source_run_dir="${remote_root}/training_runs/${phase1_run_id}/seed${seed}"
        [[ -f "${source_run_dir}/COMPLETED" ]] || exit 66
        for step in 1000 2000 3000; do
            [[ -f "${source_run_dir}/checkpoint-${step}/adapter.safetensors" && -f "${source_run_dir}/checkpoint-${step}/decoder_lora.safetensors" ]] || exit 66
        done
    else
        [[ ! -e "${remote_root}/training_runs/${phase1_run_id}" && ! -e "${remote_root}/training_runs/${phase1_run_id}_smoke" ]] || exit 74
    fi
    [[ ! -e "${remote_root}/training_runs/${phase2_run_id}" && ! -e "${remote_root}/training_runs/${phase2_run_id}_smoke" ]] || exit 74
    [[ ! -e "${status_file}" && ! -e "${summary_file}" ]] || exit 74
    command -v nvidia-smi >/dev/null 2>&1 || exit 69
    mkdir -p "${workspace_runs}" "${remote_root}/training_runs" "${remote_root}/protocols"
    admit_gpu_set "${gpu_ids}"
}

prepare_sparse_train_validation() {
    current_phase="prepare_sparse_train_validation"
    write_status running "${current_phase}" "${phase1_run_id}"
    "${python}" "${density_selector}" \
        --input-root "${mthv2_root}" --output-root "${sparse_root}" \
        --max-regions "${max_regions}" --splits train,validation \
        > "${workspace_runs}/${phase1_run_id}.sparse-selection.log" 2>&1
    "${python}" "${manifest_auditor}" \
        --train-manifest "${sparse_root}/train/manifest.jsonl" \
        --validation-manifest "${sparse_root}/validation/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch --without-test \
        --dataset-label "MTHv2_sparse24_q32" \
        --protocol-label "glm_ocr_mthv2_sparse24_q32_layout_only_v1" \
        --output "${phase1_protocol}" \
        > "${workspace_runs}/${phase1_run_id}.train-protocol.log" 2>&1
}

common_ddp_args() {
    local run_id="$1"
    local dataset_root="$2"
    local protocol_file="$3"
    local aux_weight="$4"
    local initial_gate="$5"
    printf '%s\n' \
        --foreground --run-id "${run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit "${gpu_utilization_limit}" --remote-root "${remote_root}" \
        --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}" \
        --dataset-root "${dataset_root}" --protocol-file "${protocol_file}" \
        --allow-count-mismatch --mode geometry --num-queries "${num_queries}" \
        --learning-rate "${learning_rate}" --decoder-adaptation lora \
        --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
        --decoder-learning-rate "${decoder_learning_rate}" --min-lr-ratio "${min_lr_ratio}" \
        --initial-residual-scale "${initial_gate}" --gate-freeze-steps 0 \
        --auxiliary-weight "${aux_weight}" --auxiliary-weight-start "${aux_weight}" \
        --auxiliary-ramp-steps 0 --layout-loss-profile "${layout_loss_profile}" \
        --generation-mode plain --max-eval-new-tokens "${max_eval_new_tokens}" --log-steps 16
}

run_phase1_smoke() {
    current_phase="phase1_smoke"
    current_run_id="${phase1_run_id}_smoke"
    write_status running "${current_phase}" "${current_run_id}"
    mapfile -t args < <(common_ddp_args "${current_run_id}" "${sparse_root}" "${phase1_protocol}" 1.0 0.0)
    args+=(--max-steps 8 --lr-schedule-steps 8 --warmup-steps 0 --validation-interval 9 --without-test --smoke --layout-only)
    bash "${ddp_launcher}" "${args[@]}" > "${workspace_runs}/${current_run_id}.pipeline.log" 2>&1
    "${python}" - "${remote_root}/training_runs/${current_run_id}/smoke/seed${seed}/smoke_summary.json" <<'PY'
import json, sys
summary=json.load(open(sys.argv[1],encoding='utf-8'))
objective=((summary.get('training') or {}).get('loss_objective') or {})
if summary.get('status')!='complete' or summary.get('checkpoint_reload') is not True:
    raise SystemExit('phase1 layout-only smoke did not complete')
if objective.get('formula')!='auxiliary_weight * L_layout' or objective.get('layout_only') is not True:
    raise SystemExit(f'unexpected phase1 smoke objective: {objective}')
if summary.get('decoder_adaptation')!='lora':
    raise SystemExit('phase1 smoke did not materialize decoder LoRA state')
print(json.dumps({'event':'glmocr_sparse_q32_phase1_smoke_ok'},separators=(',',':')))
PY
}

run_phase1_training() {
    current_phase="phase1_training"
    current_run_id="${phase1_run_id}"
    write_status running "${current_phase}" "${current_run_id}"
    mapfile -t args < <(common_ddp_args "${phase1_run_id}" "${sparse_root}" "${phase1_protocol}" 1.0 0.0)
    args+=(--max-steps "${phase1_steps}" --lr-schedule-steps "${phase1_steps}" \
        --warmup-steps 216 --validation-interval "${phase1_validation_interval}" \
        --defer-validation --without-test --layout-only)
    bash "${ddp_launcher}" "${args[@]}" > "${workspace_runs}/${phase1_run_id}.pipeline.log" 2>&1
    "${python}" - "${remote_root}/training_runs/${phase1_run_id}/seed${seed}/summary.json" "${remote_root}/training_runs/${phase1_run_id}/seed${seed}/metadata.json" <<'PY'
import json, sys
summary=json.load(open(sys.argv[1],encoding='utf-8'))
metadata=json.load(open(sys.argv[2],encoding='utf-8'))
objective=((summary.get('training') or {}).get('loss_objective') or {})
if summary.get('status')!='complete' or metadata.get('status')!='complete': raise SystemExit('phase1 training incomplete')
if summary.get('test_manifest_read') is not False or metadata.get('test_manifest_read') is not False: raise SystemExit('phase1 training read test')
if objective.get('formula')!='auxiliary_weight * L_layout' or objective.get('layout_only') is not True: raise SystemExit(f'unexpected phase1 objective: {objective}')
if [int(x) for x in (summary.get('training') or {}).get('checkpoint_steps',[])] != [1000,2000,3000]: raise SystemExit('phase1 checkpoints are not 1000/2000/3000')
print(json.dumps({'event':'glmocr_sparse_q32_phase1_training_ok','checkpoint_steps':[1000,2000,3000]},separators=(',',':')))
PY
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
    local result="${torch_lib}"
    local system_lib="/usr/local/cuda/targets/${target_arch}/lib"
    [[ -d "${system_lib}" ]] && result="${system_lib}:${result}"
    for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
        local component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
        [[ -d "${component_lib}" ]] && result="${result}:${component_lib}"
    done
    printf '%s' "${result}"
}

run_validation_evaluations() {
    local run_id="$1"
    local dataset_root="$2"
    local protocol_file="$3"
    local steps_csv="$4"
    local initial_gate="$5"
    local aux_weight="$6"
    local layout_only_flag="$7"
    local run_dir="${remote_root}/training_runs/${run_id}/seed${seed}"
    local validation_root_override="${8:-}"
    local validation_root="${validation_root_override:-${run_dir}/layout-validation}"
    local validation_log_dir="${validation_root}/logs"
    mkdir -p "${validation_root}" "${validation_log_dir}" "${run_dir}/logs" "${remote_root}/training_runs/${run_id}/logs"
    admit_gpu_set "${validation_gpu_ids}"
    IFS=',' read -r -a eval_gpus <<< "${validation_gpu_ids}"
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
            export TMPDIR="${run_dir}/tmp/layout-validation-${step}"
            export HF_HOME="${TMPDIR}/huggingface"
            export TRANSFORMERS_CACHE="${HF_HOME}"
            export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
            export LD_LIBRARY_PATH="${libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
            mkdir -p "${TMPDIR}" "${HF_HOME}"
            cd "${code_root}"
            eval_args=(
                --mode geometry --model-path "${model_dir}"
                --train-manifest "${dataset_root}/train/manifest.jsonl"
                --validation-manifest "${dataset_root}/validation/manifest.jsonl"
                --protocol-file "${protocol_file}" --output-dir "${eval_dir}"
                --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0
                --num-queries "${num_queries}" --seed "${seed}"
                --experiment-label "${run_id}_validation_step${step}"
                --learning-rate "${learning_rate}" --decoder-adaptation lora
                --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0
                --decoder-learning-rate "${decoder_learning_rate}"
                --min-lr-ratio "${min_lr_ratio}" --residual-scale-cap 0.03
                --initial-residual-scale "${initial_gate}" --auxiliary-weight "${aux_weight}"
                --auxiliary-weight-start "${aux_weight}" --max-grad-norm 1.0
                --max-pixels 1003520 --max-eval-new-tokens "${max_eval_new_tokens}"
                --validation-interval 2 --log-steps 16 --adapter-precision fp32
                --layout-loss-profile "${layout_loss_profile}" --query-assignment hungarian
                --processor-mode fast --generation-mode plain
                --eval-checkpoint-dir "${run_dir}/checkpoint-${step}" --eval-only
            )
            [[ "${layout_only_flag}" == "1" ]] && eval_args+=(--layout-only)
            exec "${python}" -m layout_ocr.train_screen "${eval_args[@]}"
        ) > "${validation_log_dir}/layout-validation.step${step}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || exit 1
}

run_phase1_test() {
    current_phase="phase1_test"
    current_run_id="${phase1_run_id}"
    write_status running "${current_phase}" "${current_run_id}"
    local phase1_validation_root="${remote_root}/training_runs/${phase1_run_id}/seed${seed}/layout-validation"
    if [[ -n "${reuse_phase1_run_id}" ]]; then
        phase1_validation_root="${remote_root}/training_runs/${phase2_run_id}/phase1-validation"
    fi
    run_validation_evaluations "${phase1_run_id}" "${sparse_root}" "${phase1_protocol}" "1000,2000,3000" 0.0 1.0 1 "${phase1_validation_root}"
    "${python}" "${iou_selector}" \
        --run-dir "${remote_root}/training_runs/${phase1_run_id}/seed${seed}" \
        --validation-root "${phase1_validation_root}" \
        --steps 1000,2000,3000 --dataset-label "MTHv2_sparse24_q32" --expected-world-size 5 \
        > "${workspace_runs}/${phase1_run_id}.layout-selection.json" 2>&1
    "${python}" "${density_selector}" \
        --input-root "${mthv2_root}" --output-root "${sparse_root}" \
        --max-regions "${max_regions}" --splits test \
        > "${workspace_runs}/${phase1_run_id}.test-selection.log" 2>&1
    "${python}" "${manifest_auditor}" \
        --train-manifest "${sparse_root}/train/manifest.jsonl" \
        --validation-manifest "${sparse_root}/validation/manifest.jsonl" \
        --test-manifest "${sparse_root}/test/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch \
        --dataset-label "MTHv2_sparse24_q32" \
        --protocol-label "glm_ocr_mthv2_sparse24_q32_layout_only_locked_test_v1" \
        --output "${phase1_test_protocol}" \
        > "${workspace_runs}/${phase1_run_id}.test-protocol.log" 2>&1
    bash "${locked_test_launcher}" --foreground --run-id "${phase1_run_id}" --seed "${seed}" \
        --gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}" \
        --mode geometry --num-queries "${num_queries}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" \
        --model-dir "${model_dir}" --dataset-root "${sparse_root}" \
        --protocol-file "${phase1_test_protocol}" \
        > "${workspace_runs}/${phase1_run_id}.locked-test.pipeline.log" 2>&1
    "${python}" - "${remote_root}/training_runs/${phase1_run_id}/seed${seed}/locked-test/locked_test_summary.json" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding='utf-8'))
metrics=payload.get('metrics') or {}
print(json.dumps({'event':'glmocr_sparse_q32_phase1_test_complete','selected_step':payload.get('selected_step'),'test_pages':payload.get('test_pages'),'layout_box_iou':metrics.get('layout_box_iou'),'layout_box_mae':metrics.get('layout_box_mae')},separators=(',',':')))
PY
}

prepare_dunhuang_train_validation() {
    current_phase="prepare_dunhuang_train_validation"
    write_status running "${current_phase}" "${phase2_run_id}"
    "${python}" "${manifest_auditor}" \
        --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
        --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch --without-test \
        --dataset-label "dunhuang_local_gazetteer_q32_v1" \
        --protocol-label "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1" \
        --output "${phase2_protocol}" \
        > "${workspace_runs}/${phase2_run_id}.train-protocol.log" 2>&1
}

run_phase2() {
    local phase1_run_dir="${remote_root}/training_runs/${phase1_run_id}/seed${seed}"
    local selected_step
    selected_step="$(${python} - "${phase1_run_dir}/selection.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding='utf-8'))['selected_step'])
PY
)"
    local phase1_checkpoint="${phase1_run_dir}/checkpoint-${selected_step}"
    [[ -f "${phase1_checkpoint}/adapter.safetensors" && -f "${phase1_checkpoint}/decoder_lora.safetensors" ]] || exit 66

    current_phase="phase2_smoke"
    current_run_id="${phase2_run_id}_smoke"
    write_status running "${current_phase}" "${current_run_id}"
    mapfile -t args < <(common_ddp_args "${current_run_id}" "${dunhuang_root}" "${phase2_protocol}" 0.4 "${phase2_gate}")
    args+=(--max-steps 8 --lr-schedule-steps 8 --warmup-steps 0 --validation-interval 9 --without-test --smoke \
        --init-checkpoint-dir "${phase1_checkpoint}" --init-checkpoint-override-residual-scale "${phase2_gate}")
    bash "${ddp_launcher}" "${args[@]}" > "${workspace_runs}/${current_run_id}.pipeline.log" 2>&1
    "${python}" - "${remote_root}/training_runs/${current_run_id}/smoke/seed${seed}/smoke_summary.json" <<'PY'
import json,sys
summary=json.load(open(sys.argv[1],encoding='utf-8'))
objective=((summary.get('training') or {}).get('loss_objective') or {})
if summary.get('status')!='complete' or summary.get('checkpoint_reload') is not True: raise SystemExit('phase2 smoke incomplete')
if objective.get('formula')!='L_official + auxiliary_weight * L_layout': raise SystemExit(f'unexpected phase2 smoke objective: {objective}')
print(json.dumps({'event':'glmocr_sparse_q32_phase2_smoke_ok'},separators=(',',':')))
PY

    current_phase="phase2_training"
    current_run_id="${phase2_run_id}"
    write_status running "${current_phase}" "${current_run_id}"
    mapfile -t args < <(common_ddp_args "${phase2_run_id}" "${dunhuang_root}" "${phase2_protocol}" 0.4 "${phase2_gate}")
    args+=(--max-steps "${phase2_steps}" --lr-schedule-steps "${phase2_steps}" --warmup-steps 216 \
        --validation-interval "${phase2_steps}" --defer-validation --without-test \
        --init-checkpoint-dir "${phase1_checkpoint}" --init-checkpoint-override-residual-scale "${phase2_gate}")
    bash "${ddp_launcher}" "${args[@]}" > "${workspace_runs}/${phase2_run_id}.pipeline.log" 2>&1
    "${python}" - "${remote_root}/training_runs/${phase2_run_id}/seed${seed}/summary.json" "${remote_root}/training_runs/${phase2_run_id}/seed${seed}/metadata.json" <<'PY'
import json,sys
summary=json.load(open(sys.argv[1],encoding='utf-8'))
metadata=json.load(open(sys.argv[2],encoding='utf-8'))
if summary.get('status')!='complete' or metadata.get('status')!='complete': raise SystemExit('phase2 training incomplete')
if summary.get('test_manifest_read') is not False or metadata.get('test_manifest_read') is not False: raise SystemExit('phase2 training read test')
if abs(float((metadata.get('adapter_config') or {}).get('initial_residual_scale',-1))-0.01)>1e-9: raise SystemExit('phase2 gate metadata is not 0.01')
if [int(x) for x in (summary.get('training') or {}).get('checkpoint_steps',[])] != [256]: raise SystemExit('phase2 checkpoint is not 256')
print(json.dumps({'event':'glmocr_sparse_q32_phase2_training_ok','checkpoint':256},separators=(',',':')))
PY

    run_validation_evaluations "${phase2_run_id}" "${dunhuang_root}" "${phase2_protocol}" "256" "${phase2_gate}" 0.4 0
    "${python}" - "${remote_root}/training_runs/${phase2_run_id}/seed${seed}" "${phase1_run_id}" <<'PY'
import json,sys
from pathlib import Path
run_dir=Path(sys.argv[1])
validation=json.loads((run_dir/'layout-validation/step-256/summary.json').read_text(encoding='utf-8')).get('validation')
payload={'status':'complete','dataset':'dunhuang_local_gazetteer_q32_v1','mode':'geometry','seed':42,'seeds':[42],
         'checkpoint_steps':[256],'selected_step':256,'selection_metric':'fixed_final_step_diagnostic',
         'selected_validation':validation,'seed_runs':{'42':str(run_dir)},'validation_evaluated':True,
         'selection_performed':False,'test_manifest_read':False,'test_used_for_selection':False,
         'source_phase1_run_id':sys.argv[2]}
text=json.dumps(payload,ensure_ascii=False,indent=2)+'\n'
(run_dir/'selection.json').write_text(text,encoding='utf-8',newline='\n')
(run_dir.parent/'selection.json').write_text(text,encoding='utf-8',newline='\n')
print(json.dumps({'event':'glmocr_sparse_q32_phase2_fixed_selection_ready','selected_step':256},separators=(',',':')))
PY

    current_phase="phase2_test"
    write_status running "${current_phase}" "${phase2_run_id}"
    "${python}" "${manifest_auditor}" \
        --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
        --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
        --test-manifest "${dunhuang_root}/test/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch \
        --dataset-label "dunhuang_local_gazetteer_q32_v1" \
        --protocol-label "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1" \
        --output "${phase2_test_protocol}" \
        > "${workspace_runs}/${phase2_run_id}.test-protocol.log" 2>&1
    bash "${locked_test_launcher}" --foreground --run-id "${phase2_run_id}" --seed "${seed}" \
        --gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}" \
        --mode geometry --num-queries "${num_queries}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" \
        --model-dir "${model_dir}" --dataset-root "${dunhuang_root}" \
        --protocol-file "${phase2_test_protocol}" \
        > "${workspace_runs}/${phase2_run_id}.locked-test.pipeline.log" 2>&1
}

write_final_summary() {
    current_phase="complete"
    write_status complete complete "${phase2_run_id}"
    "${python}" - "${summary_file}" "${phase1_run_id}" "${phase2_run_id}" "${remote_root}" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); root=Path(sys.argv[4])
phase1=root/'training_runs'/sys.argv[2]/'seed42'
phase2=root/'training_runs'/sys.argv[3]/'seed42'
phase1_test=json.loads((phase1/'locked-test/locked_test_summary.json').read_text(encoding='utf-8'))
phase2_test=json.loads((phase2/'locked-test/locked_test_summary.json').read_text(encoding='utf-8'))
selection=json.loads((phase1/'selection.json').read_text(encoding='utf-8'))
selected=int(selection['selected_step'])
payload={
 'status':'complete','session':out.stem.replace('.summary',''),'seed':42,'num_queries':32,'max_regions':24,
 'gpu_ids':'0,1,2,3,4','test_used_for_selection':False,
 'phase1':{
  'run_id':sys.argv[2],'run_dir':str(phase1),'selection':str(phase1/'selection.json'),
  'adapter_checkpoint':str(phase1/f'checkpoint-{selected}/adapter.safetensors'),
  'decoder_lora_checkpoint':str(phase1/f'checkpoint-{selected}/decoder_lora.safetensors'),
  'test_summary':str(phase1/'locked-test/locked_test_summary.json'),'metrics':phase1_test.get('metrics') or {},
  'selected_step':phase1_test.get('selected_step'),'test_used_for_selection':phase1_test.get('test_used_for_selection'),
 },
 'phase2':{
  'run_id':sys.argv[3],'run_dir':str(phase2),'adapter_checkpoint':str(phase2/'checkpoint-256/adapter.safetensors'),
  'decoder_lora_checkpoint':str(phase2/'checkpoint-256/decoder_lora.safetensors'),
  'test_summary':str(phase2/'locked-test/locked_test_summary.json'),'metrics':phase2_test.get('metrics') or {},
  'selected_step':phase2_test.get('selected_step'),'test_used_for_selection':phase2_test.get('test_used_for_selection'),
  'initial_residual_scale':0.01,
 },
}
out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'event':'glmocr_sparse_q32_pipeline_complete','summary':str(out),'phase1_layout_box_iou':payload['phase1']['metrics'].get('layout_box_iou'),'phase2_layout_box_iou':payload['phase2']['metrics'].get('layout_box_iou'),'phase2_cer':payload['phase2']['metrics'].get('cer')},separators=(',',':')))
PY
    cat "${summary_file}"
}

run_inner() {
    preflight
    if [[ -n "${reuse_phase1_run_id}" ]]; then
        run_phase1_test
    else
        prepare_sparse_train_validation
        run_phase1_smoke
        run_phase1_training
        run_phase1_test
    fi
    prepare_dunhuang_train_validation
    run_phase2
    write_final_summary
}

if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || exit 69
    tmux has-session -t "${session}" 2>/dev/null && exit 73
    preflight
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    mkdir -p "${workspace_runs}"
    command_line="$(printf '%q ' bash "${script_path}" --foreground)"
    tmux new-session -d -s "${session}" \
        "cd $(printf '%q' "${code_root}") && exec ${command_line} >$(printf '%q' "${workspace_runs}/${session}.log") 2>&1"
    printf '{"event":"glmocr_sparse_q32_pipeline_armed","session":"%s","run_id":"%s","gpu_ids":"%s","status":"%s","log":"%s"}\n' \
        "${session}" "${phase1_run_id}" "${gpu_ids}" "${status_file}" "${workspace_runs}/${session}.log"
else
    run_inner
fi
