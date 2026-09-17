#!/usr/bin/env bash
# Run one selection-locked q32 test for a completed training run on BSCC.
# Test reads the held-out split only after validation selection is complete.
#SBATCH --job-name=glmocr_q32_locked_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=240:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
env_dir="${experiment_root}/envs/glm-ocr-py311-cu128"
nvidia_env="${BSCC_NVIDIA_ENV:-${env_dir}}"
model_dir="${workspace}/models/GLM-OCR-ca5d8b3"
dataset_root="${BSCC_Q32_DATASET_ROOT:-${workspace}/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"
python="${env_dir}/bin/python"
locked_test_launcher="${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh"
metrics_summarizer="${code_root}/tools/summarize_glmocr_final_metrics.py"
manifest_auditor="${code_root}/tools/audit_mthv2_manifest.py"

run_id="${GLMOCR_BSCC_TEST_RUN_ID:-}"
session="${GLMOCR_BSCC_TEST_SESSION:-}"
mode="${GLMOCR_BSCC_TEST_MODE:-}"
seed="${GLMOCR_BSCC_SEED:-42}"
gpu_ids="${GLMOCR_BSCC_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0,1,2,3}}}"
num_queries=32
max_eval_new_tokens=1536
dataset_label="dunhuang_local_gazetteer_q32_v1"
protocol_label="glm_ocr_dunhuang_local_gazetteer_group_isolated_v1"
expected_test_manifest_sha256="da94dbd7e372e5e206c7c340be881e40f11eabff1b12df26668f9a983e7f782d"
expected_test_pages=59
expected_reference_characters=14475

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ "${mode}" == "geometry" || "${mode}" == "content_only" ]] || exit 64
[[ "${seed}" =~ ^[0-9]+$ && "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]] || exit 64

group_root="${experiment_root}/training_runs/${run_id}"
run_dir="${group_root}/seed${seed}"
selection_file="${group_root}/selection.json"
protocol_file="${experiment_root}/protocols/${run_id}.test_locked.json"
test_manifest="${dataset_root}/test/manifest.jsonl"
train_manifest="${dataset_root}/train/manifest.jsonl"
validation_manifest="${dataset_root}/validation/manifest.jsonl"
status_file="${workspace}/runs/${session}.status.json"
summary_file="${workspace}/runs/${session}.summary.json"
launcher_log="${workspace}/runs/${session}.sbatch.log"
current_phase="preflight"

write_status() {
    local status="$1"
    local phase="$2"
    mkdir -p "${workspace}/runs"
    printf '{"status":"%s","phase":"%s","run_id":"%s","mode":"%s","session":"%s","dataset":"%s","gpu_ids":"%s","world_size":4,"updated_at":"%s","test_used_for_selection":false}\n' \
        "${status}" "${phase}" "${run_id}" "${mode}" "${session}" \
        "${dataset_label}" "${gpu_ids}" "$(date -u +%FT%TZ)" > "${status_file}"
}

on_error() {
    local rc=$?
    trap - ERR
    write_status failed "${current_phase}_failed" || true
    exit "${rc}"
}
trap on_error ERR

preflight() {
    [[ -x "${python}" && -f "${locked_test_launcher}" && -f "${metrics_summarizer}" && -f "${manifest_auditor}" ]] || exit 66
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || exit 66
    [[ -f "${train_manifest}" && -f "${validation_manifest}" && -f "${test_manifest}" ]] || exit 66
    [[ -f "${run_dir}/summary.json" && -f "${run_dir}/metadata.json" && -f "${run_dir}/selection.json" && -f "${run_dir}/COMPLETED" ]] || exit 66
    [[ ! -e "${protocol_file}" && ! -e "${run_dir}/locked-test" && ! -e "${run_dir}/locked-test-shards" ]] || exit 74
    [[ ! -e "${status_file}" && ! -e "${summary_file}" ]] || exit 74
    [[ "$(jq -r '.selected_step' "${selection_file}")" == "200" ]] || exit 64
}

make_locked_protocol() {
    current_phase="test_protocol"
    write_status running "${current_phase}"
    "${python}" "${manifest_auditor}" \
        --train-manifest "${train_manifest}" \
        --validation-manifest "${validation_manifest}" \
        --test-manifest "${test_manifest}" \
        --num-queries "${num_queries}" \
        --dataset-label "${dataset_label}" \
        --protocol-label "${protocol_label}" \
        --allow-count-mismatch \
        --output "${protocol_file}" \
        > "${workspace}/runs/${session}.test-protocol-audit.log" 2>&1
    [[ "$(jq -r '.split_pages.train' "${protocol_file}")" == "240" ]] || exit 64
    [[ "$(jq -r '.split_pages.validation' "${protocol_file}")" == "80" ]] || exit 64
    [[ "$(jq -r '.split_pages.test' "${protocol_file}")" == "59" ]] || exit 64
    [[ "$(jq -r '.test_manifest_read' "${protocol_file}")" == "true" ]] || exit 64
    [[ "$(jq -r '.test_used_for_selection' "${protocol_file}")" == "false" ]] || exit 64
    digest="$(sha256sum "${test_manifest}" | awk '{print $1}')"
    [[ "${digest}" == "${expected_test_manifest_sha256}" ]] || exit 64
    [[ "$(jq -r '.manifest_stats.test.manifest_sha256' "${protocol_file}")" == "${digest}" ]] || exit 64
}

run_locked_test() {
    current_phase="locked_test"
    write_status running "${current_phase}"
    export GLMOCR_A100_NVIDIA_ENV="${nvidia_env}"
    source "/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311/env.sh"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    set +u
    conda activate "${env_dir}"
    set -u
    export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
    export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
    bash "${locked_test_launcher}" --foreground \
        --run-id "${run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" \
        --mode "${mode}" --num-queries "${num_queries}" \
        --max-eval-new-tokens "${max_eval_new_tokens}" \
        --remote-root "${experiment_root}" --code-root "${code_root}" \
        --env-dir "${env_dir}" --model-dir "${model_dir}" \
        --dataset-root "${dataset_root}" --protocol-file "${protocol_file}" \
        > "${launcher_log}" 2>&1
    "${python}" "${metrics_summarizer}" \
        --manifest "${test_manifest}" \
        --predictions "${run_dir}/locked-test/test_predictions.jsonl" \
        --output "${run_dir}/locked-test/final_metrics.json" \
        --expected-pages "${expected_test_pages}" \
        > "${workspace}/runs/${session}.final-metrics.log" 2>&1
    [[ "$(jq -r '.status' "${run_dir}/locked-test/locked_test_summary.json")" == "complete" ]] || exit 1
    [[ "$(jq -r '.split' "${run_dir}/locked-test/locked_test_summary.json")" == "test" ]] || exit 1
    [[ "$(jq -r '.selected_step' "${run_dir}/locked-test/locked_test_summary.json")" == "200" ]] || exit 1
    [[ "$(jq -r '.test_pages' "${run_dir}/locked-test/locked_test_summary.json")" == "59" ]] || exit 1
    [[ "$(jq -r '.test_pages_total' "${run_dir}/locked-test/locked_test_summary.json")" == "59" ]] || exit 1
    [[ "$(jq -r '.test_shard_count' "${run_dir}/locked-test/locked_test_summary.json")" == "4" ]] || exit 1
    [[ "$(jq -r '.test_used_for_selection' "${run_dir}/locked-test/locked_test_summary.json")" == "false" ]] || exit 1
    [[ "$(jq -r '.pages' "${run_dir}/locked-test/final_metrics.json")" == "59" ]] || exit 1
    [[ "$(jq -r '.raw_reference_characters' "${run_dir}/locked-test/final_metrics.json")" == "14475" ]] || exit 1
    jq -n --arg run_id "${run_id}" --arg mode "${mode}" --arg session "${session}" \
        --slurpfile metrics "${run_dir}/locked-test/final_metrics.json" \
        '{status:"complete",phase:"locked_test_complete",run_id:$run_id,mode:$mode,session:$session,dataset:"dunhuang_local_gazetteer_q32_v1",split:"test",selected_step:200,test_pages:59,max_eval_new_tokens:1536,test_used_for_selection:false,final_metrics:$metrics[0]}' \
        > "${summary_file}"
    write_status complete locked_test_complete
}

preflight
mkdir -p "${workspace}/runs" "${experiment_root}/protocols"
make_locked_protocol
run_locked_test
