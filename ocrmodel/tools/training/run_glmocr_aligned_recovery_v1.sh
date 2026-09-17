#!/usr/bin/env bash
# One immutable pipeline: CPU checks -> two-step smoke -> 1024 train -> locked test.
set -Eeuo pipefail
root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code="${GLMOCR_A100_CODE_ROOT:-${root}/code/ocrmodel}"
python="${GLMOCR_A100_ENV:-${root}/envs/glmocr_a100_py311_cu128}/bin/python"
dataset="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
run_id="${1:-glmocr_aligned_recovery_v1_base1024_260912}"
[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
group="${root}/training_runs/${run_id}"
smoke_id="${run_id}_smoke2"
skip_smoke="${GLMOCR_ALIGNED_SKIP_SMOKE:-0}"
diagnostic_steps="${GLMOCR_ALIGNED_DIAGNOSTIC_STEPS:-0,512,1024}"
if (( skip_smoke == 1 )); then
    [[ ! -e "${group}" ]] || {
        printf '{"status":"failed","error":"run_already_exists"}\n'; exit 74;
    }
else
    [[ ! -e "${group}" && ! -e "${root}/training_runs/${smoke_id}" ]] || {
        printf '{"status":"failed","error":"run_already_exists"}\n'; exit 74;
    }
fi
mkdir -p "${group}" "${root}/protocols"
phase() {
    printf '{"status":"%s","phase":"%s","run_id":"%s","updated_at":"%s"}\n' \
        "$1" "$2" "${run_id}" "$(date -u +%FT%TZ)" > "${group}/pipeline_status.json"
}
current_phase="preflight"
trap 'phase failed "${current_phase}"' ERR
phase running "${current_phase}"
export GLMOCR_DDP_TIMEOUT_SECONDS=86400
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code}/src:${code}"
env_dir="$(dirname "$(dirname "${python}")")"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
cuda_libraries="${env_dir}/lib/python3.11/site-packages/torch/lib"
cuda_arch="$(uname -m)"
[[ "${cuda_arch}" != "arm64" ]] || cuda_arch="aarch64"
system_cuda="/usr/local/cuda/targets/${cuda_arch}-linux/lib"
[[ ! -d "${system_cuda}" ]] || cuda_libraries="${system_cuda}:${cuda_libraries}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    [[ ! -d "${lib}" ]] || cuda_libraries="${cuda_libraries}:${lib}"
done
export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
common=(--seed 42 --gpu-ids 0,1,2,3,4 --decoder-adaptation lora
    --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0
    --learning-rate 5e-5 --decoder-learning-rate 1e-6 --min-lr-ratio 0.5
    --layout-loss-profile full --auxiliary-weight 0.2 --initial-residual-scale 0
    --generation-mode plain --repeat-cycle-penalty 0 --repeat-force-eos-steps 0
    --max-eval-new-tokens 768 --recovery-mode aligned_recovery_v1
    --without-test --foreground)
if (( skip_smoke == 0 )); then
    # Local/remote CPU checks do not load any dataset or GPU model.
    "${python}" "${code}/tools/test_aligned_recovery.py" > "${group}/cpu_checks.log" 2>&1
    "${python}" -m torch.distributed.run --standalone --nproc_per_node=2 \
        "${code}/tools/check_aligned_recovery_ddp.py" > "${group}/cpu_ddp_checks.log" 2>&1
    smoke_validation="${root}/protocols/${smoke_id}.validation5.seed42.jsonl"
    rm -f "${smoke_validation}"
    "${python}" "${code}/tools/subset_mthv2_manifest.py" \
        --input "${dataset}/validation/manifest.jsonl" \
        --output "${smoke_validation}" --count 5 --seed 42 \
        > "${group}/smoke_validation_subset.json" 2>&1
    current_phase="smoke"
    phase running "${current_phase}"
    bash "${code}/tools/training/run_glmocr_mthv2_ddp.sh" "${common[@]}" \
        --run-id "${smoke_id}" --max-steps 2 --lr-schedule-steps 2 --warmup-steps 1 \
        --aligned-rollout-interval 1 --validation-interval 2 --log-steps 1 \
        --validation-manifest "${smoke_validation}" \
        --protocol-file "${root}/protocols/${smoke_id}.no_test.json"
    "${python}" "${code}/tools/finalize_aligned_recovery.py" \
        --run-dir "${root}/training_runs/${smoke_id}/seed42" --smoke
fi
current_phase="training_and_validation"
phase running "${current_phase}"
bash "${code}/tools/training/run_glmocr_mthv2_ddp.sh" "${common[@]}" \
    --run-id "${run_id}" --experiment-label aligned_recovery_v1_base1024 \
    --max-steps 1024 --lr-schedule-steps 1024 --warmup-steps 102 \
    --aligned-rollout-interval 4 --validation-interval 512 --diagnostic-steps "${diagnostic_steps}" \
    --protocol-file "${root}/protocols/${run_id}.no_test.json"
"${python}" "${code}/tools/finalize_aligned_recovery.py" --run-dir "${group}/seed42"
current_phase="locked_test"
phase running "${current_phase}"
bash "${code}/tools/training/run_glmocr_mthv2_locked_test.sh" \
    --run-id "${run_id}" --seed 42 --gpu-ids 0,1,2,3,4 --foreground
phase complete complete
printf '{"status":"complete","run_id":"%s"}\n' "${run_id}"
