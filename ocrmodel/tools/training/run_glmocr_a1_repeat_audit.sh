#!/usr/bin/env bash
# Evaluate existing A1 checkpoints with the inference repeat guard enabled and
# disabled.  This is validation-only and never opens the MTHv2 test manifest.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dataset_root="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
old_screen_id="${GLMOCR_A1_OLD_SCREEN_ID:-glmocr_fast_screen_260910_r3_A1}"
validation_screen_id="${GLMOCR_A1_VALIDATION_SCREEN_ID:-glmocr_fast_screen_260910_r3}"
audit_id="${GLMOCR_A1_REPEAT_AUDIT_ID:-glmocr_a1_repeat_audit_260910_v1}"
gpu_id="${GLMOCR_A1_AUDIT_GPU_ID:-0}"

python="${env_dir}/bin/python"
base_runs="${remote_root}/training_runs"
checkpoint_root="${base_runs}/${old_screen_id}/seed42"
validation_manifest="${base_runs}/${validation_screen_id}/validation64_seed42.jsonl"
protocol_file="${base_runs}/${validation_screen_id}/protocol_validation64_no_test.json"
audit_root="${base_runs}/${audit_id}"

[[ -x "${python}" && -f "${model_dir}/config.json" ]] || {
    echo '{"error":"audit_source_or_environment_missing"}' >&2
    exit 66
}
[[ -f "${validation_manifest}" && -f "${protocol_file}" ]] || {
    echo '{"error":"audit_validation_protocol_missing"}' >&2
    exit 66
}
[[ ! -e "${audit_root}" ]] || {
    printf '{"error":"audit_output_exists","path":"%s"}\n' "${audit_root}" >&2
    exit 74
}
command -v nvidia-smi >/dev/null 2>&1 || { echo '{"error":"nvidia_smi_missing"}' >&2; exit 69; }
gpu_utilization="$(nvidia-smi -i "${gpu_id}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "${gpu_utilization}" =~ ^[0-9]+$ && ${gpu_utilization} -lt 50 ]] || {
    printf '{"error":"gpu_admission_failed","gpu":"%s","utilization":"%s"}\n' "${gpu_id}" "${gpu_utilization}" >&2
    exit 75
}
mkdir -p "${audit_root}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${remote_root}/code/ocrmodel/src:${remote_root}/code/ocrmodel${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
cuda_library_path="/usr/local/cuda/targets/x86_64-linux/lib:${torch_lib}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    if [[ -d "${component_lib}" ]]; then
        cuda_library_path="${cuda_library_path}:${component_lib}"
    fi
done
export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

run_eval() {
    local step="$1"
    local mode="$2"
    local output_dir="${audit_root}/step${step}_${mode}"
    local checkpoint_dir="${checkpoint_root}/checkpoint-${step}"
    [[ -f "${checkpoint_dir}/adapter.safetensors" && -f "${checkpoint_dir}/decoder_lora.safetensors" ]] || {
        printf '{"error":"audit_checkpoint_missing","checkpoint":"%s"}\n' "${checkpoint_dir}" >&2
        exit 66
    }
    args=(
        --eval-only
        --mode geometry
        --model-path "${model_dir}"
        --train-manifest "${dataset_root}/train/manifest.jsonl"
        --validation-manifest "${validation_manifest}"
        --protocol-file "${protocol_file}"
        --output-dir "${output_dir}"
        --max-steps 256
        --lr-schedule-steps 256
        --warmup-steps 32
        --num-queries 512
        --seed 42
        --experiment-label "a1_repeat_audit_step${step}_${mode}"
        --decoder-adaptation lora
        --decoder-lora-rank 8
        --decoder-lora-alpha 8
        --decoder-learning-rate 1e-6
        --auxiliary-weight 0.2
        --auxiliary-weight-start 0.2
        --adapter-precision fp32
        --layout-loss-profile full
        --query-assignment hungarian
        --max-pixels 1003520
        --max-eval-new-tokens 768
        --diagnostic-steps 1
        --audit-prompt-prefix
        --eval-checkpoint-dir "${checkpoint_dir}"
        --repeat-recent-window 96
        --repeat-min-cycle-length 8
        --repeat-max-cycle-length 32
        --repeat-cycle-repeats 3
        --repeat-cycle-penalty 1.0
        --repeat-force-eos-steps 0
    )
    if [[ "${mode}" == "guard_on" ]]; then
        args+=(
            --text-repeat-suppression
            --text-ul-weight 0.1
            --text-eos-loss-weight 0.05
            --repeat-cycle-penalty 2.0
            --repeat-force-eos-steps 16
        )
    fi
    "${python}" -m layout_ocr.train_screen "${args[@]}" > "${output_dir}.log" 2>&1
}

for step in 128 256; do
    run_eval "${step}" guard_on
    run_eval "${step}" guard_off
done

"${python}" - "${audit_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for step in (128, 256):
    for mode in ("guard_on", "guard_off"):
        path = root / f"step{step}_{mode}" / "summary.json"
        if not path.is_file():
            raise SystemExit(f"missing audit summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload.get("validation") or {}
        rows.append({
            "step": step,
            "mode": mode,
            "run_dir": str(path.parent),
            "cer": validation.get("cer"),
            "insertions": validation.get("insertions"),
            "deletions": validation.get("deletions"),
            "substitutions": validation.get("substitutions"),
            "repeated_cycle_page_rate": validation.get("repeated_cycle_page_rate"),
            "generation_limit_hit_rate": validation.get("generation_limit_hit_rate"),
            "generation_eos_hit_rate": validation.get("generation_eos_hit_rate"),
            "teacher_forced_validation_loss": validation.get("teacher_forced_validation_loss"),
            "prompt_target_audit": validation.get("prompt_target_audit"),
            "repeat_guard_enabled": payload.get("repeat_guard_enabled"),
        })
summary = {
    "status": "complete",
    "audit_id": root.name,
    "seed": 42,
    "validation_pages": 64,
    "test_manifest_read": False,
    "test_used_for_selection": False,
    "selection_metric": "guard_off_validation_cer",
    "rows": rows,
}
(root / "repeat_audit_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
PY
