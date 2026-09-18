#!/usr/bin/env bash
# Stage-2 判定性消融 + 输入分辨率扫描（全部 eval-only，80 页 validation，5 臂并行）。
#
# 动机一：残差通道到底有没有用。既有对照都是跨 mode 的（content_only 无残差 vs
# layout_ot 有），混淆了布局注入与残差写回。这里固定同一个 checkpoint、同一个
# mode，只改 content_gate 这一个标量：0（残差完全关闭）与 atanh(0.03)（钉在上限，
# 即当前 cap 下该通道能产生的最大影响）。对照是已跑过的 step-0 恒等（gate=父 run
# 末值 0.012923，validation CER 0.207331）。
#
# 动机二：输入分辨率。所有页面原生 310–367 万像素，而 max_pixels=1003520 把每一页
# 都下采样到约 1/3。基座模型那一臂是干净探针——它没有任何分辨率相关的适配，若
# 提高分辨率对它无益，说明信息丢失不是瓶颈；若有益，则此前所有适配器层面的调参
# 都是在补偿一个上游缺陷。
#
# 判定一律用按页配对 bootstrap（tools/analyze_cer_significance.py）的置信区间，
# 不用 80 页上的点估计。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dunhuang_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"

parent_run_id="glmocr_dunhuang_local_sem_adapter_gate001_lr1e5_dec1e6_256_from_giou20x10000_20260918_v1"
parent_checkpoint="${GLMOCR_STAGE2_ABLATION_PARENT_CHECKPOINT:-${remote_root}/training_runs/${parent_run_id}/seed42/checkpoint-256}"
continuation_run_id="glmocr_dunhuang_local_sem_adapter_continue256_val64_gate0013_lr1e5_dec1e6_20260918_v1"

run_id="${GLMOCR_STAGE2_ABLATION_RUN_ID:-glmocr_dunhuang_stage2_ablation_resolution_20260918_v1}"
session="${GLMOCR_STAGE2_ABLATION_SESSION:-glmocr_stage2_ablation_20260918_v1}"
seed="${GLMOCR_STAGE2_ABLATION_SEED:-42}"
num_queries=32
max_eval_new_tokens=1536
eval_generation_mode="loop_recovery"
baseline_pixels=1003520
native_pixels="${GLMOCR_STAGE2_ABLATION_NATIVE_PIXELS:-4000000}"
mid_pixels=2000000
residual_cap=0.03
foreground="${GLMOCR_STAGE2_ABLATION_FOREGROUND:-0}"

train_protocol="${remote_root}/protocols/${continuation_run_id}.train_validation_no_test.json"
# 已跑过的对照，用同一批 80 页做配对比较
baseline_identity_predictions="${remote_root}/training_runs/${continuation_run_id}/seed42/parallel-validation/step-0/validation_predictions.jsonl"
baseline_zeroshot_predictions="${remote_root}/zeroshot_eval/glmocr_dunhuang_local_q32_zeroshot_baseline_20260918_v1/validation/validation_predictions.jsonl"

ablation_root="${remote_root}/stage2_ablation/${run_id}"
gate_zero_checkpoint="${ablation_root}/checkpoints/gate0"
gate_cap_checkpoint="${ablation_root}/checkpoints/gate-at-cap"
python="${env_dir}/bin/python3"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

setup_environment() {
    # Standalone single-process evaluation bypasses the DDP wrapper, so the
    # CUDA/HF environment it would otherwise set has to be reproduced here.
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

preflight() {
    [[ -x "${python}" ]] || { echo "python missing: ${python}" >&2; exit 64; }
    [[ -f "${parent_checkpoint}/adapter.safetensors" ]] || { echo "parent checkpoint missing" >&2; exit 64; }
    [[ -f "${train_protocol}" ]] || { echo "train protocol missing: ${train_protocol}" >&2; exit 64; }
    [[ -f "${baseline_identity_predictions}" ]] || { echo "step-0 baseline predictions missing" >&2; exit 64; }
    [[ -f "${baseline_zeroshot_predictions}" ]] || { echo "zero-shot baseline predictions missing" >&2; exit 64; }
    if [[ -e "${ablation_root}/arms" ]]; then
        echo "output directory already exists: ${ablation_root}/arms" >&2
        exit 74
    fi
}

# Rewrite one scalar in the adapter checkpoint.  effective_residual_scale() is
# tanh(content_gate) clamped to +/- residual_cap, so the raw value has to be the
# inverse-tanh of the intended scale.
build_gate_checkpoint() {
    local target="$1" scale="$2"
    # Rewriting a safetensors file imports torch, so this needs the CUDA
    # library path even though it never touches a GPU.
    setup_environment
    # Copy the whole checkpoint: load_adapter_checkpoint requires
    # adapter_config.json alongside adapter.safetensors to decide whether the
    # stored gate may be clamped.
    mkdir -p "${target}"
    cp -r "${parent_checkpoint}/." "${target}/"
    "${python}" - "${target}/adapter.safetensors" "${scale}" <<'PY'
import math, sys
from safetensors.torch import load_file, save_file

path, scale = sys.argv[1], float(sys.argv[2])
state = load_file(path)
if "content_gate" not in state:
    raise SystemExit("adapter checkpoint has no content_gate scalar")
if abs(scale) >= 1.0:
    raise SystemExit(f"residual scale must be inside tanh's range: {scale}")
state["content_gate"] = state["content_gate"].new_full((), math.atanh(scale))
save_file(state, path, metadata={"format": "pt"})
print(f"set content_gate raw={math.atanh(scale):.12f} -> scale={math.tanh(math.atanh(scale)):.12f}")
PY
}

# launch_layout_ot <gpu> <checkpoint> <pixels> <out_dir>
launch_layout_ot() {
    local gpu="$1" checkpoint="$2" pixels="$3" out="$4"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        cd "${code_root}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode layout_ot --model-path "${model_dir}" \
            --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
            --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
            --protocol-file "${train_protocol}" \
            --output-dir "${out}" \
            --per-device-batch-size 1 --gradient-accumulation-steps 1 \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 --min-lr-ratio 0.1 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_stage2_ablation_$(basename "${out}")" \
            --learning-rate 1e-5 \
            --decoder-adaptation lora --decoder-lora-rank 8 --decoder-lora-alpha 8 \
            --decoder-lora-dropout 0 --decoder-learning-rate 1e-6 \
            --residual-scale-cap "${residual_cap}" --initial-residual-scale 0 \
            --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
            --gate-freeze-steps 0 --max-grad-norm 1.0 \
            --layout-loss-profile ocr_only --query-assignment hungarian \
            --processor-mode fast --adapter-precision fp32 \
            --box-head-mlp --query-refine-layers 1 --sem-adapter-mlp \
            --max-pixels "${pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --generation-mode "${eval_generation_mode}" \
            --validation-interval 1 --log-steps 16 \
            --eval-checkpoint-dir "${checkpoint}" --eval-only
    ) > "${out}.log" 2>&1
}

# launch_zero_shot <gpu> <pixels> <out_dir>
launch_zero_shot() {
    local gpu="$1" pixels="$2" out="$3"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        cd "${code_root}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode content_only --model-path "${model_dir}" \
            --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
            --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
            --protocol-file "${train_protocol}" \
            --output-dir "${out}" \
            --per-device-batch-size 1 --gradient-accumulation-steps 1 \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 --min-lr-ratio 0.1 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_stage2_ablation_$(basename "${out}")" \
            --learning-rate 1e-5 --max-grad-norm 1.0 \
            --decoder-adaptation frozen \
            --residual-scale-cap "${residual_cap}" --initial-residual-scale 0 \
            --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
            --layout-loss-profile ocr_only --query-assignment hungarian \
            --processor-mode fast --adapter-precision fp32 \
            --max-pixels "${pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --generation-mode "${eval_generation_mode}" \
            --validation-interval 1 --log-steps 16 \
            --eval-only
    ) > "${out}.log" 2>&1
}

run_arms() {
    local arms="${ablation_root}/arms"
    mkdir -p "${arms}"
    build_gate_checkpoint "${gate_zero_checkpoint}" 0
    build_gate_checkpoint "${gate_cap_checkpoint}" "${residual_cap}"
    echo "{\"event\":\"glmocr_stage2_ablation_started\",\"run_id\":\"${run_id}\",\"arms\":5}"

    local pids=()
    launch_layout_ot 0 "${gate_zero_checkpoint}" "${baseline_pixels}" "${arms}/A-gate0-px1m" & pids+=("$!")
    launch_layout_ot 1 "${gate_cap_checkpoint}" "${baseline_pixels}" "${arms}/B-gatecap-px1m" & pids+=("$!")
    launch_layout_ot 2 "${parent_checkpoint}" "${native_pixels}" "${arms}/C-gate0013-px4m" & pids+=("$!")
    launch_zero_shot 3 "${native_pixels}" "${arms}/D-zeroshot-px4m" & pids+=("$!")
    launch_zero_shot 4 "${mid_pixels}" "${arms}/E-zeroshot-px2m" & pids+=("$!")

    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || {
        echo "{\"event\":\"glmocr_stage2_ablation_failed\",\"error\":\"arm_failed\"}" >&2
        exit 1
    }
}

summarize() {
    setup_environment
    "${python}" - "${ablation_root}" "${baseline_pixels}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_pixels = int(sys.argv[2])
arms = root / "arms"
payload = {"status": "complete", "arms": {}}
for arm_dir in sorted(p for p in arms.iterdir() if p.is_dir()):
    summary = json.loads((arm_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["validation"]
    payload["arms"][arm_dir.name] = {
        "pages": metrics["pages"],
        "cer": metrics["cer"],
        "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "substitutions": metrics["substitutions"],
        "insertion_share": (
            metrics["insertions"]
            / max(1, metrics["insertions"] + metrics["deletions"] + metrics["substitutions"])
        ),
        "generation_limit_hits": metrics["generation_limit_hits"],
        "repeated_trigram_rate": metrics["repeated_trigram_rate"],
        "mean_new_tokens": metrics["generation_mean_new_tokens"],
        "mode": summary["mode"],
        "decoder_adaptation": summary["decoder_adaptation"],
        "eval_checkpoint_dir": summary["eval_checkpoint_dir"],
        "raw_content_gate": metrics.get("raw_content_gate"),
        "effective_residual_scale": metrics.get("effective_residual_scale"),
        "residual_relative_norm": metrics.get("residual_relative_norm"),
        "parameters_unchanged": summary["parameters_unchanged"],
        "predictions": str(arm_dir / "validation_predictions.jsonl"),
    }
payload["baseline_pixels"] = baseline_pixels
(root / "ablation_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
for name, row in payload["arms"].items():
    print(
        f"{name:24s} CER {row['cer']:.6f}  gate_raw={row['raw_content_gate']}  "
        f"res_scale={row['effective_residual_scale']}  lim={row['generation_limit_hits']}"
    )
PY
}

main() {
    preflight
    if (( foreground == 1 )); then
        mkdir -p "${ablation_root}"
        run_arms
        summarize
        return
    fi
    mkdir -p "${ablation_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    tmux new-session -d -s "${session}" \
        "bash $(printf '%q' "${script_path}") --foreground > ${ablation_root}/pipeline.log 2>&1"
    echo "{\"event\":\"glmocr_stage2_ablation_armed\",\"session\":\"${session}\",\"root\":\"${ablation_root}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --run-id) run_id="$2"; shift 2 ;;
        --native-pixels) native_pixels="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
