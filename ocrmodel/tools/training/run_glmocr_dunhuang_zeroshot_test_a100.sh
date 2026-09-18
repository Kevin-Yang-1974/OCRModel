#!/usr/bin/env bash
# GLM-OCR 官方基座在地敦煌 59 页 test（以及 80 页 validation）上的 zero-shot 基线。
#
# 为什么是 content_only + 无 checkpoint：adapter 在 content_only 模式下直接
# `merged = visual_tokens`（layout_ocr/adapter.py），而 initial_residual_scale=0
# 让 content_gate = tanh(atanh(0)) = 0，于是 PreMergeLayoutAdapter 对视觉特征
# 严格恒等——这条路径就是官方权重本身的识别行为。train_screen 也把
# checkpoint-free --eval-only 限定在这个组合上（"prompt-only content_only baseline"）。
#
# 与 layout_ot 锁定 test 的可比性：两边都走 layout_ocr.metrics.aggregate_ocr_metrics，
# 生成侧参数（max_pixels / max_eval_new_tokens / generation_mode / seed /
# processor_mode）逐项对齐，页面集合由同一份 test 协议锁定（59 页 + image sha256）。
# 差别只在 mode：zero-shot 不做布局注入，这正是要测量的量。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dunhuang_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"

run_id="${GLMOCR_DH_ZEROSHOT_RUN_ID:-glmocr_dunhuang_local_q32_zeroshot_baseline_20260918_v1}"
seed="${GLMOCR_DH_ZEROSHOT_SEED:-42}"
num_queries="${GLMOCR_DH_ZEROSHOT_NUM_QUERIES:-32}"
max_eval_new_tokens="${GLMOCR_DH_ZEROSHOT_MAX_EVAL_NEW_TOKENS:-1536}"
eval_generation_mode="${GLMOCR_DH_ZEROSHOT_GENERATION_MODE:-loop_recovery}"
eval_max_pixels="${GLMOCR_DH_ZEROSHOT_MAX_PIXELS:-1003520}"
gpu_test="${GLMOCR_DH_ZEROSHOT_GPU_TEST:-0}"
gpu_validation="${GLMOCR_DH_ZEROSHOT_GPU_VALIDATION:-1}"
foreground="${GLMOCR_DH_ZEROSHOT_FOREGROUND:-0}"

eval_root="${remote_root}/zeroshot_eval/${run_id}"
test_protocol="${GLMOCR_DH_ZEROSHOT_TEST_PROTOCOL:-${remote_root}/protocols/glmocr_dunhuang_local_sem_adapter_continue256_val64_gate0013_lr1e5_dec1e6_20260918_v1.test_locked.json}"
train_protocol="${GLMOCR_DH_ZEROSHOT_TRAIN_PROTOCOL:-${remote_root}/protocols/glmocr_dunhuang_local_sem_adapter_continue256_val64_gate0013_lr1e5_dec1e6_20260918_v1.train_validation_no_test.json}"

python="${env_dir}/bin/python3"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

setup_eval_environment() {
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
    [[ -f "${model_dir}/model.safetensors" ]] || { echo "model missing: ${model_dir}" >&2; exit 64; }
    [[ -f "${dunhuang_root}/test/manifest.jsonl" ]] || { echo "test manifest missing" >&2; exit 64; }
    [[ -f "${dunhuang_root}/validation/manifest.jsonl" ]] || { echo "validation manifest missing" >&2; exit 64; }
    [[ -f "${test_protocol}" ]] || { echo "test protocol missing: ${test_protocol}" >&2; exit 64; }
    [[ -f "${train_protocol}" ]] || { echo "train protocol missing: ${train_protocol}" >&2; exit 64; }
    if [[ -e "${eval_root}/test" || -e "${eval_root}/validation" ]]; then
        echo "output directory already exists under ${eval_root}" >&2
        exit 74
    fi
}

# launch_test <gpu> <protocol> <out_dir>
# The test split cannot go through train_screen: it loads its evaluation records
# from --validation-manifest and validate_records rejects records whose split is
# not 'validation'.  tools.evaluate_glmocr_zeroshot_test exists for this split.
launch_test() {
    local gpu="$1" protocol="$2" out="$3"
    (
        setup_eval_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        cd "${code_root}"
        exec "${python}" -m tools.evaluate_glmocr_zeroshot_test \
            --model-path "${model_dir}" \
            --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
            --test-manifest "${dunhuang_root}/test/manifest.jsonl" \
            --protocol-file "${protocol}" \
            --output-dir "${out}" \
            --mode content_only --seed "${seed}" --num-queries "${num_queries}" \
            --max-pixels "${eval_max_pixels}" \
            --max-eval-new-tokens "${max_eval_new_tokens}" \
            --generation-mode "${eval_generation_mode}"
    ) > "${eval_root}/test.log" 2>&1
}

# launch_validation <gpu> <protocol> <out_dir>
launch_validation() {
    local gpu="$1" protocol="$2" out="$3"
    (
        setup_eval_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        cd "${code_root}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode content_only --model-path "${model_dir}" \
            --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
            --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
            --protocol-file "${protocol}" \
            --output-dir "${out}" \
            --per-device-batch-size 1 --gradient-accumulation-steps 1 \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 --min-lr-ratio 0.1 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_dh_zeroshot_validation" \
            --learning-rate 1e-5 --max-grad-norm 1.0 \
            --decoder-adaptation frozen \
            --residual-scale-cap 0.03 --initial-residual-scale 0 \
            --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
            --layout-loss-profile ocr_only --query-assignment hungarian \
            --processor-mode fast --adapter-precision fp32 \
            --max-pixels "${eval_max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --generation-mode "${eval_generation_mode}" \
            --validation-interval 1 --log-steps 1 \
            --eval-only
    ) > "${eval_root}/validation.log" 2>&1
}

run_evals() {
    if [[ "${GLMOCR_DH_ZEROSHOT_SKIP_GPU_CHECK:-0}" != "1" ]]; then
        local busy
        busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | wc -l)"
        if (( busy > 0 )); then
            echo "{\"event\":\"glmocr_dh_zeroshot_gpu_busy\",\"compute_apps\":${busy}}" >&2
            exit 69
        fi
    fi
    echo "{\"event\":\"glmocr_dh_zeroshot_started\",\"run_id\":\"${run_id}\",\"test_gpu\":\"${gpu_test}\",\"validation_gpu\":\"${gpu_validation}\"}"

    local pids=()
    launch_test "${gpu_test}" "${test_protocol}" "${eval_root}/test" &
    pids+=("$!")
    launch_validation "${gpu_validation}" "${train_protocol}" "${eval_root}/validation" &
    pids+=("$!")

    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || {
        echo "{\"event\":\"glmocr_dh_zeroshot_failed\",\"error\":\"eval_failed\"}" >&2
        tail -n 40 "${eval_root}/test.log" >&2 || true
        exit 1
    }
}

summarize() {
    setup_eval_environment
    "${python}" - "${eval_root}" "${run_id}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]
payload = {"status": "complete", "run_id": run_id, "baseline": "glm_ocr_official_zeroshot"}
sources = {
    "test": ("test/zeroshot_test_summary.json", "metrics"),
    "validation": ("validation/summary.json", "validation"),
}
for split, (relative, metrics_key) in sources.items():
    summary = json.loads((root / relative).read_text(encoding="utf-8"))
    metrics = summary[metrics_key]
    payload[split] = {
        "pages": metrics["pages"],
        "cer": metrics["cer"],
        "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "substitutions": metrics["substitutions"],
        "insertion_share": (
            metrics["insertions"]
            / max(1, metrics["insertions"] + metrics["deletions"] + metrics["substitutions"])
        ),
        "exact_page_rate": metrics["exact_page_rate"],
        "generation_limit_hits": metrics["generation_limit_hits"],
        "repeated_trigram_rate": metrics["repeated_trigram_rate"],
        "mean_new_tokens": metrics["generation_mean_new_tokens"],
        "mode": summary["mode"],
        "decoder_adaptation": summary["decoder_adaptation"],
        "eval_checkpoint_dir": summary["eval_checkpoint_dir"],
        "parameters_unchanged": summary["parameters_unchanged"],
        "training_updates": summary["training_updates"],
        "test_manifest_read": summary["test_manifest_read"],
    }
(root / "zeroshot_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
}

main() {
    preflight
    if (( foreground == 1 )); then
        mkdir -p "${eval_root}"
        run_evals
        summarize
        return
    fi
    local session="${GLMOCR_DH_ZEROSHOT_SESSION:-glmocr_dh_zeroshot_20260918_v1}"
    mkdir -p "${eval_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    tmux new-session -d -s "${session}" \
        "bash $(printf '%q' "${script_path}") --foreground > ${eval_root}/pipeline.log 2>&1"
    echo "{\"event\":\"glmocr_dh_zeroshot_armed\",\"session\":\"${session}\",\"eval_root\":\"${eval_root}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --run-id) run_id="$2"; shift 2 ;;
        --gpu-test) gpu_test="$2"; shift 2 ;;
        --gpu-validation) gpu_validation="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
