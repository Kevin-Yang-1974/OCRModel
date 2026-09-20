#!/usr/bin/env bash
# 训练行检测器（FCOS ResNet50），供布局注意力偏置使用预测行框。
#
# 为什么需要：阶段 2 证明「偏置到整行」保住了字框偏置的收益（删除 −41%、插入 −49%、
# 替换不动，对 noroute 配对 CI 显著），但那些行框是**标注真值**，不可部署；而布局分支
# 自己的预测质量不够，替换不了。所以框从这里来 —— 只从图像出。
#
# 协议（AGENTS.md 第 10 条）：train 训练、validation 选点、**test 不碰**。
# 索引工具在代码里拒绝 test，本脚本读的索引因此只可能来自 train/validation。
#
# 数据实测：MTH1000 是 99.8% **竖排**，行框中位 96px 宽 × 1035px 高（宽高比 >10:1）。
# 所以选 FCOS（无锚框）而不是锚框式检测头：后者要对这种比例调锚点，调错的表现恰好是
# **漏掉细列**，而那正是路由偏置最需要的部分。
#
# 两个模式：
#   --calibrate  100 页跑 1 轮，只报每张图的秒数，用来在决定 epoch 预算之前先量一次。
#                1324 页×1200×2400 下每 epoch 可能 40–70 分钟，12 轮就是 8–14 小时，
#                没有实测就定预算是在赌。
#   默认         全量训练。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
index_dir="${GLMOCR_DET_INDEX:-${remote_root}/line_detection/index_v1}"

run_id="${GLMOCR_DET_RUN_ID:-line_detector_20260920_v1}"
session_override="${GLMOCR_DET_SESSION:-}"
gpu="${GLMOCR_DET_GPU:-0}"
epochs="${GLMOCR_DET_EPOCHS:-8}"
batch_size="${GLMOCR_DET_BATCH:-4}"
min_size="${GLMOCR_DET_MIN_SIZE:-1000}"
max_size="${GLMOCR_DET_MAX_SIZE:-2000}"
learning_rate="${GLMOCR_DET_LR:-0.005}"
workers="${GLMOCR_DET_WORKERS:-8}"
foreground="${GLMOCR_DET_FOREGROUND:-0}"

out_root="${remote_root}/line_detection/${run_id}"
python="${env_dir}/bin/python3"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

setup_environment() {
    local cuda_arch cuda_libraries
    cuda_arch="$(uname -m)"
    [[ "${cuda_arch}" != "arm64" ]] || cuda_arch="aarch64"
    cuda_libraries="${env_dir}/lib/python3.11/site-packages/torch/lib"
    local system_cuda="/usr/local/cuda/targets/${cuda_arch}-linux/lib"
    # Without the system CUDA directory first, torch fails on an undefined cupti symbol.  This is
    # the piece every ad-hoc invocation forgets and the launchers have to carry.
    [[ ! -d "${system_cuda}" ]] || cuda_libraries="${system_cuda}:${cuda_libraries}"
    local component lib
    for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
        lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
        [[ ! -d "${lib}" ]] || cuda_libraries="${cuda_libraries}:${lib}"
    done
    export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    # Deliberately *not* HF_HUB_OFFLINE: the COCO weights for the backbone come off the network
    # once, and that is the only download this run makes.
    export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
    export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
}

preflight() {
    [[ -x "${python}" ]] || { echo "python missing: ${python}" >&2; exit 64; }
    [[ -f "${index_dir}/lines_train.jsonl" ]] \
        || { echo "train index missing: ${index_dir}/lines_train.jsonl" >&2; exit 64; }
    [[ -f "${index_dir}/lines_validation.jsonl" ]] \
        || { echo "validation index missing: ${index_dir}/lines_validation.jsonl" >&2; exit 64; }
    [[ -f "${code_root}/tools/train_line_detector.py" ]] \
        || { echo "trainer not synced" >&2; exit 64; }
    [[ ! -e "${out_root}/best.pt" ]] || { echo "output exists: ${out_root}/best.pt" >&2; exit 74; }
}

calibrate() {
    local out="${out_root}_calibrate"
    mkdir -p "${out}"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        cd "${code_root}"
        /usr/bin/time -v "${python}" tools/train_line_detector.py \
            --train-index "${index_dir}/lines_train.jsonl" \
            --validation-index "${index_dir}/lines_validation.jsonl" \
            --output-dir "${out}" \
            --epochs 1 --batch-size "${batch_size}" --workers "${workers}" \
            --min-size "${min_size}" --max-size "${max_size}" \
            --learning-rate "${learning_rate}" \
            --limit-train 100 --limit-validation 20
    ) > "${out}.log" 2>&1
    grep -E '"epoch"|Elapsed|Maximum resident' "${out}.log" | tail -5
    echo "{\"event\":\"line_detector_calibrated\",\"log\":\"${out}.log\",\"pages\":100}"
}

train() {
    mkdir -p "${out_root}"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        cd "${code_root}"
        exec "${python}" tools/train_line_detector.py \
            --train-index "${index_dir}/lines_train.jsonl" \
            --validation-index "${index_dir}/lines_validation.jsonl" \
            --output-dir "${out_root}" \
            --epochs "${epochs}" --batch-size "${batch_size}" --workers "${workers}" \
            --min-size "${min_size}" --max-size "${max_size}" \
            --learning-rate "${learning_rate}"
    ) > "${out_root}.log" 2>&1
}

main() {
    preflight
    if [[ "${foreground}" == "1" ]]; then
        if (( calibrate_only == 1 )); then calibrate; else train; fi
        return
    fi
    local session="${session_override:-glmocr_line_detector_$(printf '%s' "${run_id}" | tr '.' '_')}"
    mkdir -p "${out_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    tmux new-session -d -s "${session}" \
        "export GLMOCR_DET_RUN_ID=$(printf '%q' "${run_id}"); "\
"export GLMOCR_DET_GPU=$(printf '%q' "${gpu}"); "\
"export GLMOCR_DET_EPOCHS=$(printf '%q' "${epochs}"); "\
"export GLMOCR_DET_BATCH=$(printf '%q' "${batch_size}"); "\
"export GLMOCR_DET_INDEX=$(printf '%q' "${index_dir}"); "\
"export GLMOCR_DET_MIN_SIZE=$(printf '%q' "${min_size}"); "\
"export GLMOCR_DET_MAX_SIZE=$(printf '%q' "${max_size}"); "\
"bash $(printf '%q' "${script_path}") --foreground $([[ ${calibrate_only} == 1 ]] && echo --calibrate) > ${out_root}_pipeline.log 2>&1"
    echo "{\"event\":\"line_detector_armed\",\"session\":\"${session}\",\"root\":\"${out_root}\",\"gpu\":\"${gpu}\",\"epochs\":\"${epochs}\",\"calibrate\":\"${calibrate_only}\"}"
}

calibrate_only=0
while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --calibrate) calibrate_only=1; shift ;;
        --gpu) gpu="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --run-id) run_id="$2"; out_root="${remote_root}/line_detection/${run_id}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
