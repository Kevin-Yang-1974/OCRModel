#!/usr/bin/env bash
# 布局分支归因：布局分支对 OCR 到底有没有贡献，贡献是否来自「布局信息」。
#
# 已确认的结构事实（layout_ocr/glm_bridge.py）：布局分支通往 OCR 输出的唯一路径是
# `merged_tokens = visual_tokens + gate * layout_context`；box/order/direction 头只经
# bridge 的 last_output 侧信道进训练损失，从不进入基座前向。因此「布局分支有效」等价于
# 「gate * layout_context 让 OCR 变好」。
#
# 但仅仅 gate=0 vs gate>0 不足以归因：同量级的任意扰动也可能带来同样效果。于是设计成
# 一条**布局质量阶梯**——把不同质量的布局分支（查询/传输机制）接到同一个 stage-2 适配器
# 上，只换 19 个 query_* 张量，其余（sem_adapter / content_norm / content_gate / 各头）
# 全部沿用父 run：
#
#   A  gate=0                    残差关闭（无布局信息进入 OCR）
#   B  gate=cap, 父 run 自己的分支   IoU 0.672（本线最优）
#   C  gate=cap, 查询机制随机化      同量级、无布局信息（内容无关对照）
#   D  gate=cap, giou10x_3000      IoU 0.4225
#   E  gate=cap, giou10x_10000     IoU 0.648
#
# 判定：
#   B < A 显著            → 布局分支的残差确实改善 OCR
#   C 不优于 A            → 该效果不是任意扰动造成的
#   D → E → B 单调改善     → 布局质量可迁移到识别，布局分支「有效」成立
#   若 C ≈ B              → 所谓效果只是同量级扰动，与布局信息无关（阴性结论）
#
# 强度取 cap(0.03) 而非父 run 的 0.0129：1M 下的消融显示 0 → 0.0129 → 0.03 单调改善，
# 取上限可最大化检出布局效应的机会。已有参照：同 ckpt、gate=0.0129、4M → 0.111608。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dunhuang_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"

parent_run_id="glmocr_dunhuang_local_sem_adapter_gate001_lr1e5_dec1e6_256_from_giou20x10000_20260918_v1"
parent_checkpoint="${GLMOCR_LAYOUT_ATTR_PARENT_CHECKPOINT:-${remote_root}/training_runs/${parent_run_id}/seed42/checkpoint-256}"
continuation_run_id="glmocr_dunhuang_local_sem_adapter_continue256_val64_gate0013_lr1e5_dec1e6_20260918_v1"
layout_runs_root="${remote_root}/training_runs"
low_iou_run="glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou10x_mlp_refine_3000_from_boxeq820_20260917_v1"
mid_iou_run="glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou10x_mlp_refine_10000_continue_from_giou10x3000_20260917_v2"

run_id="${GLMOCR_LAYOUT_ATTR_RUN_ID:-glmocr_dunhuang_layout_branch_attribution_20260919_v1}"
session="${GLMOCR_LAYOUT_ATTR_SESSION:-glmocr_layout_attr_20260919_v1}"
seed=42
num_queries=32
max_eval_new_tokens=1536
eval_generation_mode="loop_recovery"
# 原生分辨率：消融已证明 1003520 把每页压到约三分之一、值 0.0977 CER，归因必须在
# 真实工作点上做。
max_pixels="${GLMOCR_LAYOUT_ATTR_MAX_PIXELS:-4000000}"
residual_cap=0.03
foreground="${GLMOCR_LAYOUT_ATTR_FOREGROUND:-0}"

train_protocol="${remote_root}/protocols/${continuation_run_id}.train_validation_no_test.json"
attr_root="${remote_root}/layout_branch_attribution/${run_id}"
python="${env_dir}/bin/python3"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

setup_environment() {
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
    [[ -f "${train_protocol}" ]] || { echo "train protocol missing" >&2; exit 64; }
    [[ -f "${layout_runs_root}/${low_iou_run}/seed42/checkpoint-3000/adapter.safetensors" ]] \
        || { echo "low-IoU layout checkpoint missing" >&2; exit 64; }
    [[ -f "${layout_runs_root}/${mid_iou_run}/seed42/checkpoint-10000/adapter.safetensors" ]] \
        || { echo "mid-IoU layout checkpoint missing" >&2; exit 64; }
    [[ ! -e "${attr_root}/arms" ]] || { echo "output exists: ${attr_root}/arms" >&2; exit 74; }
}

# build_arm_checkpoint <target> <gate_scale> <query_source>
#   query_source: keep | random | <path to another adapter.safetensors>
build_arm_checkpoint() {
    local target="$1" gate_scale="$2" query_source="$3"
    setup_environment
    mkdir -p "${target}"
    cp -r "${parent_checkpoint}/." "${target}/"
    "${python}" - "${target}/adapter.safetensors" "${gate_scale}" "${query_source}" <<'PY'
import math, sys

import torch
from safetensors.torch import load_file, save_file

path, scale, source = sys.argv[1], float(sys.argv[2]), sys.argv[3]
state = load_file(path)
QUERY_PREFIXES = ("query_seed", "query_attention.", "query_norm.", "query_refine.")
query_keys = [k for k in state if k.startswith(QUERY_PREFIXES)]
if len(query_keys) != 19:
    raise SystemExit(f"expected 19 query tensors, found {len(query_keys)}")

if source == "random":
    # Same mean/std per tensor, no learned structure: keeps the residual magnitude
    # in the same range so the comparison is about information, not scale.
    generator = torch.Generator().manual_seed(20260919)
    for key in query_keys:
        tensor = state[key]
        noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
        state[key] = (noise * tensor.float().std() + tensor.float().mean()).to(tensor.dtype)
elif source != "keep":
    donor = load_file(source)
    missing = [k for k in query_keys if k not in donor]
    mismatch = [k for k in query_keys if k in donor and donor[k].shape != state[k].shape]
    if missing or mismatch:
        raise SystemExit(f"donor incompatible: missing={missing[:3]} mismatch={mismatch[:3]}")
    for key in query_keys:
        state[key] = donor[key].to(state[key].dtype)

if abs(scale) >= 1.0:
    raise SystemExit(f"residual scale must be inside tanh's range: {scale}")
state["content_gate"] = state["content_gate"].new_full((), math.atanh(scale))
save_file(state, path, metadata={"format": "pt"})
print(f"{path}: gate_raw={math.atanh(scale):.12f} query_source={source}")
PY
}

# launch <gpu> <checkpoint> <out_dir> <label>
launch() {
    local gpu="$1" checkpoint="$2" out="$3" label="$4"
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
            --experiment-label "glmocr_layout_attr_${label}" \
            --learning-rate 1e-5 \
            --decoder-adaptation lora --decoder-lora-rank 8 --decoder-lora-alpha 8 \
            --decoder-lora-dropout 0 --decoder-learning-rate 1e-6 \
            --residual-scale-cap "${residual_cap}" --initial-residual-scale 0 \
            --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
            --gate-freeze-steps 0 --max-grad-norm 1.0 \
            --layout-loss-profile ocr_only --query-assignment hungarian \
            --processor-mode fast --adapter-precision fp32 \
            --box-head-mlp --query-refine-layers 1 --sem-adapter-mlp \
            --max-pixels "${max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --generation-mode "${eval_generation_mode}" \
            --validation-interval 1 --log-steps 16 \
            --eval-checkpoint-dir "${checkpoint}" --eval-only
    ) > "${out}.log" 2>&1
}

run_arms() {
    local arms="${attr_root}/arms" ckpts="${attr_root}/checkpoints"
    mkdir -p "${arms}" "${ckpts}"
    build_arm_checkpoint "${ckpts}/A-gate0"            0                keep
    build_arm_checkpoint "${ckpts}/B-gatecap-parent"   "${residual_cap}" keep
    build_arm_checkpoint "${ckpts}/C-gatecap-randquery" "${residual_cap}" random
    build_arm_checkpoint "${ckpts}/D-gatecap-iou042"   "${residual_cap}" \
        "${layout_runs_root}/${low_iou_run}/seed42/checkpoint-3000/adapter.safetensors"
    build_arm_checkpoint "${ckpts}/E-gatecap-iou065"   "${residual_cap}" \
        "${layout_runs_root}/${mid_iou_run}/seed42/checkpoint-10000/adapter.safetensors"
    echo "{\"event\":\"glmocr_layout_attr_started\",\"run_id\":\"${run_id}\",\"arms\":5,\"max_pixels\":${max_pixels}}"

    local pids=()
    launch 0 "${ckpts}/A-gate0"              "${arms}/A-gate0"              "gate0" & pids+=("$!")
    launch 1 "${ckpts}/B-gatecap-parent"     "${arms}/B-gatecap-parent"     "gatecap_parent" & pids+=("$!")
    launch 2 "${ckpts}/C-gatecap-randquery"  "${arms}/C-gatecap-randquery"  "gatecap_randquery" & pids+=("$!")
    launch 3 "${ckpts}/D-gatecap-iou042"     "${arms}/D-gatecap-iou042"     "gatecap_iou042" & pids+=("$!")
    launch 4 "${ckpts}/E-gatecap-iou065"     "${arms}/E-gatecap-iou065"     "gatecap_iou065" & pids+=("$!")

    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || {
        echo "{\"event\":\"glmocr_layout_attr_failed\",\"error\":\"arm_failed\"}" >&2
        exit 1
    }
}

summarize() {
    setup_environment
    "${python}" - "${attr_root}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
arms = root / "arms"
labelled = {
    "A-gate0": "residual off (no layout information)",
    "B-gatecap-parent": "layout branch of this run (IoU 0.672)",
    "C-gatecap-randquery": "randomised query machinery (no layout information)",
    "D-gatecap-iou042": "giou10x_3000 layout branch (IoU 0.4225)",
    "E-gatecap-iou065": "giou10x_10000 layout branch (IoU 0.648)",
}
payload = {"status": "complete", "arms": {}}
for name, label in labelled.items():
    summary = json.loads((arms / name / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["validation"]
    payload["arms"][name] = {
        "label": label,
        "pages": metrics["pages"],
        "cer": metrics["cer"],
        "substitutions": metrics["substitutions"],
        "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "generation_limit_hits": metrics["generation_limit_hits"],
        "raw_content_gate": metrics.get("raw_content_gate"),
        "effective_residual_scale": metrics.get("effective_residual_scale"),
        "residual_relative_norm": metrics.get("residual_relative_norm"),
        "predictions": str(arms / name / "validation_predictions.jsonl"),
    }
(root / "attribution_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
for name, row in payload["arms"].items():
    print(
        f"{name:22s} CER {row['cer']:.6f}  res_norm={row['residual_relative_norm']}  "
        f"scale={row['effective_residual_scale']}"
    )
PY
}

main() {
    preflight
    if (( foreground == 1 )); then
        mkdir -p "${attr_root}"
        run_arms
        summarize
        return
    fi
    mkdir -p "${attr_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    tmux new-session -d -s "${session}" \
        "export GLMOCR_LAYOUT_ATTR_RUN_ID=$(printf '%q' "${run_id}"); "\
"export GLMOCR_LAYOUT_ATTR_MAX_PIXELS=$(printf '%q' "${max_pixels}"); "\
"bash $(printf '%q' "${script_path}") --foreground > ${attr_root}/pipeline.log 2>&1"
    echo "{\"event\":\"glmocr_layout_attr_armed\",\"session\":\"${session}\",\"root\":\"${attr_root}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --run-id) run_id="$2"; shift 2 ;;
        --max-pixels) max_pixels="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
