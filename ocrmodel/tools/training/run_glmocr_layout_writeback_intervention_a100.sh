#!/usr/bin/env bash
# 写回干预矩阵：判定布局分支的增益来自「布局信息」还是「任意同幅度扰动」。
#
# 背景（见 plans/LAYOUT_FUSION_REDESIGN_REVIEW.md）：
# 受控对比里 W−N 显著（CI [-0.001691,-0.000368]，P=0.999），即布局分支整体有可测增益；
# 但随机查询机制（写回探针 lc_flat=0.008）与训练分支（lc_flat=0.3895）CER 几乎相同。
# lc_flat 显示两臂的 patch 共有分量量级相当，因此「随机 ≈ 训练」正是「增益来自共有偏置
# 而非空间结构」所预言的结果。本矩阵判定这一假说。
#
# 全部是 eval-only，同一个 checkpoint、同一协议、同一残差缩放路径，只改写回张量：
#
#   full      H                      基线
#   zero      0                      地板（等价 gate=0）
#   global    mean_p H 广播          只剩 patch 共有偏置
#   spatial   H - mean_p H           只剩空间结构（幅度不同，按构造）
#   shuffle   patch 轴固定置换       范数/flatness 与 full 逐位相同，只破坏空间对应
#   noise     flatness+范数匹配高斯  随机内容，幅度匹配
#   global_perm  H̄ 的隐藏维固定置换  保留范数与页特异性，只破坏学到的方向
#   spatial_scaled  (H-H̄) 按页缩放到 full 幅度  去掉 spatial 的幅度混淆
#
# 第二轮（v3，含 global_perm）判定页级分量的「内容」是否重要。
# 第三轮（含 spatial_scaled）判定逐 patch 分量是否只是「太轻」：v2 实测 spatial 臂
# inj/vt=0.0072 而 full 是 0.0206（差 2.9 倍），所以「逐 patch 无用」与「逐 patch 太弱」
# 未被分开。见 docs/LAYOUT_WRITEBACK_INTERVENTION_RESULT.md 第四节。
# 可用 GLMOCR_INTERVENE_ARMS 覆盖臂集合，只跑需要的新臂 + 同批参照臂。
#
# shuffle 是最锐利的对照：向量多重集与 full 完全相同，任何边缘统计都不变。
#
# 判定：
#   global ≈ full ≫ spatial 且 shuffle ≈ full → 增益与布局信息无关，方案 §二 作废
#   spatial ≈ full ≫ global                  → 空间信息确实被利用，值得改注入点
#   full ≈ zero                              → 与 W−N 显著性矛盾，先复查 W−N 的归因
#
# full 臂同时开启 seam 传递函数探针（GLMOCR_MERGER_ATTENUATION），测量假说 C。
# 不读取 test。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dunhuang_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"

# 与 W 臂同谱系：stage-2 从 giou20x_10000 起，gate warm-start 0.01（收敛约 0.013）。
checkpoint="${GLMOCR_INTERVENE_CHECKPOINT:-${remote_root}/training_runs/glmocr_dunhuang_local_sem_adapter_gate001_lr1e5_dec1e6_256_from_giou20x10000_20260918_v1/seed42/checkpoint-256}"
protocol_run="glmocr_mthv2_sem_adapter_stage2_gate001_lr1e5_256_from_giou20x10000_20260918_v1"
protocol_file="${GLMOCR_INTERVENE_PROTOCOL:-${remote_root}/protocols/${protocol_run}.train_validation_no_test.json}"

run_id="${GLMOCR_INTERVENE_RUN_ID:-glmocr_layout_writeback_intervention_20260919_v1}"
# The tmux session name is derived from the run id below rather than fixed: a fixed
# name makes a second launch kill the first one's still-running session, because
# main() clears the session before recreating it.
session_override="${GLMOCR_INTERVENE_SESSION:-}"
seed=42
num_queries=32
max_pixels="${GLMOCR_INTERVENE_MAX_PIXELS:-4000000}"
max_eval_new_tokens=1536
eval_generation_mode="loop_recovery"
residual_cap=0.03
foreground="${GLMOCR_INTERVENE_FOREGROUND:-0}"
gpu_slots="${GLMOCR_INTERVENE_GPUS:-0,1,2,3,4}"

intervention_root="${remote_root}/writeback_intervention/${run_id}"
python="${env_dir}/bin/python3"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"
# 臂顺序即调度顺序；full 放最后，它的传递函数探针要多跑一次 merger。
arms="${GLMOCR_INTERVENE_ARMS:-zero global spatial shuffle noise global_perm full}"

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
    [[ -f "${checkpoint}/adapter.safetensors" ]] || { echo "checkpoint missing: ${checkpoint}" >&2; exit 64; }
    [[ -f "${protocol_file}" ]] || { echo "protocol missing: ${protocol_file}" >&2; exit 64; }
    [[ -d "${dunhuang_root}" ]] || { echo "dataset missing: ${dunhuang_root}" >&2; exit 64; }
    # 干预臂只可用于 eval-only：训练中应用会针对被破坏的信号训练。
    [[ -f "${code_root}/src/layout_ocr/writeback_intervention.py" ]] \
        || { echo "intervention module not synced" >&2; exit 64; }
    [[ ! -e "${intervention_root}/arms" ]] || { echo "output exists: ${intervention_root}/arms" >&2; exit 74; }
}

# launch <arm> <gpu>
# train_screen.py refuses to start when --output-dir already exists, so nothing may
# pre-create the arm directory: the log and the two probe streams live as siblings.
launch() {
    local arm="$1" gpu="$2"
    local out="${intervention_root}/arms/${arm}"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export GLMOCR_LAYOUT_INTERVENE="${arm}"
        export GLMOCR_ADAPTER_PROBE="${out}.probe.jsonl"
        # seam 传递函数只在 full 臂上测：其余臂的写回张量不是真实布局上下文。
        if [[ "${arm}" == "full" ]]; then
            export GLMOCR_MERGER_ATTENUATION="${out}.attenuation.jsonl"
        fi
        cd "${code_root}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode layout_ot --model-path "${model_dir}" \
            --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
            --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
            --protocol-file "${protocol_file}" \
            --output-dir "${out}" \
            --per-device-batch-size 1 --gradient-accumulation-steps 1 \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 --min-lr-ratio 0.1 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_layout_intervene_${arm}" \
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
    mkdir -p "${intervention_root}/arms"
    IFS=',' read -r -a slots <<< "${gpu_slots}"
    local total="${#slots[@]}"
    echo "{\"event\":\"glmocr_layout_intervene_started\",\"run_id\":\"${run_id}\",\"arms\":\"${arms}\",\"gpus\":\"${gpu_slots}\",\"max_pixels\":${max_pixels}}"

    local -a pids=() labels=() failed=0
    local index=0 arm pid
    for arm in ${arms}; do
        local gpu="${slots[$(( index % total ))]}"
        launch "${arm}" "${gpu}" &
        pids+=("$!"); labels+=("${arm}")
        index=$(( index + 1 ))
        # 超过 GPU 数就等一批，避免同卡重叠。
        if (( index % total == 0 )); then
            local i
            for i in "${!pids[@]}"; do
                wait "${pids[$i]}" || { echo "{\"event\":\"glmocr_layout_intervene_arm_failed\",\"arm\":\"${labels[$i]}\"}" >&2; failed=1; }
            done
            pids=(); labels=()
        fi
    done
    for pid in "${pids[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || { echo "{\"event\":\"glmocr_layout_intervene_failed\",\"error\":\"arm_failed\"}" >&2; exit 1; }
}

summarize() {
    setup_environment
    "${python}" - "${intervention_root}" <<'PY'
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
arms_dir = root / "arms"
labels = {
    "full": "H (baseline)",
    "zero": "0 (floor, gate=0)",
    "global": "mean_p H broadcast (patch-common only)",
    "spatial": "H - mean_p H (spatial structure only)",
    "shuffle": "patch-axis permutation (content destroyed, statistics identical)",
    "noise": "flatness+norm matched Gaussian (random content)",
}
payload = {"status": "complete", "arms": {}}
missing = []
for arm, label in labels.items():
    arm_dir = arms_dir / arm
    summary_path = arm_dir / "summary.json"
    if not summary_path.exists():
        # Report a partial matrix instead of dying: the arms that did finish are
        # still usable evidence.
        missing.append(arm)
        payload["arms"][arm] = {"label": label, "status": "missing"}
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["validation"]
    probe_path = arm_dir.with_name(arm_dir.name + ".probe.jsonl")
    probe = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines()] if probe_path.exists() else []
    record = {
        "label": label,
        "pages": metrics["pages"],
        "cer": metrics["cer"],
        "substitutions": metrics["substitutions"],
        "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "generation_limit_hits": metrics["generation_limit_hits"],
        "effective_residual_scale": metrics.get("effective_residual_scale"),
        "predictions": str(arm_dir / "validation_predictions.jsonl"),
    }
    if probe:
        record["probe"] = {
            "records": len(probe),
            "lc_flat": statistics.median([r["lc_flat"] for r in probe]),
            # 真实注入幅度 = alpha * lc_norm_over_vt，不是 lc_norm_over_vt 本身。
            "inj_over_vt": statistics.median(
                [r["inj_over_vt"] for r in probe if r.get("inj_over_vt") is not None]
            ),
            "lc_global_share": statistics.median([r["lc_global_share"] for r in probe]),
            "lc_spatial_share": statistics.median([r["lc_spatial_share"] for r in probe]),
            "intervention_seen": sorted({r["intervention"] for r in probe}),
        }
    attenuation_path = arm_dir.with_name(arm_dir.name + ".attenuation.jsonl")
    if attenuation_path.exists():
        rows = [json.loads(line) for line in attenuation_path.read_text(encoding="utf-8").splitlines()]
        record["merger_attenuation"] = {
            "records": len(rows),
            "in_rel": statistics.median([r["in_rel"] for r in rows]),
            "out_rel": statistics.median([r["out_rel"] for r in rows]),
            "attenuation": statistics.median([r["attenuation"] for r in rows]),
        }
    payload["arms"][arm] = record

(root / "intervention_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
if missing:
    payload["status"] = "partial"
    print(f"missing arms: {missing}")
print(f"{'arm':10s} {'CER':>10s} {'edits':>7s} {'scale':>7s} {'lc_flat':>8s} {'inj/vt':>8s} {'gshare':>7s} {'sshare':>7s}")
for arm, row in payload["arms"].items():
    if row.get("status") == "missing":
        print(f"{arm:10s} {'MISSING':>10s}")
        continue
    probe = row.get("probe", {})
    edits = row["substitutions"] + row["insertions"] + row["deletions"]
    print(
        f"{arm:10s} {row['cer']:10.6f} {edits:7d} {row['effective_residual_scale'] or 0:7.4f} "
        f"{probe.get('lc_flat', float('nan')):8.4f} {probe.get('inj_over_vt', float('nan')):8.4f} "
        f"{probe.get('lc_global_share', float('nan')):7.3f} {probe.get('lc_spatial_share', float('nan')):7.3f}"
    )
for arm, row in payload["arms"].items():
    if row.get("status") == "missing":
        continue
    if "merger_attenuation" in row:
        m = row["merger_attenuation"]
        print(f"seam: in_rel={m['in_rel']:.5f} out_rel={m['out_rel']:.5f} attenuation={m['attenuation']:.5f}")
PY
}

main() {
    preflight
    # Derive the session from the run id so two concurrent runs cannot collide.
    local session="${session_override:-glmocr_intervene_$(printf '%s' "${run_id}" | tr '.' '_')}"
    if (( foreground == 1 )); then
        run_arms
        summarize
        return
    fi
    mkdir -p "${intervention_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    # Every override the foreground phase reads must be re-exported here; the tmux
    # child is a fresh shell and inherits only what is named.  Omitting ARMS here
    # silently ran the default arm list instead of the requested one.
    tmux new-session -d -s "${session}" \
        "export GLMOCR_INTERVENE_RUN_ID=$(printf '%q' "${run_id}"); "\
"export GLMOCR_INTERVENE_GPUS=$(printf '%q' "${gpu_slots}"); "\
"export GLMOCR_INTERVENE_ARMS=$(printf '%q' "${arms}"); "\
"export GLMOCR_INTERVENE_MAX_PIXELS=$(printf '%q' "${max_pixels}"); "\
"export GLMOCR_INTERVENE_CHECKPOINT=$(printf '%q' "${checkpoint}"); "\
"export GLMOCR_INTERVENE_PROTOCOL=$(printf '%q' "${protocol_file}"); "\
"bash $(printf '%q' "${script_path}") --foreground > ${intervention_root}/pipeline.log 2>&1"
    echo "{\"event\":\"glmocr_layout_intervene_armed\",\"session\":\"${session}\",\"root\":\"${intervention_root}\",\"arms\":\"${arms}\",\"gpus\":\"${gpu_slots}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --run-id) run_id="$2"; intervention_root="${remote_root}/writeback_intervention/${run_id}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
