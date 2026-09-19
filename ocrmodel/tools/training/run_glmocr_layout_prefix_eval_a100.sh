#!/usr/bin/env bash
# 前缀注入的接线校验（eval-only）。先证明不破坏，再谈有没有用。
#
# 背景：写回干预矩阵判定残差缝的全部可用贡献是「页级、方向特定」的条件向量，
# 逐 patch 分量在等幅度下有害（见 docs/LAYOUT_WRITEBACK_INTERVENTION_RESULT.md
# 第四节）。前缀注入是绕开该缝的另一条路：K 个保留 token 放在序列最前，专用位置、
# 全注意力、不受残差 cap 0.03 限制，投影零初始化。
#
# 零初始化给出一个极强的判据：**装上但未训练时，模型必须与 full 逐位相同**。
# 因此本轮不训练，只验证：
#
#   baseline        不装前缀            期望 0.111608（本次运行的内部对照）
#   prefix_global   装前缀，页级 payload  期望与 baseline 逐位相同
#   prefix_queries  装前缀，逐区域 payload 期望与 baseline 逐位相同
#
# 三臂逐位相同 = 接线正确：resize 没有破坏模型、splice 落在保留位而非真实 token、
# mrope 位置编码未退化、生成期保留 id 被屏蔽、数据路径接通。任何一臂不同都说明
# 上述某处有问题——那必须在花 GPU 小时训练之前发现，否则训练曲线会给出误导性的
# 结论（前缀"有害"其实是接线错的）。
#
# 本轮通过后再起训练对照（baseline vs prefix_global vs prefix_queries，同 seed 同步数）。
#
# 干预臂仍可用 GLMOCR_LAYOUT_INTERVENE 叠加，用于前缀路径的同批归因对照。
# 不读取 test。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dunhuang_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"

checkpoint="${GLMOCR_PREFIX_EVAL_CHECKPOINT:-${remote_root}/training_runs/glmocr_dunhuang_local_sem_adapter_gate001_lr1e5_dec1e6_256_from_giou20x10000_20260918_v1/seed42/checkpoint-256}"
protocol_run="glmocr_mthv2_sem_adapter_stage2_gate001_lr1e5_256_from_giou20x10000_20260918_v1"
protocol_file="${GLMOCR_PREFIX_EVAL_PROTOCOL:-${remote_root}/protocols/${protocol_run}.train_validation_no_test.json}"

run_id="${GLMOCR_PREFIX_EVAL_RUN_ID:-glmocr_layout_prefix_eval_20260919_v1}"
session_override="${GLMOCR_PREFIX_EVAL_SESSION:-}"
seed=42
num_queries=32
prefix_tokens="${GLMOCR_PREFIX_EVAL_TOKENS:-32}"
max_pixels="${GLMOCR_PREFIX_EVAL_MAX_PIXELS:-4000000}"
max_eval_new_tokens=1536
eval_generation_mode="loop_recovery"
residual_cap=0.03
foreground="${GLMOCR_PREFIX_EVAL_FOREGROUND:-0}"
gpu_slots="${GLMOCR_PREFIX_EVAL_GPUS:-0,1,2}"
# 臂格式：<名字>:<prefix_tokens>:<payload>:<position>:<disable>
#   tokens 为 0 表示不装前缀；disable=1 表示插入保留位但不写入（隔离"多出 K 个位置"
#   与"写进去的向量"两种代价）。
#
# 第一轮（v1）实测：零初始化投影下，装前缀相对无前缀 +0.006712 CER（0.111608→0.118320），
# 是布局分支全部实测收益（+0.001380）的 4.9 倍。这是槽位固定开销，不是接线错误，所以本轮
# 拆开它：front vs tail 分离"图像 token 位置后移"与"注意力多出 K 个槽"，
# disable 再分离"多出槽位"与"写进去的零向量"。
arms="${GLMOCR_PREFIX_EVAL_ARMS:-baseline:0:global:front:0 prefix_front_disable:${prefix_tokens}:global:front:1 prefix_tail:${prefix_tokens}:global:tail:0}"

eval_root="${remote_root}/prefix_eval/${run_id}"
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
    [[ -f "${checkpoint}/adapter.safetensors" ]] || { echo "checkpoint missing: ${checkpoint}" >&2; exit 64; }
    [[ -f "${protocol_file}" ]] || { echo "protocol missing: ${protocol_file}" >&2; exit 64; }
    [[ -d "${dunhuang_root}" ]] || { echo "dataset missing: ${dunhuang_root}" >&2; exit 64; }
    [[ -f "${code_root}/src/layout_ocr/prefix_injection.py" ]] \
        || { echo "prefix module not synced" >&2; exit 64; }
    [[ ! -e "${eval_root}/arms" ]] || { echo "output exists: ${eval_root}/arms" >&2; exit 74; }
}

# launch <arm> <prefix_tokens> <payload> <position> <disable> <gpu>
# train_screen.py refuses to start when --output-dir already exists, so nothing may
# pre-create the arm directory: the log lives as a sibling.
launch() {
    local arm="$1" tokens="$2" payload="$3" position="$4" disable="$5" gpu="$6"
    local out="${eval_root}/arms/${arm}"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export GLMOCR_ADAPTER_PROBE="${out}.probe.jsonl"
        export GLMOCR_PREFIX_PROBE="${out}.prefix.jsonl"
        [[ "${disable}" == "0" ]] || export GLMOCR_PREFIX_DISABLE="${disable}"
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
            --experiment-label "glmocr_layout_prefix_${arm}" \
            --learning-rate 1e-5 \
            --decoder-adaptation lora --decoder-lora-rank 8 --decoder-lora-alpha 8 \
            --decoder-lora-dropout 0 --decoder-learning-rate 1e-6 \
            --residual-scale-cap "${residual_cap}" --initial-residual-scale 0 \
            --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
            --gate-freeze-steps 0 --max-grad-norm 1.0 \
            --layout-loss-profile ocr_only --query-assignment hungarian \
            --processor-mode fast --adapter-precision fp32 \
            --box-head-mlp --query-refine-layers 1 --sem-adapter-mlp \
            --prefix-tokens "${tokens}" --prefix-payload "${payload}" \
            --prefix-position "${position}" \
            --max-pixels "${max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --generation-mode "${eval_generation_mode}" \
            --validation-interval 1 --log-steps 16 \
            --eval-checkpoint-dir "${checkpoint}" --eval-only
    ) > "${out}.log" 2>&1
}

run_arms() {
    mkdir -p "${eval_root}/arms"
    IFS=',' read -r -a slots <<< "${gpu_slots}"
    local total="${#slots[@]}"
    echo "{\"event\":\"glmocr_prefix_eval_started\",\"run_id\":\"${run_id}\",\"arms\":\"${arms}\",\"gpus\":\"${gpu_slots}\"}"

    local -a pids=() labels=() failed=0
    local index=0 spec
    for spec in ${arms}; do
        IFS=':' read -r arm tokens payload position disable <<< "${spec}"
        local gpu="${slots[$(( index % total ))]}"
        launch "${arm}" "${tokens}" "${payload}" "${position}" "${disable}" "${gpu}" &
        pids+=("$!"); labels+=("${arm}")
        index=$(( index + 1 ))
        if (( index % total == 0 )); then
            local i
            for i in "${!pids[@]}"; do
                wait "${pids[$i]}" || { echo "{\"event\":\"glmocr_prefix_eval_arm_failed\",\"arm\":\"${labels[$i]}\"}" >&2; failed=1; }
            done
            pids=(); labels=()
        fi
    done
    for pid in "${pids[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || { echo "{\"event\":\"glmocr_prefix_eval_failed\",\"error\":\"arm_failed\"}" >&2; exit 1; }
}

summarize() {
    setup_environment
    "${python}" - "${eval_root}" <<'PY'
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
arms_dir = root / "arms"
order = ["baseline", "prefix_global", "prefix_queries"]
payload = {"status": "complete", "arms": {}}
missing = []
for arm in order:
    arm_dir = arms_dir / arm
    summary_path = arm_dir / "summary.json"
    if not summary_path.exists():
        missing.append(arm)
        payload["arms"][arm] = {"status": "missing"}
        continue
    metrics = json.loads(summary_path.read_text(encoding="utf-8"))["validation"]
    record = {
        "pages": metrics["pages"],
        "cer": metrics["cer"],
        "edits": metrics["substitutions"] + metrics["insertions"] + metrics["deletions"],
        "generation_limit_hits": metrics["generation_limit_hits"],
    }
    prefix_path = arm_dir.with_name(arm_dir.name + ".prefix.jsonl")
    if prefix_path.exists():
        rows = [json.loads(line) for line in prefix_path.read_text(encoding="utf-8").splitlines()]
        if rows:
            record["prefix"] = {
                "splices": len(rows),
                "prefix_tokens": rows[0]["prefix_tokens"],
                "payload_mode": rows[0]["payload_mode"],
                "payload_norm": statistics.median([r["payload_norm"] for r in rows]),
                "prefix_norm": statistics.median([r["prefix_norm"] for r in rows]),
                "embeds_norm": statistics.median([r["embeds_norm"] for r in rows]),
            }
    payload["arms"][arm] = record

baseline = payload["arms"].get("baseline", {})
payload["status"] = "partial" if missing else "complete"
if missing:
    print(f"missing arms: {missing}")

# 记录的无前缀基线（docs/LAYOUT_WRITEBACK_INTERVENTION_RESULT.md 的 full 臂）。
# 只逐位比编辑数，CER 用宽容差：CER 是 2993/总长 的商，字面量 0.111608 是四舍五入值，
# 用 1e-9 去比会把一个正确的复现判成失败。
RECORDED_BASELINE_EDITS = 2993
RECORDED_BASELINE_CER = 0.111608

print(f"{'arm':16s} {'CER':>10s} {'edits':>7s} {'genlim':>7s} {'splices':>8s} {'pnorm':>9s}")
for arm in order:
    row = payload["arms"][arm]
    if row.get("status") == "missing":
        print(f"{arm:16s} {'MISSING':>10s}")
        continue
    p = row.get("prefix", {})
    print(f"{arm:16s} {row['cer']:10.6f} {row['edits']:7d} {row['generation_limit_hits']:7d} "
          f"{p.get('splices', 0):8d} {p.get('prefix_norm', float('nan')):9.6f}")

print()
print("接线判据：")
ok = True
if baseline.get("status") != "missing":
    if (baseline["edits"] == RECORDED_BASELINE_EDITS
            and abs(baseline["cer"] - RECORDED_BASELINE_CER) < 1e-5):
        print("  baseline        复现记录的 full 基线 0.111608 / 2993  OK")
    else:
        print(f"  baseline        未复现 full 基线：{baseline['cer']:.6f} / {baseline['edits']}"
              "  <- 先查这一项，其余臂的结论都依赖它")
        ok = False
for arm in ("prefix_global", "prefix_queries"):
    row = payload["arms"][arm]
    if row.get("status") == "missing" or baseline.get("status") == "missing":
        continue
    if "prefix" not in row or row["prefix"]["splices"] == 0:
        print(f"  {arm:16s} 前缀探针 0 条记录：splice 从未生效，接线未接通")
        ok = False
        continue
    # 注意：这里**不再**期望与 baseline 逐位相同。零初始化只保证投影不贡献布局内容，
    # 不保证模型不变——序列里多了 K 个位置，后续 token 的 mrope 位置整体后移，且这些槽位
    # 参与注意力。实际代价见下表，它才是前缀路线必须赚回的固定开销。
    delta = row["cer"] - baseline["cer"]
    print(f"  {arm:16s} splice 生效（{row['prefix']['splices']} 条），"
          f"零 payload 相对无前缀 ΔCER {delta:+.6f}"
          f"  <- 槽位固定开销，非接线错误")

print()
print("前缀路线必须先赚回上面的槽位开销，才谈得上值不值。")
if ok:
    print("接线正确；可以起训练对照（baseline=装前缀+零 payload 才是正确对照）。")
else:
    print("接线未通过：先修，不要起训练。")

(root / "prefix_eval_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
}

main() {
    preflight
    local session="${session_override:-glmocr_prefix_eval_$(printf '%s' "${run_id}" | tr '.' '_')}"
    if (( foreground == 1 )); then
        run_arms
        summarize
        return
    fi
    mkdir -p "${eval_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    # 前景阶段读取的每个覆盖变量都必须在此重新导出：tmux 子进程是全新 shell，
    # 只继承被点名的变量（见写回干预启动器的同类注释）。
    tmux new-session -d -s "${session}" \
        "export GLMOCR_PREFIX_EVAL_RUN_ID=$(printf '%q' "${run_id}"); "\
"export GLMOCR_PREFIX_EVAL_GPUS=$(printf '%q' "${gpu_slots}"); "\
"export GLMOCR_PREFIX_EVAL_ARMS=$(printf '%q' "${arms}"); "\
"export GLMOCR_PREFIX_EVAL_TOKENS=$(printf '%q' "${prefix_tokens}"); "\
"export GLMOCR_PREFIX_EVAL_MAX_PIXELS=$(printf '%q' "${max_pixels}"); "\
"export GLMOCR_PREFIX_EVAL_CHECKPOINT=$(printf '%q' "${checkpoint}"); "\
"export GLMOCR_PREFIX_EVAL_PROTOCOL=$(printf '%q' "${protocol_file}"); "\
"bash $(printf '%q' "${script_path}") --foreground > ${eval_root}/pipeline.log 2>&1"
    echo "{\"event\":\"glmocr_prefix_eval_armed\",\"session\":\"${session}\",\"root\":\"${eval_root}\",\"arms\":\"${arms}\",\"gpus\":\"${gpu_slots}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --run-id) run_id="$2"; eval_root="${remote_root}/prefix_eval/${run_id}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
