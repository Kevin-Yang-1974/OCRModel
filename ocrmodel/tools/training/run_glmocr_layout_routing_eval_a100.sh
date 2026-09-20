#!/usr/bin/env bash
# 布局注意力路由的偏置强度扫描（eval-only，MTHv2 sparse24）。
#
# 问题：布局信息能不能帮到识别？前六次测的都是「信息够不够」——往序列里加东西，
# 全部中性或有害（见 plans/LAYOUT_ATTENTION_ROUTING.md 第二节）。本轮测的是
# 「注意力对不对」：不动序列，只在解码步给落在目标字框内的视觉 token 加一个
# 加性偏置，看识别变不变。判据是 paired bootstrap 的 CI，不是点估计。
#
# 四个臂只差一个数：--layout-routing-bias（注意力 logits 上的强度）。
#   bias0  关闭路由，必须复现本 checkpoint 记录的 validation CER 0.486484
#   bias1  偏置 +1
#   bias2  偏置 +2
#   bias4  偏置 +4
#
# 指针用 synced。**不要改成 step**：实测这个 checkpoint 上「第 t 步 → 第 t 个字框」
# 相对真实阅读位置中位数偏 6 字、p90 偏 547 字、最大偏 1271 字（生成触顶的页会
# 多写 4 倍长度），按步索引的指针会把偏置打到好几列之外。synced 用真值 page_text
# 把已生成的字对齐回去，得到模型真正读到的位置——这是上界臂该有的语义。
#
# 字框来自 manifest.char.jsonl（tools/prepare_mthv2_char_manifest.py 生成）。
# 顺序锚定在 page_text 上，属于评测真值，**不是推理输入**。
#
# 不读取 test。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
sparse_root="${GLMOCR_MTHV2_SPARSE_Q32_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24}"

source_run="${GLMOCR_ROUTING_SOURCE_RUN:-glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1}"
checkpoint="${GLMOCR_ROUTING_CHECKPOINT:-${remote_root}/training_runs/${source_run}/seed42/checkpoint-3000}"
protocol_file="${GLMOCR_ROUTING_PROTOCOL:-${remote_root}/protocols/${source_run}.train_validation_no_test.json}"

run_id="${GLMOCR_ROUTING_RUN_ID:-glmocr_layout_routing_eval_20260920_v1}"
session_override="${GLMOCR_ROUTING_SESSION:-}"
seed=42
num_queries=32
max_pixels="${GLMOCR_ROUTING_MAX_PIXELS:-1003520}"
max_eval_new_tokens=1536
layout_loss_profile="history_box_equalized_v2"
pointer="${GLMOCR_ROUTING_POINTER:-synced}"
foreground="${GLMOCR_ROUTING_FOREGROUND:-0}"
gpu_slots="${GLMOCR_ROUTING_GPUS:-0,1,2,3}"

# 臂格式：<名字>:<bias>。字框通道与指针两臂相同，只有强度在动。
arms="${GLMOCR_ROUTING_ARMS:-bias0:0 bias1:1 bias2:2 bias4:4}"

# 记录的无路由基线：该 checkpoint 的 validation 选择记录（selection.json）。
RECORDED_BASELINE_CER=0.4864838911028953

eval_root="${remote_root}/routing_eval/${run_id}"
python="${env_dir}/bin/python3"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

manifests=(--train-manifest "${sparse_root}/train/manifest.char.jsonl" \
    --validation-manifest "${sparse_root}/validation/manifest.char.jsonl")

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
    [[ -f "${sparse_root}/validation/manifest.char.jsonl" ]] \
        || { echo "character manifest missing; run tools/prepare_mthv2_char_manifest.py first" >&2; exit 64; }
    [[ -f "${code_root}/src/layout_ocr/attention_routing.py" ]] \
        || { echo "attention routing module not synced" >&2; exit 64; }
    [[ ! -e "${eval_root}/arms" ]] || { echo "output exists: ${eval_root}/arms" >&2; exit 74; }
}

# launch <arm> <bias> <gpu>
# train_screen.py refuses to start when --output-dir already exists, so nothing may
# pre-create the arm directory: the log lives as a sibling.
launch() {
    local arm="$1" bias="$2" gpu="$3"
    local out="${eval_root}/arms/${arm}"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export GLMOCR_ADAPTER_PROBE="${out}.adapter.jsonl"
        export GLMOCR_ROUTING_PROBE="${out}.routing.jsonl"
        cd "${code_root}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode geometry --model-path "${model_dir}" \
            "${manifests[@]}" \
            --protocol-file "${protocol_file}" --output-dir "${out}" \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_layout_routing_${arm}" \
            --learning-rate 2.5e-5 --decoder-adaptation lora \
            --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
            --decoder-learning-rate 5e-6 --min-lr-ratio 0.1 \
            --initial-residual-scale 0.0 --auxiliary-weight 1.0 \
            --auxiliary-weight-start 1.0 --max-grad-norm 1.0 \
            --max-pixels "${max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --validation-interval 2 --log-steps 16 --adapter-precision fp32 \
            --layout-loss-profile "${layout_loss_profile}" --query-assignment hungarian \
            --processor-mode fast --generation-mode plain --layout-only \
            --layout-routing-bias "${bias}" --layout-routing-pointer "${pointer}" \
            --eval-checkpoint-dir "${checkpoint}" --eval-only
    ) > "${out}.log" 2>&1
}

run_arms() {
    mkdir -p "${eval_root}/arms"
    IFS=',' read -r -a slots <<< "${gpu_slots}"
    local total="${#slots[@]}"
    echo "{\"event\":\"glmocr_routing_eval_started\",\"run_id\":\"${run_id}\",\"arms\":\"${arms}\",\"pointer\":\"${pointer}\",\"gpus\":\"${gpu_slots}\"}"

    local -a pids=() labels=() failed=0
    local index=0 spec
    for spec in ${arms}; do
        IFS=':' read -r arm bias <<< "${spec}"
        local gpu="${slots[$(( index % total ))]}"
        launch "${arm}" "${bias}" "${gpu}" &
        pids+=("$!"); labels+=("${arm}")
        index=$(( index + 1 ))
        if (( index % total == 0 )); then
            local i
            for i in "${!pids[@]}"; do
                wait "${pids[$i]}" || { echo "{\"event\":\"glmocr_routing_eval_arm_failed\",\"arm\":\"${labels[$i]}\"}" >&2; failed=1; }
            done
            pids=(); labels=()
        fi
    done
    for pid in "${pids[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || { echo "{\"event\":\"glmocr_routing_eval_failed\",\"error\":\"arm_failed\"}" >&2; exit 1; }
}

summarize() {
    setup_environment
    "${python}" - "${eval_root}" "${RECORDED_BASELINE_CER}" <<'PY'
import json
import sys
from pathlib import Path

# The judge is the paired bootstrap, never the point estimate.  Imported rather
# than re-implemented so the arm comparison and every other CER comparison in this
# project are the same test.
from tools.analyze_cer_significance import cer, load_predictions, paired_bootstrap

root = Path(sys.argv[1])
recorded_baseline = float(sys.argv[2])
ITERATIONS = 10000

arms_dir = root / "arms"
# Discovered rather than hardcoded: a fixed list reports an arm it does not know
# about as MISSING, which reads as a failed run when the run in fact succeeded.
order = [
    name for name in sorted(
        (p.name for p in arms_dir.iterdir() if p.is_dir()),
        key=lambda name: int(name.replace("bias", "")) if name.startswith("bias") else -1,
    )
] if arms_dir.exists() else []
payload = {"status": "complete", "pointer": None, "iterations": ITERATIONS, "arms": {}}
missing = []
for arm in order:
    summary_path = arms_dir / arm / "summary.json"
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
        "layout_routing": metrics.get("layout_routing"),
    }
    probe_path = arms_dir / (arm + ".routing.jsonl")
    if probe_path.exists():
        rows = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines()]
        if rows:
            record["routing_pages"] = len(rows)
            payload["pointer"] = rows[0]["pointer"]
    payload["arms"][arm] = record

# Paired bootstrap against the zero-bias arm.  The CI is for
# CER(bias0) - CER(arm), so a positive interval means the arm is better.
baseline_predictions = arms_dir / "bias0" / "validation_predictions.jsonl"
payload["comparisons"] = {}
if baseline_predictions.is_file():
    rows_a = load_predictions(baseline_predictions)
    payload["comparisons"]["cer_bias0"] = cer(rows_a)
    for arm in order:
        if arm == "bias0":
            continue
        path = arms_dir / arm / "validation_predictions.jsonl"
        if not path.is_file():
            continue
        rows_b = load_predictions(path)
        low, high, share = paired_bootstrap(rows_a, rows_b, ITERATIONS, 0)
        payload["comparisons"][arm] = {
            "cer": cer(rows_b),
            # Same direction as the interval, so a reader cannot take the point
            # estimate and the CI to mean opposite things.
            "delta_cer_bias0_minus_arm": cer(rows_a) - cer(rows_b),
            "ci_low": low,
            "ci_high": high,
            "p_bias0_not_worse": share,
            "significant": low > 0 or high < 0,
            "favours": "arm" if low > 0 else ("bias0" if high < 0 else "neither"),
        }

payload["status"] = "partial" if missing else "complete"
if missing:
    print(f"missing arms: {missing}")

print(f"{'arm':8s} {'CER':>10s} {'edits':>7s} {'genlim':>7s} {'biased%':>8s} {'steps':>7s}")
for arm in order:
    row = payload["arms"][arm]
    if row.get("status") == "missing":
        print(f"{arm:8s} {'MISSING':>10s}")
        continue
    routing = row.get("layout_routing") or {}
    print(f"{arm:8s} {row['cer']:10.6f} {row['edits']:7d} {row['generation_limit_hits']:7d} "
          f"{100 * routing.get('biased_fraction', 0.0):7.1f}% {routing.get('decoding_steps', 0):7d}")

print()
print("接线判据：")
baseline = payload["arms"].get("bias0")
ok = True
if baseline is None or baseline.get("status") == "missing":
    # A missing control has to be reported rather than indexed: the whole run's
    # verdict is a comparison against it, and an absent arm is not a neutral one.
    print("  bias0           缺少 bias0 臂：没有无偏置对照，无法判定")
    ok = False
else:
    routing = baseline.get("layout_routing")
    if routing is None:
        print("  bias0           无 layout_routing 段：路由未装上，bias0 不是路由基线")
        ok = False
    elif routing["decoding_steps"] == 0:
        print("  bias0           路由已装但一步未记录：接线未接通")
        ok = False
    else:
        print(f"  bias0           路由已装并逐步记录（{routing['decoding_steps']} 步），"
              f"bias=0 故偏置为零 —— 这是正确的无偏置对照")
    delta = baseline["cer"] - recorded_baseline
    if abs(delta) < 1e-5:
        print(f"  bias0           复现记录的 validation CER {recorded_baseline:.6f}  OK")
    else:
        print(f"  bias0           未复现记录基线：{baseline['cer']:.6f} vs {recorded_baseline:.6f}"
              f"（Δ{delta:+.6f}）<- 先查这一项，其余臂的结论都依赖它")
for arm in [name for name in order if name != "bias0"]:
    row = payload["arms"][arm]
    if row.get("status") == "missing":
        continue
    routing = row.get("layout_routing") or {}
    fraction = routing.get("biased_fraction", 0.0)
    if fraction < 0.9:
        print(f"  {arm:8s} 只有 {fraction:.1%} 的解码步拿到偏置：接线未接通，本臂无效")
        ok = False
    else:
        print(f"  {arm:8s} {fraction:.1%} 的解码步拿到偏置，"
              f"平均命中 {routing.get('mean_boxes_hit')} 个视觉 token")

print()
print(f"配对 bootstrap（{ITERATIONS} 次重采样，按页配对）：")
print("  CI 是 CER(bias0) - CER(arm)，正区间表示该臂更好；点估计不作判据。")
comparisons = payload.get("comparisons") or {}
if "cer_bias0" not in comparisons:
    print("  没有 bias0 的预测文件，无法做配对比较")
    ok = False
else:
    for arm in [name for name in order if name != "bias0"]:
        stats = comparisons.get(arm)
        if stats is None:
            continue
        tag = "显著" if stats["significant"] else "不显著"
        print(f"  {arm:8s} ΔCER(bias0-arm) {stats['delta_cer_bias0_minus_arm']:+.6f}  "
              f"CI [{stats['ci_low']:+.6f}, {stats['ci_high']:+.6f}]  {tag}"
              f"  （利好：{stats['favours']}）")

print()
if not ok:
    print("接线未通过：先修，不要读结果。")
elif any(stats["significant"] for stats in comparisons.values() if isinstance(stats, dict)):
    print("至少一条 bias>0 臂的配对 CI 不含零 → 注意力路由有效。")
else:
    print("四条臂两两比较均不显著 → 在这个分辨率与这个 checkpoint 上，注意力路由收口。")
    print("注意：结论仅在「模型有能力利用该信号」的前提下成立，需与分辨率对照一起读。")

(root / "routing_eval_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
)

PY
}

main() {
    preflight
    local session="${session_override:-glmocr_routing_eval_$(printf '%s' "${run_id}" | tr '.' '_')}"
    if (( foreground == 1 )); then
        run_arms
        summarize
        return
    fi
    mkdir -p "${eval_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    # 前台阶段读取的每个覆盖变量都必须在此重新导出：tmux 子进程是全新 shell，
    # 只继承被点名的变量（见写回干预启动器的同类注释）。
    tmux new-session -d -s "${session}" \
        "export GLMOCR_ROUTING_RUN_ID=$(printf '%q' "${run_id}"); "\
"export GLMOCR_ROUTING_GPUS=$(printf '%q' "${gpu_slots}"); "\
"export GLMOCR_ROUTING_ARMS=$(printf '%q' "${arms}"); "\
"export GLMOCR_ROUTING_POINTER=$(printf '%q' "${pointer}"); "\
"export GLMOCR_ROUTING_MAX_PIXELS=$(printf '%q' "${max_pixels}"); "\
"export GLMOCR_ROUTING_CHECKPOINT=$(printf '%q' "${checkpoint}"); "\
"export GLMOCR_ROUTING_PROTOCOL=$(printf '%q' "${protocol_file}"); "\
"bash $(printf '%q' "${script_path}") --foreground > ${eval_root}/pipeline.log 2>&1"
    echo "{\"event\":\"glmocr_routing_eval_armed\",\"session\":\"${session}\",\"root\":\"${eval_root}\",\"arms\":\"${arms}\",\"pointer\":\"${pointer}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --run-id) run_id="$2"; eval_root="${remote_root}/routing_eval/${run_id}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
