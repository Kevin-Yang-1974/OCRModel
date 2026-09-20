#!/usr/bin/env bash
# 阶段 2：oracle_line —— 行框偏置能不能保住字框偏置的收益（eval-only，MTHv2 sparse24，4M）。
#
# 这是方案里「决定整条路线生死」的一步，不是可选诊断。
#
# 已知：字框偏置（`bias2`）把 CER 从 0.169924 压到 0.135977，配对 CI 不含零，**收益全在
# 删除数减半**（1082→680）与插入数减半（1948→942），替换数几乎不动（4048→4042）。见
# docs/LAYOUT_ATTENTION_ROUTING_RESULT.md。
#
# 问题：那个收益来自**精确定位到字**。行框退到「大约在第几行」，字级精度正是很可能被丢掉
# 的那部分。若行框无改善趋势，则「可部署（无真值字级定位）」的路线在原则上走不通，
# 应停止，而不是继续投入注意力跟踪（plans/LAYOUT_ATTENTION_TRACKING.md §5 阶段 2）。
#
# 四个臂：
#   noroute    不装路由 —— 原始基准，应复现记录的 0.169924
#   char2      GT 字框 + synced 指针，B=2 —— 既有机制，应复现记录的 0.135977
#   line025    GT 行框 + 同一指针，B=0.25
#   line050    GT 行框 + 同一指针，B=0.50
#
# 行框覆盖的 token 远多于字框，**字框 B=2 的有效区间不能直接迁移**，所以行臂只试两个小幅
# 候选（方案 §4.3）。B=4 压力实验暂缓（§5：先证明低增益方案有用）。
# 每个臂的 report 都记 `mean_boxes_hit`（平均命中 token 数）与 `biased_fraction`，
# 用于判断「行框覆盖更多 keys」是否把偏置推到了另一个剂量区间。
#
# 地图来源（行框/字框）都是评测真值，**不是推理输入**（AGENTS.md 第 3 条允许 e oracle 诊断）。
# 不读取 test。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
sparse_root="${GLMOCR_MTHV2_SPARSE_Q32_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24}"

source_run="${GLMOCR_ORACLE_SOURCE_RUN:-glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1}"
checkpoint="${GLMOCR_ORACLE_CHECKPOINT:-${remote_root}/training_runs/${source_run}/seed42/checkpoint-3000}"
protocol_file="${GLMOCR_ORACLE_PROTOCOL:-${remote_root}/protocols/${source_run}.train_validation_no_test.json}"

run_id="${GLMOCR_ORACLE_RUN_ID:-glmocr_layout_oracle_line_20260920_v1}"
session_override="${GLMOCR_ORACLE_SESSION:-}"
seed=42
num_queries=32
# 4M，与记录的 0.169924 / 0.135977 同一分辨率，否则那两条记录值不可比。
max_pixels="${GLMOCR_ORACLE_MAX_PIXELS:-4000000}"
max_eval_new_tokens=1536
layout_loss_profile="history_box_equalized_v2"
pointer="${GLMOCR_ORACLE_POINTER:-synced}"
# The detector's boxes, for the pred_static box source.  Written by
# tools/predict_lines_for_routing.py after the detector is trained; required by, and only by,
# the arms whose boxes come from an image rather than the annotation.
predicted_lines="${GLMOCR_ORACLE_PREDICTED_LINES:-}"
foreground="${GLMOCR_ORACLE_FOREGROUND:-0}"
gpu_slots="${GLMOCR_ORACLE_GPUS:-0,1,2,3}"

# 臂格式：<名字>:<bias|none>:<框来源>
arms="${GLMOCR_ORACLE_ARMS:-noroute:none:none char2:2:char line025:0.25:line line050:0.5:line}"

# 记录值（4M，149 页 validation，见 docs/LAYOUT_ATTENTION_ROUTING_RESULT.md §5）。
RECORDED_NOROUTE_CER=0.169924
RECORDED_CHAR2_CER=0.135977
RECORDED_MAX_PIXELS=4000000

eval_root="${remote_root}/oracle_line_eval/${run_id}"
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
    grep -q "BOX_SOURCES" "${code_root}/src/layout_ocr/attention_routing.py" \
        || { echo "routing module has no box source support: not synced" >&2; exit 64; }
    [[ ! -e "${eval_root}/arms" ]] || { echo "output exists: ${eval_root}/arms" >&2; exit 74; }
}

# launch <arm> <bias|none> <box_source> <gpu>
# train_screen.py refuses to start when --output-dir already exists, so nothing may
# pre-create the arm directory: the log lives as a sibling.
launch() {
    local arm="$1" bias="$2" box_source="$3" gpu="$4"
    local out="${eval_root}/arms/${arm}"
    local -a route_flags=()
    if [[ "${bias}" != "none" ]]; then
        route_flags=(--layout-routing-bias "${bias}"
                     --layout-routing-pointer "${pointer}"
                     --layout-routing-box-source "${box_source}")
        if [[ "${box_source}" == "pred_static" ]]; then
            [[ -n "${predicted_lines}" ]] \
                || { echo "pred_static needs GLMOCR_ORACLE_PREDICTED_LINES" >&2; exit 64; }
            route_flags+=(--layout-routing-predicted-lines "${predicted_lines}")
        fi
    fi
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
            --experiment-label "glmocr_oracle_line_${arm}" \
            --learning-rate 2.5e-5 --decoder-adaptation lora \
            --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
            --decoder-learning-rate 5e-6 --min-lr-ratio 0.1 \
            --initial-residual-scale 0.0 --auxiliary-weight 1.0 \
            --auxiliary-weight-start 1.0 --max-grad-norm 1.0 \
            --max-pixels "${max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --validation-interval 2 --log-steps 16 --adapter-precision fp32 \
            --layout-loss-profile "${layout_loss_profile}" --query-assignment hungarian \
            --processor-mode fast --generation-mode plain --layout-only \
            --eval-checkpoint-dir "${checkpoint}" --eval-only \
            "${route_flags[@]}"
    ) > "${out}.log" 2>&1
}

run_arms() {
    mkdir -p "${eval_root}/arms"
    IFS=',' read -r -a slots <<< "${gpu_slots}"
    local total="${#slots[@]}"
    echo "{\"event\":\"glmocr_oracle_line_eval_started\",\"run_id\":\"${run_id}\",\"arms\":\"${arms}\",\"pointer\":\"${pointer}\",\"max_pixels\":\"${max_pixels}\"}"

    local -a pids=() labels=() failed=0
    local index=0 spec
    for spec in ${arms}; do
        IFS=':' read -r arm bias box_source <<< "${spec}"
        local gpu="${slots[$(( index % total ))]}"
        launch "${arm}" "${bias}" "${box_source}" "${gpu}" &
        pids+=("$!"); labels+=("${arm}")
        index=$(( index + 1 ))
        if (( index % total == 0 )); then
            local i
            for i in "${!pids[@]}"; do
                wait "${pids[$i]}" || { echo "{\"event\":\"glmocr_oracle_line_arm_failed\",\"arm\":\"${labels[$i]}\"}" >&2; failed=1; }
            done
            pids=(); labels=()
        fi
    done
    for pid in "${pids[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || { echo "{\"event\":\"glmocr_oracle_line_eval_failed\",\"error\":\"arm_failed\"}" >&2; exit 1; }
}

summarize() {
    setup_environment
    "${python}" - "${eval_root}" "${RECORDED_NOROUTE_CER}" "${RECORDED_CHAR2_CER}" \
        "${max_pixels}" "${RECORDED_MAX_PIXELS}" <<'PY'
import json
import sys
from pathlib import Path

from tools.analyze_cer_significance import cer, load_predictions, paired_bootstrap

root = Path(sys.argv[1])
recorded_noroute, recorded_char2 = float(sys.argv[2]), float(sys.argv[3])
max_pixels, recorded_max_pixels = int(sys.argv[4]), int(sys.argv[5])
ITERATIONS = 10000

arms_dir = root / "arms"
# Discovered, not hardcoded: a fixed list reports an arm it does not know about as missing,
# which reads as a failed run when the run in fact succeeded.
order = [p.name for p in sorted(arms_dir.iterdir()) if p.is_dir()] if arms_dir.exists() else []
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
        "substitutions": metrics["substitutions"],
        "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "edits": metrics["substitutions"] + metrics["insertions"] + metrics["deletions"],
        "generation_limit_hits": metrics["generation_limit_hits"],
        "generation_mean_new_tokens": metrics.get("generation_mean_new_tokens"),
        "layout_routing": metrics.get("layout_routing"),
    }
    probe_path = arms_dir / (arm + ".routing.jsonl")
    if probe_path.exists():
        rows = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            payload["pointer"] = rows[0]["pointer"]
            record["box_source"] = rows[0].get("box_source")
    payload["arms"][arm] = record

print(f"{'arm':9s} {'CER':>10s} {'sub':>6s} {'ins':>6s} {'del':>6s} {'genlim':>7s} "
      f"{'tokens/step':>11s}")
for arm in order:
    row = payload["arms"][arm]
    if row.get("status") == "missing":
        print(f"{arm:9s} {'MISSING':>10s}")
        continue
    routing = row.get("layout_routing") or {}
    print(f"{arm:9s} {row['cer']:10.6f} {row['substitutions']:6d} {row['insertions']:6d} "
          f"{row['deletions']:6d} {row['generation_limit_hits']:7d} "
          f"{routing.get('mean_boxes_hit', 0.0):11.2f}")

ok = True
print()
print("接线判据：")
if "noroute" not in payload["arms"] or payload["arms"]["noroute"].get("status") == "missing":
    print("  缺少 noroute 臂：没有对照就无法判定")
    ok = False
else:
    if max_pixels != recorded_max_pixels:
        print(f"  本分辨率 {max_pixels} 无记录基线可复现（记录值是 {recorded_max_pixels} 下的）；"
              f"noroute 自身即对照")
    else:
        delta = payload["arms"]["noroute"]["cer"] - recorded_noroute
        if abs(delta) < 1e-5:
            print(f"  noroute 复现记录的 CER {recorded_noroute:.6f}  OK")
        else:
            print(f"  noroute 未复现记录基线：{payload['arms']['noroute']['cer']:.6f} vs "
                  f"{recorded_noroute:.6f}（Δ{delta:+.6f}）<- 先查这一项")
    # The char arm is the arm the recorded gain came from; if it does not reproduce here,
    # a null result for the line arms means nothing.
    if "char2" in payload["arms"] and payload["arms"]["char2"].get("status") != "missing":
        delta = payload["arms"]["char2"]["cer"] - recorded_char2
        if max_pixels == recorded_max_pixels and abs(delta) < 1e-5:
            print(f"  char2 复现记录的 CER {recorded_char2:.6f}  OK（既有机制在本页集上成立）")
        elif max_pixels == recorded_max_pixels:
            print(f"  char2 未复现记录值：{payload['arms']['char2']['cer']:.6f} vs "
                  f"{recorded_char2:.6f}（Δ{delta:+.6f}）<- 行臂的零结果不可读")
            ok = False
for arm in order:
    row = payload["arms"][arm]
    if row.get("status") == "missing" or arm == "noroute":
        continue
    routing = row.get("layout_routing") or {}
    fraction = routing.get("biased_fraction", 0.0)
    if fraction < 0.9:
        print(f"  {arm:9s} 只有 {fraction:.1%} 的解码步拿到偏置：接线未接通，本臂无效")
        ok = False
    else:
        print(f"  {arm:9s} {fraction:.1%} 的解码步拿到偏置，"
              f"平均命中 {routing.get('mean_boxes_hit'):.1f} 个视觉 token"
              f"（框来源 {routing.get('box_source')}）")

print()
print(f"配对 bootstrap（{ITERATIONS} 次重采样，按页配对），基准 noroute：")
print("  CI 是 CER(noroute) - CER(arm)，正区间表示该臂更好。")
baseline = arms_dir / "noroute" / "validation_predictions.jsonl"
if not baseline.is_file():
    print("  没有 noroute 的预测文件，无法做配对比较")
    ok = False
else:
    rows_a = load_predictions(baseline)
    payload["comparisons"] = {}
    for arm in order:
        if arm == "noroute":
            continue
        path = arms_dir / arm / "validation_predictions.jsonl"
        if not path.is_file():
            continue
        rows_b = load_predictions(path)
        low, high, share = paired_bootstrap(rows_a, rows_b, ITERATIONS, 0)
        stats = {
            "cer": cer(rows_b),
            "delta_noroute_minus_arm": cer(rows_a) - cer(rows_b),
            "relative": (cer(rows_a) - cer(rows_b)) / cer(rows_a) if cer(rows_a) else None,
            "ci_low": low,
            "ci_high": high,
            "p_noroute_not_worse": share,
            "significant": low > 0 or high < 0,
            "favours": "arm" if low > 0 else ("noroute" if high < 0 else "neither"),
        }
        payload["comparisons"][arm] = stats
        tag = "显著" if stats["significant"] else "不显著"
        rel = "" if stats["relative"] is None else f"  ({stats['relative']:+.1%})"
        print(f"  {arm:9s} CER {stats['cer']:.6f}  ΔCER(noroute-arm) "
              f"{stats['delta_noroute_minus_arm']:+.6f}{rel}  "
              f"CI [{low:+.6f}, {high:+.6f}]  {tag}")

print()
print("阶段 2 的判定（plan §5：先确认 oracle_line 的 CER、插入/删除有改善趋势且无触顶恶化）：")
line_arms = [arm for arm in order if arm.startswith("line")]
if not line_arms:
    print("  没有行框臂")
    ok = False
else:
    improved = [
        arm for arm in line_arms
        if payload.get("comparisons", {}).get(arm, {}).get("delta_noroute_minus_arm", 0.0) > 0
    ]
    worsened_limits = [
        arm for arm in line_arms
        if payload["arms"][arm].get("generation_limit_hits", 0)
        > payload["arms"].get("noroute", {}).get("generation_limit_hits", 0)
    ]
    print(f"  CER 有改善的臂：{improved or '无'}")
    print(f"  触顶恶化的臂：{worsened_limits or '无'}")
    if not improved:
        print("  -> 行框偏置无改善趋势：字级精度正是收益所在，"
              "「可部署（无真值字级定位）」的路线在原则上走不通，应停止")
    elif worsened_limits:
        print("  -> 有改善但有臂触顶恶化：剂量区间需下调，先减 Bmax 再判")
    else:
        # Says nothing about the predicted map: this block judges the *oracle* arms, and the
        # deployable form is a separate question judged separately below.  An earlier version
        # ended this line with "worth continuing into the predicted map", which read as an
        # endorsement of the exact arm that then came out null.
        print("  -> oracle 行框偏置有改善趋势且无触顶恶化：真值行框这条路成立。"
              "注意这只说明 oracle 有效，不说明预测地图有效——见下面的静态地图判定。")
    # The recorded gain was concentrated in deletions; a line bias that only moves
    # substitutions has not reproduced the mechanism.
    noroute_row = payload["arms"].get("noroute", {})
    for arm in line_arms:
        row = payload["arms"][arm]
        if row.get("status") == "missing":
            continue
        print(f"  {arm:9s} 删除 {noroute_row.get('deletions')} -> {row['deletions']}，"
              f"插入 {noroute_row.get('insertions')} -> {row['insertions']}，"
              f"替换 {noroute_row.get('substitutions')} -> {row['substitutions']}")

# ---------------------------------------------------------------------------------------
# The predicted-map arms answer a different question from the oracle ones above, and the
# first version of this script did not judge them at all: its arm filter was `line*`, so a
# static arm could collapse on the token limit and go unmentioned while the verdict line
# talked about the oracle.
#
# What the static arms test is whether the *deployable* form of the bias exists. An oracle
# line box says which line to look at; a predicted map biasing every line says only "look at
# text". The dose is reported alongside, because the two are not comparable otherwise and the
# plan asks for the static and dynamic arms to be matched on total bias weight -- without
# that, a static arm that looks better may only be looking at more of the page.
# ---------------------------------------------------------------------------------------
def box_source_of(arm):
    row = payload["arms"].get(arm, {})
    return row.get("box_source") or (row.get("layout_routing") or {}).get("box_source")


static_arms = [arm for arm in order if box_source_of(arm) == "pred_static" or arm.startswith("static")]
if static_arms:
    print()
    print("静态预测地图的判定（只从图像出框、无指针、无标注）：")
    noroute_row = payload["arms"].get("noroute", {})
    noroute_limits = noroute_row.get("generation_limit_hits", 0)
    improved_static = []
    for arm in static_arms:
        row = payload["arms"][arm]
        if row.get("status") == "missing":
            print(f"  {arm:13s} MISSING")
            continue
        routing = row.get("layout_routing") or {}
        bias = routing.get("bias")
        tokens = routing.get("mean_boxes_hit")
        # The dose that has to be compared, not the bias value: the same B over a page-wide
        # union is a different intervention from the same B over one line.
        mass = (bias * tokens) if isinstance(bias, (int, float)) and isinstance(tokens, (int, float)) else None
        stats = payload.get("comparisons", {}).get(arm, {})
        delta = stats.get("delta_noroute_minus_arm")
        if delta is not None and delta > 0 and stats.get("significant"):
            improved_static.append(arm)
        limits = row.get("generation_limit_hits", 0)
        # Built outside the f-string: a backslash inside an f-string expression is a syntax
        # error on the 3.11 that runs this, and nesting one is how that got here.
        delta_text = "n/a" if delta is None else f"{delta:+.6f}"
        ci_text = (
            "n/a"
            if delta is None
            else "[{:+.6f}, {:+.6f}]".format(stats["ci_low"], stats["ci_high"])
        )
        mass_text = "n/a" if mass is None else f"{mass:.1f}"
        tokens_text = 0.0 if tokens is None else tokens
        print(f"  {arm:13s} CER {row['cer']:.6f}  ΔCER {delta_text}  CI {ci_text}  "
              f"tokens/step {tokens_text:.1f}  total bias/step {mass_text}  触顶 {limits}"
              + ("  <- 触顶恶化" if limits > noroute_limits else ""))
    if not improved_static:
        print("  -> 没有一个静态臂显著优于 noroute：**收益不来自「多看文字」，而来自「看对那一行」**。")
        print("     静态地图这条路走不通；可部署形态必须逐步知道读的是哪一行，")
        print("     即预测框 + 行状态（predmap_track），而它的前置是 gtmap_track。")
    else:
        print(f"  -> 静态预测地图有改善：{improved_static}")

payload["status"] = "partial" if missing else "complete"
if missing:
    print(f"\nmissing arms: {missing}")
if not ok:
    print("\n接线未通过：先修，不要读结果。")

(root / "oracle_line_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
)
PY
}

main() {
    preflight
    local session="${session_override:-glmocr_oracle_line_$(printf '%s' "${run_id}" | tr '.' '_')}"
    if (( foreground == 1 )); then
        run_arms
        summarize
        return
    fi
    mkdir -p "${eval_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    # Every override the foreground stage reads has to be re-exported here.  tmux starts a fresh
    # shell and inherits only what is named, so a variable that is set in this shell but missing
    # from this list is silently absent inside -- which is how a pred_static run came to abort on
    # its own missing-file guard while the arms that did not need the file went ahead.  The list
    # below is checked against the sources that read it rather than trusted.
    local tmux_env
    tmux_env="export GLMOCR_ORACLE_RUN_ID=$(printf '%q' "${run_id}"); "\
"export GLMOCR_ORACLE_GPUS=$(printf '%q' "${gpu_slots}"); "\
"export GLMOCR_ORACLE_ARMS=$(printf '%q' "${arms}"); "\
"export GLMOCR_ORACLE_POINTER=$(printf '%q' "${pointer}"); "\
"export GLMOCR_ORACLE_MAX_PIXELS=$(printf '%q' "${max_pixels}"); "\
"export GLMOCR_ORACLE_CHECKPOINT=$(printf '%q' "${checkpoint}"); "\
"export GLMOCR_ORACLE_PROTOCOL=$(printf '%q' "${protocol_file}"); "\
"export GLMOCR_ORACLE_PREDICTED_LINES=$(printf '%q' "${predicted_lines}"); "
    local required name
    for name in GLMOCR_ORACLE_PREDICTED_LINES GLMOCR_ORACLE_ARMS GLMOCR_ORACLE_POINTER \
                GLMOCR_ORACLE_MAX_PIXELS GLMOCR_ORACLE_CHECKPOINT GLMOCR_ORACLE_PROTOCOL; do
        case "${tmux_env}" in
            *"export ${name}="*) ;;
            *) echo "re-export missing for ${name}: the tmux shell would not see it" >&2; exit 64 ;;
        esac
    done
    tmux new-session -d -s "${session}" \
        "${tmux_env}bash $(printf '%q' "${script_path}") --foreground > ${eval_root}/pipeline.log 2>&1"
    echo "{\"event\":\"glmocr_oracle_line_eval_armed\",\"session\":\"${session}\",\"root\":\"${eval_root}\",\"arms\":\"${arms}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --run-id) run_id="$2"; eval_root="${remote_root}/oracle_line_eval/${run_id}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
