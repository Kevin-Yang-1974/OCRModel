#!/usr/bin/env bash
# 阶段 3：win50_g4 的预登记确认 —— 在 509 页锁定 test 上各跑一次，不做任何阈值/后处理调整。
#
# 这是 docs/LAYOUT_TRACKING_WINSHARE_RESULT.md §3.1 里那条「预登记的单一确认」：
# `win50_g4`（跟踪行 + 下一行份额 0.5 + 门控 4.0，真值行框地图）在 149 页 validation 上把 CER 从
# 0.139722 降到 0.133313，删除 771→552，按页/按卷两种配对 bootstrap 都不含零。它是 5 臂网格里
# 选出来的，区间偏乐观，所以需要一次在**没参与选点的锁定 test** 上的预登记确认。
#
# 预登记口径（本脚本写死，先于 test 数据提交）：
#   * 主比较：win50_g4 vs noroute —— 路由干预的整体效果（装了路由 vs 不装）。
#   * 次比较：win50_g4 vs raw6 —— 在跟踪臂之上，份额 0→0.5 + 门控 6→4 的边际效果。
#     raw6 与 win50_g4 只差这两个变量，是隔离「份额+门控」的正确对照。
#   * 页集：MTHv2 sparse24 锁定 test（509 页），4M、num_queries=32 —— 与 validation 上那条
#     0.133313 的记录同分辨率、同协议，只是换页集。checkpoint 仍是 seed42/checkpoint-3000
#     （该 source run 唯一存在的 checkpoint，换 checkpoint 的确认不可得，故上锁定 test）。
#   * 判定口径：CER 点估计 + 按页/按卷两档配对 bootstrap，全部预登记；不调阈值、不做后处理。
#
# 地图仍是真值行框 ⇒ 即便确认，这也只是**诊断臂的确认**，不改变「可部署形态（预测地图）仍无改善」。
# test 不参与任何选点（AGENTS.md 第 3、10 条）。三个臂在同一轮、同一 5 卡集合里一起跑。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
sparse_root="${GLMOCR_MTHV2_SPARSE_Q32_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24}"

source_run="${GLMOCR_ORACLE_SOURCE_RUN:-glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1}"
checkpoint="${GLMOCR_ORACLE_CHECKPOINT:-${remote_root}/training_runs/${source_run}/seed42/checkpoint-3000}"
protocol_file="${GLMOCR_ORACLE_TEST_PROTOCOL:-${remote_root}/protocols/${source_run}.test_locked.json}"

run_id="${GLMOCR_ORACLE_TEST_RUN_ID:-glmocr_layout_test_confirm_20260921_v1}"
session_override="${GLMOCR_ORACLE_TEST_SESSION:-}"
seed=42
num_queries=32
# 4M，与记录的 0.139722 / 0.133313 同一分辨率，否则那两条记录值不可比。
max_pixels="${GLMOCR_ORACLE_MAX_PIXELS:-4000000}"
max_eval_new_tokens=1536
layout_loss_profile="history_box_equalized_v2"
pointer="synced"
# Stage 1 的冻结探针配置，与 winshare 四轮完全一致。
probe_layers="8"
probe_heads="2,3,8,10,11,12,14,15"
tracking_confidence="6.0"
foreground="${GLMOCR_ORACLE_TEST_FOREGROUND:-0}"
summarize_only=0
gpu_slots="${GLMOCR_ORACLE_TEST_GPUS:-0,1,2}"

# 三个臂，字段见 run_glmocr_layout_oracle_line_eval_a100.sh 的臂格式说明。
#   noroute   不装路由 —— 无路由基准，test 上第一次算（没有可复用的同分辨率 test 基线）。
#   raw6      跟踪行 + 原门控 6.0、无份额 —— win50_g4 只改份额(0→0.5)与门控(6→4)的对照。
#   win50_g4  跟踪行 + 份额 0.5 + 门控 4.0 —— 待确认臂。
arms="${GLMOCR_ORACLE_TEST_ARMS:-noroute:none:none raw6:1:line:tracked:regions:6::0:0:1 win50_g4:1:line:tracked:regions:4::0:0.5:1}"

eval_root="${remote_root}/oracle_line_eval/${run_id}"
python="${env_dir}/bin/python3"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

# test 才是被评测的 split；train/validation manifest 仍要读（指纹与 train_records 依赖）。
manifests=(--train-manifest "${sparse_root}/train/manifest.char.jsonl" \
    --validation-manifest "${sparse_root}/validation/manifest.char.jsonl" \
    --eval-split test \
    --test-manifest "${sparse_root}/test/manifest.char.jsonl")

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
    [[ -f "${sparse_root}/test/manifest.char.jsonl" ]] \
        || { echo "test character manifest missing; run tools/prepare_mthv2_char_manifest.py first" >&2; exit 64; }
    grep -q "test-manifest" "${code_root}/src/layout_ocr/train_screen.py" \
        || { echo "train_screen.py has no test-split support: not synced" >&2; exit 64; }
    [[ ! -e "${eval_root}/arms" ]] || { echo "output exists: ${eval_root}/arms" >&2; exit 74; }
}

# launch <arm> <bias|none> <box_source> <line_source> <line_map> <confidence> <predicted>
#        <corrected> <next_scale> <switch_bar> <gpu>
launch() {
    local arm="$1" bias="$2" box_source="$3" line_source="$4" line_map="$5" confidence="$6"
    local arm_predicted="$7" arm_corrected="$8" arm_next_scale="$9" arm_switch_bar="${10}" gpu="${11}"
    local arm_lines="${arm_predicted:-}"
    local out="${eval_root}/arms/${arm}"
    local -a route_flags=()
    if [[ "${bias}" != "none" ]]; then
        route_flags=(--layout-routing-bias "${bias}"
                     --layout-routing-pointer "${pointer}"
                     --layout-routing-box-source "${box_source}"
                     --layout-routing-line-source "${line_source}")
        route_flags+=(--layout-routing-line-map "${line_map}")
        if [[ "${line_source}" == "tracked" ]]; then
            route_flags+=(--layout-attention-probe
                          --layout-attention-probe-layers "${probe_layers}"
                          --layout-attention-probe-heads "${probe_heads}"
                          --layout-tracking-confidence "${confidence}")
            if [[ "${arm_corrected}" == "1" ]]; then
                route_flags+=(--layout-tracking-corrected-confidence)
            fi
            if [[ -n "${arm_next_scale}" && "${arm_next_scale}" != "0" ]]; then
                route_flags+=(--layout-routing-next-line-scale "${arm_next_scale}")
            fi
            if [[ -n "${arm_switch_bar}" && "${arm_switch_bar}" != "1" ]]; then
                route_flags+=(--layout-tracking-switch-bar "${arm_switch_bar}")
            fi
        fi
    fi
    printf '{"arm":"%s","bias":"%s","box_source":"%s","line_source":"%s","line_map":"%s","confidence":"%s","corrected":%s,"next_line_scale":%s,"switch_bar":%s}\n' \
        "${arm}" "${bias}" "${box_source}" "${line_source}" "${line_map}" "${confidence}" \
        "${arm_corrected:-0}" "${arm_next_scale:-0}" "${arm_switch_bar:-1}" > "${out}.arm.json"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export GLMOCR_ADAPTER_PROBE="${out}.adapter.jsonl"
        export GLMOCR_ROUTING_PROBE="${out}.routing.jsonl"
        export GLMOCR_ATTENTION_PROBE="${out}.attention.jsonl"
        cd "${code_root}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode geometry --model-path "${model_dir}" \
            "${manifests[@]}" \
            --protocol-file "${protocol_file}" --output-dir "${out}" \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_oracle_test_${arm}" \
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
    echo "{\"event\":\"glmocr_oracle_test_confirm_started\",\"run_id\":\"${run_id}\",\"arms\":\"${arms}\",\"max_pixels\":\"${max_pixels}\"}"

    local -a pids=() labels=() failed=0
    local index=0 spec
    for spec in ${arms}; do
        IFS=':' read -r arm bias box_source line_source line_map confidence arm_predicted arm_corrected arm_next_scale arm_switch_bar <<< "${spec}"
        line_source="${line_source:-pointer}"
        line_map="${line_map:-regions}"
        confidence="${confidence:-${tracking_confidence}}"
        arm_corrected="${arm_corrected:-0}"
        local gpu="${slots[$(( index % total ))]}"
        launch "${arm}" "${bias}" "${box_source}" "${line_source}" "${line_map}" "${confidence}" \
            "${arm_predicted}" "${arm_corrected}" "${arm_next_scale}" "${arm_switch_bar}" "${gpu}" &
        pids+=("$!"); labels+=("${arm}")
        index=$(( index + 1 ))
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "{\"event\":\"glmocr_oracle_test_arm_failed\",\"arm\":\"${labels[$i]}\"}" >&2; failed=1; }
    done
    (( failed == 0 )) || { echo "{\"event\":\"glmocr_oracle_test_confirm_failed\",\"error\":\"arm_failed\"}" >&2; exit 1; }
}

summarize() {
    setup_environment
    "${python}" - "${eval_root}" <<'PY'
import json
import sys
from pathlib import Path

from tools.analyze_cer_significance import cer, group_indices, load_predictions, paired_bootstrap

root = Path(sys.argv[1])
ITERATIONS = 10000
SEED = 0

arms_dir = root / "arms"
order = [p.name for p in sorted(arms_dir.iterdir()) if p.is_dir()] if arms_dir.exists() else []

payload = {"status": "complete", "iterations": ITERATIONS, "arms": {}, "comparisons": {}}

for arm in order:
    summary_path = arms_dir / arm / "summary.json"
    if not summary_path.exists():
        payload["arms"][arm] = {"status": "missing"}
        continue
    metrics = json.loads(summary_path.read_text(encoding="utf-8")).get("test")
    if metrics is None:
        payload["arms"][arm] = {"status": "no_test_summary"}
        continue
    record = {
        "pages": metrics["pages"],
        "cer": metrics["cer"],
        "substitutions": metrics["substitutions"],
        "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "edits": metrics["substitutions"] + metrics["insertions"] + metrics["deletions"],
        "generation_limit_hits": metrics["generation_limit_hits"],
    }
    arm_config_path = arms_dir / (arm + ".arm.json")
    record["arm_config"] = (
        json.loads(arm_config_path.read_text(encoding="utf-8"))
        if arm_config_path.exists()
        else None
    )
    probe_path = arms_dir / (arm + ".routing.jsonl")
    if probe_path.exists():
        rows = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            steps = sum(int(row.get("decoding_steps", 0)) for row in rows)
            biased = sum(int(row.get("biased_steps", 0)) for row in rows)
            record["routing"] = {
                "bias": rows[0].get("bias"),
                "box_source": rows[0].get("box_source"),
                "line_source": rows[0].get("line_source"),
                "line_map": rows[0].get("line_map"),
                "biased_fraction": biased / max(1, steps),
                "next_line_scale": rows[0].get("next_line_scale"),
                "mean_boxes_hit": (
                    sum(row["mean_boxes_hit"] for row in rows if row.get("mean_boxes_hit") is not None)
                    / max(1, sum(1 for row in rows if row.get("mean_boxes_hit") is not None))
                ),
            }
    payload["arms"][arm] = record


def arm_note(row):
    config = row.get("arm_config") or {}
    routing = row.get("routing") or {}
    parts = []
    if routing.get("line_source") == "tracked":
        parts.append(f"gate {config.get('confidence', '?')}")
        scale = config.get("next_line_scale")
        if scale not in (None, 0, "0"):
            parts.append(f"next+{scale}")
    return "  ".join(parts)


print(f"{'arm':9s} {'CER':>10s} {'sub':>6s} {'ins':>6s} {'del':>6s} {'genlim':>7s}  note")
for arm in order:
    row = payload["arms"][arm]
    if row.get("status") in ("missing", "no_test_summary"):
        print(f"{arm:9s} {row['status']:>10s}")
        continue
    print(f"{arm:9s} {row['cer']:10.6f} {row['substitutions']:6d} {row['insertions']:6d} "
          f"{row['deletions']:6d} {row['generation_limit_hits']:7d}  {arm_note(row)}")

ok = True
print()
print("接线判据：")
if "noroute" not in payload["arms"] or payload["arms"]["noroute"].get("status") == "missing":
    print("  缺少 noroute 臂：没有对照就无法判定")
    ok = False
for arm in order:
    row = payload["arms"][arm]
    if row.get("status") in ("missing", "no_test_summary"):
        continue
    routing = row.get("routing") or {}
    if routing.get("line_source") != "tracked":
        continue
    # tracked 臂的偏置覆盖率必须高（winshare 四轮是 96.5%）；低覆盖率说明探针/门控没接上。
    fraction = routing.get("biased_fraction", 0.0)
    if fraction < 0.9:
        print(f"  {arm:11s} 偏置覆盖率 {fraction:.1%} < 90% <- 接线有问题，不要读结果")
        ok = False
    else:
        print(f"  {arm:11s} {fraction:.1%} 的解码步拿到偏置，平均命中 "
              f"{routing.get('mean_boxes_hit', 0.0):.1f} 个 token（地图 {routing.get('line_map')}）")


def compare(arm, control):
    """CI for CER(control) - CER(arm)：正区间表示臂更好，与项目文档同向。"""
    pa = arms_dir / arm / "test_predictions.jsonl"
    pb = arms_dir / control / "test_predictions.jsonl"
    if not pa.is_file() or not pb.is_file():
        return None
    rows_arm = load_predictions(pa)
    rows_control = load_predictions(pb)
    page_ids = [row[0] for row in rows_arm]
    out = {
        "cer_arm": cer(rows_arm),
        "cer_control": cer(rows_control),
        "difference_control_minus_arm": cer(rows_control) - cer(rows_arm),
        "relative": (
            (cer(rows_control) - cer(rows_arm)) / cer(rows_control) if cer(rows_control) else None
        ),
        "intervals": {},
    }
    for key in ("page", "volume"):
        low, high, share = paired_bootstrap(
            rows_control, rows_arm, ITERATIONS, SEED, group_indices(page_ids, key)
        )
        out["intervals"][key] = {"low": low, "high": high, "share_at_or_below_zero": share}
    return out


print()
print(f"配对 bootstrap（{ITERATIONS} 次，Δ = CER(对照) - CER(臂)，正区间表示臂更好）：")
# 预登记的两条比较：臂 vs 对照。raw6 是 win50_g4 只改「份额+门控」的对照。
pairs = [("win50_g4", "noroute", "主比较"), ("win50_g4", "raw6", "次比较")]
for label_a, label_b, tag in pairs:
    stats = compare(label_a, label_b)
    if stats is None:
        print(f"  {label_a} vs {label_b}（{tag}）：缺少预测文件，跳过")
        continue
    payload["comparisons"][f"{label_a}_vs_{label_b}"] = stats
    print(f"  {label_a} vs {label_b}（{tag}）:")
    print(f"    CER 对照({label_b}) {stats['cer_control']:.6f}   臂({label_a}) {stats['cer_arm']:.6f}   "
          f"Δ {stats['difference_control_minus_arm']:+.6f}"
          + ("" if stats["relative"] is None else f"  ({stats['relative']:+.1%})"))
    for key in ("page", "volume"):
        interval = stats["intervals"][key]
        low, high, share = interval["low"], interval["high"], interval["share_at_or_below_zero"]
        verdict = "显著" if (low > 0 or high < 0) else "不显著"
        print(f"    {key:>6} 95% CI [{low:+.6f}, {high:+.6f}]  P(<=0)={share:.3f}  {verdict}")

payload["status"] = "partial" if any(
    payload["arms"].get(a, {}).get("status") in ("missing", "no_test_summary") for a in order
) else "complete"
if not ok:
    print("\n接线未通过：先修，不要读结果。")

(root / "test_confirm_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
)
PY
}

main() {
    if (( summarize_only == 1 )); then
        summarize
        return
    fi
    preflight
    local session="${session_override:-glmocr_oracle_test_$(printf '%s' "${run_id}" | tr '.' '_')}"
    if (( foreground == 1 )); then
        run_arms
        summarize
        return
    fi
    mkdir -p "${eval_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    # tmux 起新 shell，只继承显式导出的变量。这里与 oracle_line 版同构：一对（名,值）驱动
    # 导出与校验，缺一个变量 tmux 内就是空、而且校验收不到——之前 pred_static 就是这么崩的。
    local -a tmux_pairs=(
        "GLMOCR_ORACLE_TEST_RUN_ID=${run_id}"
        "GLMOCR_ORACLE_TEST_GPUS=${gpu_slots}"
        "GLMOCR_ORACLE_TEST_ARMS=${arms}"
        "GLMOCR_ORACLE_MAX_PIXELS=${max_pixels}"
        "GLMOCR_ORACLE_CHECKPOINT=${checkpoint}"
        "GLMOCR_ORACLE_TEST_PROTOCOL=${protocol_file}"
        "GLMOCR_ORACLE_SOURCE_RUN=${source_run}"
        "GLMOCR_MTHV2_SPARSE_Q32_ROOT=${sparse_root}"
        "GLMOCR_A100_ROOT=${remote_root}"
        "GLMOCR_A100_CODE_ROOT=${code_root}"
        "GLMOCR_A100_MODEL=${model_dir}"
        "GLMOCR_A100_ENV=${env_dir}"
        "GLMOCR_A100_NVIDIA_ENV=${nvidia_env}"
    )
    local tmux_env="" pair
    for pair in "${tmux_pairs[@]}"; do
        tmux_env+="export ${pair%%=*}=$(printf '%q' "${pair#*=}"); "
    done
    local -a required=(
        GLMOCR_ORACLE_TEST_ARMS GLMOCR_ORACLE_TEST_PROTOCOL
        GLMOCR_MTHV2_SPARSE_Q32_ROOT GLMOCR_A100_CODE_ROOT
    )
    local name
    for name in "${required[@]}"; do
        case "${tmux_env}" in
            *"export ${name}="*) ;;
            *) echo "re-export missing for ${name}: the tmux shell would not see it" >&2; exit 64 ;;
        esac
    done
    tmux new-session -d -s "${session}" \
        "${tmux_env}bash $(printf '%q' "${script_path}") --foreground > ${eval_root}/pipeline.log 2>&1"
    echo "{\"event\":\"glmocr_oracle_test_confirm_armed\",\"session\":\"${session}\",\"root\":\"${eval_root}\",\"arms\":\"${arms}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --summarize-only) summarize_only=1; shift ;;
        --run-id) run_id="$2"; eval_root="${remote_root}/oracle_line_eval/${run_id}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
