#!/usr/bin/env bash
# 注意力探针的接线与成本评测（probe-only，不装路由，MTHv2 sparse24）。
#
# 问题：模型自己的注意力里，有没有一个可读的「我现在读到哪一行」信号？
# 上一步（plans/LAYOUT_ATTENTION_ROUTING.md）证明「用真值字框给注意力加偏置」有效；
# 但那需要真值字框，不可部署。可部署的版本必须自己预测位置，而它只有在注意力本身
# 带位置信号时才可能。本轮只观测、不干预，先把这件事测出来。
#
# 两个臂只差一个开关：
#   noroute      不装探针（原始基准）
#   probe_only   装探针，只记录，不写入任何东西
#
# **判据 1（接线）**：两臂生成的 token 必须逐页完全相同。探针不改序列、不改掩码、
# 不改权重，所以任何差异都是探针装错了 —— 在这种情况下定位结论一律不读，先修。
#
# **判据 2（成本）**：probe_only 减 noroute 的每页耗时、峰值显存。阶段 0 的工程目标
# 是额外推理时延不超过 20%，这是待测目标，不是既有性能。超了就减层/减头/降频。
#
# **判据 3（信号是否存在）**：summary 里 layout_attention_probe.mean_visual_mass。
# 若 m_t 始终很低（模型主要靠 token 间注意力、不怎么看图），说明连信号都没有，
# 阶段 1 不必做 —— 这正是阶段 1 更根本的失败模式。
# 注意 m_t 是生成长度的函数：视觉 key 固定、文本 key 随生成增长，m_t 单调下降。
# 本脚本只报原始值；判「信号是否存在」要按固定窗口或 lse_vis-lse_test 的差值归一，
# 由 tools/analyze_attention_localization.py 离线处理，不要拿裸 m_t 直接下结论。
#
# **阶段 0 与阶段 1 的页数**：阶段 0 取 validation 的前 4 页（--probe-stage 0）；
# 阶段 1 取预先固定的开发子集（--probe-stage 1）。子集按真值长度分层等距抽取，
# 覆盖短页与长页，**不按实验收益挑页**，并把 select/check 两组页号落盘，
# 供离线评测按预登记口径分别报数。不读取 test。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
sparse_root="${GLMOCR_MTHV2_SPARSE_Q32_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24}"

source_run="${GLMOCR_PROBE_SOURCE_RUN:-glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1}"
checkpoint="${GLMOCR_PROBE_CHECKPOINT:-${remote_root}/training_runs/${source_run}/seed42/checkpoint-3000}"
protocol_file="${GLMOCR_PROBE_PROTOCOL:-${remote_root}/protocols/${source_run}.train_validation_no_test.json}"

run_id="${GLMOCR_PROBE_RUN_ID:-glmocr_layout_attention_probe_20260920_v1}"
session_override="${GLMOCR_PROBE_SESSION:-}"
stage="${GLMOCR_PROBE_STAGE:-0}"
seed=42
num_queries=32
max_pixels="${GLMOCR_PROBE_MAX_PIXELS:-4000000}"
max_eval_new_tokens=1536
layout_loss_profile="history_box_equalized_v2"
foreground="${GLMOCR_PROBE_FOREGROUND:-0}"
gpu_slots="${GLMOCR_PROBE_GPUS:-0,1}"

# 层/头可覆盖。默认 0,4,8,12 与全部头 —— 阶段 1 的挑头要在 select 页上做，
# 在 check 页上只用固定选择，不得每页用真值挑头。
probe_layers="${GLMOCR_PROBE_LAYERS:-0,4,8,12}"
probe_heads="${GLMOCR_PROBE_HEADS:-}"
# A previous run's validation_subset_stage.jsonl.  Pages it already measured are left out
# of this draw, so a re-run is not judged on the pages that produced the result it is
# re-examining.  Empty means no exclusion (a first run has nothing to exclude).
subset_exclude="${GLMOCR_PROBE_SUBSET_EXCLUDE:-}"

eval_root="${remote_root}/attention_probe/${run_id}"
python="${env_dir}/bin/python3"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

# 阶段 0 的 4 页与阶段 1 的开发子集页数。两者都是预先固定的数量，不是「跑到够为止」。
case "${stage}" in
    0) page_count=4; select_pages=2 ;;
    1) page_count=28; select_pages=14 ;;
    *) echo "unknown stage: ${stage} (expected 0 or 1)" >&2; exit 64 ;;
esac

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

# The measured pages are written once and reused by both arms, so the two arms score
# the same pages even if the manifest is regenerated between launches.
subset_manifest() {
    setup_environment
    mkdir -p "${eval_root}"
    "${python}" - "${sparse_root}/validation/manifest.char.jsonl" "${eval_root}" \
        "${page_count}" "${select_pages}" "${subset_exclude}" <<'PY'
import json
import sys
from pathlib import Path

source, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
count, select_count = int(sys.argv[3]), int(sys.argv[4])
exclude_path = sys.argv[5]
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) < count:
    raise SystemExit(f"validation manifest has {len(rows)} pages, need {count}")

# Pages a previous run already measured, so a re-run can be judged on pages that did not
# produce the result being re-examined.  Excluding them is what makes the evidence
# independent of that result; it does not make the sample a blind one, and the
# pre-registration says so.
excluded = set()
if exclude_path:
    prior = Path(exclude_path)
    if not prior.is_file():
        raise SystemExit(f"--subset-exclude file not found: {prior}")
    excluded = {
        json.loads(line)["page_id"]
        for line in prior.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

# The subset is fixed by a rule, not by an outcome.  Sorting by reference length and
# taking an even stride across the range covers short pages and long pages without
# anyone looking at how the model did on them first -- the plan's requirement is that
# pages are not chosen by experimental gain.
ordered = [
    index
    for index in sorted(
        range(len(rows)), key=lambda index: (len(rows[index].get("page_text") or ""), index)
    )
    if rows[index]["page_id"] not in excluded
]
if len(ordered) < count:
    raise SystemExit(
        f"{len(ordered)} pages remain after excluding {len(excluded)}, need {count}"
    )
stride = len(ordered) / count
picked = [ordered[int(index * stride)] for index in range(count)]
chosen = [rows[index] for index in sorted(picked)]
overlap = excluded & {row["page_id"] for row in chosen}
if overlap:
    raise SystemExit(f"subset overlaps the excluded set: {sorted(overlap)[:3]}")

(out_dir / f"validation_subset_stage.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen), encoding="utf-8"
)
# select/check split: every other page by the length order, so both halves span the
# same length range and the check half is not "the long pages" by accident.
by_length = sorted(range(len(chosen)), key=lambda index: (len(chosen[index].get("page_text") or ""), index))
select = {by_length[index] for index in range(len(by_length)) if index % 2 == 0}
select = sorted(select)[:select_count]
select_ids = [chosen[index]["page_id"] for index in select]
check_ids = [row["page_id"] for index, row in enumerate(chosen) if index not in set(select)]
(out_dir / "select_pages.txt").write_text("".join(f"{page}\n" for page in select_ids), encoding="utf-8")
(out_dir / "check_pages.txt").write_text("".join(f"{page}\n" for page in check_ids), encoding="utf-8")
print(json.dumps({
    "event": "glmocr_probe_subset_ready",
    "pages": len(chosen),
    "select": len(select_ids),
    "check": len(check_ids),
    "excluded": len(excluded),
    "length_range": [len(chosen[by_length[0]].get("page_text") or ""),
                     len(chosen[by_length[-1]].get("page_text") or "")],
    # Printed so disjointness from the excluded set can be checked from the log rather
    # than taken on trust.
    "page_ids": sorted(row["page_id"] for row in chosen),
}, ensure_ascii=False))
PY
}

preflight() {
    [[ -x "${python}" ]] || { echo "python missing: ${python}" >&2; exit 64; }
    [[ -f "${checkpoint}/adapter.safetensors" ]] || { echo "checkpoint missing: ${checkpoint}" >&2; exit 64; }
    [[ -f "${protocol_file}" ]] || { echo "protocol missing: ${protocol_file}" >&2; exit 64; }
    [[ -f "${sparse_root}/validation/manifest.char.jsonl" ]] \
        || { echo "character manifest missing; run tools/prepare_mthv2_char_manifest.py first" >&2; exit 64; }
    [[ -f "${code_root}/src/layout_ocr/attention_probe.py" ]] \
        || { echo "attention probe module not synced" >&2; exit 64; }
    [[ -f "${code_root}/tools/analyze_attention_localization.py" ]] \
        || { echo "localization analyzer not synced" >&2; exit 64; }
    [[ ! -e "${eval_root}/arms" ]] || { echo "output exists: ${eval_root}/arms" >&2; exit 74; }
}

# launch <arm> <gpu> <probe?>
# train_screen.py refuses to start when --output-dir already exists, so nothing may
# pre-create the arm directory: the log lives as a sibling.
launch() {
    local arm="$1" gpu="$2" with_probe="$3"
    local out="${eval_root}/arms/${arm}"
    local -a probe_flags=()
    if [[ "${with_probe}" == "yes" ]]; then
        probe_flags=(--layout-attention-probe --layout-attention-probe-layers "${probe_layers}")
        [[ -z "${probe_heads}" ]] || probe_flags+=(--layout-attention-probe-heads "${probe_heads}")
    fi
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export GLMOCR_ADAPTER_PROBE="${out}.adapter.jsonl"
        # The probe's report path.  Left set on the noroute arm too, which then writes
        # nothing: an empty file is the evidence that the arm really had no probe.
        export GLMOCR_ATTENTION_PROBE="${out}.attention.jsonl"
        cd "${code_root}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode geometry --model-path "${model_dir}" \
            --train-manifest "${sparse_root}/train/manifest.char.jsonl" \
            --validation-manifest "${eval_root}/validation_subset_stage.jsonl" \
            --protocol-file "${protocol_file}" --output-dir "${out}" \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_attention_probe_${arm}" \
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
            "${probe_flags[@]}"
    ) > "${out}.log" 2>&1
}

run_arms() {
    mkdir -p "${eval_root}/arms"
    IFS=',' read -r -a slots <<< "${gpu_slots}"
    local total="${#slots[@]}"
    echo "{\"event\":\"glmocr_probe_eval_started\",\"run_id\":\"${run_id}\",\"stage\":\"${stage}\",\"pages\":\"${page_count}\",\"layers\":\"${probe_layers}\",\"heads\":\"${probe_heads:-all}\"}"

    local -a pids=() labels=() failed=0
    local index=0 spec with_probe
    for spec in "noroute:no" "probe_only:yes"; do
        IFS=':' read -r arm with_probe <<< "${spec}"
        local gpu="${slots[$(( index % total ))]}"
        launch "${arm}" "${gpu}" "${with_probe}" &
        pids+=("$!"); labels+=("${arm}")
        index=$(( index + 1 ))
        if (( index % total == 0 )); then
            local i
            for i in "${!pids[@]}"; do
                wait "${pids[$i]}" || { echo "{\"event\":\"glmocr_probe_eval_arm_failed\",\"arm\":\"${labels[$i]}\"}" >&2; failed=1; }
            done
            pids=(); labels=()
        fi
    done
    for pid in "${pids[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || { echo "{\"event\":\"glmocr_probe_eval_failed\",\"error\":\"arm_failed\"}" >&2; exit 1; }
}

summarize() {
    setup_environment
    "${python}" - "${eval_root}" "${stage}" "${max_pixels}" "${page_count}" <<'PY'
import json
import sys
from pathlib import Path

root, stage, max_pixels, page_count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
arms_dir = root / "arms"
ITERATIONS = 10000
# Stage 0's engineering target.  A target, not a measurement: it is what the stage is
# allowed to proceed on, and it is fixed before the numbers exist.
LATENCY_BUDGET = 0.20

payload = {"status": "complete", "stage": stage, "pages": page_count, "arms": {}}
missing = []
for arm in ("noroute", "probe_only"):
    path = arms_dir / arm / "summary.json"
    if not path.exists():
        missing.append(arm)
        payload["arms"][arm] = {"status": "missing"}
        continue
    metrics = json.loads(path.read_text(encoding="utf-8"))["validation"]
    payload["arms"][arm] = {
        "pages": metrics["pages"],
        "cer": metrics["cer"],
        "edits": metrics["substitutions"] + metrics["insertions"] + metrics["deletions"],
        "generation_limit_hits": metrics["generation_limit_hits"],
        "generation_mean_new_tokens": metrics.get("generation_mean_new_tokens"),
        "seconds": metrics.get("seconds"),
        "seconds_per_page": metrics.get("seconds_per_page"),
        "cuda_peak_memory_bytes": metrics.get("cuda_peak_memory_bytes"),
        "layout_attention_probe": metrics.get("layout_attention_probe"),
    }

probe_path = arms_dir / "probe_only.attention.jsonl"
noroute_path = arms_dir / "noroute.attention.jsonl"
payload["probe_file_pages"] = (
    len(probe_path.read_text(encoding="utf-8").strip().splitlines()) if probe_path.exists() else 0
)
payload["noroute_probe_file_bytes"] = noroute_path.stat().st_size if noroute_path.exists() else None

print(f"{'arm':11s} {'CER':>10s} {'edits':>7s} {'genlim':>7s} {'s/page':>8s} {'peakMiB':>9s}")
for arm in ("noroute", "probe_only"):
    row = payload["arms"][arm]
    if row.get("status") == "missing":
        print(f"{arm:11s} {'MISSING':>10s}")
        continue
    per_page = row.get("seconds_per_page") or 0.0
    peak = row.get("cuda_peak_memory_bytes")
    print(f"{arm:11s} {row['cer']:10.6f} {row['edits']:7d} {row['generation_limit_hits']:7d} "
          f"{per_page:8.2f} {(peak / 2**20 if peak else 0):9.1f}")

ok = True
print()
print("接线判据（两臂生成的 token 必须逐页完全相同）：")
base, probed = payload["arms"].get("noroute"), payload["arms"].get("probe_only")
if base is None or base.get("status") == "missing" or probed is None or probed.get("status") == "missing":
    print("  缺少臂：没有对照就无法判定")
    ok = False
else:
    def rows(arm):
        path = arms_dir / arm / "validation_predictions.jsonl"
        return {json.loads(line)["page_id"]: json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    a, b = rows("noroute"), rows("probe_only")
    differing = [page for page in a if page in b and a[page]["prediction"] != b[page]["prediction"]]
    if set(a) != set(b):
        print(f"  两臂页数不同：{len(a)} vs {len(b)}")
        ok = False
    elif differing:
        print(f"  {len(differing)} 页生成不同，例如 {differing[:3]}")
        print("  -> 探针写入了前向。定位结论一律不读，先修探针")
        ok = False
    else:
        print(f"  {len(a)} 页逐页逐位相同：探针只观测，未改变生成")
    # 探针必须真的记到了东西：没记录和「没有信号」是两件事。
    probe = probed.get("layout_attention_probe")
    if probe is None:
        print("  probe_only 的 summary 里没有 layout_attention_probe 段：探针未装上")
        ok = False
    elif probe["decoding_steps"] == 0:
        print("  探针已装但一步未记录：接线未接通")
        ok = False
    else:
        print(f"  探针逐步记录（{probe['decoding_steps']} 步 / {probe['pages_with_steps']} 页），"
              f"层 {probe['layers']}，头 {probe['heads'] or '全部'}")
        if probe["grid_missing_steps"]:
            print(f"  警告：{probe['grid_missing_steps']} 步没有 patch 网格，行概率缺失")
            ok = False
        if probe["emitted_missing_steps"]:
            print(f"  警告：{probe['emitted_missing_steps']} 步没有生成文本，离线对齐会缺")
        if probe.get("emitted_join_mismatch"):
            print(f"  **逐步文本拼不回完整解码**（{probe['emitted_join_mismatch']} 页）："
                  f"字符到步的映射不可信，离线定位结果一律不读")
            ok = False
        if probe["transform_failed"]:
            print(f"  **探针变换失败**（这些层的数字不可信）：{probe['transform_failed']}")
            ok = False
    # Absent is correct, not suspicious: nothing in the control arm calls write_probe,
    # so the file is never created.  Only a *non-empty* file means the control was
    # really running a probe.  (An earlier version of this check failed a correct run
    # by treating "does not exist" as "wrote a report".)
    if noroute_path.exists() and noroute_path.stat().st_size > 0:
        print(f"  noroute 臂写出了探针文件（{noroute_path.stat().st_size} 字节）：它不该装探针")
        ok = False

print()
print("成本（阶段 0 工程目标：额外推理时延不超过 20%，这是待测目标而非既有性能）：")
if base and probed and base.get("status") != "missing" and probed.get("status") != "missing":
    a_s, b_s = base.get("seconds_per_page"), probed.get("seconds_per_page")
    if a_s and b_s:
        ratio = (b_s - a_s) / a_s
        payload["latency_overhead"] = ratio
        verdict = "在预算内" if ratio <= LATENCY_BUDGET else "超出预算"
        print(f"  每页 {a_s:.2f}s -> {b_s:.2f}s，{ratio:+.1%}  {verdict}（预算 {LATENCY_BUDGET:.0%}）")
        if ratio > LATENCY_BUDGET:
            print("  -> 先减层/减头或降观测频率，再进入阶段 1")
    a_m, b_m = base.get("cuda_peak_memory_bytes"), probed.get("cuda_peak_memory_bytes")
    if a_m and b_m:
        payload["peak_memory_overhead_bytes"] = b_m - a_m
        print(f"  峰值显存 {a_m / 2**20:.1f}MiB -> {b_m / 2**20:.1f}MiB，"
              f"{(b_m - a_m) / 2**20:+.1f}MiB")

print()
print("信号是否存在（原始 m_t；判读前必须先归一，见下）：")
probe = (payload["arms"].get("probe_only") or {}).get("layout_attention_probe")
if probe and probe.get("mean_visual_mass") is not None:
    payload["mean_visual_mass"] = probe["mean_visual_mass"]
    print(f"  平均视觉总质量 m_t = {probe['mean_visual_mass']:.6f}")
    print("  注意：m_t 是生成长度的函数，不是「看图程度」的纯度量 —— 视觉 key 固定、")
    print("  文本 key 随生成增长，m_t 单调下降。裸 m_t 低不等于没有信号。")
    print("  下一步：tools/analyze_attention_localization.py 用 lse_vis - lse_text 的")
    print("  差值按窗口归一，再看行定位准确率与置信度曲线。")
else:
    print("  没有 m_t：探针未记录到步骤")
    ok = False

print()
print(f"离线定位评测（阶段 1 的主指标，在 check 页上报告，不用 select 页）：")
analyzer = root / "localization.json"
print(f"  ${sys.executable} tools/analyze_attention_localization.py \\")
print(f"      --probe {probe_path} \\")
print(f"      --predictions {arms_dir / 'probe_only' / 'validation_predictions.jsonl'} \\")
print(f"      --manifest {root / 'validation_subset_stage.jsonl'} \\")
print(f"      --select-pages {root / 'select_pages.txt'} --check-pages {root / 'check_pages.txt'} \\")
# Space-separated, because the analyzer takes nargs="+": printing the Python list
# repr produces a command that fails when pasted.
layers_arg = " ".join(str(layer) for layer in (probe["layers"] if probe else [0, 4, 8, 12]))
print(f"      --layers {layers_arg} --output {analyzer}")

payload["status"] = "partial" if missing else "complete"
if missing:
    print(f"\nmissing arms: {missing}")
if not ok:
    print("\n接线未通过：先修，不要读定位结论。")

(root / "attention_probe_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
)
PY
}

main() {
    preflight
    local session="${session_override:-glmocr_attention_probe_$(printf '%s' "${run_id}" | tr '.' '_')}"
    if (( foreground == 1 )); then
        subset_manifest
        run_arms
        summarize
        return
    fi
    mkdir -p "${eval_root}"
    tmux kill-session -t "${session}" 2>/dev/null || true
    # 前台阶段读取的每个覆盖变量都必须在此重新导出：tmux 子进程是全新 shell，
    # 只继承被点名的变量（见路由启动器的同类注释）。
    tmux new-session -d -s "${session}" \
        "export GLMOCR_PROBE_RUN_ID=$(printf '%q' "${run_id}"); "\
"export GLMOCR_PROBE_STAGE=$(printf '%q' "${stage}"); "\
"export GLMOCR_PROBE_GPUS=$(printf '%q' "${gpu_slots}"); "\
"export GLMOCR_PROBE_LAYERS=$(printf '%q' "${probe_layers}"); "\
"export GLMOCR_PROBE_HEADS=$(printf '%q' "${probe_heads}"); "\
"export GLMOCR_PROBE_SUBSET_EXCLUDE=$(printf '%q' "${subset_exclude}"); "\
"export GLMOCR_PROBE_MAX_PIXELS=$(printf '%q' "${max_pixels}"); "\
"export GLMOCR_PROBE_CHECKPOINT=$(printf '%q' "${checkpoint}"); "\
"export GLMOCR_PROBE_PROTOCOL=$(printf '%q' "${protocol_file}"); "\
"bash $(printf '%q' "${script_path}") --foreground > ${eval_root}/pipeline.log 2>&1"
    echo "{\"event\":\"glmocr_probe_eval_armed\",\"session\":\"${session}\",\"root\":\"${eval_root}\",\"stage\":\"${stage}\",\"layers\":\"${probe_layers}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --stage) stage="$2"; shift 2 ;;
        --run-id) run_id="$2"; eval_root="${remote_root}/attention_probe/${run_id}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

main
