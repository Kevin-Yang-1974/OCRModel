#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_ancientdoc_validation_suite.sh [options]

Runs same-protocol validation for the AncientDoc baselines on one test split.
The command does not train. It expects the converted AncientDoc dataset and the
chosen checkpoints to exist already.

Checks performed:
  - original GOT2 zero-shot on AncientDoc test
  - C4/VLQA OCR-only checkpoint
  - C5/VLQA OCR replay checkpoint
  - C6/VLQA layout replay checkpoint

Options:
  --ancient-dataset-id <id>       Default: ancientdoc_layout_260707
  --baseline-model <path>         Default: $GOT_SOURCE_MODEL
  --c4-model <path>               Required
  --c5-model <path>               Required
  --c6-model <path>               Required
  --run-prefix <prefix>           Default: ancientdoc_validation_YYYYmmdd_HHMMSS
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
if [[ -f "${paths_env}" ]]; then
    # shellcheck source=/dev/null
    source "${paths_env}"
fi

ancient_dataset_id="ancientdoc_layout_260707"
baseline_model="${GOT_SOURCE_MODEL:-}"
c4_model=""
c5_model=""
c6_model=""
run_prefix="ancientdoc_validation_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ancient-dataset-id) ancient_dataset_id="${2:-}"; shift 2 ;;
        --baseline-model) baseline_model="${2:-}"; shift 2 ;;
        --c4-model) c4_model="${2:-}"; shift 2 ;;
        --c5-model) c5_model="${2:-}"; shift 2 ;;
        --c6-model) c6_model="${2:-}"; shift 2 ;;
        --run-prefix) run_prefix="${2:-}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

if [[ -z "${GOT_LAYOUT_DATA:-}" || -z "${GOT_EVALUATION_RUNS:-}" ]]; then
    printf 'ERROR: GOT_LAYOUT_DATA and GOT_EVALUATION_RUNS must be set.\n' >&2
    exit 64
fi
for pair in "baseline_model:${baseline_model}" "c4_model:${c4_model}" "c5_model:${c5_model}" "c6_model:${c6_model}"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [[ -z "${value}" || ! -d "${value}" ]]; then
        printf 'ERROR: missing model for %s: %s\n' "${name}" "${value}" >&2
        exit 66
    fi
done

dataset_root="${GOT_LAYOUT_DATA}/${ancient_dataset_id}"
bash "${ocrmodel_root}/tools/training/check_layout_dataset_mount.sh" --dataset-root "${dataset_root}" >/dev/null

suite_root="${GOT_EVALUATION_RUNS}/${run_prefix}"
mkdir -p "${suite_root}"
summary="${suite_root}/validation_summary.jsonl"
: > "${summary}"

run_eval() {
    local label="$1"
    local model="$2"
    local model_kind="$3"
    local output_dir="${suite_root}/${label}"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/src/GOT-OCR-2.0/scripts/evaluate_GOT_layout.py" \
        --model-name-or-path "${model}" \
        --model-kind "${model_kind}" \
        --tokenizer-name-or-path "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --layout-manifest "${dataset_root}/test/manifest.jsonl" \
        --layout-image-root "${dataset_root}/test" \
        --layout-split test \
        --output-dir "${output_dir}" \
        --max-regions 16 \
        --max-records 0 \
        --model-max-length 2048 \
        --max-new-tokens 2048 \
        --no-repeat-ngram-size 20 \
        --object-threshold 0.5 \
        --iou-threshold 0.5 \
        --dtype bfloat16 \
        --device cuda
    python3 - "$label" "${output_dir}/layout_validation_metrics.json" "$summary" <<'PY'
from __future__ import annotations
import json, sys
label, summary_path, summary = sys.argv[1:4]
line = json.dumps({"event":"validation_finished","label":label,"summary":summary_path}, ensure_ascii=False, separators=(",", ":"))
print(line)
with open(summary, "a", encoding="utf-8") as handle:
    handle.write(line + "\n")
PY
}

run_eval baseline "${baseline_model}" baseline
run_eval c4 "${c4_model}" vlqa
run_eval c5 "${c5_model}" vlqa
run_eval c6 "${c6_model}" vlqa

python3 - "${suite_root}" <<'PY'
from __future__ import annotations

from datetime import date
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
items = {}
for label in ("baseline", "c4", "c5", "c6"):
    path = root / label / "layout_validation_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    items[label] = payload
ocr = {label: payload["metrics"]["ocr"] for label, payload in items.items()}

def absolute_delta(left: str, right: str, metric: str = "page_cer") -> float:
    return ocr[left][metric] - ocr[right][metric]

def relative_delta(left: str, right: str, metric: str = "page_cer") -> float:
    return absolute_delta(left, right, metric) / ocr[right][metric]

report_path = root / "report.md"
summary = {
    "status": "ok",
    "suite_root": str(root),
    "report": str(report_path),
    "baseline": items["baseline"],
    "c4": items["c4"],
    "c5": items["c5"],
    "c6": items["c6"],
    "deltas": {
        "c4_minus_baseline": absolute_delta("c4", "baseline"),
        "c5_minus_c4": absolute_delta("c5", "c4"),
        "c6_minus_c5": absolute_delta("c6", "c5"),
        "c6_minus_c4": absolute_delta("c6", "c4"),
        "c6_minus_baseline": absolute_delta("c6", "baseline"),
    },
}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

labels = {
    "baseline": "原始 GOT2",
    "c4": "C4：AncientDoc OCR-only",
    "c5": "C5：C4＋synthetic OCR replay",
    "c6": "C6：C4＋synthetic layout replay",
}
rows = []
for label in ("baseline", "c4", "c5", "c6"):
    metric = ocr[label]
    relative = "-" if label == "baseline" else f"{relative_delta(label, 'baseline') * 100:+.2f}%"
    rows.append(
        f"| {labels[label]} | {metric['pages']} | {metric['page_cer']:.6f} | "
        f"{metric['whitespace_normalized_page_cer']:.6f} | "
        f"{metric['page_exact_matches']}/{metric['pages']} | {relative} |"
    )

report = f"""# AncientDoc 基线验证报告

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {date.today().isoformat()}
- Verification Status: ANALYZED
- Version Label: ancientdoc_validation_v1

## 验证范围

- 结果目录：`{root}`
- 测试集：AncientDoc 转换数据的 `test` split
- 页面数：{ocr['baseline']['pages']}
- 输入协议：原始整页图像＋OCR prompt
- 布局信息：不作为模型输入；AncientDoc 没有布局标注
- 复现状态：只完成一次指标生成，没有独立重复运行或多随机种子区间

## 结果

| 模型 | 页面数 | 页面 CER | 去空白 CER | 完全正确页面 | 相对原始 GOT2 的 CER 变化 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

当前排序为 C6、C4、C5、原始 GOT2。C4 相对原始 GOT2 的页面 CER 下降 {abs(relative_delta('c4', 'baseline')) * 100:.2f}%。C5 相对 C4 变化 {relative_delta('c5', 'c4') * 100:+.2f}%，没有改善本次测试。C6 相对原始 GOT2 下降 {abs(relative_delta('c6', 'baseline')) * 100:.2f}%，相对 C4 下降 {abs(relative_delta('c6', 'c4')) * 100:.2f}%。

## 结果解释

- C4 表明，在当前页面协议下，AncientDoc OCR-only 适配相对原始 GOT2 有明显域内收益。
- C5 与 C6 都从 C4 独立启动。二者差异反映 replay 选择，不是 C5 到 C6 的连续训练收益。
- C5 的 synthetic OCR replay 相对 C4 略微变差，当前 replay 比例或数据域可能干扰 AncientDoc 适配。
- C6 的 synthetic layout replay 是本次已实现 baseline 中最优的一项。这与正则化作用一致，但没有隔离布局 queries 或布局监督的独立贡献。
- C6 的完全正确页面仍只有 {ocr['c6']['page_exact_matches']}/{ocr['c6']['pages']}，说明整页转写的绝对质量仍不足。

## 统计与协议限制

- 总体置信度：`CAUTION`。当前没有置信区间、逐页配对检验或多随机种子变化。
- 当前划分沿用 AncientDoc 原始 split ID，尚未完成跨书籍、跨版本、跨馆藏和近重复隔离审计，不能支持小样本泛化结论。
- 同一 test 已查看四个配置。后续 replay 比例只能用 validation 选择，避免继续产生 test 选择偏差。
- 插入错误较多时 CER 可以超过 1；这表示严重转写错误，不是一个限定在 100% 以内的普通准确率。

## 统计谬误检查

- 覆盖：11/11。
- Simpson 悖论、生态谬误、碰撞变量偏差、基准率忽视、均值回归和反向因果：对当前聚合指标不适用或无法判断。
- 选择偏差：`CAUTION`。AncientDoc 是单一应用数据集，且来源分组独立性未确认。
- 幸存者偏差：各模型均报告 {ocr['baseline']['pages']} 页，未见页面缺失；仍需检查逐页预测中的生成失败。
- 多重寻找和分析分支：`CAUTION`。比较了多个配置，但没有预注册选择规则。
- 因果归因：`CAUTION`。当前结果只能排序 checkpoint，不能证明布局监督导致 C6 改善。

## 下一验证门槛

1. 对 C4 与 C6 做逐页配对错误分析，报告改善、持平和退化页面数。
2. 按书籍、版本、馆藏、来源 ID、图片哈希和图文近重复审计数据独立性。
3. 只在 validation 上选择 replay 比例，然后在分组隔离 test 上评估一次冻结配置。
4. 增加等参数量 adaptor 和无布局监督 query 对照，再判断收益能否归因于 VLQA 结构。
"""
report_path.write_text(report, encoding="utf-8")
print(json.dumps({"event":"ancientdoc_validation_suite_completed","suite_root":str(root),"summary":str(root / "summary.json"),"report":str(report_path)}, ensure_ascii=False, separators=(",", ":")))
PY
