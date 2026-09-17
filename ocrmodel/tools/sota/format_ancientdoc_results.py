#!/usr/bin/env python3
"""Render an AncientDoc SOTA result summary using the existing RESULTS.md layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


MODELS = (
    ("paddleocr_vl_1_6", "PaddleOCR-VL-1.6"),
    ("mineru2_5_pro", "MinerU2.5-Pro-2605-1.2B"),
    ("opendoc_0_1b", "OpenDoc-0.1B"),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def integer(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{int(value):,}"


def metrics_row(metrics: dict[str, Any]) -> list[str]:
    statuses = metrics.get("status_counts") or {}
    pages = int(metrics.get("pages") or 0)
    status = f"ok={int(statuses.get('ok', 0))}"
    return [
        str(pages),
        f"`{status}`",
        number(metrics.get("micro_page_cer")),
        number(metrics.get("whitespace_stripped_micro_page_cer")),
        number(metrics.get("macro_page_cer")),
        number(metrics.get("mean_normalized_edit_distance")),
        number(metrics.get("mean_edit_distance")),
        f"{int(metrics.get('exact_matches') or 0)}/{pages}",
        number(metrics.get("mean_latency_seconds"), 4),
        number(metrics.get("p95_latency_seconds"), 4),
        number(metrics.get("pages_per_second"), 6),
        number(metrics.get("peak_memory_mib"), 2),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-label", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paddle-gpu", required=True)
    parser.add_argument("--mineru-gpu", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=1536)
    args = parser.parse_args()

    results: dict[str, dict[str, Any]] = {}
    protocols: dict[str, dict[str, Any]] = {}
    load_statuses: dict[str, dict[str, Any]] = {}
    for key, _ in MODELS:
        model_root = args.run_root / key / "test"
        metrics_path = model_root / "unified_metrics.json"
        protocol_path = model_root / "protocol.json"
        load_path = model_root / "load_status.json"
        metrics = read_json(metrics_path)
        protocol = read_json(protocol_path)
        load_status = read_json(load_path)
        if int(metrics.get("pages") or 0) != 516:
            raise ValueError(f"{key} metrics do not contain all 516 AncientDoc pages")
        if (load_status.get("status"), metrics.get("failed_pages")) != ("loaded", 0):
            raise ValueError(f"{key} is not a complete loaded result: {load_status.get('status')}")
        results[key] = metrics
        protocols[key] = protocol
        load_statuses[key] = load_status

    pages = int(results[MODELS[0][0]]["pages"])
    manifest_hash = sha256(args.manifest)
    source_label_hash = sha256(args.source_label)
    model_rows = []
    for key, display in MODELS:
        row = metrics_row(results[key])
        model_rows.append("| " + " | ".join([display, *row]) + " |")

    domain_label = f"AncientDoc（split5，{pages} 页）"
    micro_rows = []
    stripped_rows = []
    length_rows = []
    evidence_rows = []
    for key, display in MODELS:
        metrics = results[key]
        micro_rows.append(f"| {display} | {number(metrics.get('micro_page_cer'))} |")
        stripped_rows.append(
            f"| {display} | {number(metrics.get('whitespace_stripped_micro_page_cer'))} |"
        )
        length_rows.append(
            f"| {display} | {integer(metrics.get('total_reference_characters'))} | "
            f"{integer(metrics.get('total_prediction_characters'))} | "
            f"{number(metrics.get('mean_prediction_reference_length_ratio'))} | "
            f"{int(metrics.get('failed_pages') or 0)} |"
        )
        model_info = protocols[key].get("model") or {}
        evidence_rows.append(
            f"| {display} | `{model_info.get('checkpoint_id', 'N/A')}` | "
            f"`{model_info.get('revision', 'N/A')}` | `{key}/test/unified_metrics.json` |"
        )

    lines = [
        "# A100 外部 SOTA zero-shot 结果（AncientDoc split5 test）",
        "",
        "## 运行记录",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- run：`{args.run_id}`",
        "- 数据集：`AncientDoc`",
        f"- 数据源：`{args.source_label}`；图像根目录：`{args.image_root}`",
        f"- split：`test`（历史 AncientDoc `split5`），共 {pages} 页",
        "- 输入：整页图像 `whole_page_image` + 固定 OCR prompt",
        f"- 生成：确定性生成，`do_sample=false`，`max_output_tokens={args.max_output_tokens}`",
        "- test 未参与模型选择、阈值选择或后处理调参：`test_used_for_selection=false`",
        f"- test manifest SHA-256：`{manifest_hash}`",
        f"- source label SHA-256：`{source_label_hash}`",
        f"- 设备：PaddleOCR-VL `cuda:{args.paddle_gpu}`，MinerU `cuda:{args.mineru_gpu}`，OpenDoc `CPU`；三个模型均以 `status=ok` 完成",
        "",
        "本次协议使用 AncientDoc split5 的 516 页 test。zero-shot 官方 checkpoint 没有 validation selection；结果只用于本数据集的外部方案对照，不与敦煌＋地方志 Q32 或 MTHv2 数值直接比较。",
        "",
        "## 总体指标",
        "",
        "| 模型 | 页数 | 状态 | micro CER | 去空白 CER | macro CER | mean NED | 平均编辑距离 | exact match | 平均延迟 (s) | P95 延迟 (s) | pages/s | 峰值显存 (MiB) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *model_rows,
        "",
        "CER 保留实际编辑距离结果，不截断到 1；因此插入较多时可以大于 1。",
        "",
        "## 按数据域指标",
        "",
        "### micro CER",
        "",
        f"| 模型 | {domain_label} |",
        "|---|---:|",
        *micro_rows,
        "",
        "### 去空白 micro CER",
        "",
        f"| 模型 | {domain_label} |",
        "|---|---:|",
        *stripped_rows,
        "",
        "## 输出长度与状态",
        "",
        "| 模型 | 参考文本字符数 | 输出字符数 | 输出/参考长度比 | 失败页 |",
        "|---|---:|---:|---:|---:|",
        *length_rows,
        "",
        "## 模型与证据",
        "",
        "| 模型 | 官方 checkpoint | revision | 本地详细指标 |",
        "|---|---|---|---|",
        *evidence_rows,
        "",
        "每个模型目录下还保留了 `protocol.json`、`load_status.json`、`run_summary.json` 和 `predictions.jsonl`。完整逐项指标（包括参考文本长度、延迟、显存和输出长度）以三个 `unified_metrics.json` 为准。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"event": "ancientdoc_results_rendered", "output": str(args.output), "pages": pages}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

