#!/usr/bin/env python3
"""Render one GLM-OCR checkpoint's AncientDoc result in the SOTA RESULTS layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any, digits: int = 6) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def integer(value: Any) -> str:
    return "N/A" if value is None else f"{int(value):,}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    test_root = args.run_root / "geometry" / "test"
    metrics = read_json(test_root / "unified_metrics.json")
    protocol = read_json(test_root / "protocol.json")
    load_status = read_json(test_root / "load_status.json")
    if metrics.get("pages") != 516 or metrics.get("failed_pages") != 0:
        raise ValueError("incomplete AncientDoc metrics")
    if load_status.get("status") != "loaded":
        raise ValueError("checkpoint load did not complete")
    if protocol.get("test_used_for_selection") is not False:
        raise ValueError("test selection boundary missing")
    source_label_path = Path(args.source_label)
    source_label_hash = (
        sha256(source_label_path)
        if source_label_path.is_file()
        else str((protocol.get("dataset") or {}).get("source_label_sha256", "N/A"))
    )

    statuses = metrics.get("status_counts") or {}
    pages = int(metrics["pages"])
    display = "GLM-OCR geometry step-1400"
    tick = chr(96)
    row = [
        display,
        str(pages),
        tick + "ok=" + str(int(statuses.get("ok", 0))) + tick,
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
    lines = [
        "# A100 GLM-OCR zero-shot 结果（AncientDoc split5 test）",
        "",
        "## 运行记录",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- run：{tick}{args.run_id}{tick}",
        "- 数据集：" + tick + "AncientDoc" + tick,
        f"- 数据源：{args.source_label}；图像根目录：{args.image_root}",
        "- split：" + tick + "test" + tick + "（历史 AncientDoc " + tick + "split5" + tick + "），共 " + str(pages) + " 页",
        "- 输入：整页图像 " + tick + "whole_page_image" + tick + " + 固定 OCR prompt " + tick + "Text Recognition:" + tick,
        "- 生成：确定性 plain greedy，" + tick + "do_sample=false" + tick + "，" + tick + "max_output_tokens=1536" + tick,
        "- adapter：" + tick + "geometry" + tick + "，" + tick + "num_queries=32" + tick + "；decoder LoRA " + tick + "rank=8" + tick + "、" + tick + "alpha=8" + tick + "、" + tick + "dropout=0" + tick,
        "- test 未参与模型选择、阈值选择或后处理调参：" + tick + "test_used_for_selection=false" + tick,
        f"- test manifest SHA-256：{tick}{sha256(args.manifest)}{tick}",
        f"- source label SHA-256：{tick}{source_label_hash}{tick}",
        "- 设备：五个 A100 分片并行完成；以 " + tick + "status=ok" + tick + " 完成",
        "",
        "本次协议使用 AncientDoc split5 的 516 页 test。step-1400 权重直接加载，不做 validation selection；结果只用于 GLM-OCR geometry checkpoint 的外部 zero-shot 对照。",
        "",
        "## 总体指标",
        "",
        "| 模型 | 页数 | 状态 | micro CER | 去空白 CER | macro CER | mean NED | 平均编辑距离 | exact match | 平均延迟 (s) | P95 延迟 (s) | pages/s | 峰值显存 (MiB) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(row) + " |",
        "",
        "CER 保留实际编辑距离结果，不截断到 1；因此插入较多时可以大于 1。",
        "",
        "## 按数据域指标",
        "",
        "### micro CER",
        "",
        f"| 模型 | AncientDoc（split5，{pages} 页） |",
        "|---|---:|",
        f"| {display} | {number(metrics.get('micro_page_cer'))} |",
        "",
        "### 去空白 micro CER",
        "",
        f"| 模型 | AncientDoc（split5，{pages} 页） |",
        "|---|---:|",
        f"| {display} | {number(metrics.get('whitespace_stripped_micro_page_cer'))} |",
        "",
        "## 输出长度与状态",
        "",
        "| 模型 | 参考文本字符数 | 输出字符数 | 输出/参考长度比 | 失败页 |",
        "|---|---:|---:|---:|---:|",
        f"| {display} | {integer(metrics.get('total_reference_characters'))} | {integer(metrics.get('total_prediction_characters'))} | {number(metrics.get('mean_prediction_reference_length_ratio'))} | {int(metrics.get('failed_pages') or 0)} |",
        "",
        "## 模型与证据",
        "",
        "| 模型 | 基础 checkpoint | 训练 checkpoint | 本地详细指标 |",
        "|---|---|---|---|",
        f"| {display} | {tick}zai-org/GLM-OCR{tick} | {tick}{args.checkpoint_dir}{tick}；{tick}num_queries=32{tick}，{tick}LoRA rank/alpha=8/8{tick} | {tick}geometry/test/unified_metrics.json{tick} |",
        "",
        "保留 " + tick + "protocol.json" + tick + "、" + tick + "load_status.json" + tick + "、" + tick + "run_summary.json" + tick + " 和 " + tick + "predictions.jsonl" + tick + "；完整逐项指标以 " + tick + "geometry/test/unified_metrics.json" + tick + " 为准。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"event": "ancientdoc_glmocr_checkpoint_results_rendered", "output": str(args.output), "pages": pages}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
