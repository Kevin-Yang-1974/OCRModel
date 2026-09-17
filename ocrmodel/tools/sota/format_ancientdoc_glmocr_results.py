#!/usr/bin/env python3
"""Render the AncientDoc report for the two supplied GLM-OCR variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


MODELS = (
    ("content_only", "GLM-OCR synthetic content-only"),
    ("geometry", "GLM-OCR synthetic geometry"),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: Any, digits: int = 6) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def integer(value: Any) -> str:
    return "N/A" if value is None else f"{int(value):,}"


def metrics_row(metrics: dict[str, Any]) -> list[str]:
    statuses = metrics.get("status_counts") or {}
    pages = int(metrics.get("pages") or 0)
    return [
        str(pages),
        f"`ok={int(statuses.get('ok', 0))}`",
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
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--num-queries", type=int, required=True)
    parser.add_argument("--decoder-lora-rank", type=int, required=True)
    parser.add_argument("--decoder-lora-alpha", type=float, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    args = parser.parse_args()

    if args.num_queries != 32:
        raise ValueError("AncientDoc synthetic-weight report requires 32 queries")
    results: dict[str, dict[str, Any]] = {}
    protocols: dict[str, dict[str, Any]] = {}
    for key, _display in MODELS:
        test_root = args.run_root / key / "test"
        metrics = read_json(test_root / "unified_metrics.json")
        protocol = read_json(test_root / "protocol.json")
        load_status = read_json(test_root / "load_status.json")
        if metrics.get("pages") != 516 or metrics.get("failed_pages") != 0:
            raise ValueError(f"incomplete metrics for {key}")
        if load_status.get("status") != "loaded":
            raise ValueError(f"checkpoint was not loaded for {key}")
        if protocol.get("test_used_for_selection") is not False:
            raise ValueError(f"test selection boundary missing for {key}")
        if (protocol.get("adapter") or {}).get("num_queries") != 32:
            raise ValueError(f"query count mismatch for {key}")
        results[key] = metrics
        protocols[key] = protocol

    pages = int(results[MODELS[0][0]]["pages"])
    model_rows = [
        "| " + " | ".join([display, *metrics_row(results[key])]) + " |"
        for key, display in MODELS
    ]
    domain_label = f"AncientDoc（split5，{pages} 页）"
    micro_rows = [
        f"| {display} | {number(results[key].get('micro_page_cer'))} |"
        for key, display in MODELS
    ]
    stripped_rows = [
        f"| {display} | {number(results[key].get('whitespace_stripped_micro_page_cer'))} |"
        for key, display in MODELS
    ]
    length_rows = [
        f"| {display} | {integer(results[key].get('total_reference_characters'))} | "
        f"{integer(results[key].get('total_prediction_characters'))} | "
        f"{number(results[key].get('mean_prediction_reference_length_ratio'))} | "
        f"{int(results[key].get('failed_pages') or 0)} |"
        for key, display in MODELS
    ]
    evidence_rows = []
    for key, display in MODELS:
        checkpoint = protocols[key].get("model") or {}
        evidence_rows.append(
            f"| {display} | `{checkpoint.get('checkpoint_id', 'N/A')}` | "
            f"synthetic weights (`num_queries=32`, `LoRA rank/alpha={args.decoder_lora_rank}/{args.decoder_lora_alpha:g}`) | "
            f"`{key}/test/unified_metrics.json` |"
        )

    lines = [
        "# A100 GLM-OCR 合成数据权重 zero-shot 结果（AncientDoc split5 test）",
        "",
        "## 运行记录",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- run：`{args.run_id}`",
        "- 数据集：`AncientDoc`",
        f"- 数据源：`{args.source_label}`；图像根目录：`{args.image_root}`",
        f"- split：`test`（历史 AncientDoc `split5`），共 {pages} 页",
        "- 输入：整页图像 `whole_page_image` + `Text Recognition:` prompt",
        f"- 生成：确定性 plain greedy，`do_sample=false`，`max_output_tokens={args.max_output_tokens}`",
        f"- adapter：`num_queries={args.num_queries}`；decoder LoRA `rank={args.decoder_lora_rank}`、`alpha={args.decoder_lora_alpha:g}`、`dropout=0`",
        "- test 未参与模型选择、阈值选择或后处理调参：`test_used_for_selection=false`",
        f"- test manifest SHA-256：`{sha256(args.manifest)}`",
        f"- source label SHA-256：`{sha256(args.source_label)}`",
        f"- 设备：两组均使用 A100 `cuda:{args.gpu_ids}`，两组串行完成；均以 `status=ok` 完成",
        "",
        "本次协议使用 AncientDoc split5 的 516 页 test。两组权重均直接加载，不做 validation selection；结果只用于合成数据训练权重的外部零样本对照。",
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
        "| 模型 | 基础 checkpoint | 合成权重配置 | 本地详细指标 |",
        "|---|---|---|---|",
        *evidence_rows,
        "",
        "每个模型目录下还保留了 `protocol.json`、`load_status.json`、`run_summary.json` 和 `predictions.jsonl`；完整逐项指标以两个 `unified_metrics.json` 为准。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"event": "ancientdoc_glmocr_results_rendered", "output": str(args.output), "pages": pages}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
