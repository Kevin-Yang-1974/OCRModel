#!/usr/bin/env python3
"""Apply the fixed cross-domain D2/D3 rule to the two completed diagnostic32 summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mthv2-summary", type=Path, required=True)
    parser.add_argument("--dunhuang-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite candidate selection: {args.output}")
    summaries = {
        "mthv2": json.loads(args.mthv2_summary.read_text(encoding="utf-8")),
        "dunhuang_local_gazetteer": json.loads(args.dunhuang_summary.read_text(encoding="utf-8")),
    }
    expected = {"mthv2", "dunhuang_local_gazetteer"}
    if {summary.get("domain") for summary in summaries.values()} != expected:
        raise ValueError("the two summaries must be the MTHv2 and Dunhuang/local-gazetteer domains")
    for summary in summaries.values():
        if (summary.get("status") != "complete" or summary.get("evaluation_stage") != "diagnostic32"
                or summary.get("diagnostic_pages") != 32):
            raise ValueError("candidate selection requires both complete 32-page summaries")
        if (summary.get("evaluation_role") != "val_tune"
                or summary.get("val_verify_manifest_read") is not False
                or summary.get("val_verify_used_for_selection") is not False):
            raise ValueError("candidate selection must use val_tune only")
        if summary.get("attention_backend") != "torch_sdpa_math":
            raise ValueError("candidate selection requires a matched torch SDPA math backend")
        if summary.get("test_manifest_read") is not False or summary.get("test_used_for_selection") is not False:
            raise ValueError("candidate selection requires explicit test-isolation fields")
        if summary.get("shared_start", {}).get("decoder_lora_loaded") is not True:
            raise ValueError("Stage D screening must retain the paired frozen decoder LoRA")
    mthv2_summary = summaries["mthv2"]
    dunhuang_summary = summaries["dunhuang_local_gazetteer"]
    if (mthv2_summary.get("source_snapshot_sha256")
            != dunhuang_summary.get("source_snapshot_sha256")
            or mthv2_summary.get("shared_start") != dunhuang_summary.get("shared_start")
            or mthv2_summary.get("diagnostic_config_sha256")
            != dunhuang_summary.get("diagnostic_config_sha256")):
        raise ValueError("cross-domain screening requires one locked source and shared starting point")

    eligible = []
    relative_changes = {}
    for candidate in ("D2", "D3"):
        changes = {}
        valid = True
        for domain, summary in summaries.items():
            baseline = float(summary["arms"]["D1"]["metrics"]["cer"])
            value = float(summary["arms"][candidate]["metrics"]["cer"])
            change = (value - baseline) / max(baseline, 1e-12)
            changes[domain] = change
            valid &= value <= baseline
        relative_changes[candidate] = changes
        if valid:
            eligible.append((sum(changes.values()) / 2.0, 0 if candidate == "D2" else 1, candidate))
    selected = min(eligible)[2] if eligible else None
    result = {
        "status": "complete",
        "evaluation_stage": "diagnostic32_screen_selection",
        "evaluation_role": "val_tune",
        "domains": {
            domain: {
                "summary": str(path.resolve()),
                "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "diagnostic_manifest_sha256": summary["diagnostic_manifest_sha256"],
                "val_tune_manifest_sha256": summary["val_tune_manifest_sha256"],
                "cer_by_arm": {arm: summary["arms"][arm]["metrics"]["cer"]
                               for arm in ("D0", "D1", "D2", "D3")},
            }
            for domain, summary, path in (
                ("mthv2", summaries["mthv2"], args.mthv2_summary),
                ("dunhuang_local_gazetteer", summaries["dunhuang_local_gazetteer"], args.dunhuang_summary),
            )
        },
        "candidate_relative_cer_change_vs_D1": relative_changes,
        "eligible_candidates": [item[2] for item in sorted(eligible)],
        "selected_candidate": selected,
        "full_val_tune_arms": ["D0", "D1"] + ([selected] if selected else []),
        "interpretation": "screening_only_64_pages_no_significance_claim",
        "attention_backend": "torch_sdpa_math",
        "source_snapshot_sha256": mthv2_summary["source_snapshot_sha256"],
        "shared_start": mthv2_summary["shared_start"],
        "diagnostic_config_sha256": mthv2_summary["diagnostic_config_sha256"],
        "val_verify_manifest_sha256": None,
        "val_verify_manifest_read": False,
        "val_verify_used_for_selection": False,
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps({"selected_candidate": selected,
                      "full_val_tune_arms": result["full_val_tune_arms"],
                      "test_manifest_read": False, "test_used_for_selection": False},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
