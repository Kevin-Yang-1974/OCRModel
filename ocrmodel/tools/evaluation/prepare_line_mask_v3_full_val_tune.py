#!/usr/bin/env python3
"""Lock full val_tune evaluation arms after the two-domain 32-page screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from layout_ocr.data import load_records
from layout_ocr.line_mask_v3_diagnostics import sha256_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_line_mask_v3_diagnostic_manifests import line_evidence

EXPECTED_PAGES = {"mthv2": 240, "dunhuang_local_gazetteer": 80}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=tuple(EXPECTED_PAGES), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-tune-manifest", type=Path, required=True)
    parser.add_argument("--screen-protocol", type=Path, required=True)
    parser.add_argument("--candidate-selection", type=Path, required=True)
    parser.add_argument("--base-model-weights-fingerprint", type=Path, required=True)
    parser.add_argument("--decoder-lora-fingerprint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace full val_tune protocol: {args.output_dir}")
    screen = json.loads(args.screen_protocol.read_text(encoding="utf-8"))
    selection = json.loads(args.candidate_selection.read_text(encoding="utf-8"))
    fingerprint = json.loads(args.base_model_weights_fingerprint.read_text(encoding="utf-8"))
    lora_fingerprint = json.loads(args.decoder_lora_fingerprint.read_text(encoding="utf-8"))
    if (screen.get("status") != "locked_before_candidate_inference"
            or screen.get("evaluation_stage") != "diagnostic32"
            or screen.get("evaluation_role") != "val_tune"
            or screen.get("dataset") != args.domain
            or screen.get("test_manifest_read") is not False
            or screen.get("test_used_for_selection") is not False
            or screen.get("val_verify_manifest_read") is not False
            or screen.get("val_verify_used_for_selection") is not False):
        raise ValueError("full val_tune requires the matching isolated 32-page screen protocol")
    if (selection.get("status") != "complete"
            or selection.get("evaluation_stage") != "diagnostic32_screen_selection"
            or selection.get("evaluation_role") != "val_tune"
            or selection.get("test_manifest_read") is not False
            or selection.get("test_used_for_selection") is not False
            or selection.get("val_verify_manifest_read") is not False
            or selection.get("val_verify_used_for_selection") is not False):
        raise ValueError("candidate-selection record is incomplete or violates data isolation")
    if (selection.get("source_snapshot_sha256") != screen.get("source_snapshot_sha256")
            or selection.get("shared_start") != screen.get("shared_start")
            or selection.get("diagnostic_config_sha256") != screen.get("diagnostic_config_sha256")):
        raise ValueError("candidate selection does not use the matching diagnostic source and weights")

    records = load_records(args.val_tune_manifest)
    if len(records) != EXPECTED_PAGES[args.domain]:
        raise ValueError(f"{args.domain} full val_tune requires {EXPECTED_PAGES[args.domain]} pages")
    if any(record.get("split", record.get("official_split")) != "validation" for record in records):
        raise ValueError("full val_tune manifest contains a record outside the validation split")
    val_tune_sha256 = sha256_file(args.val_tune_manifest)
    if val_tune_sha256 != screen.get("val_tune_manifest_sha256"):
        raise ValueError("full val_tune source manifest differs from the screened protocol")
    if sha256_file(args.train_manifest) != screen.get("train_manifest_sha256"):
        raise ValueError("full val_tune source train manifest differs from the screened protocol")
    if (fingerprint.get("model_weights_sha256")
            != screen.get("shared_start", {}).get("base_model_weights_sha256")
            or fingerprint.get("model_revision")
            != screen.get("shared_start", {}).get("model_revision")):
        raise ValueError("base-model fingerprint differs from the screened protocol")
    if (lora_fingerprint.get("decoder_lora_loaded") is not True
            or lora_fingerprint.get("decoder_lora_sha256")
            != screen.get("shared_start", {}).get("decoder_lora_sha256")
            or lora_fingerprint.get("checkpoint_path")
            != screen.get("shared_start", {}).get("decoder_lora_checkpoint")
            or sha256_file(args.decoder_lora_fingerprint)
            != screen.get("decoder_lora_fingerprint_manifest_sha256")):
        raise ValueError("frozen decoder LoRA fingerprint differs from the screened protocol")

    selected = selection.get("selected_candidate")
    expected_arms = ["D0", "D1"] + ([selected] if selected else [])
    if selection.get("full_val_tune_arms") != expected_arms:
        raise ValueError("full val_tune arms do not follow the locked 32-page selection rule")
    domain_selection = selection.get("domains", {}).get(args.domain, {})
    if (domain_selection.get("val_tune_manifest_sha256") != val_tune_sha256
            or domain_selection.get("diagnostic_manifest_sha256")
            != screen.get("diagnostic_manifest_sha256")):
        raise ValueError("candidate screen belongs to a different domain or val_tune manifest")
    screen_ids = set(screen.get("selection", {}).get("selected_page_ids", []))
    page_ids = [str(record["page_id"]) for record in records]
    if not screen_ids or not screen_ids.issubset(set(page_ids)):
        raise ValueError("screened page sample is not contained in the full val_tune manifest")

    config_path = Path(__file__).resolve().parents[2] / "configs/line_mask_v3/diagnostic.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != screen.get("diagnostic_config_sha256"):
        raise ValueError("diagnostic configuration differs from the screen protocol")
    active_arms = {name: config["arms"][name] for name in expected_arms}
    trace_ids = screen.get("selection", {}).get("full_mask_trace_page_ids", [])
    if len(trace_ids) != 3 or not set(trace_ids).issubset(screen_ids):
        raise ValueError("full-mask trace pages must be the three previously locked sample pages")

    # Re-derive the line evidence over the full page set rather than copying the
    # screen's: the screen only covered 32 pages, and the full run must not
    # supervise a page whose line grouping was never checked.
    evidence = line_evidence(records)
    if (evidence["spatial_targets"] != screen.get("spatial_targets")
            or evidence["box_granularity"] != (screen.get("line_evidence") or {}).get(
                "box_granularity")
            or evidence["line_source"] != (screen.get("line_evidence") or {}).get("line_source")):
        raise ValueError(
            "the full val_tune page set has different line evidence from the 32-page screen; "
            "the screened arm cannot be extended to it"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    full_manifest = args.output_dir / "val-tune-full.jsonl"
    with full_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    evidence_path = args.output_dir / "line-evidence.json"
    evidence_path.write_text(
        json.dumps({"status": "locked_after_32_page_screen_before_full_inference",
                    "evaluation_stage": "full_val_tune",
                    "evaluation_role": "val_tune",
                    "dataset": args.domain,
                    "diagnostic_manifest_sha256": sha256_file(full_manifest),
                    "val_tune_manifest_sha256": val_tune_sha256,
                    **evidence,
                    "test_manifest_read": False,
                    "test_used_for_selection": False},
                   ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    protocol = {
        "status": "locked_after_32_page_screen_before_full_inference",
        "evaluation_stage": "full_val_tune",
        "evaluation_role": "val_tune",
        "dataset": args.domain,
        "seed": screen["seed"],
        "source_group_field": screen.get("source_group_field"),
        "diagnostic_config": screen["diagnostic_config"],
        "diagnostic_config_sha256": screen["diagnostic_config_sha256"],
        "source_snapshot_sha256": screen["source_snapshot_sha256"],
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": screen["train_manifest_sha256"],
        "val_tune_manifest": str(args.val_tune_manifest.resolve()),
        "val_tune_manifest_sha256": val_tune_sha256,
        "evaluation_manifest": str(full_manifest.resolve()),
        "evaluation_manifest_sha256": sha256_file(full_manifest),
        "diagnostic_manifest_sha256": sha256_file(full_manifest),
        "val_tune_pages": len(records),
        "diagnostic_pages": len(records),
        "screen_manifest_sha256": screen["diagnostic_manifest_sha256"],
        "screen_protocol_sha256": sha256_file(args.screen_protocol),
        "line_evidence": evidence,
        "spatial_targets": evidence["spatial_targets"],
        "line_evidence_sha256": sha256_file(evidence_path),
        "line_evidence_file": str(evidence_path.resolve()),
        "candidate_selection_sha256": sha256_file(args.candidate_selection),
        "candidate_selected_from_screen": selected,
        "arms": active_arms,
        "selection": {
            "selected_page_ids": page_ids,
            "full_mask_trace_page_ids": sorted(trace_ids),
            "screen_sample_page_ids": sorted(screen_ids),
        },
        "overlap_audit": screen["overlap_audit"],
        "shared_start": screen["shared_start"],
        "base_model_weights_fingerprint_manifest_sha256": (
            screen["base_model_weights_fingerprint_manifest_sha256"]
        ),
        "decoder_lora_fingerprint_manifest_sha256": (
            screen["decoder_lora_fingerprint_manifest_sha256"]
        ),
        "generation": screen["generation"],
        "val_verify_manifest_sha256": None,
        "val_verify_manifest_read": False,
        "val_verify_used_for_selection": False,
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"status": "locked", "domain": args.domain,
                      "evaluation_stage": protocol["evaluation_stage"],
                      "evaluation_role": protocol["evaluation_role"],
                      "pages": len(records), "arms": expected_arms,
                      "source_snapshot_sha256": protocol["source_snapshot_sha256"],
                      "val_verify_manifest_read": False,
                      "test_manifest_read": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
