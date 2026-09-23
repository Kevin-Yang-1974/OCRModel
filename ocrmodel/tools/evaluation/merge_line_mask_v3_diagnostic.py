#!/usr/bin/env python3
"""Merge one domain's five frozen line-mask v3 shards and verify exact val_tune coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from layout_ocr.data import load_records
from layout_ocr.line_mask_v3_diagnostics import sha256_file, source_group
from layout_ocr.metrics import aggregate_ocr_metrics

ARMS = ("D0", "D1", "D2", "D3")
VAL_TUNE_PAGES = {"mthv2": 240, "dunhuang_local_gazetteer": 80}
# Kept in step with the evaluator: only these two spatial target kinds can be
# read, and a protocol naming anything else is refused rather than merged.
SPATIAL_TARGET_LINE_SOURCES = {
    "character_boxes": "annotation",
    "line_regions": "region_textline",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--val-tune-manifest", "--validation-manifest", dest="val_tune_manifest",
                        type=Path, required=True)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--line-evidence", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root / "src"))
    records = load_records(args.val_tune_manifest)
    protocol = json.loads(args.diagnostic_protocol.read_text(encoding="utf-8"))
    evaluation_stage = protocol.get("evaluation_stage")
    if evaluation_stage == "diagnostic32":
        expected_pages, expected_status, active_arms = 32, "locked_before_candidate_inference", list(ARMS)
    elif evaluation_stage == "full_val_tune":
        expected_pages = VAL_TUNE_PAGES.get(protocol.get("dataset"))
        expected_status = "locked_after_32_page_screen_before_full_inference"
        candidate = protocol.get("candidate_selected_from_screen")
        if candidate not in (None, "D2", "D3"):
            raise ValueError("full val_tune protocol selected an unsupported candidate")
        active_arms = ["D0", "D1"] + ([candidate] if candidate else [])
    else:
        raise ValueError("merge protocol must name diagnostic32 or full_val_tune")
    if expected_pages is None or len(records) != expected_pages or protocol.get("status") != expected_status:
        raise ValueError("merge manifest does not match the locked val_tune stage")
    if (protocol.get("evaluation_role") != "val_tune"
            or protocol.get("val_verify_manifest_read") is not False
            or protocol.get("val_verify_used_for_selection") is not False):
        raise ValueError("merge protocol does not explicitly mark val_tune and val_verify isolation")
    if not protocol.get("source_snapshot_sha256"):
        raise ValueError("merge protocol must lock an immutable source snapshot")
    if protocol.get("diagnostic_manifest_sha256") != sha256_file(args.val_tune_manifest):
        raise ValueError("val_tune subset fingerprint differs from the locked protocol")
    line_evidence = json.loads(args.line_evidence.read_text(encoding="utf-8"))
    if (sha256_file(args.line_evidence) != protocol.get("line_evidence_sha256")
            or (protocol.get("line_evidence") or {}).get("box_granularity")
            != line_evidence.get("box_granularity")
            # Matched against the stage being merged, not one fixed string: the
            # evidence carries whichever stage wrote it, and full_val_tune writes
            # its own status.
            or line_evidence.get("status") != expected_status
            or line_evidence.get("spatial_targets") not in SPATIAL_TARGET_LINE_SOURCES):
        raise ValueError("merge line evidence differs from the locked protocol")
    if protocol.get("test_manifest_read") is not False or protocol.get("test_used_for_selection") is not False:
        raise ValueError("test isolation fields are not locked false")
    expected_ids = [str(record["page_id"]) for record in records]
    if protocol.get("selection", {}).get("selected_page_ids") != expected_ids:
        raise ValueError("val_tune subset order differs from the locked protocol")

    if list(protocol.get("arms", {})) != active_arms:
        raise ValueError("merge active arms differ from the locked protocol")
    merged: dict[str, list[dict]] = {arm: [] for arm in active_arms}
    shard_ids = set()
    source_snapshot_hashes = set()
    worker_settings = set()
    worker_identity_fields = (
        "diagnostic_manifest_sha256", "train_manifest_sha256", "val_tune_manifest_sha256",
        "mask_checkpoint_sha256", "base_model_weights_sha256",
        "base_model_weights_fingerprint_manifest_sha256", "decoder_lora_sha256",
        "decoder_lora_fingerprint_manifest_sha256", "decoder_lora_loaded",
        "diagnostic_config_sha256", "line_evidence_sha256", "line_source",
        "spatial_targets", "spatial_target_granularity",
        "source_snapshot_sha256", "attention_backend",
        "model_revision", "evaluation_stage",
    )
    shared_start = protocol.get("shared_start", {})
    expected_worker_identity = {
        "diagnostic_manifest_sha256": protocol["diagnostic_manifest_sha256"],
        "train_manifest_sha256": protocol["train_manifest_sha256"],
        "val_tune_manifest_sha256": protocol["val_tune_manifest_sha256"],
        "mask_checkpoint_sha256": shared_start.get("line_mask_head_sha256"),
        "base_model_weights_sha256": shared_start.get("base_model_weights_sha256"),
        "base_model_weights_fingerprint_manifest_sha256": (
            protocol.get("base_model_weights_fingerprint_manifest_sha256")
        ),
        "decoder_lora_sha256": shared_start.get("decoder_lora_sha256"),
        "decoder_lora_fingerprint_manifest_sha256": (
            protocol.get("decoder_lora_fingerprint_manifest_sha256")
        ),
        "decoder_lora_loaded": True,
        "diagnostic_config_sha256": protocol["diagnostic_config_sha256"],
        "line_evidence_sha256": protocol.get("line_evidence_sha256"),
        "line_source": (protocol.get("line_evidence") or {}).get("line_source"),
        "spatial_targets": (protocol.get("line_evidence") or {}).get("spatial_targets"),
        "spatial_target_granularity": (protocol.get("line_evidence") or {}).get(
            "box_granularity"
        ),
        "source_snapshot_sha256": protocol["source_snapshot_sha256"],
        "attention_backend": protocol["generation"]["attention_backend"],
        "model_revision": shared_start.get("model_revision"),
        "evaluation_stage": evaluation_stage,
    }
    for shard_index in range(5):
        shard_root = args.run_root / f"shard-{shard_index}"
        status = json.loads((shard_root / "worker_status.json").read_text(encoding="utf-8"))
        worker_protocol = json.loads((shard_root / "worker_protocol.json").read_text(encoding="utf-8"))
        # The worker reports the pages it actually processed -- its own shard --
        # so it is compared against the shard's allocation, not the whole set.
        # Comparing against ``len(expected_ids)`` rejected every shard of a
        # five-way split (7 against 32), which is what failed the v4 merge.
        expected_shard_ids = expected_ids[shard_index::5]
        if (status.get("status") != "complete" or status.get("evaluation_stage") != evaluation_stage
                or status.get("pages") != len(expected_shard_ids)
                or worker_protocol.get("status") != "running"):
            raise RuntimeError(
                f"shard {shard_index} is incomplete: status={status.get('status')} "
                f"pages={status.get('pages')} (expected {len(expected_shard_ids)})"
            )
        if worker_protocol.get("source_snapshot_sha256") != protocol.get("source_snapshot_sha256"):
            raise RuntimeError(f"shard {shard_index} used a different source snapshot")
        for field, expected_value in expected_worker_identity.items():
            if worker_protocol.get(field) != expected_value:
                raise RuntimeError(f"shard {shard_index} differs from locked {field}")
        source_snapshot_hashes.add(worker_protocol["source_snapshot_sha256"])
        worker_settings.add(tuple(worker_protocol.get(field) for field in worker_identity_fields))
        if (worker_protocol.get("evaluation_role") != "val_tune"
                or worker_protocol.get("val_verify_manifest_read") is not False
                or worker_protocol.get("val_verify_used_for_selection") is not False
                or worker_protocol.get("test_manifest_read") is not False
                or worker_protocol.get("test_used_for_selection") is not False):
            raise RuntimeError(f"shard {shard_index} violated data-use isolation")
        if worker_protocol.get("page_ids") != expected_shard_ids:
            raise RuntimeError(f"shard {shard_index} has a different preregistered page allocation")
        shard_ids.update(expected_shard_ids)
        if (worker_protocol.get("evaluation_stage") != evaluation_stage
                or worker_protocol.get("active_arms") != active_arms):
            raise RuntimeError(f"shard {shard_index} used a different evaluation stage or arms")
        for arm in active_arms:
            arm_status = json.loads((shard_root / f"status-{arm}.json").read_text(encoding="utf-8"))
            if arm_status.get("status") != "complete" or arm_status.get("pages") != len(expected_shard_ids):
                raise RuntimeError(f"shard {shard_index} arm {arm} is incomplete")
            rows = read_rows(shard_root / f"predictions-{arm}.jsonl")
            row_ids = [str(row.get("page_id")) for row in rows]
            if row_ids != expected_shard_ids:
                raise RuntimeError(f"shard {shard_index} arm {arm} has missing, duplicate, or reordered pages")
            for row in rows:
                if (row.get("evaluation_stage") != evaluation_stage
                        or row.get("reads_ground_truth_for_routing") is not False
                        or row.get("test_manifest_read") is not False
                        or row.get("test_used_for_selection") is not False):
                    raise RuntimeError(f"protocol violation in {shard_index}/{arm}/{row.get('page_id')}")
            merged[arm].extend(rows)
    if shard_ids != set(expected_ids) or len(shard_ids) != expected_pages:
        raise RuntimeError(f"five shards do not provide exact {expected_pages}-page coverage")
    if len(source_snapshot_hashes) != 1 or len(worker_settings) != 1:
        raise RuntimeError("five shards do not share one source and model protocol")
    locked_worker_settings = dict(zip(worker_identity_fields, next(iter(worker_settings))))

    record_by_id = {str(record["page_id"]): record for record in records}
    summary_arms = {}
    for arm in active_arms:
        rows = merged[arm]
        if len(rows) != expected_pages or {str(row["page_id"]) for row in rows} != set(expected_ids):
            raise RuntimeError(f"arm {arm} does not cover the complete diagnostic subset")
        for row in rows:
            if row["reference"] != record_by_id[str(row["page_id"])]["page_text"]:
                raise RuntimeError(f"reference mismatch for {arm}/{row['page_id']}")
        metrics = aggregate_ocr_metrics(
            ((row["reference"], row["prediction"]) for row in rows), Counter()
        )
        metrics.update({
            "eos_pages": sum(bool(row["generation_eos_hit"]) for row in rows),
            "generation_limit_hits": sum(bool(row["generation_limit_hit"]) for row in rows),
            "loop_pages": sum(bool(row["repetition"].get("repeated_cycle_detected")) for row in rows),
        })
        metrics["loop_rate"] = metrics["loop_pages"] / max(1, metrics["pages"])
        quality_rows = [row["alignment"] for row in rows]
        quality_count = sum(int(row["reliable_spatial_character_count"]) for row in quality_rows)
        quality_means = {}
        for name in ("continuous_line_iou", "in_line_probability_mass", "continuous_mass_ratio", "binary_line_iou"):
            numerator = sum(
                float(row[name]) * int(row["reliable_spatial_character_count"])
                for row in quality_rows if row.get(name) is not None
            )
            denominator = sum(
                int(row["reliable_spatial_character_count"])
                for row in quality_rows if row.get(name) is not None
            )
            quality_means[name] = numerator / denominator if denominator else None
        groups: dict[str, list[dict]] = {}
        for row in rows:
            group, _ = source_group(record_by_id[str(row["page_id"])],
                                    protocol.get("source_group_field"))
            groups.setdefault(group, []).append(row)
        source_metrics = {}
        for group, group_rows in sorted(groups.items()):
            group_result = aggregate_ocr_metrics(
                ((row["reference"], row["prediction"]) for row in group_rows), Counter()
            )
            source_metrics[group] = {"pages": len(group_rows), "cer": group_result["cer"]}
        # The granularity is locked per domain, so every row must agree with the
        # protocol.  A mixed arm would make the line IoU mean two different things.
        granularities = {row.get("spatial_target_granularity") for row in rows}
        locked_granularity = protocol["line_evidence"]["box_granularity"]
        if granularities != {locked_granularity}:
            raise RuntimeError(
                f"arm {arm} spatial target granularity {sorted(map(str, granularities))} "
                "differs from the locked line evidence"
            )
        summary_arms[arm] = {
            "metrics": metrics,
            "line_quality": quality_means,
            "reliable_spatial_positions": quality_count,
            "alignment_coverage_mean": sum(row["alignment"]["alignment_coverage"] for row in rows) / len(rows),
            "ambiguous_token_count": sum(row["alignment"]["ambiguous_token_count"] for row in rows),
            "multi_character_token_count": sum(row["alignment"]["multi_character_token_count"] for row in rows),
            "cross_line_token_count": sum(row["alignment"]["cross_line_token_count"] for row in rows),
            "insertion_token_count": sum(row["alignment"]["insertion_token_count"] for row in rows),
            # A line-level manifest has no inner-line geometry, so its targets
            # cannot localise a character -- report that rather than letting the
            # reader compare its line IoU against the char manifest's directly.
            "spatial_target_granularity": locked_granularity,
            "pages_with_unavailable_line_mapping": sum(
                1 for row in rows if not row["window_report"]["lines"]
            ),
            "source_group_metrics": source_metrics,
        }

    output = {
        "status": "complete",
        "domain": protocol["dataset"],
        "evaluation_stage": evaluation_stage,
        "diagnostic_pages": expected_pages,
        "active_arms": active_arms,
        "source_snapshot_sha256": next(iter(source_snapshot_hashes)),
        "attention_backend": protocol["generation"]["attention_backend"],
        "page_ids": expected_ids,
        "evaluation_role": "val_tune",
        "diagnostic_manifest_sha256": sha256_file(args.val_tune_manifest),
        "train_manifest_sha256": protocol["train_manifest_sha256"],
        "val_tune_manifest_sha256": protocol["val_tune_manifest_sha256"],
        "val_verify_manifest_sha256": None,
        "val_verify_manifest_read": False,
        "val_verify_used_for_selection": False,
        "source_code": {
            "runtime_sha256": sha256_file(code_root / "src/layout_ocr/line_mask_runtime.py"),
            "head_sha256": sha256_file(code_root / "src/layout_ocr/line_mask_head.py"),
            "evaluator_sha256": sha256_file(code_root / "tools/evaluation/diagnose_line_mask_v3.py"),
            "merger_sha256": sha256_file(Path(__file__)),
        },
        "shared_start": protocol["shared_start"],
        "diagnostic_config_sha256": locked_worker_settings["diagnostic_config_sha256"],
        "arms": summary_arms,
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
    output_path = args.run_root / "summary.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite merged result: {output_path}")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "complete", "domain": output["domain"],
                      "evaluation_stage": evaluation_stage,
                      "cer_by_arm": {arm: summary_arms[arm]["metrics"]["cer"] for arm in active_arms},
                      "test_manifest_read": False, "test_used_for_selection": False},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
