#!/usr/bin/env python3
"""Lock the two preregistered, stratified 32-page val_tune samples for line-mask v3.

Also locks each domain's *line evidence*: which manifest field supplies the line
grouping that ``target_mode='line'`` reads, and at what granularity.  The two
domains differ -- MTHv2's char manifest carries a per-character ``line_index``,
while the Dunhuang/local-gazetteer manifest is textline-level with no character
array at all -- and the evaluator refuses to run an arm whose manifest does not
match the evidence locked here.  A domain with no verified line evidence is
recorded as such rather than falling back to a guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PIL import Image

from layout_ocr.data import load_records
from layout_ocr.line_mask_v3_diagnostics import (
    layout_region_count,
    sha256_file,
    source_group,
    stratified_sample,
)

EXPECTED_VAL_TUNE_PAGES = {"mthv2": 240, "dunhuang_local_gazetteer": 80}
EXPECTED_TRAIN_MANIFEST_SHA256 = {
    "mthv2": "1016198040944e39329712eb2a7bdfe6db7526134b91d750c83afa91d854f9b3",
    "dunhuang_local_gazetteer": "00ae8c30fc12046586cce836897af26b7a701a749fa10ff805bb4ae8022fb29d",
}
EXPECTED_VAL_TUNE_MANIFEST_SHA256 = {
    "mthv2": "efc29e22b42fb81c8915282f935ef6c77ddcdcb7169f7a48bae6471c6c0cd315",
    "dunhuang_local_gazetteer": "e20d2f9b07e535ccfab03c95a8f81222f75ee4f5a793e27dabb28338f48316e2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=tuple(EXPECTED_VAL_TUNE_PAGES), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-tune-manifest", "--validation-manifest", dest="val_tune_manifest",
                        type=Path, required=True)
    parser.add_argument("--base-model-weights-fingerprint", type=Path, required=True)
    parser.add_argument("--decoder-lora-fingerprint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-group-field")
    return parser.parse_args()


def split_name(record: dict) -> str | None:
    return record.get("split", record.get("official_split"))


def line_evidence(records: list[dict]) -> dict:
    """Decide how a domain's line grouping may be read, and record why.

    Both domains are evaluated with ``target_mode='line'``, but they can supply
    line geometry by different routes and the evaluator must not be allowed to
    guess which.  A manifest that carries ``characters[].line_index`` is read
    through ``annotation`` (exact, and the only shape the v2 head was trained
    on).  A line-level manifest with no ``characters`` array is read through
    ``region_textline``, which walks ``page_text`` against the regions'
    reading-order concatenation.

    The concatenation is compared raw, exactly as the runtime walks it, because
    the line boxes are indexed by ``page_text`` position and the evaluator aligns
    generated tokens to those same positions.  Anything else -- a non-textline
    layout level, regions that do not reproduce ``page_text`` -- leaves the
    domain without line evidence; the evaluator then refuses the arm rather than
    supervising it with a grouping nobody verified.  A refused domain is
    *recorded* rather than raised, so a run covering both domains can still be
    locked and report the one whose lines are not locatable.
    """

    annotated = sum(
        1 for record in records
        if any(isinstance(entry, dict) and "line_index" in entry
               for entry in (record.get("characters") or []))
    )
    if annotated:
        if annotated != len(records):
            raise ValueError(
                f"manifest mixes {annotated}/{len(records)} pages with a per-character "
                "'line_index'; partial line annotation cannot drive one evaluation arm"
            )
        return {
            "line_source": "annotation",
            "box_granularity": "character",
            "spatial_targets": "character_boxes",
            "pages": len(records),
            "pages_with_line_mapping": len(records),
            "unmappable_pages": [],
            "whitespace_insensitive_concat": None,
        }

    # From here on the manifest has no usable per-character line evidence, so
    # every remaining failure is a *reported* shortfall for the locked sample
    # rather than an error -- the two domains are handled by one code path and a
    # domain that genuinely cannot supply line geometry must still be lockable,
    # so the run can proceed on the domain that can and report the other.
    if any(record.get("characters") for record in records):
        return _no_line_evidence(
            records, "manifest carries a 'characters' array without any 'line_index'"
        )
    levels = {str(record.get("layout_level")) for record in records}
    if levels != {"textline"}:
        return _no_line_evidence(
            records,
            f"manifest is not textline-level (layout_level={sorted(levels)})",
        )

    # A wrong line grouping poisons every spatial target on the page, so one
    # unreadable page disqualifies the domain rather than being skipped.
    #
    # The comparison is a raw one, not whitespace-insensitive: the boxes below
    # are indexed by ``page_text`` position, and the evaluator aligns generated
    # tokens to those same raw positions, so a region set that only reproduces
    # the *stripped* text would shift every line boundary by the whitespace it
    # dropped.  This deliberately mirrors ``region_line_targets`` exactly, so a
    # domain locked as having line evidence is one the runtime can actually read.
    unmappable: list[dict] = []
    for record in records:
        page_text = str(record.get("page_text") or "")
        regions = sorted(record.get("regions") or [],
                         key=lambda region: int(region["reading_order"]))
        joined = "".join(str(region.get("text") or "") for region in regions)
        if joined != page_text:
            unmappable.append({
                "page_id": str(record.get("page_id")),
                "reason": "reading-order region text does not reproduce page_text",
            })
    if unmappable:
        return _no_line_evidence(
            records, "at least one page's regions do not reproduce page_text",
            unmappable=unmappable,
        )
    return {
        "line_source": "region_textline",
        "box_granularity": "textline",
        "spatial_targets": "line_regions",
        "pages": len(records),
        "pages_with_line_mapping": len(records),
        "unmappable_pages": [],
        "whitespace_insensitive_concat": True,
    }


def _strip(text: str) -> str:
    return "".join(text.split())


def _no_line_evidence(records: list[dict], reason: str,
                      unmappable: list[dict] | None = None) -> dict:
    return {
        "line_source": None,
        "box_granularity": None,
        "spatial_targets": None,
        "pages": len(records),
        "pages_with_line_mapping": 0,
        "unmappable_pages": unmappable or [
            {"page_id": str(record.get("page_id")), "reason": reason} for record in records
        ],
        "whitespace_insensitive_concat": None,
        "unavailable_reason": reason,
    }


def dhash64(path: Path) -> int:
    with Image.open(path) as source:
        image = source.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = list(image.getdata())
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | int(values[row * 9 + col] > values[row * 9 + col + 1])
    return result


def image_fingerprints(records: list[dict], label: str) -> dict[str, tuple[str, int]]:
    fingerprints = {}
    for index, record in enumerate(records, 1):
        page_id = str(record["page_id"])
        path = Path(record["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} image for {page_id}: {path}")
        fingerprints[page_id] = (sha256_file(path), dhash64(path))
        if index % 250 == 0 or index == len(records):
            print(json.dumps({"phase": "image_audit", "split": label, "pages": index,
                              "total": len(records)}, ensure_ascii=False), flush=True)
    return fingerprints


def duplicate_groups(fingerprints: dict[str, tuple[str, int]]) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for page_id, (digest, _) in fingerprints.items():
        by_hash[digest].append(page_id)
    return [sorted(page_ids) for page_ids in by_hash.values() if len(page_ids) > 1]


def overlap_audit(train: list[dict], val_tune: list[dict], source_group_field: str | None) -> dict:
    train_ids = {str(record["page_id"]) for record in train}
    val_tune_ids = {str(record["page_id"]) for record in val_tune}
    if train_ids & val_tune_ids:
        raise ValueError(f"train/val_tune page ID overlap: {sorted(train_ids & val_tune_ids)[:10]}")

    train_groups = {source_group(record, source_group_field)[0] for record in train}
    val_tune_groups = {source_group(record, source_group_field)[0] for record in val_tune}
    shared_groups = sorted(train_groups & val_tune_groups)

    print(json.dumps({"phase": "image_audit_start", "train_pages": len(train),
                      "val_tune_pages": len(val_tune)}, ensure_ascii=False), flush=True)
    train_hashes = image_fingerprints(train, "train")
    val_tune_hashes = image_fingerprints(val_tune, "val_tune")

    train_by_sha: dict[str, list[str]] = defaultdict(list)
    for page_id, (digest, _) in train_hashes.items():
        train_by_sha[digest].append(page_id)
    exact_cross_split = [
        {"val_tune_page_id": page_id, "train_page_ids": sorted(train_by_sha[digest])}
        for page_id, (digest, _) in val_tune_hashes.items()
        if digest in train_by_sha
    ]

    near_cross_split = []
    for val_id, (val_sha, val_hash) in val_tune_hashes.items():
        for train_id, (train_sha, train_hash) in train_hashes.items():
            if val_sha != train_sha and (val_hash ^ train_hash).bit_count() <= 4:
                near_cross_split.append({
                    "val_tune_page_id": val_id,
                    "train_page_id": train_id,
                    "dhash_hamming_distance": (val_hash ^ train_hash).bit_count(),
                })
    near_cross_split.sort(key=lambda pair: (pair["dhash_hamming_distance"],
                                            pair["val_tune_page_id"], pair["train_page_id"]))
    return {
        "train_val_tune_page_id_overlap": False,
        "shared_source_groups": shared_groups,
        "train_internal_exact_image_duplicates": duplicate_groups(train_hashes),
        "val_tune_internal_exact_image_duplicates": duplicate_groups(val_tune_hashes),
        "exact_image_duplicates_between_splits": exact_cross_split,
        "near_duplicate_screen": {
            "method": "64-bit difference hash on 9x8 grayscale thumbnails",
            "candidate_threshold_hamming_distance": 4,
            "interpretation": "screening candidates only; manually inspect before treating as duplicates",
            "train_val_tune_candidate_count": len(near_cross_split),
            "train_val_tune_candidates": near_cross_split[:200],
            "candidate_list_truncated": len(near_cross_split) > 200,
        },
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace locked protocol directory: {args.output_dir}")
    train = load_records(args.train_manifest)
    val_tune = load_records(args.val_tune_manifest)
    if len(val_tune) != EXPECTED_VAL_TUNE_PAGES[args.domain]:
        raise ValueError(
            f"{args.domain} expects {EXPECTED_VAL_TUNE_PAGES[args.domain]} val_tune pages, "
            f"found {len(val_tune)}"
        )
    val_tune_sha256 = sha256_file(args.val_tune_manifest)
    train_sha256 = sha256_file(args.train_manifest)
    if train_sha256 != EXPECTED_TRAIN_MANIFEST_SHA256[args.domain]:
        raise ValueError(f"{args.domain} diagnosis requires its fixed train manifest")
    if val_tune_sha256 != EXPECTED_VAL_TUNE_MANIFEST_SHA256[args.domain]:
        raise ValueError(f"{args.domain} diagnosis requires its fixed full val_tune manifest")
    if any(split_name(record) != "train" for record in train):
        raise ValueError("train manifest contains a non-train record")
    if any(split_name(record) != "validation" for record in val_tune):
        raise ValueError("val_tune input must use the existing validation split")
    if args.source_group_field:
        for record in train + val_tune:
            source_group(record, args.source_group_field)
    audit = overlap_audit(train, val_tune, args.source_group_field)
    evidence = line_evidence(val_tune)
    selected, selection = stratified_sample(
        val_tune, count=32, seed=42, source_group_field=args.source_group_field
    )
    if not selection["layout_density_quartile_coverage_complete"]:
        raise ValueError("32-page diagnostic sample must include all available layout-density quartiles")
    selected_above_sparse_cap = sum(layout_region_count(record) > 24 for record in selected)
    if args.domain == "mthv2" and selected_above_sparse_cap < 8:
        raise ValueError(
            "MTHv2 diagnostic32 must include at least eight pages above the historical sparse24 cap"
        )
    selection["selected_pages_above_sparse24_region_cap"] = selected_above_sparse_cap
    selection["historical_sparse24_region_cap"] = 24 if args.domain == "mthv2" else None
    config_path = Path(__file__).resolve().parents[2] / "configs" / "line_mask_v3" / "diagnostic.json"
    diagnostic_config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(config_path)
    source_fingerprint_path = Path(__file__).resolve().parents[3] / "source-fingerprint.json"
    source_snapshot_sha256 = None
    if source_fingerprint_path.is_file():
        source_snapshot_sha256 = json.loads(
            source_fingerprint_path.read_text(encoding="utf-8")
        )["source_tree_sha256"]
    base_model_fingerprint = None
    base_model_fingerprint = json.loads(
        args.base_model_weights_fingerprint.read_text(encoding="utf-8")
    )
    if (base_model_fingerprint.get("model_revision")
            != diagnostic_config["shared_start"]["model_revision"]
            or base_model_fingerprint.get("decoder_lora_loaded") is not False
            or len(str(base_model_fingerprint.get("model_weights_sha256", ""))) != 64):
        raise ValueError("raw GLM-OCR weight fingerprint is invalid or uses a different revision")
    decoder_lora_fingerprint = json.loads(
        args.decoder_lora_fingerprint.read_text(encoding="utf-8")
    )
    if (decoder_lora_fingerprint.get("model_revision")
            != diagnostic_config["shared_start"]["model_revision"]
            or decoder_lora_fingerprint.get("decoder_lora_loaded") is not True
            or decoder_lora_fingerprint.get("rank") != 8
            or decoder_lora_fingerprint.get("alpha") != 8.0
            or decoder_lora_fingerprint.get("dropout") != 0.0
            or decoder_lora_fingerprint.get("decoder_lora_sha256")
            != "ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5"):
        raise ValueError("decoder LoRA fingerprint differs from the paired v2 training checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "selected-val-tune-32.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    protocol = {
        "status": "locked_before_candidate_inference",
        "evaluation_stage": "diagnostic32",
        "dataset": args.domain,
        "evaluation_role": "val_tune",
        "seed": 42,
        "source_group_field": args.source_group_field,
        "diagnostic_config": "configs/line_mask_v3/diagnostic.json",
        "diagnostic_config_sha256": config_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "shared_start": {
            **diagnostic_config["shared_start"],
            "model_path": base_model_fingerprint["model_path"],
            "base_model_weights_sha256": base_model_fingerprint["model_weights_sha256"],
            "decoder_lora_loaded": True,
            "decoder_lora_checkpoint": decoder_lora_fingerprint["checkpoint_path"],
            "decoder_lora_sha256": decoder_lora_fingerprint["decoder_lora_sha256"],
            "decoder_lora_rank": decoder_lora_fingerprint["rank"],
            "decoder_lora_alpha": decoder_lora_fingerprint["alpha"],
            "decoder_lora_dropout": decoder_lora_fingerprint["dropout"],
            "base_model_weights_fingerprint_manifest_sha256": sha256_file(
                args.base_model_weights_fingerprint
            ),
            "decoder_lora_fingerprint_manifest_sha256": sha256_file(args.decoder_lora_fingerprint),
        },
        "base_model_weights_fingerprint_manifest_sha256": sha256_file(
            args.base_model_weights_fingerprint
        ),
        "decoder_lora_fingerprint_manifest_sha256": sha256_file(args.decoder_lora_fingerprint),
        "val_tune_pages": len(val_tune),
        "diagnostic_pages": len(selected),
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": train_sha256,
        "val_tune_manifest": str(args.val_tune_manifest.resolve()),
        "val_tune_manifest_sha256": val_tune_sha256,
        "diagnostic_manifest": str(manifest_path.resolve()),
        "diagnostic_manifest_sha256": sha256_file(manifest_path),
        "val_verify_manifest_sha256": None,
        "val_verify_manifest_read": False,
        "val_verify_used_for_selection": False,
        "selection": selection,
        "line_evidence": evidence,
        "spatial_targets": evidence["spatial_targets"],
        "overlap_audit": audit,
        "generation": {
            "input": "whole-page image plus the fixed Text Recognition prompt",
            "processor": "fast",
            "max_pixels": 4000000,
            "max_new_tokens": 1536,
            "sampling": "greedy",
            "backbone_precision": "bfloat16",
            "head_precision": "float32",
            "attention_backend": "torch_sdpa_math",
        },
        "arms": diagnostic_config["arms"],
        "selection_rule": (
            "A D2/D3 candidate is eligible only when it is no worse than D1 in both val_tune domains; "
            "among eligible candidates choose the minimum equal-domain mean relative CER change, "
            "breaking ties in favor of D2. Otherwise retain D0/D1 for full val_tune."
        ),
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
    protocol_path = args.output_dir / "protocol.json"
    # A separate file the evaluator hashes and compares, so the arm cannot run
    # against a manifest whose line geometry nobody verified.  Written before the
    # protocol so its digest can be recorded in the protocol itself.
    evidence_path = args.output_dir / "line-evidence.json"
    evidence_path.write_text(
        json.dumps({"status": "locked_before_candidate_inference",
                    "evaluation_stage": "diagnostic32",
                    "evaluation_role": "val_tune",
                    "dataset": args.domain,
                    "diagnostic_manifest_sha256": protocol["diagnostic_manifest_sha256"],
                    "val_tune_manifest_sha256": val_tune_sha256,
                    **evidence,
                    "test_manifest_read": False,
                    "test_used_for_selection": False},
                   ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    protocol["line_evidence_sha256"] = sha256_file(evidence_path)
    protocol["line_evidence_file"] = str(evidence_path.resolve())
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
                             encoding="utf-8")
    print(json.dumps({"status": "locked", "domain": args.domain,
                      "evaluation_role": protocol["evaluation_role"],
                      "diagnostic_pages": len(selected),
                      "manifest_sha256": protocol["diagnostic_manifest_sha256"],
                      "selected_page_ids": selection["selected_page_ids"],
                      "trace_page_ids": selection["full_mask_trace_page_ids"],
                      "spatial_targets": evidence["spatial_targets"],
                      "line_source": evidence["line_source"],
                      "test_manifest_read": False, "test_used_for_selection": False},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
