#!/usr/bin/env python3
"""Run the preregistered frozen-weight line-mask v3 val_tune diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from layout_ocr.data import load_records, prepare_inference_inputs, prepare_training_inputs
from layout_ocr.line_mask_head import LineMaskConfig, LineMaskHead
from layout_ocr.line_mask_runtime import LineMaskRuntime
from layout_ocr.line_mask_v3_diagnostics import align_nonspace, sha256_file
from layout_ocr.lora import inject_decoder_lora, load_lora_state_dict
from layout_ocr.mask_targets import build_mask_targets
from layout_ocr.metrics import aggregate_ocr_metrics
from layout_ocr.stabilization import repetition_diagnostics
from layout_ocr.train_screen import configure_deterministic_execution

EXPECTED_HEAD_SHA256 = "d56023f270fd51812b58272141f8903bbd2ca1b437b8565b20ab5d12c7f59cac"
EXPECTED_DECODER_LORA_SHA256 = "ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5"
VAL_TUNE_PAGES = {"mthv2": 240, "dunhuang_local_gazetteer": 80}
# How a domain's locked line evidence is read by ``build_mask_targets``.  A
# domain whose evidence is absent or unverified has no entry and is refused.
SPATIAL_TARGET_LINE_SOURCES = {
    "character_boxes": "annotation",
    "line_regions": "region_textline",
}
ARMS = {
    "D0": {"enabled": False, "bias": 0.0, "layer_scope": "all"},
    "D1": {"enabled": True, "bias": 1.0, "layer_scope": "all"},
    "D2": {"enabled": True, "bias": 0.5, "layer_scope": "all"},
    "D3": {"enabled": True, "bias": 1.0, "layer_scope": "latter_half"},
}


def eos_ids(model, processor) -> list[int]:
    values = set()
    for value in (processor.tokenizer.eos_token_id, model.generation_config.eos_token_id):
        if value is not None:
            values.update(value if isinstance(value, (tuple, list)) else [value])
    return sorted(int(value) for value in values)


def load_diagnostic_backbone(model_path: Path, lora_checkpoint: Path, device: str):
    """Load the pinned raw GLM-OCR snapshot and its paired frozen decoder LoRA."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    configure_deterministic_execution()
    random.seed(42)
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, local_files_only=True)
    processor.image_processor.size = {
        **processor.image_processor.size,
        "longest_edge": 4000000,
    }
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    model.to(device)
    inject_decoder_lora(model, rank=8, alpha=8, dropout=0)
    lora_weights = load_file(str(lora_checkpoint / "decoder_lora.safetensors"))
    if not all(bool(torch.isfinite(value).all()) for value in lora_weights.values()):
        raise FloatingPointError("paired decoder LoRA contains non-finite weights")
    load_lora_state_dict(model, lora_weights)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, processor


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-model-weights-fingerprint", type=Path, required=True)
    parser.add_argument("--decoder-lora-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-lora-fingerprint", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path, required=True)
    parser.add_argument("--val-tune-manifest", "--validation-manifest", dest="val_tune_manifest",
                        type=Path, required=True)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--line-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _line_mapping_incomplete(window_report: dict) -> bool:
    """Whether a page ended up with no usable line geometry.

    A page with no line members has no line box to supervise against, so the
    count is reported rather than folded into the averaged alignment coverage.
    A partial mapping cannot reach here: ``region_line_targets`` refuses a page
    whose regions do not cover ``page_text`` exactly, and the locked evidence
    file already rejected such a page set before the run could start.
    """

    return not window_report.get("lines")


def decode_pieces(tokenizer, token_ids: list[int]) -> list[str]:
    """Per-token decoded text, whose concatenation is exactly ``decode(token_ids)``.

    Decoding each token independently is wrong for a byte-level BPE tokenizer: a
    character whose UTF-8 bytes are split across tokens decodes to U+FFFD on its
    own, so the concatenation never matches the full decode and every downstream
    per-token comparison silently fails -- which is what happened to every page
    of the first v3 attempt (``token_character_mapping_reliable`` was false
    everywhere and all line-IoU statistics were empty).

    Decoding successive prefixes and diffing them is *also* wrong: while a
    prefix ends mid-character the decoder emits U+FFFD for the incomplete tail,
    and completing the character replaces that placeholder rather than appending
    to it, so the diff drops or duplicates text.

    What is exact: for prefix ``i``, the characters the decoder can already
    resolve form a common prefix of that decode and the full decode.  Taking the
    longest common prefix length at each step and emitting ``full[cursor:k]``
    therefore telescopes to the full decode exactly, and attributes each
    character to the token that completes it -- the right owner for a character
    assembled from several tokens.
    """

    full = _decode_prefix(tokenizer, token_ids)
    pieces: list[str] = []
    cursor = 0
    for index in range(len(token_ids)):
        prefix = _decode_prefix(tokenizer, token_ids[: index + 1])
        limit = min(len(prefix), len(full))
        resolved = 0
        while resolved < limit and prefix[resolved] == full[resolved]:
            resolved += 1
        pieces.append(full[cursor:resolved])
        cursor = resolved
    if cursor < len(full):  # defensive: only reachable if the decode is unstable
        pieces[-1] += full[cursor:]
    return pieces


def _decode_prefix(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True)


def _quality(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = predicted.float()
    target = target.float()
    soft_intersection = torch.minimum(predicted, target).sum()
    soft_union = torch.maximum(predicted, target).sum().clamp_min(1e-8)
    predicted_binary = predicted >= 0.5
    target_binary = target >= 0.5
    binary_intersection = (predicted_binary & target_binary).sum().float()
    binary_union = (predicted_binary | target_binary).sum().float().clamp_min(1)
    return {
        "continuous_line_iou": float(soft_intersection / soft_union),
        "in_line_probability_mass": float((predicted * target).sum() / predicted.sum().clamp_min(1e-8)),
        "continuous_mass_ratio": float(predicted.sum() / target.sum().clamp_min(1e-8)),
        "binary_line_iou": float(binary_intersection / binary_union),
    }


def _target_by_reference_char(targets, device) -> dict[int, torch.Tensor]:
    result = {}
    for token_index, span in enumerate(targets.char_spans):
        if span is None or token_index >= targets.spatial_valid.shape[1]:
            continue
        if not bool(targets.spatial_valid[0, token_index]):
            continue
        mask = targets.mask[0, token_index]
        if not bool(mask.sum() > 0):
            continue
        for char_index in range(span[0], span[1]):
            result.setdefault(char_index, mask.to(device=device))
    return result


def _token_annotations(record: dict, token_ids: list[int], tokenizer, prediction: str,
                       trace_by_position: dict[int, dict], targets,
                       eos_set: set[int]) -> tuple[list[dict], dict]:
    pieces = decode_pieces(tokenizer, token_ids)
    piece_text = "".join(pieces)
    normalized_piece = "".join(char for char in piece_text if not char.isspace())
    normalized_full = "".join(char for char in prediction if not char.isspace())
    # The prediction must be exactly what these token ids decode to; if it is not,
    # the page's token->position attribution is unsound and no per-token spatial
    # statistic may be reported from it.
    token_map_reliable = normalized_piece == normalized_full
    mapping_failure = None
    if not token_map_reliable:
        mapping_failure = (
            f"per-token decode ({len(normalized_piece)} non-space chars) does not reproduce "
            f"the prediction ({len(normalized_full)}); token ids and text are out of sync"
        )
    alignment = align_nonspace(record["page_text"], prediction)
    ref_positions = alignment["reference_positions"]
    pred_positions = alignment["prediction_positions"]

    aligned_ref = alignment["prediction_aligned_reference_position"]
    operations = alignment["operations"]
    candidate_refs = alignment["candidate_reference_positions"]
    ambiguous = alignment["ambiguous"]

    normalized_token_ids = []
    for token_index, piece in enumerate(pieces):
        normalized_token_ids.extend([token_index] * sum(not char.isspace() for char in piece))
    if token_map_reliable and len(normalized_token_ids) != len(pred_positions):
        token_map_reliable = False
    if not token_map_reliable:
        normalized_token_ids = [None] * len(pred_positions)

    chars = record.get("characters") or []
    char_line_ids = [
        (entry.get("line_index") if isinstance(entry, dict) else None) for entry in chars
    ]
    target_by_char = _target_by_reference_char(targets, targets.mask.device)
    tokens_by_index: list[list[int]] = [[] for _ in token_ids]
    for char_index, token_index in enumerate(normalized_token_ids):
        if token_index is not None:
            tokens_by_index[token_index].append(char_index)

    page_steps = []
    for token_index, token_id in enumerate(token_ids):
        char_indices = tokens_by_index[token_index]
        mapped_ref_positions = sorted({
            ref_positions[aligned_ref[index]]
            for index in char_indices
            if aligned_ref[index] is not None
        })
        line_ids = {
            int(char_line_ids[index])
            for index in mapped_ref_positions
            if index < len(char_line_ids) and char_line_ids[index] is not None
        }
        lines_complete = all(index < len(char_line_ids) and char_line_ids[index] is not None
                             for index in mapped_ref_positions)
        cross_line = len(line_ids) > 1 if mapped_ref_positions and lines_complete else None
        char_operations = [operations[index] for index in char_indices]
        ambiguous_positions = [
            {
                "prediction_char_index": pred_positions[index],
                "candidate_reference_char_indices": [ref_positions[candidate]
                                                     for candidate in candidate_refs[index]],
            }
            for index in char_indices if ambiguous[index]
        ]

        trace = trace_by_position.get(token_index, {})
        row = {
            "generation_position": token_index,
            "token_id": int(token_id),
            "is_eos": int(token_id) in eos_set,
            "decoded_piece": pieces[token_index],
            "multi_character_token": len(char_indices) > 1,
            "token_character_alignment_reliable": token_map_reliable,
            "character_operations": char_operations,
            "insertion": "I" in char_operations,
            "substitution": "S" in char_operations,
            "alignment_ambiguous": bool(ambiguous_positions),
            "ambiguous_positions": ambiguous_positions,
            "reference_character_indices": mapped_ref_positions,
            "cross_line_token": cross_line,
            "mask_applied": bool(trace.get("mask_applied", False)),
            "mask_source_generation_position": trace.get("mask_source_generation_position"),
            "mask_update_gate": trace.get("update_gate"),
            "mask_update_gate_raw": trace.get("update_gate_raw"),
            "predicted_mask_change_mean_abs": trace.get("mask_change_mean_abs"),
            "binary_area": trace.get("binary_area", 0),
            "routing_bias": trace.get("bias"),
            "injection_layers": trace.get("injection_layers"),
            "line_quality": None,
        }

        # A single unambiguous matched character with exact spatial annotation is
        # the only token eligible for line-local quality statistics.
        if (token_map_reliable and len(char_indices) == 1 and char_operations == ["M"]
                and not ambiguous_positions and len(mapped_ref_positions) == 1
                and mapped_ref_positions[0] in target_by_char and token_index in trace_by_position):
            ref_index = mapped_ref_positions[0]
            target_token = next((index for index, span in enumerate(targets.char_spans)
                                 if span is not None and span[0] <= ref_index < span[1]
                                 and targets.alignment_status[index] == "exact"
                                 and bool(targets.spatial_valid[0, index])), None)
            predicted_mask = trace.get("continuous_mask")
            if target_token is not None and predicted_mask is not None:
                row["line_quality"] = _quality(
                    predicted_mask.to(device=targets.mask.device),
                    targets.mask[0, target_token],
                )
        page_steps.append(row)

    eligible_quality = [row["line_quality"] for row in page_steps if row["line_quality"] is not None]
    denominator = len(pred_positions)
    page_alignment = {
        "decoded_token_count": len(token_ids),
        "decoded_nonspace_characters": len(pred_positions),
        "token_character_mapping_reliable": token_map_reliable,
        "token_character_mapping_failure": mapping_failure,
        "reliable_spatial_character_count": len(eligible_quality),
        "alignment_coverage": len(eligible_quality) / max(1, denominator),
        "ambiguous_token_count": sum(row["alignment_ambiguous"] for row in page_steps),
        "multi_character_token_count": sum(row["multi_character_token"] for row in page_steps),
        "cross_line_token_count": sum(row["cross_line_token"] is True for row in page_steps),
        "insertion_token_count": sum(row["insertion"] for row in page_steps),
        "continuous_line_iou": (sum(value["continuous_line_iou"] for value in eligible_quality)
                                / len(eligible_quality) if eligible_quality else None),
        "in_line_probability_mass": (sum(value["in_line_probability_mass"] for value in eligible_quality)
                                     / len(eligible_quality) if eligible_quality else None),
        "continuous_mass_ratio": (sum(value["continuous_mass_ratio"] for value in eligible_quality)
                                  / len(eligible_quality) if eligible_quality else None),
        "binary_line_iou": (sum(value["binary_line_iou"] for value in eligible_quality)
                            / len(eligible_quality) if eligible_quality else None),
    }
    return page_steps, page_alignment


def main() -> None:
    args = parse_args()
    if args.shard_count != 5 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("the preregistered diagnosis requires five disjoint shards")
    if not args.device.startswith("cuda:"):
        raise ValueError("the fixed-weight diagnosis requires a CUDA worker")
    args.code_root = args.code_root.resolve()
    sys.path.insert(0, str(args.code_root / "src"))

    protocol = json.loads(args.diagnostic_protocol.read_text(encoding="utf-8"))
    evaluation_stage = protocol.get("evaluation_stage")
    stage_expectations = {
        "diagnostic32": (32, "locked_before_candidate_inference"),
        "full_val_tune": (VAL_TUNE_PAGES.get(protocol.get("dataset")),
                          "locked_after_32_page_screen_before_full_inference"),
    }
    if evaluation_stage not in stage_expectations:
        raise ValueError("protocol evaluation_stage must be diagnostic32 or full_val_tune")
    expected_page_count, expected_status = stage_expectations[evaluation_stage]
    if expected_page_count is None:
        raise ValueError("full val_tune protocol names an unsupported dataset")
    records = load_records(args.val_tune_manifest)
    if (len(records) != expected_page_count
            or any(record.get("split", record.get("official_split")) != "validation"
                   for record in records)):
        raise ValueError(f"{evaluation_stage} must contain {expected_page_count} val_tune pages")
    if (protocol.get("status") != expected_status
            or protocol.get("evaluation_role") != "val_tune"
            or protocol.get("val_verify_manifest_read") is not False
            or protocol.get("val_verify_used_for_selection") is not False
            or protocol.get("test_manifest_read") is not False
            or protocol.get("test_used_for_selection") is not False
            or protocol.get("diagnostic_manifest_sha256") != sha256_file(args.val_tune_manifest)):
        raise ValueError("protocol does not lock this isolated val_tune manifest")
    if protocol.get("selection", {}).get("selected_page_ids") != [str(r["page_id"]) for r in records]:
        raise ValueError("val_tune manifest page order or membership differs from locked protocol")
    # The arm reads line-level targets, so the manifest must actually supply a
    # line grouping.  The locked evidence says which field does that for this
    # domain; a manifest that does not match it is refused rather than falling
    # back to `auto`, which would silently produce per-character targets.
    line_evidence = json.loads(args.line_evidence.read_text(encoding="utf-8"))
    if (sha256_file(args.line_evidence) != protocol.get("line_evidence_sha256")
            # The evidence file carries the status of the stage that wrote it, so
            # it is checked against this stage rather than against one fixed
            # string -- otherwise every full_val_tune worker would reject the
            # evidence its own prepare step just wrote.
            or line_evidence.get("status") != expected_status
            or line_evidence.get("evaluation_stage") != evaluation_stage
            or line_evidence.get("evaluation_role") != "val_tune"
            or line_evidence.get("dataset") != protocol.get("dataset")
            or line_evidence.get("diagnostic_manifest_sha256")
            != protocol.get("diagnostic_manifest_sha256")
            or line_evidence.get("test_manifest_read") is not False
            or line_evidence.get("test_used_for_selection") is not False):
        raise ValueError("line evidence does not match the locked val_tune protocol")
    spatial_targets = line_evidence.get("spatial_targets")
    if spatial_targets not in SPATIAL_TARGET_LINE_SOURCES:
        raise ValueError(
            f"{protocol.get('dataset')} has no verified line evidence "
            f"({line_evidence.get('unavailable_reason') or spatial_targets!r}); the D1-D3 arms "
            "supervise line targets and cannot run without it"
        )
    line_source = SPATIAL_TARGET_LINE_SOURCES[spatial_targets]
    # The evidence is computed over the domain's whole val_tune manifest, while a
    # stage evaluates either a subset of it (diagnostic32) or all of it
    # (full_val_tune).  The invariant is therefore that the *source* manifest is
    # fully mapped -- comparing against this stage's own page count would fail on
    # the 32-page subset, whose pages are a sample drawn from those 240/80.
    if (line_evidence.get("pages") != protocol.get("val_tune_pages")
            or line_evidence.get("pages_with_line_mapping") != line_evidence.get("pages")
            or line_evidence.get("unmappable_pages")):
        raise ValueError(
            "line evidence does not cover every page of the locked val_tune manifest"
        )
    # The evidence covers the source manifest, but this stage's required pages
    # must still be a bounded subset of what the evidence actually walked.  An
    # oversized batch would fail the per-page walk anyway; catching it here names
    # the cause instead of surfacing as a bare IndexError deep in the loop.
    stage_pages = len(records)
    if stage_pages > line_evidence["pages"]:
        raise ValueError(
            f"{evaluation_stage} evaluates {stage_pages} pages but the locked line evidence "
            f"covers only {line_evidence['pages']}"
        )
    base_fingerprint = json.loads(
        args.base_model_weights_fingerprint.read_text(encoding="utf-8")
    )
    lora_fingerprint = json.loads(args.decoder_lora_fingerprint.read_text(encoding="utf-8"))
    shared_start = protocol.get("shared_start", {})
    if (base_fingerprint.get("model_revision") != shared_start.get("model_revision")
            or base_fingerprint.get("model_weights_sha256")
            != shared_start.get("base_model_weights_sha256")
            or Path(base_fingerprint.get("model_path", "")).resolve() != args.model_path.resolve()
            or sha256_file(args.base_model_weights_fingerprint)
            != protocol.get("base_model_weights_fingerprint_manifest_sha256")):
        raise ValueError("raw GLM-OCR model does not match the locked base weight fingerprint")
    if (shared_start.get("decoder_lora_loaded") is not True
            or lora_fingerprint.get("decoder_lora_loaded") is not True
            or lora_fingerprint.get("decoder_lora_sha256") != EXPECTED_DECODER_LORA_SHA256
            or lora_fingerprint.get("decoder_lora_sha256") != shared_start.get("decoder_lora_sha256")
            or lora_fingerprint.get("checkpoint_path") != str(args.decoder_lora_checkpoint.resolve())
            or lora_fingerprint.get("model_revision") != shared_start.get("model_revision")
            or lora_fingerprint.get("rank") != 8
            or lora_fingerprint.get("alpha") != 8.0
            or lora_fingerprint.get("dropout") != 0.0
            or sha256_file(args.decoder_lora_fingerprint)
            != protocol.get("decoder_lora_fingerprint_manifest_sha256")):
        raise ValueError("diagnostic decoder LoRA does not match the paired v2 checkpoint")
    if sha256_file(args.decoder_lora_checkpoint / "decoder_lora.safetensors") != EXPECTED_DECODER_LORA_SHA256:
        raise ValueError("decoder LoRA checkpoint SHA mismatch")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)

    config_path = args.code_root / "configs/line_mask_v3/diagnostic.json"
    if sha256_file(config_path) != protocol.get("diagnostic_config_sha256"):
        raise ValueError("diagnostic config differs from the configuration locked with the manifest")
    diagnostic_config = json.loads(config_path.read_text(encoding="utf-8"))
    if diagnostic_config.get("shared_start", {}).get("line_mask_head_sha256") != EXPECTED_HEAD_SHA256:
        raise ValueError("diagnostic config does not name the fixed epoch8/step3456 head")
    expected_arms = {
        "D0": {"mask": "disabled", "threshold": None, "injection_layers": [], "bias": 0.0},
        "D1": {"mask": "hard", "threshold": 0.5, "injection_layers": "all", "bias": 1.0},
        "D2": {"mask": "hard", "threshold": 0.5, "injection_layers": "all", "bias": 0.5},
        "D3": {"mask": "hard", "threshold": 0.5,
               "injection_layers": "indices_ge_floor_layers_div_2", "bias": 1.0},
    }
    if diagnostic_config.get("arms") != expected_arms:
        raise ValueError("diagnostic config differs from the preregistered D0-D3 arms")
    protocol_arms = protocol.get("arms")
    if not isinstance(protocol_arms, dict):
        raise TypeError("protocol arms must be a locked object")
    if evaluation_stage == "diagnostic32":
        expected_active_arms = list(ARMS)
    else:
        candidate = protocol.get("candidate_selected_from_screen")
        expected_active_arms = ["D0", "D1"] + ([candidate] if candidate else [])
        if candidate not in (None, "D2", "D3"):
            raise ValueError("full val_tune selected an unsupported screening candidate")
    if list(protocol_arms) != expected_active_arms:
        raise ValueError("active arms differ from the preregistered stage selection")
    if any(protocol_arms[name] != expected_arms[name] for name in expected_active_arms):
        raise ValueError("active diagnostic arm settings differ from the locked config")
    active_arms = {name: ARMS[name] for name in expected_active_arms}
    if diagnostic_config.get("generation", {}).get("attention_backend") != "torch_sdpa_math":
        raise ValueError("diagnosis requires torch SDPA with the math backend")
    source_fingerprint_path = args.code_root.parent / "source-fingerprint.json"
    if not source_fingerprint_path.is_file():
        raise FileNotFoundError("diagnostic code must come from the immutable run source snapshot")
    source_snapshot_sha256 = json.loads(
        source_fingerprint_path.read_text(encoding="utf-8")
    )["source_tree_sha256"]
    if protocol.get("source_snapshot_sha256") != source_snapshot_sha256:
        raise ValueError("diagnostic source snapshot differs from the locked protocol")

    checkpoint_sha = sha256_file(args.mask_checkpoint)
    if checkpoint_sha != EXPECTED_HEAD_SHA256:
        raise ValueError(f"line-mask checkpoint SHA mismatch: {checkpoint_sha}")
    checkpoint = torch.load(args.mask_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 8 or checkpoint.get("step") != 3456:
        raise ValueError("D arms must share the v2 epoch8/step3456 head")
    config = LineMaskConfig(**checkpoint["config"])

    random.seed(42)
    torch.manual_seed(42)
    model, processor = load_diagnostic_backbone(
        args.model_path, args.decoder_lora_checkpoint, args.device
    )
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise ValueError("diagnosis requires Transformers SDPA before forcing the math kernel")
    head = LineMaskHead(config).to(args.device)
    head.load_state_dict(checkpoint["head"], strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in head.state_dict().values()):
        raise FloatingPointError("selected v2 line-mask checkpoint contains non-finite weights")
    head.eval()
    eos_set = set(eos_ids(model, processor))
    if not eos_set:
        raise ValueError("model has no EOS token id")

    shard_records = records[args.shard_index::args.shard_count]
    if not shard_records:
        raise ValueError("five-way val_tune sharding produced an empty worker shard")
    if any(record.get("split", record.get("official_split")) == "test" for record in shard_records):
        raise ValueError("test data must not enter this diagnosis")

    layer_count = len(model.model.language_model.layers)
    if layer_count < 1:
        raise RuntimeError("could not locate the model decoder layers")
    protocol_copy = {
        "status": "running",
        "domain": protocol["dataset"],
        "evaluation_stage": evaluation_stage,
        "evaluation_pages": len(records),
        "active_arms": expected_active_arms,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "page_ids": [str(record["page_id"]) for record in shard_records],
        "evaluation_role": "val_tune",
        "diagnostic_manifest_sha256": sha256_file(args.val_tune_manifest),
        "train_manifest_sha256": protocol["train_manifest_sha256"],
        "val_tune_manifest_sha256": protocol["val_tune_manifest_sha256"],
        "val_verify_manifest_sha256": None,
        "val_verify_manifest_read": False,
        "val_verify_used_for_selection": False,
        "mask_checkpoint_sha256": checkpoint_sha,
        "base_model_weights_sha256": base_fingerprint["model_weights_sha256"],
        "base_model_weights_fingerprint_manifest_sha256": sha256_file(
            args.base_model_weights_fingerprint
        ),
        "decoder_lora_sha256": lora_fingerprint["decoder_lora_sha256"],
        "decoder_lora_fingerprint_manifest_sha256": sha256_file(args.decoder_lora_fingerprint),
        "decoder_lora_checkpoint": str(args.decoder_lora_checkpoint.resolve()),
        "decoder_lora_loaded": True,
        "shared_start": protocol["shared_start"],
        "diagnostic_config_sha256": protocol["diagnostic_config_sha256"],
        "line_evidence_sha256": sha256_file(args.line_evidence),
        "line_source": line_source,
        "spatial_targets": spatial_targets,
        "spatial_target_granularity": line_evidence["box_granularity"],
        "pages_with_unavailable_line_mapping": 0,
        "source_snapshot_sha256": source_snapshot_sha256,
        "attention_backend": diagnostic_config["generation"]["attention_backend"],
        "model_revision": diagnostic_config["shared_start"]["model_revision"],
        "source_code": {
            "runtime_sha256": sha256_file(args.code_root / "src/layout_ocr/line_mask_runtime.py"),
            "head_sha256": sha256_file(args.code_root / "src/layout_ocr/line_mask_head.py"),
            "evaluator_sha256": sha256_file(Path(__file__)),
        },
        "decoder_layers": layer_count,
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
    write_json(args.output_root / "worker_protocol.json", protocol_copy)

    trace_page_ids = set(protocol["selection"]["full_mask_trace_page_ids"])
    all_rows: dict[str, list[dict]] = {arm: [] for arm in active_arms}
    all_steps: dict[str, list[dict]] = {arm: [] for arm in active_arms}
    started = time.time()
    for page_number, record in enumerate(shard_records, 1):
        inputs = prepare_inference_inputs(processor, {"image_path": record["image_path"]},
                                          torch.device(args.device))
        prompt_length = int(inputs["input_ids"].shape[1])
        page_traces = {}
        for arm_name, arm in active_arms.items():
            runtime = LineMaskRuntime(model, processor.tokenizer, head, bias=arm["bias"],
                                      layer_scope=arm["layer_scope"])
            runtime.enabled = arm["enabled"]
            runtime.set_page(inputs, str(record["page_id"]),
                             capture_trace=arm["enabled"])
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                generated = model.generate(
                    **inputs,
                    max_new_tokens=1536,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=sorted(eos_set),
                )
            token_tensor = generated[0, prompt_length:]
            token_ids = [int(token) for token in token_tensor.tolist()]
            prediction = processor.tokenizer.decode(token_tensor, skip_special_tokens=True)
            trace_by_position = {int(step["generation_position"]): step
                                 for step in runtime.trace_steps}
            if arm["enabled"] and len(token_ids) > 1:
                expected_positions = set(range(1, len(token_ids)))
                if set(trace_by_position) != expected_positions:
                    raise RuntimeError(
                        f"{arm_name} cached-decode trace positions differ from generated tokens: "
                        f"expected={sorted(expected_positions)}, got={sorted(trace_by_position)}"
                    )
                if any(not step["mask_applied"] for step in runtime.trace_steps):
                    raise RuntimeError(f"{arm_name} failed to apply a mask on a cached decode step")

            training = prepare_training_inputs(processor, record, torch.device(args.device), eos_set)
            target_start = int((training["labels"] == -100).int().cumprod(1).sum())
            target_ids = [int(value) for value in training["input_ids"][0, target_start:].tolist()]
            targets = build_mask_targets(
                processor.tokenizer, record, target_ids, eos_set, runtime.xywh,
                target_mode="line", line_source=line_source, raster_mode="hard",
            )
            # GT is used only after generation to score masks and character alignment.
            page_steps, alignment_summary = _token_annotations(
                record, token_ids, processor.tokenizer, prediction,
                trace_by_position, targets, eos_set,
            )
            row = {
                "page_id": str(record["page_id"]),
                "reference": record["page_text"],
                "prediction": prediction,
                "generation_tokens": len(token_ids),
                "generation_eos_hit": any(token in eos_set for token in token_ids),
                "generation_limit_hit": not any(token in eos_set for token in token_ids),
                "repetition": repetition_diagnostics(prediction),
                "arm": arm_name,
                "evaluation_stage": evaluation_stage,
                "evaluation_role": "val_tune",
                "routing_report": runtime.route.report(),
                "alignment": alignment_summary,
                "spatial_targets": spatial_targets,
                "spatial_target_granularity": targets.line_evidence,
                "window_report": targets.window_report,
                "token_steps": page_steps,
                "reads_ground_truth_for_routing": False,
                "test_manifest_read": False,
                "test_used_for_selection": False,
            }
            all_rows[arm_name].append(row)
            all_steps[arm_name].extend(page_steps)
            if arm_name != "D0" and str(record["page_id"]) in trace_page_ids:
                tensors = [step["continuous_mask"].detach().cpu().numpy().astype(np.float16)
                           for step in runtime.trace_steps]
                trace_dir = args.output_root / "full-mask-traces" / arm_name
                trace_dir.mkdir(parents=True, exist_ok=True)
                safe_name = hashlib.sha256(str(record["page_id"]).encode("utf-8")).hexdigest()[:16]
                mask_array = (np.stack(tensors) if tensors else
                              np.zeros((0, int(runtime.xywh.shape[1])), dtype=np.float16))
                np.savez_compressed(
                    trace_dir / f"{safe_name}.npz",
                    page_id=np.asarray(str(record["page_id"])),
                    generation_positions=np.asarray([step["generation_position"]
                                                     for step in runtime.trace_steps], dtype=np.int32),
                    token_ids=np.asarray([token_ids[step["generation_position"]]
                                          for step in runtime.trace_steps], dtype=np.int32),
                    source_generation_positions=np.asarray([
                        step["mask_source_generation_position"] for step in runtime.trace_steps
                    ], dtype=np.int32),
                    update_gates=np.asarray([step["update_gate"] for step in runtime.trace_steps],
                                            dtype=np.float32),
                    mask_change_mean_abs=np.asarray([
                        step["mask_change_mean_abs"] for step in runtime.trace_steps
                    ], dtype=np.float32),
                    binary_areas=np.asarray([step["binary_area"] for step in runtime.trace_steps],
                                            dtype=np.int32),
                    continuous_masks=mask_array,
                )
            page_traces[arm_name] = runtime.route.report()
            runtime.remove()
            del targets, training, runtime
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            prediction_path = args.output_root / f"predictions-{arm_name}.jsonl"
            with prediction_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            write_json(args.output_root / f"progress-{arm_name}.json", {
                "status": "running", "pages": page_number, "total": len(shard_records),
                "last_page_id": str(record["page_id"]), "time": time.time(),
            })

        print(json.dumps({"domain": protocol["dataset"], "shard": args.shard_index,
                          "page": page_number, "total": len(shard_records),
                          "page_id": str(record["page_id"]), "arm_routes": page_traces},
                         ensure_ascii=False), flush=True)

    for arm_name, rows in all_rows.items():
        metrics = aggregate_ocr_metrics(((row["reference"], row["prediction"]) for row in rows), Counter())
        metrics.update({
            "eos_pages": sum(row["generation_eos_hit"] for row in rows),
            "generation_limit_hits": sum(row["generation_limit_hit"] for row in rows),
            "loop_pages": sum(bool(row["repetition"].get("repeated_cycle_detected")) for row in rows),
        })
        metrics["loop_rate"] = metrics["loop_pages"] / max(1, metrics["pages"])
        aligned_steps = [step for step in all_steps[arm_name] if step["line_quality"] is not None]
        summary = {
            "status": "complete",
            "domain": protocol["dataset"],
            "evaluation_stage": evaluation_stage,
            "diagnostic_pages": len(records),
            "arm": arm_name,
            "metrics": metrics,
            "mean_continuous_line_iou": (
                sum(step["line_quality"]["continuous_line_iou"] for step in aligned_steps)
                / len(aligned_steps) if aligned_steps else None
            ),
            "reliable_spatial_positions": len(aligned_steps),
            "mean_alignment_coverage": sum(row["alignment"]["alignment_coverage"] for row in rows)
            / max(1, len(rows)),
            "decoder_layers": layer_count,
            "injection_layers": list(range(layer_count)) if arm_name in ("D1", "D2")
            else (list(range(layer_count // 2, layer_count)) if arm_name == "D3" else []),
            "bias": arm["bias"],
            "mask_threshold": config.threshold,
            "elapsed_seconds": time.time() - started,
            "evaluation_role": "val_tune",
            "diagnostic_manifest_sha256": sha256_file(args.val_tune_manifest),
            "val_tune_manifest_sha256": protocol_copy["val_tune_manifest_sha256"],
            "val_verify_manifest_sha256": None,
            "val_verify_manifest_read": False,
            "val_verify_used_for_selection": False,
            "mask_checkpoint_sha256": checkpoint_sha,
            "base_model_weights_sha256": protocol_copy["base_model_weights_sha256"],
            "base_model_weights_fingerprint_manifest_sha256": (
                protocol_copy["base_model_weights_fingerprint_manifest_sha256"]
            ),
            "decoder_lora_sha256": protocol_copy["decoder_lora_sha256"],
            "decoder_lora_fingerprint_manifest_sha256": (
                protocol_copy["decoder_lora_fingerprint_manifest_sha256"]
            ),
            "decoder_lora_loaded": True,
            "attention_backend": protocol_copy["attention_backend"],
            "line_evidence_sha256": protocol_copy["line_evidence_sha256"],
            "line_source": line_source,
            "spatial_targets": spatial_targets,
            "spatial_target_granularity": line_evidence["box_granularity"],
            # Spatial supervision is only meaningful where the manifest actually
            # supplied a line box.  ``alignment_coverage`` counts positions the
            # diagnostic rejected; these count the geometry that was never there.
            "pages_with_unavailable_line_mapping": sum(
                1 for row in rows if _line_mapping_incomplete(row["window_report"])
            ),
            "supervised_spatial_tokens": sum(
                int(row["alignment"]["reliable_spatial_character_count"]) for row in rows
            ),
            "test_manifest_read": False,
            "test_used_for_selection": False,
        }
        write_json(args.output_root / f"summary-{arm_name}.json", summary)
        write_json(args.output_root / f"status-{arm_name}.json", {
            "status": "complete", "arm": arm_name, "pages": len(rows),
            "evaluation_stage": evaluation_stage, "time": time.time(),
        })
    write_json(args.output_root / "worker_status.json", {
        "status": "complete", "shard_index": args.shard_index,
        "pages": len(shard_records), "evaluation_stage": evaluation_stage, "time": time.time(),
    })


if __name__ == "__main__":
    main()
