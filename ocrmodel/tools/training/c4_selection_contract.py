#!/usr/bin/env python3
"""Validate and resolve a frozen AncientDoc C4 checkpoint selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Sequence


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_safetensors_structure(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > file_size - 8:
            raise ValueError(f"Invalid safetensors header size: {path}")
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid safetensors JSON header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not an object: {path}")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ValueError(f"Safetensors checkpoint has no tensors: {path}")
    data_bytes = file_size - 8 - header_size
    for key, value in tensors.items():
        if not isinstance(value, dict):
            raise ValueError(f"Invalid safetensors entry for {key}: {path}")
        offsets = value.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_bytes
        ):
            raise ValueError(f"Invalid safetensors offsets for {key}: {path}")
    required_fragments = ("model.layers.", "mm_projector_vary.", "layout_adapter.")
    missing = [
        fragment
        for fragment in required_fragments
        if not any(fragment in key for key in tensors)
    ]
    if missing:
        raise ValueError(f"Incomplete C4 safetensors state; missing {missing}: {path}")
    return {"tensor_count": len(tensors), "header_bytes": header_size}


def checkpoint_files(model_path: Path) -> tuple[Path, Path]:
    model_path = model_path.expanduser().resolve()
    config = model_path / "config.json"
    weights = next((model_path / name for name in WEIGHT_NAMES if (model_path / name).is_file()), None)
    if not config.is_file() or weights is None or weights.stat().st_size < 1:
        raise FileNotFoundError(
            f"Incomplete checkpoint under {model_path}: config={config.is_file()}, weights={weights}"
        )
    config_payload = read_json(config)
    if config_payload.get("use_vlqa") is not True:
        raise ValueError(f"Selected C4 checkpoint does not enable VLQA: {config}")
    if int(config_payload.get("vlqa_num_queries", -1)) < 1:
        raise ValueError(f"Selected C4 checkpoint has invalid vlqa_num_queries: {config}")
    if weights.name == "model.safetensors":
        validate_safetensors_structure(weights)
    return config, weights


def checkpoint_provenance(model_path: Path) -> dict[str, Any]:
    model_path = model_path.expanduser().resolve()
    config, weights = checkpoint_files(model_path)
    result = {
        "model_path": str(model_path),
        "config_path": str(config),
        "config_sha256": file_sha256(config),
        "weights_path": str(weights),
        "weights_sha256": file_sha256(weights),
        "weights_bytes": weights.stat().st_size,
    }
    if weights.name == "model.safetensors":
        result["safetensors"] = validate_safetensors_structure(weights)
    return result


def load_c4_selection(
    selection_path: Path,
    *,
    verify_hashes: bool = True,
    require_formal_validation: bool = True,
) -> dict[str, Any]:
    selection_path = selection_path.expanduser().resolve()
    payload = read_json(selection_path)
    if payload.get("status") != "ok" or payload.get("purpose") != "c4_checkpoint_selection":
        raise ValueError(f"Not a completed C4 selection: {selection_path}")
    if payload.get("selection_split") != "validation":
        raise ValueError("C4 selection must use split='validation'.")
    if payload.get("test_used_for_selection") is not False:
        raise ValueError("C4 selection must explicitly report test_used_for_selection=false.")
    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("C4 selection has no evaluator protocol.")
    required_protocol = {
        "prompt": "OCR: ",
        "decoding": "greedy",
        "max_new_tokens": 2048,
        "no_repeat_ngram_size": 20,
        "batch_size": 1,
        "layout_metadata_as_model_input": False,
    }
    for key, expected in required_protocol.items():
        if evaluator.get(key) != expected:
            raise ValueError(
                f"C4 selection evaluator mismatch for {key}: {evaluator.get(key)!r} != {expected!r}"
            )
    if require_formal_validation and int(evaluator.get("max_records", -1)) != 0:
        raise ValueError("Replay training requires a full validation C4 selection (max_records=0).")

    selected = payload.get("selected")
    candidates = payload.get("candidates")
    if not isinstance(selected, dict) or not isinstance(candidates, list):
        raise ValueError("C4 selection must contain selected and candidates.")
    model_value = selected.get("model_path")
    step_value = selected.get("optimizer_step")
    metrics = selected.get("validation_metrics")
    provenance = selected.get("provenance")
    if not isinstance(model_value, str) or not isinstance(step_value, int):
        raise ValueError("Selected C4 model path or optimizer step is invalid.")
    if not isinstance(metrics, dict) or not isinstance(provenance, dict):
        raise ValueError("Selected C4 metrics or provenance is missing.")
    page_cer = float(metrics.get("page_cer", float("nan")))
    whitespace_cer = float(metrics.get("whitespace_normalized_page_cer", float("nan")))
    if not math.isfinite(page_cer) or not math.isfinite(whitespace_cer):
        raise ValueError("Selected C4 validation CER is not finite.")
    matching = [
        item for item in candidates
        if isinstance(item, dict)
        and item.get("model_path") == model_value
        and item.get("optimizer_step") == step_value
    ]
    if len(matching) != 1:
        raise ValueError("Selected C4 checkpoint is not a unique candidate.")

    c4_run_root = Path(str(payload.get("c4_run_root", ""))).expanduser().resolve()
    model_root = c4_run_root / "p2" / "model"
    model_path = Path(model_value).expanduser().resolve()
    if model_path != model_root and model_path.parent != model_root:
        raise ValueError("Selected C4 model is outside its declared parent C4 run.")
    parent_metrics_path = model_root / "layout_training_metrics.json"
    parent_metrics = read_json(parent_metrics_path)
    if parent_metrics.get("layout_stage") != "p2":
        raise ValueError("Parent C4 run does not report layout_stage='p2'.")
    if int(parent_metrics.get("global_step", -1)) < step_value:
        raise ValueError("Selected C4 step exceeds the completed parent optimizer steps.")
    name_match = CHECKPOINT_RE.fullmatch(model_path.name)
    if name_match is not None and int(name_match.group(1)) != step_value:
        raise ValueError("Selected checkpoint directory and optimizer step disagree.")
    if name_match is not None:
        trainer_state_path = model_path / "trainer_state.json"
        trainer_state = read_json(trainer_state_path)
        if int(trainer_state.get("global_step", -1)) != step_value:
            raise ValueError("Selected checkpoint trainer_state step disagrees.")
        if verify_hashes and provenance.get("trainer_state_sha256") != file_sha256(
            trainer_state_path
        ):
            raise ValueError("Selected C4 trainer_state provenance mismatch.")
    current = checkpoint_provenance(model_path)
    for key in ("config_sha256", "weights_sha256", "weights_bytes"):
        if verify_hashes and provenance.get(key) != current[key]:
            raise ValueError(f"Selected C4 checkpoint provenance mismatch for {key}.")

    return {
        "selection_path": str(selection_path),
        "c4_run_root": str(c4_run_root),
        "selected_model_path": str(model_path),
        "selected_step": step_value,
        "validation_page_cer": page_cer,
        "validation_whitespace_page_cer": whitespace_cer,
        "validation_exact_matches": metrics.get("page_exact_matches"),
        "config_sha256": current["config_sha256"],
        "weights_sha256": current["weights_sha256"],
        "weights_bytes": current["weights_bytes"],
        "manifest": payload.get("manifest"),
        "dataset_root": payload.get("dataset_root"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--allow-partial-validation", action="store_true")
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    resolved = load_c4_selection(
        args.selection,
        verify_hashes=not args.skip_hash_verification,
        require_formal_validation=not args.allow_partial_validation,
    )
    if args.format == "lines":
        for key in (
            "selected_model_path",
            "selected_step",
            "validation_page_cer",
            "validation_whitespace_page_cer",
            "config_sha256",
            "weights_sha256",
            "c4_run_root",
            "selection_path",
        ):
            print(resolved[key])
    else:
        print(json.dumps(resolved, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
