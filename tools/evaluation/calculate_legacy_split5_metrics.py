#!/usr/bin/env python3
"""Compute compact, reproducible metrics from one GOT myeval JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import Levenshtein as _levenshtein

    EDITOPS_BACKEND = "python-Levenshtein"
except ImportError:
    try:
        from rapidfuzz.distance import Levenshtein as _levenshtein

        EDITOPS_BACKEND = "rapidfuzz"
    except ImportError:
        _levenshtein = None
        EDITOPS_BACKEND = "pure_python_dp_tie_break_sub_delete_insert"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_line_endings(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def compact_text(value: str) -> str:
    value = normalize_line_endings(value)
    value = "".join(ch for ch in value if not ch.isspace())
    return value.replace("\\n", "").replace("\\r", "")


def edit_counts(reference: str, hypothesis: str) -> Dict[str, int]:
    if _levenshtein is not None:
        operations = _levenshtein.editops(reference, hypothesis)
        substitutions = sum(op[0] == "replace" for op in operations)
        deletions = sum(op[0] == "delete" for op in operations)
        insertions = sum(op[0] == "insert" for op in operations)
    else:
        substitutions, deletions, insertions = _pure_edit_counts(reference, hypothesis)
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "edit_distance": substitutions + deletions + insertions,
    }


def _pure_edit_counts(reference: str, hypothesis: str) -> Tuple[int, int, int]:
    """Exact unit-cost Levenshtein counts with bounded per-page memory."""
    n = len(reference)
    m = len(hypothesis)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0

    # One byte per DP parent direction; score rows are discarded as we go.
    parents = [bytearray(m + 1) for _ in range(n + 1)]
    for column in range(1, m + 1):
        parents[0][column] = 3  # insert
    previous = list(range(m + 1))
    for row in range(1, n + 1):
        current = [row] + [0] * m
        parents[row][0] = 2  # delete
        source_char = reference[row - 1]
        for column in range(1, m + 1):
            if source_char == hypothesis[column - 1]:
                current[column] = previous[column - 1]
                parents[row][column] = 0  # match
                continue
            substitute = previous[column - 1] + 1
            delete = previous[column] + 1
            insert = current[column - 1] + 1
            if substitute <= delete and substitute <= insert:
                current[column] = substitute
                parents[row][column] = 1
            elif delete <= insert:
                current[column] = delete
                parents[row][column] = 2
            else:
                current[column] = insert
                parents[row][column] = 3
        previous = current

    substitutions = deletions = insertions = 0
    row, column = n, m
    while row or column:
        direction = parents[row][column]
        if direction == 0:
            row -= 1
            column -= 1
        elif direction == 1:
            substitutions += 1
            row -= 1
            column -= 1
        elif direction == 2:
            deletions += 1
            row -= 1
        elif direction == 3:
            insertions += 1
            column -= 1
        else:
            raise RuntimeError(f"invalid edit direction {direction} at {row},{column}")
    return substitutions, deletions, insertions


def row_metrics(reference: str, hypothesis: str) -> Dict[str, float | int]:
    # Preserve the strip + editops macro metric used by the legacy batch script.
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    counts = edit_counts(reference, hypothesis)
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    distance = counts["edit_distance"]
    cer = distance / ref_len if ref_len else (0.0 if hyp_len == 0 else 1.0)
    matches = ref_len - counts["substitutions"] - counts["deletions"]
    precision = matches / hyp_len if hyp_len else 0.0
    recall = matches / ref_len if ref_len else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **counts,
        "reference_characters": ref_len,
        "hypothesis_characters": hyp_len,
        "cer": cer,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def micro_metrics(rows: Iterable[Tuple[str, str]], transform) -> Dict[str, Any]:
    total = {"substitutions": 0, "deletions": 0, "insertions": 0, "edit_distance": 0}
    ref_chars = 0
    hyp_chars = 0
    for reference, hypothesis in rows:
        reference = transform(reference)
        hypothesis = transform(hypothesis)
        counts = edit_counts(reference, hypothesis)
        for key in total:
            total[key] += counts[key]
        ref_chars += len(reference)
        hyp_chars += len(hypothesis)
    matches = ref_chars - total["substitutions"] - total["deletions"]
    precision = matches / hyp_chars if hyp_chars else 0.0
    recall = matches / ref_chars if ref_chars else 0.0
    return {
        "evaluated_pages": len(rows),
        "reference_characters": ref_chars,
        "hypothesis_characters": hyp_chars,
        **total,
        "cer": total["edit_distance"] / ref_chars if ref_chars else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    with args.labels.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(predictions, list) or not isinstance(labels, list):
        raise TypeError("predictions and labels must be JSON lists")
    if len(predictions) != len(labels):
        raise ValueError(f"prediction/label count mismatch: {len(predictions)} != {len(labels)}")

    pairs: List[Tuple[str, str]] = []
    page_rows: List[Dict[str, Any]] = []
    token_lengths: List[int] = []
    tokenizer_error = None
    tokenizer = None
    if args.tokenizer is not None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(args.tokenizer), trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:  # Keep metrics usable if tokenizer loading changes.
            tokenizer_error = f"{type(exc).__name__}: {exc}"

    for index, (prediction, label_record) in enumerate(zip(predictions, labels)):
        reference = str(label_record.get("conversations", [{}, {"value": ""}])[1].get("value", ""))
        answer = str(prediction.get("answer", ""))
        pairs.append((reference, answer))
        metrics = row_metrics(reference, answer)
        token_length = None
        if tokenizer is not None:
            token_length = len(tokenizer.encode(answer, add_special_tokens=False))
            token_lengths.append(token_length)
        page_rows.append(
            {
                "index": index,
                "image": prediction.get("image", label_record.get("image", "")),
                "metrics": metrics,
                "token_length": token_length,
            }
        )

    macro = {
        "evaluated_pages": len(page_rows),
        "cer": statistics.fmean(float(row["metrics"]["cer"]) for row in page_rows),
        "precision": statistics.fmean(float(row["metrics"]["precision"]) for row in page_rows),
        "recall": statistics.fmean(float(row["metrics"]["recall"]) for row in page_rows),
        "f1": statistics.fmean(float(row["metrics"]["f1"]) for row in page_rows),
    }
    raw = micro_metrics(pairs, normalize_line_endings)
    compact = micro_metrics(pairs, compact_text)
    exact_pages = sum(row["metrics"]["edit_distance"] == 0 for row in page_rows)
    empty_answers = sum(not answer.strip() for _, answer in pairs)
    worst = sorted(page_rows, key=lambda row: float(row["metrics"]["cer"]), reverse=True)[:10]

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "editops_backend": EDITOPS_BACKEND,
        "predictions_sha256": sha256_file(args.predictions),
        "labels_sha256": sha256_file(args.labels),
        "coverage": {
            "prediction_pages": len(predictions),
            "label_pages": len(labels),
            "exact_match_pages": exact_pages,
            "empty_answer_pages": empty_answers,
        },
        "metrics_page_macro_legacy_editops": macro,
        "metrics_corpus_micro_raw_line_endings": raw,
        "metrics_corpus_micro_compact": compact,
        "tokenization": {
            "tokenizer_path": str(args.tokenizer) if args.tokenizer else None,
            "tokenizer_error": tokenizer_error,
            "max_answer_tokens": max(token_lengths) if token_lengths else None,
            "pages_at_or_above_4096_tokens": sum(length >= 4096 for length in token_lengths),
        },
        "worst_pages_by_macro_cer": worst,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "report.txt").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2))
        handle.write("\n")
    print(
        "METRICS_OK pages={} macro_cer={:.6f} raw_cer={:.6f} compact_cer={:.6f} output_bytes={}".format(
            len(predictions), macro["cer"], raw["cer"], compact["cer"], summary_path.stat().st_size
        )
    )


if __name__ == "__main__":
    main()
