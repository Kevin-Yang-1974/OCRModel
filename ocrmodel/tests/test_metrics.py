from collections import Counter

from layout_ocr.metrics import (
    aggregate_ocr_metrics,
    levenshtein_alignment,
    levenshtein_error_counts,
)


def test_levenshtein_alignment_counts_matches() -> None:
    distance, matches = levenshtein_alignment("天地玄黄", "天玄黄")
    assert distance == 1
    assert matches == Counter("天玄黄")


def test_low_frequency_recall_reports_diagnostic_not_k_shot_protocol() -> None:
    metrics = aggregate_ocr_metrics(
        [("甲乙丙", "甲丙")], Counter({"甲": 1, "乙": 3, "丙": 10})
    )
    assert metrics["cer"] == 1 / 3
    assert metrics["low_frequency_k1_character_types"] == 1
    assert metrics["low_frequency_k1_recall"] == 1.0
    assert metrics["low_frequency_k3_character_types"] == 2
    assert metrics["low_frequency_k3_recall"] == 0.5
    # Keep compatibility with summaries written by the first screen.
    assert metrics["r2_k1_recall"] == 1.0
    assert metrics["r2_k3_recall"] == 0.5


def test_error_components_sum_to_character_errors() -> None:
    counts = levenshtein_error_counts("甲乙丙", "甲丁")
    assert counts == {"deletions": 1, "substitutions": 1}
    metrics = aggregate_ocr_metrics([("甲乙丙", "甲丁")], Counter())
    assert metrics["character_errors"] == 2
    assert metrics["insertions"] + metrics["deletions"] + metrics["substitutions"] == 2
