from __future__ import annotations

from collections import Counter
from typing import Iterable


def levenshtein_alignment(reference: str, prediction: str) -> tuple[int, Counter[str]]:
    rows = len(reference) + 1
    cols = len(prediction) + 1
    costs = [[0] * cols for _ in range(rows)]
    moves = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], moves[i][0] = i, "D"
    for j in range(1, cols):
        costs[0][j], moves[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1])
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            best = min((substitution, "M"), (deletion, "D"), (insertion, "I"))
            costs[i][j], moves[i][j] = best
    matches: Counter[str] = Counter()
    i, j = len(reference), len(prediction)
    while i or j:
        move = moves[i][j]
        if move == "M":
            if reference[i - 1] == prediction[j - 1]:
                matches[reference[i - 1]] += 1
            i -= 1
            j -= 1
        elif move == "D":
            i -= 1
        else:
            j -= 1
    return costs[-1][-1], matches


def aggregate_ocr_metrics(
    pairs: Iterable[tuple[str, str]], train_character_counts: Counter[str]
) -> dict[str, float | int | None]:
    """Aggregate CER and diagnostics for characters seen at most *k* times.

    These are low-frequency diagnostics, not a controlled K-shot protocol:
    they do not constrain support/query examples or guarantee symbol-level
    disjointness.  The legacy ``r2_k*`` aliases remain for old summaries.
    """
    pairs = list(pairs)
    total_edits = 0
    total_characters = 0
    exact = 0
    rare_reference = {k: 0 for k in (1, 3, 5)}
    rare_matches = {k: 0 for k in (1, 3, 5)}
    for reference, prediction in pairs:
        reference = "".join(reference.split())
        prediction = "".join(prediction.split())
        edits, matches = levenshtein_alignment(reference, prediction)
        total_edits += edits
        total_characters += len(reference)
        exact += reference == prediction
        reference_counts = Counter(reference)
        for k in (1, 3, 5):
            rare = {char for char, count in train_character_counts.items() if 0 < count <= k}
            rare_reference[k] += sum(reference_counts[char] for char in rare)
            rare_matches[k] += sum(matches[char] for char in rare)
    result: dict[str, float | int | None] = {
        "pages": len(pairs),
        "reference_characters": total_characters,
        "character_errors": total_edits,
        "cer": total_edits / max(1, total_characters),
        "exact_page_rate": exact / max(1, len(pairs)),
    }
    for k in (1, 3, 5):
        rare_types = sum(
            1 for count in train_character_counts.values() if 0 < count <= k
        )
        recall = (
            rare_matches[k] / rare_reference[k] if rare_reference[k] else None
        )
        result[f"low_frequency_k{k}_character_types"] = rare_types
        result[f"low_frequency_k{k}_reference_characters"] = rare_reference[k]
        result[f"low_frequency_k{k}_recall"] = recall
        result[f"r2_k{k}_reference_characters"] = rare_reference[k]
        result[f"r2_k{k}_recall"] = recall
    return result
