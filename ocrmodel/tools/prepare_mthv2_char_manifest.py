#!/usr/bin/env python3
"""Add a per-character box channel to a converted MTHv2 manifest.

The converted MTHv2 pages already carry official character boxes, but only inside
the per-page annotation files: the manifest exposes textline regions and nothing
below them.  A spatial routing signal needs one box per *character*, indexed by
reading order, so this tool derives that channel and writes it into a new manifest
alongside the existing one.

The order is not stored anywhere and has to be recovered.  Geometry alone is not
enough -- characters near a column boundary land in the neighbouring line, which
gets 97% of lines right and silently misplaces the rest.  The line ``text`` is the
anchor that closes the gap: concatenating the lines in reading order reproduces the
page text exactly (verified on every page of every MTHv2 split), and the length of
each line is known, so a monotone alignment between "characters sorted
geometrically" and "the page text" cannot let a character cross a line boundary.

What the alignment must get right is count and order, not identity: the source
writes ``#`` for glyphs it could not identify, and a box is still a box.  Identity
mismatches are counted and reported so a page whose annotation is offset shows up
as a number rather than as a silently wrong label.

The output is index-aligned with ``page_text``: entry ``i`` is the box of the
character the decoder emits at generation step ``i``.  A character with no box gets
``null`` rather than an interpolated guess -- 0.2% of characters, and fabricating
one would put a made-up location under an arm whose whole point is spatial truth.

Each entry also carries ``alignment_status`` (``exact``, ``placeholder`` for an
unidentified ``#`` glyph, ``mismatch`` for an identity drift, or ``missing`` for no
box) and ``source_index`` (the character's index in the annotation's ``characters``
list).  The mask head reads ``alignment_status`` to distinguish a placeholder box --
which is still the right *location* -- from a box that belongs to a different glyph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

# Needleman-Wunsch scores.  A mismatch is penalised less than a gap so that a
# placeholder or a variant glyph is paired rather than skipped: skipping it would
# shift every following character by one position, which is exactly the error this
# alignment exists to prevent.
MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_SCORE = -1

CHAR_SOURCE = "mthv2_official_char_annotation"
DEFAULT_OUTPUT_NAME = "manifest.char.jsonl"
SPLITS = ("train", "validation", "test")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _center(box: Sequence[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_distance(box: Sequence[float], point: tuple[float, float]) -> float:
    """Distance from a point to an axis-aligned box; zero when it is inside."""

    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return (dx * dx + dy * dy) ** 0.5


def _reads_downwards(line: dict[str, Any]) -> bool:
    """Whether the line's characters advance vertically.

    ``unknown`` is resolved from the box shape: a line taller than it is wide is a
    vertical column, which is how the converter itself infers the direction.  An
    explicit direction always wins, because the converter may have taken it from
    the source rather than from the geometry.
    """

    direction = line.get("writing_direction", "unknown")
    if direction == "horizontal_ltr":
        return False
    if direction == "vertical_rtl":
        return True
    box = line["bbox_px"]
    return (box[3] - box[1]) >= (box[2] - box[0])


def order_characters(
    lines: Sequence[dict[str, Any]], characters: Sequence[dict[str, Any]]
) -> list[int]:
    """Return source indices in a geometric reading order.

    Each character goes to the line whose box is nearest its centre, then the
    characters inside a line are ordered along that line's reading direction.  This
    is the initial ordering the alignment refines, so it is allowed to be wrong at
    the margins.
    """

    ranked: list[tuple[int, float, int]] = []
    for index, char in enumerate(characters):
        point = _center(char["bbox_xyxy_px"])
        distances = [_box_distance(line["bbox_px"], point) for line in lines]
        owner = min(range(len(lines)), key=lambda line: (distances[line], line))
        along = point[1] if _reads_downwards(lines[owner]) else point[0]
        ranked.append((owner, along, index))
    ranked.sort()
    return [item[2] for item in ranked]


def align(target: str, source: str) -> list[int | None]:
    """Map every target character to a source character, or to ``None``.

    Monotone (order-preserving) alignment, so the result cannot pair two target
    characters with the same source or move backwards through either sequence.
    Both sequences may skip: a target character with no annotated box yields
    ``None``, and an annotated box with no matching text is dropped.
    """

    n, m = len(target), len(source)
    # Only the traceback matrix is kept in full; the score rows roll, because a
    # 800-character page would otherwise hold two 640k-entry integer matrices.
    move = [bytearray(m + 1) for _ in range(n + 1)]
    previous = [j * GAP_SCORE for j in range(m + 1)]
    for j in range(1, m + 1):
        move[0][j] = 2  # left: the source character is unmatched
    for i in range(1, n + 1):
        current = [0] * (m + 1)
        current[0] = i * GAP_SCORE
        move[i][0] = 1  # up: the target character is unmatched
        target_char = target[i - 1]
        for j in range(1, m + 1):
            best = previous[j - 1] + (
                MATCH_SCORE if target_char == source[j - 1] else MISMATCH_SCORE
            )
            step = 0  # diagonal: pair the two
            if previous[j] + GAP_SCORE > best:
                best, step = previous[j] + GAP_SCORE, 1
            if current[j - 1] + GAP_SCORE > best:
                best, step = current[j - 1] + GAP_SCORE, 2
            current[j] = best
            move[i][j] = step
        previous = current

    pairs: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        step = move[i][j]
        if step == 0:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs


def page_characters(record: dict[str, Any], annotation: dict[str, Any]) -> tuple[list, dict[str, int]]:
    """Build the ``page_text``-indexed character channel for one page.

    Returns the per-character entries and the page's statistics.  Raises when the
    annotation does not reproduce ``page_text``, because the indices the entries
    are placed at would then refer to a different string than the one the decoder
    is asked to produce.
    """

    lines = sorted(annotation["textlines"], key=lambda line: int(line["reading_order"]))
    joined = "".join(line["text"] for line in lines)
    page_text = record["page_text"]
    if joined != page_text:
        raise ValueError(
            f"{record['page_id']}: textline texts do not reproduce page_text "
            f"({len(joined)} vs {len(page_text)} characters)"
        )
    if not lines:
        raise ValueError(f"{record['page_id']}: page has no textlines")

    # Which line each page_text position belongs to, so an entry can carry it.
    line_of: list[int] = []
    for line_index, line in enumerate(lines):
        line_of.extend([line_index] * len(line["text"]))

    characters = annotation.get("characters") or []
    ordered = order_characters(lines, characters)
    source = "".join(characters[index]["character"] for index in ordered)
    pairs = align(page_text, source)

    width, height = (float(value) for value in annotation["page_size"])
    entries: list[dict[str, Any]] = []
    matched = 0
    mismatches = 0
    for index, pair in enumerate(pairs):
        if pair is None:
            entries.append(
                {
                    "bbox": None,
                    "line_index": line_of[index] if index < len(line_of) else None,
                    "source_index": None,
                    "alignment_status": "missing",
                }
            )
            continue
        matched += 1
        source_index = ordered[pair]
        source_char = source[pair]
        mismatches += int(source_char != page_text[index])
        if source_char == page_text[index]:
            status = "exact"
        elif source_char == "#":
            status = "placeholder"
        else:
            status = "mismatch"
        box = characters[source_index]["bbox_xyxy_px"]
        entries.append(
            {
                "bbox": [
                    round(float(box[0]) / width, 8),
                    round(float(box[1]) / height, 8),
                    round(float(box[2]) / width, 8),
                    round(float(box[3]) / height, 8),
                ],
                "line_index": line_of[index] if index < len(line_of) else None,
                "source_index": source_index,
                "alignment_status": status,
            }
        )
    stats = {
        "characters": len(page_text),
        "matched": matched,
        "unmatched": len(page_text) - matched,
        "order_mismatches": mismatches,
        "boxes": len(characters),
    }
    return entries, stats


def augment_split(
    dataset_root: Path,
    annotation_root: Path,
    split: str,
    output_name: str,
) -> dict[str, Any]:
    """Write ``split``'s augmented manifest and return the split's statistics."""

    manifest_path = dataset_root / split / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    records = _load_jsonl(manifest_path)
    totals = {"pages": len(records), "characters": 0, "matched": 0, "unmatched": 0,
              "order_mismatches": 0, "pages_with_unmatched": 0}
    output: list[dict[str, Any]] = []
    for record in records:
        annotation_path = annotation_root / split / str(record["annotation_file"])
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"{annotation_path} is missing; a subset dataset keeps its annotations "
                "in the parent, so pass --annotation-root"
            )
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        entries, stats = page_characters(record, annotation)
        augmented = dict(record)
        augmented["char_source"] = CHAR_SOURCE
        augmented["characters"] = entries
        augmented["char_stats"] = stats
        output.append(augmented)
        for key in ("characters", "matched", "unmatched", "order_mismatches"):
            totals[key] += stats[key]
        totals["pages_with_unmatched"] += int(stats["unmatched"] > 0)

    target = manifest_path.with_name(output_name)
    target.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output),
        encoding="utf-8",
    )
    totals["manifest"] = str(target)
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        help="where annotation_file paths resolve; defaults to --dataset-root, and "
        "must be the parent dataset for a subset such as q32_sparse24",
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--report", type=Path, help="optional JSON statistics output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    annotation_root = (args.annotation_root or dataset_root).resolve()
    report = {
        "dataset_root": str(dataset_root),
        "annotation_root": str(annotation_root),
        "char_source": CHAR_SOURCE,
        "splits": {},
    }
    for split in args.splits:
        stats = augment_split(dataset_root, annotation_root, split, args.output_name)
        report["splits"][split] = stats
        matched = stats["matched"]
        coverage = matched / max(1, stats["characters"])
        print(
            f"{split:11s} pages={stats['pages']:5d} characters={stats['characters']:7d} "
            f"coverage={coverage:.4%} order_mismatches={stats['order_mismatches']:5d} "
            f"pages_with_unmatched={stats['pages_with_unmatched']:4d}"
        )
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
