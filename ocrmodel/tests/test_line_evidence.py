"""Tests for the locked line evidence that gates the v3 ``line`` spatial target.

``target_mode='line'`` needs a line grouping for whatever manifest it is handed,
and the two evaluation domains supply one by different routes: MTHv2 through a
per-character ``line_index``, Dunhuang/local-gazetteer through textline regions
with no character array at all.  The things worth pinning are that the evidence
classifier picks the right route, that "no line evidence" is a recorded outcome
rather than a silent guess, and that what the classifier accepts is exactly what
``region_line_targets`` can read -- a domain locked as locatable must not then
raise, or raise in a way the classifier would not have predicted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "evaluation"))

from layout_ocr.mask_targets import LineTargetError, region_line_targets
from prepare_line_mask_v3_diagnostic_manifests import line_evidence


def _region_page(page_id: str, lines, *, layout_level="textline", separator="",
                 page_text=None, characters=None) -> dict:
    record = {
        "page_id": page_id,
        "page_text": page_text if page_text is not None
        else "".join(text for _, (text, _) in sorted(lines)),
        "layout_level": layout_level,
        "page_text_separator": separator,
        "bbox_format": "xyxy_normalized",
        "regions": [
            {"reading_order": order, "text": text, "bbox": list(box),
             "layout_level": "textline", "layout_type": "REGION"}
            for order, (text, box) in lines
        ],
    }
    if characters is not None:
        record["characters"] = characters
    return record


def _char_page(page_id: str, *, indexed=True) -> dict:
    characters = []
    for index in range(8):
        entry = {"bbox": [(index % 4) / 4, 0.0, (index % 4 + 1) / 4, 0.5],
                 "alignment_status": "exact"}
        if indexed:
            entry["line_index"] = index // 4
        characters.append(entry)
    return {
        "page_id": page_id,
        "page_text": "甲乙丙丁戊己庚辛",
        "layout_level": "textline",
        "characters": characters,
    }


def test_a_char_manifest_is_read_through_the_annotation_path():
    evidence = line_evidence([_char_page(f"mth{index}") for index in range(4)])
    assert evidence["line_source"] == "annotation"
    assert evidence["spatial_targets"] == "character_boxes"
    assert evidence["box_granularity"] == "character"
    assert evidence["pages_with_line_mapping"] == 4


def test_a_textline_manifest_is_read_through_the_region_path():
    records = [
        _region_page(f"dh{index}", [
            (0, ("須菩提如恒河", [0.10, 0.15, 0.20, 0.80])),
            (1, ("如是沙等恒河", [0.25, 0.15, 0.35, 0.80])),
        ])
        for index in range(4)
    ]
    evidence = line_evidence(records)
    assert evidence["line_source"] == "region_textline"
    assert evidence["spatial_targets"] == "line_regions"
    assert evidence["box_granularity"] == "textline"
    assert evidence["pages_with_line_mapping"] == 4
    assert evidence["unmappable_pages"] == []


def test_a_domain_without_line_geometry_is_recorded_not_guessed():
    """A domain needing no line boxes must still be lockable.

    The two domains are handled by one code path, so a domain that genuinely
    cannot supply line geometry is reported as unavailable -- with a reason and
    the offending pages -- rather than being raised out of, which would make a
    run covering both domains impossible to lock.
    """

    record = _region_page("p", [(0, ("甲乙", [0.1, 0.1, 0.2, 0.9]))],
                          page_text="甲丙")
    evidence = line_evidence([record])
    assert evidence["line_source"] is None
    assert evidence["spatial_targets"] is None
    assert evidence["pages_with_line_mapping"] == 0
    assert evidence["unavailable_reason"]
    assert evidence["unmappable_pages"] == [
        {"page_id": "p", "reason": "reading-order region text does not reproduce page_text"}
    ]

    not_textline = line_evidence([_region_page(
        "p", [(0, ("甲乙", [0.1, 0.1, 0.2, 0.9]))], layout_level="region"
    )])
    assert not_textline["line_source"] is None
    assert "not textline-level" in not_textline["unavailable_reason"]

    no_char_index = line_evidence([_char_page("p", indexed=False)])
    assert no_char_index["line_source"] is None
    assert "without any 'line_index'" in no_char_index["unavailable_reason"]


def test_a_partly_indexed_character_set_raises_rather_than_locking():
    """Only some pages carrying line_index is corrupt input, not a missing feature."""

    with pytest.raises(ValueError, match="mixes"):
        line_evidence([_char_page("indexed"), _char_page("plain", indexed=False)])


def _real_format_page(page_id: str, lines, *, layout_level="textline", separator="",
                      page_text=None, characters=None) -> dict:
    """A page shaped like ``glmocr_compat``: one region per text line."""

    record = {
        "page_id": page_id,
        "page_text": page_text if page_text is not None
        else "".join(text for _, (text, _) in sorted(lines)),
        "layout_level": layout_level,
        "page_text_separator": separator,
        "bbox_format": "xyxy_normalized",
        "regions": [
            {"reading_order": order, "text": text, "bbox": list(box),
             "layout_level": "textline", "layout_type": "REGION",
             "writing_direction": "vertical_rtl"}
            for order, (text, box) in lines
        ],
    }
    if characters is not None:
        record["characters"] = characters
    return record


def test_a_line_level_page_builds_a_whole_line_target_end_to_end():
    """The real Dunhuang-shaped record reaches ``build_mask_targets`` intact."""

    import torch

    from layout_ocr.mask_targets import build_mask_targets

    class _Tokenizer:
        def __init__(self, chars):
            self.mapping = {20 + index: char for index, char in enumerate(chars)}

        def decode(self, ids, skip_special_tokens=True):
            return "".join(self.mapping.get(int(index), "") for index in ids)

    text = "須菩提如恒河中所有沙數如是沙等恒河於意云何"
    record = _real_format_page("dh", [
        (0, (text[:11], [0.10, 0.15, 0.20, 0.80])),
        (1, (text[11:], [0.25, 0.15, 0.35, 0.80])),
    ])
    assert line_evidence([record])["spatial_targets"] == "line_regions"

    cells = len(text)
    width = 1.0 / cells
    xywh = torch.tensor(
        [[(index + 0.5) * width, 0.5, width, 1.0] for index in range(cells)],
        dtype=torch.float32,
    ).unsqueeze(0)
    targets = build_mask_targets(
        _Tokenizer(list(text)), record, list(range(20, 20 + cells)), {13}, xywh,
        target_mode="line", line_source="region_textline", raster_mode="hard",
    )

    assert targets.line_evidence == "textline"
    assert bool(targets.spatial_valid[0, :cells].all())
    assert targets.window_report["window_fallbacks"] == 0
    assert targets.window_report["lines"] == 2
    # Each line token covers its own region, and the two regions differ here.
    assert int((targets.mask[0, 0] > 0).sum()) > 0
    assert targets.mask[0, 0].tolist() != targets.mask[0, cells - 1].tolist()


SPATIAL_TARGET_LINE_SOURCES = {"character_boxes": "annotation", "line_regions": "region_textline"}
# The status each stage locks into its own protocol and evidence files.
STAGE_STATUS = {
    "diagnostic32": "locked_before_candidate_inference",
    "full_val_tune": "locked_after_32_page_screen_before_full_inference",
}


def _as_stage_evidence(evidence: dict, stage: str) -> dict:
    """Tag a classifier result the way the prepare step writes it to disk."""

    return {**evidence, "status": STAGE_STATUS[stage], "evaluation_stage": stage}


def _diagnostic_guard(evidence: dict, protocol: dict) -> tuple[bool, str | None]:
    """Mirror of the evaluator's line-evidence guard, as a testable predicate."""

    stage = protocol.get("evaluation_stage")
    expected_status = STAGE_STATUS.get(stage)
    if expected_status is None:
        return False, None
    # Both the protocol and the evidence carry their own stage's status.  Pinning
    # the evidence to one fixed string made every full_val_tune worker reject the
    # evidence its own prepare step had just written.
    if evidence.get("status") != expected_status or evidence.get("evaluation_stage") != stage:
        return False, None
    spatial_targets = evidence.get("spatial_targets")
    if spatial_targets not in SPATIAL_TARGET_LINE_SOURCES:
        return False, None
    if (evidence.get("pages") != protocol.get("val_tune_pages")
            or evidence.get("pages_with_line_mapping") != evidence.get("pages")
            or evidence.get("unmappable_pages")):
        return False, None
    if protocol.get("stage_pages", 0) > evidence["pages"]:
        return False, None
    return True, SPATIAL_TARGET_LINE_SOURCES[spatial_targets]


def test_both_stages_accept_their_own_evidence_status():
    """diagnostic32 and full_val_tune lock different status strings."""

    evidence_by_stage = {
        "diagnostic32": {"status": STAGE_STATUS["diagnostic32"],
                         "evaluation_stage": "diagnostic32"},
        "full_val_tune": {"status": STAGE_STATUS["full_val_tune"],
                          "evaluation_stage": "full_val_tune"},
    }
    for stage, fields in evidence_by_stage.items():
        for probed_stage in STAGE_STATUS:
            evidence = {"spatial_targets": "line_regions", "pages": 80,
                        "pages_with_line_mapping": 80, "unmappable_pages": [],
                        "evaluation_stage": stage, **fields}
            protocol = {"evaluation_stage": probed_stage, "val_tune_pages": 80,
                        "stage_pages": 80}
            accepted = _diagnostic_guard(evidence, protocol)[0]
            assert accepted == (stage == probed_stage), (stage, probed_stage)


def test_a_stage_cannot_evaluate_more_pages_than_the_evidence_covers():
    """An oversized batch is refused up front, not deep inside the page loop."""

    evidence = {"status": STAGE_STATUS["diagnostic32"], "evaluation_stage": "diagnostic32",
                "spatial_targets": "line_regions", "pages": 32,
                "pages_with_line_mapping": 32, "unmappable_pages": []}
    protocol = {"evaluation_stage": "diagnostic32", "val_tune_pages": 32, "stage_pages": 32}
    assert _diagnostic_guard(evidence, protocol)[0] is True
    protocol["stage_pages"] = 33
    assert _diagnostic_guard(evidence, protocol)[0] is False


def test_evidence_covers_the_source_manifest_not_the_stage_subset():
    """The guard's coverage check must compare against the source manifest.

    The evidence is computed once over the domain's whole val_tune manifest (240
    MTHv2 / 80 Dunhuang pages), while a stage evaluates either all of it or a
    32-page stratified subset.  Comparing ``pages_with_line_mapping`` against the
    *stage's* page count therefore rejects the diagnostic32 stage -- which is
    exactly what it did on the real v2 run, for both domains at once.
    """

    records = [_region_page(f"dh{index}", [(0, ("甲乙丙丁", [0.1, 0.1, 0.2, 0.9]))])
               for index in range(4)]
    evidence = _as_stage_evidence(line_evidence(records), "diagnostic32")
    assert evidence["pages"] == 4 and evidence["pages_with_line_mapping"] == 4

    # diagnostic32 over a 240-page domain: protocol page count is the source
    # manifest's, so the guard accepts even though only 32 pages are evaluated.
    stage = {"evaluation_stage": "diagnostic32", "stage_pages": 4}
    assert _diagnostic_guard(evidence, {**stage, "val_tune_pages": 4}) == (True, "region_textline")
    assert _diagnostic_guard(evidence, {**stage, "val_tune_pages": 32}) == (False, None)
    # A domain with no verified line evidence is still refused.
    refused = line_evidence([_region_page("p", [(0, ("甲乙", [0.1, 0.1, 0.2, 0.9]))],
                                          page_text="甲丙")])
    assert _diagnostic_guard(_as_stage_evidence(refused, "diagnostic32"),
                             {**stage, "stage_pages": 1}) == (False, None)


def test_the_guard_accepts_both_real_domain_manifest_shapes():
    """The two domains resolve to different line sources through the same guard.

    The real v2b failure was a guard bug, not a data bug: the evidence files were
    correct for both domains.  This pins that neither manifest shape is rejected.
    """

    stage = {"evaluation_stage": "diagnostic32", "val_tune_pages": 4, "stage_pages": 4}

    mthv2_evidence = _as_stage_evidence(
        line_evidence([_char_page(f"mth{index}") for index in range(4)]), "diagnostic32")
    assert _diagnostic_guard(mthv2_evidence, stage) == (True, "annotation")

    dunhuang_evidence = _as_stage_evidence(line_evidence([
        _region_page(f"dh{index}", [
            (0, ("須菩提如恒河", [0.10, 0.15, 0.20, 0.80])),
            (1, ("如是沙等恒河", [0.25, 0.15, 0.35, 0.80])),
        ]) for index in range(4)
    ]), "diagnostic32")
    assert _diagnostic_guard(dunhuang_evidence, stage) == (True, "region_textline")


def test_the_evidence_and_the_runtime_walk_agree_on_every_fixture():
    """A domain locked as locatable must be one the runtime can actually read.

    The classifier and ``region_line_targets`` duplicate the same concatenation
    check by necessity -- one to lock a protocol, one to build a target -- so this
    pins that they cannot drift apart.
    """

    pages = [
        ("clean", _region_page("a", [(0, ("甲乙丙丁", [0.1, 0.1, 0.2, 0.9]))])),
        ("two lines", _region_page("b", [
            (0, ("甲乙丙丁", [0.1, 0.1, 0.2, 0.9])),
            (1, ("戊己庚辛", [0.3, 0.1, 0.4, 0.9])),
        ])),
        ("newline separator", _region_page(
            "c", [(0, ("甲乙丙丁", [0.1, 0.1, 0.2, 0.9]))],
            separator="\n", page_text="甲乙丙丁\n",
        )),
        ("trailing text", _region_page("d", [(0, ("甲乙", [0.1, 0.1, 0.2, 0.9]))],
                                       page_text="甲乙丙丁")),
        ("empty region text", _region_page("e", [(0, ("", [0.1, 0.1, 0.2, 0.9]))])),
    ]
    for label, record in pages:
        locatable = line_evidence([record])["line_source"] == "region_textline"
        try:
            region_line_targets(record)
            readable = True
        except LineTargetError:
            readable = False
        assert locatable == readable, label
