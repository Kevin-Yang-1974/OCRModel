"""Tests for the offline attention-localization scoring.

The tool's job is to turn a probe report into one number that a stage gate is decided
on, so the properties worth pinning are the ones that would let a wrong number pass as
a right one:

* the alignment, which decides which reference character a step is scored against --
  an off-by-one there moves the truth to the neighbouring line and the result still
  looks like a plausible accuracy;
* the character-to-step mapping, which follows the emitted text rather than the step
  index, because a token can carry several characters;
* the refusal to score a page whose manifest has no character boxes, which is the
  Dunhuang case today: "no signal" and "no truth" must not produce the same report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_attention_localization import (  # noqa: E402
    align,
    baselines,
    char_lines,
    combine,
    expected_in_line_pos,
    main,
    score_page,
    steps_for_head_selection,
)

# Two vertical columns side by side, each read top to bottom.
COLUMN_LEFT = {
    "bbox": [0.0, 0.0, 0.5, 1.0],
    "reading_order": 0,
    "writing_direction": "vertical_rtl",
}
COLUMN_RIGHT = {
    "bbox": [0.5, 0.0, 1.0, 1.0],
    "reading_order": 1,
    "writing_direction": "vertical_rtl",
}
REGIONS = [COLUMN_LEFT, COLUMN_RIGHT]


def _char(box):
    return {"bbox": box}


def _row(line, confidence=0.9, layer=0, head=0, in_line_pos=0.5, m_t=0.5):
    """One head's readout: a distribution over the two columns plus background."""

    probs = [0.0, 0.0, 0.0]
    probs[line] = confidence
    probs[-1] = 1.0 - confidence
    return {
        "layer": layer,
        "head": head,
        "m_t": m_t,
        "lse_vis": 1.0,
        "lse_text": 0.5,
        "entropy_norm": 0.3,
        "line_probs": probs,
        "argmax_line": line,
        "top_line_mass": confidence,
        "background_mass": probs[-1],
        "in_line_pos": in_line_pos,
    }


def _report(page_id, steps):
    """``steps`` maps a decode step to (emitted text, predicted line, confidence)."""

    records = [
        {"step": step, "text_keys": 10, "heads": [_row(line, confidence)]}
        for step, (_, line, confidence) in sorted(steps.items())
        for _ in [0]
    ]
    for record in records:
        record["emitted"] = steps[record["step"]][0]
    return {
        "page_id": page_id,
        "steps": records,
        "visual_tokens": 4,
        "num_regions": 2,
        "decoding_steps": len(records),
        "grid_missing_steps": 0,
        "emitted_missing_steps": 0,
    }


def test_the_alignment_maps_generated_characters_to_reference_positions():
    # 丁 was emitted where the reference has nothing: a true insertion.
    mapping = align("甲丁乙丙", "甲乙丙")
    assert mapping == [0, None, 1, 2]


def test_a_misread_character_is_a_substitution_not_an_insertion():
    """A wrong glyph still says where the model was reading, so it keeps its line.

    On a tie, aligning the character to a reference position is what makes the
    localization question answerable at all: calling every misread character an
    insertion would drop exactly the steps the readout is meant to cover.
    """

    assert align("甲丁丙", "甲乙丙") == [0, 1, 2]


def test_the_alignment_of_a_skipped_character_leaves_the_position_unused():
    # 乙 was dropped by the model, so 丙 still maps to its own reference position.
    mapping = align("甲丙", "甲乙丙")
    assert mapping == [0, 2]


def test_char_lines_uses_the_region_boxes_not_the_manifest_order():
    """The probe's labels are reading-order indices, so the truth has to be too."""

    characters = [
        _char([0.1, 0.1, 0.3, 0.2]),  # left column
        _char([0.6, 0.1, 0.8, 0.2]),  # right column
        _char([0.1, 0.6, 0.3, 0.7]),  # left column, lower
    ]
    assert char_lines(REGIONS, characters) == [0, 1, 0]
    # Same regions, manifest order reversed: the labels must not move.
    reversed_regions = list(reversed(REGIONS))
    assert char_lines(reversed_regions, characters) == [0, 1, 0]


def test_a_character_outside_every_region_has_no_line():
    """An unmatched character is not a line, so it is not scored."""

    assert char_lines(REGIONS, [_char([2.0, 2.0, 3.0, 3.0]), {"bbox": None}]) == [-1, -1]


def test_a_step_is_scored_against_the_line_of_what_it_emitted():
    """The first column holds 甲乙 and the second 丙丁."""

    report = _report(
        "p0",
        {
            1: ("甲", 0, 0.9),
            2: ("乙", 0, 0.9),
            3: ("丙", 1, 0.9),
            4: ("丁", 1, 0.9),
        },
    )
    characters = [
        _char([0.1, 0.05, 0.3, 0.15]),  # 甲, left
        _char([0.1, 0.25, 0.3, 0.35]),  # 乙, left
        _char([0.6, 0.05, 0.8, 0.15]),  # 丙, right
        _char([0.6, 0.25, 0.8, 0.35]),  # 丁, right
    ]
    page = score_page(
        report, "甲乙丙丁", "甲乙丙丁", REGIONS, characters, layers=None, heads=None,
        aggregate="mean",
    )
    assert page["scored"] == 4
    assert [row["truth"] for row in page["rows"]] == [0, 0, 1, 1]
    assert page["inserted"] == 0
    assert page["unbounded_chars"] == 0


def test_the_emitted_text_decides_the_step_not_the_step_index():
    """One token can carry several characters, so the step index is not the position."""

    # Step 1 emits two characters at once; the alignment has to consume both.
    report = _report("p0", {1: ("甲乙", 0, 0.9), 2: ("丙", 1, 0.9)})
    characters = [
        _char([0.1, 0.05, 0.3, 0.15]),
        _char([0.1, 0.25, 0.3, 0.35]),
        _char([0.6, 0.05, 0.8, 0.15]),
    ]
    page = score_page(
        report, "甲乙丙", "甲乙丙", REGIONS, characters, layers=None,
        heads=None, aggregate="mean",
    )
    assert [row["truth"] for row in page["rows"]] == [0, 0, 1]
    # The step covering two reference characters has no single line by observation,
    # so it is counted rather than quietly averaged.
    assert page["ambiguous_steps"] == 1


def test_an_inserted_character_is_not_scored_against_a_line():
    """Scoring an insertion would score the model against its own hallucination.

    The extra character is inserted at the end, where it cannot be read as a
    substitution for anything the reference has.
    """

    report = _report("p0", {1: ("甲", 0, 0.9), 2: ("乙", 0, 0.9), 3: ("Z", 1, 0.9)})
    characters = [_char([0.1, 0.05, 0.3, 0.15]), _char([0.1, 0.25, 0.3, 0.35])]
    page = score_page(
        report, "甲乙", "甲乙", REGIONS, characters, layers=None, heads=None,
        aggregate="mean",
    )
    assert page["inserted"] == 1
    assert [row["truth"] for row in page["rows"]] == [0, 0]


def test_a_step_without_emitted_text_is_reported_not_skipped():
    report = _report("p0", {1: ("甲", 0, 0.9)})
    del report["steps"][0]["emitted"]
    characters = [_char([0.1, 0.05, 0.3, 0.15])]
    page = score_page(
        report, "甲", "甲", REGIONS, characters, layers=None, heads=None, aggregate="mean"
    )
    assert page["steps_missing_text"] == 1
    assert page["scored"] == 0


def test_combine_averages_the_distributions_not_the_argmax():
    """Averaging ``argmax_line`` would throw away the confidence the stage turns on."""

    rows = [_row(0, confidence=0.9), _row(1, confidence=0.9)]
    readout = combine(rows, "mean")
    assert readout["line_probs"][0] == pytest.approx(0.45)
    assert readout["line_probs"][1] == pytest.approx(0.45)
    assert readout["top_line_mass"] == pytest.approx(0.45)
    # The per-head variant picks the loudest head instead, for comparison only.
    assert combine(rows, "best")["argmax_line"] in (0, 1)


def test_the_vertical_axis_is_the_one_measured():
    """A vertical column is read top to bottom, so the position axis is y."""

    position, axis = expected_in_line_pos(COLUMN_LEFT, [0.1, 0.75, 0.3, 0.85])
    assert axis == 1
    assert position == pytest.approx(0.80)
    position, axis = expected_in_line_pos(
        {"bbox": [0.0, 0.0, 1.0, 0.2], "writing_direction": "horizontal_ltr"},
        [0.75, 0.05, 0.85, 0.15],
    )
    assert axis == 0
    assert position == pytest.approx(0.80)


def test_row_breaks_are_marked_and_scored_separately():
    report = _report(
        "p0",
        {1: ("甲", 0, 0.9), 2: ("乙", 0, 0.9), 3: ("丙", 1, 0.9), 4: ("丁", 1, 0.9)},
    )
    characters = [
        _char([0.1, 0.05, 0.3, 0.15]),
        _char([0.1, 0.25, 0.3, 0.35]),
        _char([0.6, 0.05, 0.8, 0.15]),
        _char([0.6, 0.25, 0.8, 0.35]),
    ]
    page = score_page(
        report, "甲乙丙丁", "甲乙丙丁", REGIONS, characters, layers=None, heads=None,
        aggregate="mean",
    )
    from analyze_attention_localization import add_row_breaks

    add_row_breaks(page["rows"])
    assert [row["row_break"] for row in page["rows"]] == [False, False, True, False]


def test_the_baselines_are_computed_on_the_same_steps():
    """Staying on the previous line is strong when line changes are rare, and it is
    the bar the attention has to clear."""

    rows = [{"step": 1, "truth": 0}, {"step": 2, "truth": 0}, {"step": 3, "truth": 1}]
    result = baselines(rows, {"p0": rows})
    # Two steps have a predecessor; one of them stays on the same line.  The prior is
    # scored on the same steps as the attention, so the comparison is paired.
    assert result["stay_previous_line"] == pytest.approx(0.5)
    assert result["stay_previous_line_steps"] == 2
    # Two reference lines over three steps: a constant-rate scan steps 0, 0, 1 -- the
    # middle step lands on 0.5 and Python's round-half-to-even sends it to 0 -- which
    # happens to be exact here.  The point is that it is *not* better than the
    # attention, not that it is a fixed number.
    assert result["uniform_scan"] == pytest.approx(1.0)


def test_head_selection_is_reported_per_head():
    report = _report("p0", {1: ("甲", 0, 0.9), 2: ("乙", 0, 0.9)})
    report["steps"][0]["heads"] = [_row(0, layer=0, head=0), _row(1, layer=0, head=1)]
    report["steps"][1]["heads"] = [_row(0, layer=0, head=0), _row(1, layer=0, head=1)]
    characters = [_char([0.1, 0.05, 0.3, 0.15]), _char([0.1, 0.25, 0.3, 0.35])]
    page = score_page(
        report, "甲乙", "甲乙", REGIONS, characters, layers=None, heads=None,
        aggregate="mean",
    )
    page["raw_by_step"] = steps_for_head_selection(report)
    from analyze_attention_localization import by_selection

    selection = by_selection([page], None, None)
    assert selection["per_head"]["0:0"]["accuracy"] == 1.0
    assert selection["per_head"]["0:1"]["accuracy"] == 0.0
    assert selection["per_layer"]["0"]["accuracy"] == pytest.approx(0.5)


def _write(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")


def _fixture(tmp_path, *, with_characters=True):
    characters = [
        _char([0.1, 0.05, 0.3, 0.15]),
        _char([0.1, 0.25, 0.3, 0.35]),
    ]
    manifest = [
        {
            "page_id": "p0",
            "regions": REGIONS,
            "page_text": "甲乙",
            **({"characters": characters} if with_characters else {}),
        }
    ]
    predictions = [{"page_id": "p0", "reference": "甲乙", "prediction": "甲乙"}]
    probe = [_report("p0", {1: ("甲", 0, 0.9), 2: ("乙", 0, 0.9)})]
    paths = {}
    for name, rows in (("manifest", manifest), ("predictions", predictions), ("probe", probe)):
        paths[name] = tmp_path / f"{name}.jsonl"
        _write(paths[name], rows)
    return paths


def test_the_tool_reports_a_page_without_the_character_channel_as_a_gap(tmp_path, capsys):
    """Dunhuang today: regions exist, character boxes do not, so there is no truth.

    Reporting an accuracy here would be reporting the model against an invented
    ground truth, which is exactly the failure the plan's data note warns about.
    """

    paths = _fixture(tmp_path, with_characters=False)
    code = main(
        [
            "--probe", str(paths["probe"]),
            "--predictions", str(paths["predictions"]),
            "--manifest", str(paths["manifest"]),
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pages_scored"] == 0
    assert report["pages_without_char_channel"] == [
        {"page_id": "p0", "reason": "no character box channel"}
    ]
    assert "all" not in report


def test_the_tool_scores_the_gate_on_the_check_pages_only(tmp_path, capsys):
    paths = _fixture(tmp_path)
    select = tmp_path / "select.txt"
    select.write_text("p0\n", encoding="utf-8")
    check = tmp_path / "check.txt"
    check.write_text("p0\n", encoding="utf-8")
    code = main(
        [
            "--probe", str(paths["probe"]),
            "--predictions", str(paths["predictions"]),
            "--manifest", str(paths["manifest"]),
            "--select-pages", str(select),
            "--check-pages", str(check),
            "--layers", "0",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["check"]["scored"] == 2
    assert report["check"]["accuracy"] == 1.0
    assert report["check"]["alignment"]["inserted_chars"] == 0
    # The step-to-token mapping is checked end to end: the text attributed to observed
    # steps has to be a suffix of what the model actually produced.
    assert report["check"]["alignment"]["pages_matching_prediction"] == 1
    assert report["selection"]["per_head"]["0:0"]["accuracy"] == 1.0
    # The gate needs both the accuracy and the coverage, and it has to beat the prior.
    assert report["check"]["gate"]["accuracy_target"] == 0.90
    assert report["check"]["beats_stay_previous_line"] in (True, False)
    assert report["check"]["gate"]["passes"] in (True, False)
