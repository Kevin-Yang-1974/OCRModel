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


def test_a_dropped_character_marks_its_neighbours():
    """The readout has to be reported where a line constraint would have to work.

    A dropped reference character is what drifting off the line looks like, so if the
    attention is least reliable right there, a tracker built on it inherits the failure.
    """

    from analyze_attention_localization import neighbourhood_flags

    # 乙 was dropped by the model: reference 1 has nothing aligned to it.
    mapping = align("甲丙", "甲乙丙")
    near_dropped, near_repeated = neighbourhood_flags("甲丙", "甲乙丙", mapping, window=3)
    # Both generated characters sit within three of the dropped position, and neither is
    # itself an insertion, so both are marked.
    assert near_dropped == [True, True]
    assert near_repeated == [False, False]

    # A window of zero leaves only exact adjacency, which no character here has.
    near_dropped, _ = neighbourhood_flags("甲丙", "甲乙丙", mapping, window=0)
    assert near_dropped == [False, False]


def test_an_insertion_is_not_itself_near_a_drop():
    """An inserted character has no reference position, so it has nothing to be near."""

    from analyze_attention_localization import neighbourhood_flags

    mapping = align("甲丁丙", "甲乙丙")  # 丁 is inserted
    near_dropped, _ = neighbourhood_flags("甲丁丙", "甲乙丙", mapping, window=3)
    assert len(near_dropped) == 3
    assert near_dropped[1] is False


def test_a_cycle_marks_its_whole_stretch():
    """The failure a line constraint answers for is a stretch the model got stuck on."""

    from analyze_attention_localization import neighbourhood_flags

    # 甲乙 repeated four times back to back is a loop; the trailing 丙丁 is not.
    generated = "甲乙甲乙甲乙甲乙丙丁"
    mapping = align(generated, generated)
    _, near_repeated = neighbourhood_flags(generated, generated, mapping, window=0)
    assert near_repeated == [True] * 8 + [False] * 2


def test_ordinary_recurring_text_is_not_a_repetition():
    """In Chinese, common trigrams recur constantly; that is the language, not a failure.

    Using ``repeated_trigram_rate``'s page-level notion as a per-character flag marked 75%
    of a validation page, which is not a measurement of anything.
    """

    from analyze_attention_localization import neighbourhood_flags

    generated = "天地玄黄宇宙洪荒日月盈昃辰宿列张"
    mapping = align(generated, generated)
    _, near_repeated = neighbourhood_flags(generated, generated, mapping, window=0)
    assert not any(near_repeated)


def test_two_repeats_are_not_yet_a_loop():
    from analyze_attention_localization import neighbourhood_flags

    generated = "甲乙甲乙丙丁戊己庚辛"
    mapping = align(generated, generated)
    _, near_repeated = neighbourhood_flags(generated, generated, mapping, window=0)
    assert not any(near_repeated)


def test_a_page_with_no_failures_has_no_marked_characters():
    from analyze_attention_localization import neighbourhood_flags

    near_dropped, near_repeated = neighbourhood_flags("甲乙丙", "甲乙丙", [0, 1, 2], window=3)
    assert near_dropped == [False] * 3
    assert near_repeated == [False] * 3


def test_the_scale_free_confidence_unit_removes_the_line_count():
    """The absolute share is not comparable across pages; the multiple of uniform is.

    With N lines nothing exceeds 1/N under a uniform distribution, so a fixed bar is much
    harder to clear on a page with more lines -- which is how coverage came to correlate
    -0.65 with the line count on the stage-1 subset.
    """

    from analyze_attention_localization import confidence_value

    two_lines = {"confidence": 0.5, "regions": 2}
    twenty_lines = {"confidence": 0.1, "regions": 20}
    # Same readout quality: half the mass on a two-line page and a tenth on a twenty-line
    # page are both 1x uniform.
    assert confidence_value(two_lines, "uniform_multiple") == pytest.approx(1.0)
    assert confidence_value(twenty_lines, "uniform_multiple") == pytest.approx(2.0)
    # Under the absolute unit the two are far apart, which is the confound.
    assert confidence_value(two_lines, "absolute") == pytest.approx(0.5)
    assert confidence_value(twenty_lines, "absolute") == pytest.approx(0.1)


def test_the_threshold_and_the_curve_use_the_same_unit():
    """A curve drawn in one unit and a verdict in another would not describe each other."""

    rows = [
        {"step": 1, "truth": 0, "pred": 0, "confidence": 0.30, "regions": 4,
         "row_break": False, "in_line_pos": 0.5, "expected_pos": 0.5, "m_t": 0.5,
         "near_dropped": False, "near_repeated": False},
        {"step": 2, "truth": 1, "pred": 1, "confidence": 0.05, "regions": 4,
         "row_break": True, "in_line_pos": 0.5, "expected_pos": 0.5, "m_t": 0.5,
         "near_dropped": False, "near_repeated": False},
    ]
    from analyze_attention_localization import curves

    # In the absolute unit only the 0.30 row clears 0.20.
    absolute = curves(rows, 0.20, "absolute")
    assert absolute["confident_steps"] == 1
    # In the scale-free unit neither does: 0.30 x 4 = 1.2x uniform and 0.05 x 4 = 0.2x.
    scale_free = curves(rows, 2.0, "uniform_multiple")
    assert scale_free["confident_steps"] == 0
    # And the bands follow the same unit.  Fixed 0.0-1.0 edges would drop both rows --
    # 1.2x and 0.2x -- outside every band, leaving an empty curve that still looked like
    # one.  Every row must land in exactly one band.
    assert sum(band["steps"] for band in scale_free["bands"].values()) == 2
    assert sum(band["steps"] for band in absolute["bands"].values()) == 2
    assert list(scale_free["bands"])[-1].endswith("+")


def test_the_report_splits_the_readout_by_failure_neighbourhood():
    report = _report("p0", {1: ("甲", 0, 0.9), 2: ("丙", 1, 0.9)})
    # Three boxes for three reference characters; the middle one is the dropped 乙.
    characters = [
        _char([0.1, 0.05, 0.3, 0.15]),
        _char([0.1, 0.25, 0.3, 0.35]),
        _char([0.1, 0.45, 0.3, 0.55]),
    ]
    # 乙 is dropped between 甲 and 丙.
    page = score_page(
        report, "甲乙丙", "甲丙", REGIONS, characters, layers=None, heads=None,
        aggregate="mean", window=3,
    )
    rows = page["rows"]
    assert [row["near_dropped"] for row in rows] == [True, True]
    assert [row["near_repeated"] for row in rows] == [False, False]

    from analyze_attention_localization import summarise

    summary = summarise(rows, 0.5, {"p0": rows})
    assert summary["near_dropped"]["chars"] == 2
    assert summary["clean"]["chars"] == 0
    assert summary["near_repeated"]["chars"] == 0


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
