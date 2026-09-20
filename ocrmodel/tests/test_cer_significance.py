"""Tests for the paired CER intervals.

Two intervals are reported from here -- one resampling pages, one resampling whole volume groups
-- and the difference between them is the point: same pages, same difference, different claim
about what counts as an independent draw. A grouping bug would show up as the two silently being
the same interval, which is exactly the failure that would make the grouped number decorative.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from analyze_cer_significance import (  # noqa: E402
    cer,
    group_indices,
    paired_bootstrap,
    volume_of,
)


def test_volume_of_reads_the_register_proxy():
    # Digits alone are not enough: P000D and P000F are different pages of one volume, and
    # merging them would pool two independent draws into one.
    assert volume_of("mthv2_mth1000_01-V001P000D") == "V001P000D"
    assert volume_of("mthv2_mth1000_02-V001P000F") == "V001P000F"
    assert volume_of("mthv2_mth1000_03-V001P0021") == "V001P0021"
    # A page with no volume prefix is pooled into one group, per the register.
    assert volume_of("mthv2_mth1000_037") == "unnumbered"


def test_group_indices_buckets_by_volume_and_pages_stay_singletons():
    ids = ["mthv2_mth1000_01-V001P000D", "mthv2_mth1000_02-V001P000F", "mthv2_mth1000_037"]
    groups = group_indices(ids, "volume")
    assert sorted(len(group) for group in groups) == [1, 1, 1]
    assert group_indices(ids, "page") is None


def test_group_indices_pools_pages_of_one_volume():
    ids = ["mthv2_mth1000_01-V001P0021", "mthv2_mth1000_02-V001P0021", "mthv2_mth1000_037"]
    groups = group_indices(ids, "volume")
    assert sorted(len(group) for group in groups) == [1, 2]


def _rows(pairs):
    return [(page_id, reference, prediction) for page_id, reference, prediction in pairs]


A = _rows(
    [
        ("mthv2_mth1000_01-V001P0021", "甲乙丙丁", "甲乙丙丁"),
        ("mthv2_mth1000_02-V001P0021", "甲乙丙丁", "甲乙丙丁"),
        ("mthv2_mth1000_037", "甲乙丙丁", "甲乙丙戊"),
    ]
)
B = _rows(
    [
        ("mthv2_mth1000_01-V001P0021", "甲乙丙丁", "甲乙丙戊"),
        ("mthv2_mth1000_02-V001P0021", "甲乙丙丁", "甲乙丙戊"),
        ("mthv2_mth1000_037", "甲乙丙丁", "甲乙丙戊"),
    ]
)


def test_cer_is_micro_over_characters():
    assert cer(A) == 1 / 12
    assert cer(B) == 3 / 12


def test_identical_runs_straddle_zero_exactly():
    low, high, share = paired_bootstrap(A, A, 500, 0, group_indices([r[0] for r in A], "volume"))
    assert (low, high, share) == (0.0, 0.0, 1.0)


def test_a_difference_present_on_every_page_shows_up_in_both_groupings():
    # The interval is for CER(a) - CER(b).  Every page carries the same one-substitution
    # difference, so no resample can put a positive value in either interval.
    ids = ["mthv2_mth1000_01-V001P0021", "mthv2_mth1000_02-V001P0021", "mthv2_mth1000_037"]
    clean = _rows([(page_id, "甲乙丙丁", "甲乙丙丁") for page_id in ids])
    one_wrong = _rows([(page_id, "甲乙丙丁", "甲乙丙戊") for page_id in ids])
    page_low, page_high, _ = paired_bootstrap(clean, one_wrong, 2000, 0, None)
    volume_low, volume_high, _ = paired_bootstrap(
        clean, one_wrong, 2000, 0, group_indices(ids, "volume")
    )
    assert (page_low, page_high) == (-0.25, -0.25)
    assert (volume_low, volume_high) == (-0.25, -0.25)


def test_the_two_groupings_can_disagree():
    """The grouped interval is only worth reporting if it can differ from the page one."""

    # One volume carries the whole effect and the other page does not, so resampling the volume
    # either brings in both its pages or neither -- a different spread from resampling pages.
    ids = ["mthv2_mth1000_01-V001P0021", "mthv2_mth1000_02-V001P0021", "mthv2_mth1000_037"]
    a = _rows([(ids[0], "甲乙丙丁", "甲乙丙丁"), (ids[1], "甲乙丙丁", "甲乙丙丁"), (ids[2], "甲乙丙丁", "甲乙丙丁")])
    b = _rows([(ids[0], "甲乙丙丁", "甲丁"), (ids[1], "甲乙丙丁", "甲乙丙丁"), (ids[2], "甲乙丙丁", "甲乙丙丁")])
    page_low, page_high, _ = paired_bootstrap(a, b, 4000, 7, None)
    volume_low, volume_high, _ = paired_bootstrap(a, b, 4000, 7, group_indices(ids, "volume"))
    assert (page_low, page_high) != (volume_low, volume_high)


def test_mismatched_page_sets_are_refused():
    other = _rows([("mthv2_mth1000_01-V001P0021", "甲乙", "甲乙")])
    try:
        paired_bootstrap(A, other, 10, 0, None)
    except ValueError as error:
        assert "same pages" in str(error) or "equal page counts" in str(error)
    else:
        raise AssertionError("a paired comparison across different pages must not proceed")
