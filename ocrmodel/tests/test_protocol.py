import pytest

from layout_ocr.protocol import PageRecord, audit_split_isolation, select_mechanism_screen


def _records(count: int = 130) -> list[PageRecord]:
    return [
        PageRecord(f"page-{index:03d}", "train", f"source-{index}", f"dup-{index}")
        for index in range(count)
    ]


def test_mechanism_screen_is_reproducible() -> None:
    first = select_mechanism_screen(_records(), pages=128, seed=42)
    second = select_mechanism_screen(_records(), pages=128, seed=42)
    assert [item.page_id for item in first] == [item.page_id for item in second]
    assert len(first) == 128


def test_split_leakage_is_reported() -> None:
    records = [
        PageRecord("a", "train", "book-1", "scan-a"),
        PageRecord("b", "validation", "book-1", "scan-b"),
        PageRecord("c", "test", "book-2", "scan-a"),
    ]
    errors = audit_split_isolation(records)
    assert any("source_group book-1" in error for error in errors)
    assert any("duplicate_group scan-a" in error for error in errors)


def test_screen_rejects_insufficient_train_pages() -> None:
    with pytest.raises(ValueError, match="requested 128"):
        select_mechanism_screen(_records(127))
