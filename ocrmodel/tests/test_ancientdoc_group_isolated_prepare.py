from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PREPARE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "preprocessing"
    / "prepare_ancientdoc_group_isolated_dataset.py"
)
AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "preprocessing"
    / "audit_ancientdoc_split_leakage.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module(PREPARE_PATH, "ancientdoc_group_prepare_under_test")
audit_module = load_module(AUDIT_PATH, "ancientdoc_group_prepare_audit_under_test")


PROMPT = "<image>\nOCR: "


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def source_record(category: str, book: str, page: int, text: str) -> dict[str, object]:
    image = f"imgs/{category}/{book}/page_{page}.png"
    return {
        "image": image,
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": text},
        ],
    }


class AncientDocGroupIsolatedPrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.source_root = self.tmp_path / "AncientDoc"
        records_by_split = {
            1: [
                source_record("catA", "bookA", 1, "book A page 1"),
                source_record("catB", "bookB", 1, "book B page 1"),
            ],
            2: [
                source_record("catA", "bookA", 2, "book A page 2"),
                source_record("catC", "bookC", 1, "book C page 1"),
            ],
            3: [source_record("catD", "bookD", 1, "book D page 1")],
        }
        for split_id, records in records_by_split.items():
            write_json(self.source_root / f"label_for_got_split{split_id}.json", records)
            for record in records:
                image_path = self.source_root / str(record["image"])
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"fake-image-" + bytes(str(record["image"]), "utf-8"))

    def test_group_isolated_prepare_keeps_books_in_one_split(self) -> None:
        output_root = self.tmp_path / "converted"
        summary = prepare.build_dataset(
            prepare.parse_args(
                [
                    "--ancientdoc-root",
                    str(self.source_root),
                    "--output-root",
                    str(output_root),
                    "--source-splits",
                    "1,2,3",
                    "--seed",
                    "17",
                    "--train-ratio",
                    "0.5",
                    "--validation-ratio",
                    "0.25",
                    "--test-ratio",
                    "0.25",
                    "--max-ratio-deviation",
                    "0.2",
                ]
            )
        )
        self.assertEqual(summary["event"], "ancientdoc_group_isolated_dataset_prepared")
        self.assertEqual(summary["total_records"], 5)
        self.assertEqual(summary["total_groups"], 4)
        self.assertLessEqual(
            summary["allocation"]["max_absolute_ratio_deviation"],
            0.2,
        )
        for split in ("train", "validation", "test"):
            self.assertTrue((output_root / split / "manifest.jsonl").is_file())

        audit_payload = audit_module.audit(
            audit_module.parse_args(
                [
                    "--dataset-root",
                    str(output_root),
                    "--skip-image-hash",
                ]
            )
        )
        self.assertEqual(audit_payload["status"], "ok")
        self.assertEqual(audit_payload["book_key_cross_split"], 0)

    def test_allocator_tracks_requested_page_ratios(self) -> None:
        pages = []
        for group_index in range(30):
            for page_index in range(1 + group_index % 4):
                pages.append(
                    prepare.SourcePage(
                        split_id=1,
                        index=len(pages),
                        record={},
                        image_relative=prepare.PurePosixPath(
                            f"imgs/cat/book{group_index}/page_{page_index}.png"
                        ),
                        image_absolute=self.source_root,
                        category="cat",
                        book=f"book{group_index}",
                        book_key=f"cat/book{group_index}",
                        page_text="x",
                    )
                )
        assignments = prepare.assign_groups(
            pages,
            seed=17,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
        )
        counts = {split: 0 for split in prepare.TARGET_SPLITS}
        for page in pages:
            counts[assignments[page.book_key]] += 1
        total = len(pages)
        self.assertLessEqual(abs(counts["train"] / total - 0.6), 0.03)
        self.assertLessEqual(abs(counts["validation"] / total - 0.2), 0.03)
        self.assertLessEqual(abs(counts["test"] / total - 0.2), 0.03)


if __name__ == "__main__":
    unittest.main()
