from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "preprocessing"
    / "audit_ancientdoc_split_leakage.py"
)
SPEC = importlib.util.spec_from_file_location("ancientdoc_leakage_audit_under_test", MODULE_PATH)
audit_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit_module
SPEC.loader.exec_module(audit_module)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_image(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def record(
    split: str,
    index: int,
    *,
    category: str,
    book: str,
    page: int,
    text: str,
) -> dict[str, object]:
    page_id = f"ancientdoc_{split}_{index:06d}_imgs_{category}_{book}_page_{page}"
    return {
        "schema_version": 2,
        "input_level": "page",
        "layout_source": "real_ancientdoc",
        "layout_annotation_status": "none",
        "bbox_format": "xyxy_normalized",
        "page_id": page_id,
        "split": split,
        "tier": "real-ancientdoc",
        "source_group_id": f"ancientdoc_reference_{split}",
        "content_id": f"ancientdoc_reference_{split}_{index:06d}",
        "image": f"images/{page_id}.png",
        "original_image": f"imgs/{category}/{book}/page_{page}.png",
        "page_text": text,
        "page_text_separator": "",
        "regions": [],
    }


class AncientDocSplitLeakageAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.dataset_root = self.tmp_path / "ancientdoc"
        specs = {
            "train": [
                record("train", 1, category="catA", book="bookA", page=1, text="same text"),
                record("train", 2, category="catB", book="bookB", page=3, text="train only"),
            ],
            "validation": [
                record("validation", 1, category="catA", book="bookA", page=2, text="valid only"),
            ],
            "test": [
                record("test", 1, category="catC", book="bookC", page=9, text="same text"),
            ],
        }
        for split, records in specs.items():
            for item in records:
                write_image(self.dataset_root / split / str(item["image"]), b"image-" + bytes(item["page_id"], "utf-8"))
            write_jsonl(self.dataset_root / split / "manifest.jsonl", records)

    def test_audit_reports_book_and_text_cross_split_overlap(self) -> None:
        payload = audit_module.audit(
            audit_module.parse_args(
                [
                    "--dataset-root",
                    str(self.dataset_root),
                    "--skip-image-hash",
                    "--max-examples",
                    "5",
                ]
            )
        )
        self.assertEqual(payload["event"], "ancientdoc_split_leakage_audit_completed")
        self.assertEqual(payload["status"], "leakage_found")
        self.assertEqual(payload["book_key_cross_split"], 1)
        self.assertEqual(payload["normalized_text_cross_split"], 1)
        output_dir = Path(payload["output_dir"])
        self.assertTrue((output_dir / "split_leakage_audit.json").is_file())
        self.assertTrue((output_dir / "split_leakage_audit.md").is_file())
        report = json.loads((output_dir / "split_leakage_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(report["checks"]["book_key"]["cross_split_value_count"], 1)
        self.assertEqual(report["checks"]["normalized_text_sha256"]["cross_split_value_count"], 1)


if __name__ == "__main__":
    unittest.main()
