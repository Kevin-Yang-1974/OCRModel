from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PREPARE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "preprocessing"
    / "prepare_mthv2_layout_dataset.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module(PREPARE_PATH, "mthv2_layout_prepare_under_test")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class MTHv2LayoutPrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.raw_root = self.tmp_path / "TKHMTH2200"
        self.split_root = self.tmp_path / "official_splits"
        self.split_root.mkdir()

        train_entries = []
        test_entries = []
        for subset_index, subset in enumerate(prepare.SUBSETS):
            for page_index in range(4):
                stem = f"page_{subset_index}_{page_index}"
                suffix = ".png" if subset == "MTH1000" else ".jpg"
                image_path = self.raw_root / subset / "img" / f"{stem}{suffix}"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (100, 200), color=(240, 240, 230)).save(image_path)

                textline_root = self.raw_root / subset / "label_textline"
                character_root = self.raw_root / subset / "label_char"
                boundary_root = self.raw_root / subset / "label_table"
                for root in (textline_root, character_root, boundary_root):
                    root.mkdir(parents=True, exist_ok=True)
                (textline_root / f"{stem}.txt").write_text(
                    "甲,乙,90,10,90,190,99,190,99,10\n"
                    "丙,10,20,10,180,20,180,20,20\n",
                    encoding="utf-8",
                )
                (character_root / f"{stem}.txt").write_text(
                    "甲 90.0 10 99 20\n乙 90 21 99 31\n", encoding="utf-8"
                )
                (boundary_root / f"{stem}.txt").write_text(
                    " ,50,0,50,200\n", encoding="utf-8"
                )
                entry = f"/official/dzj/{subset}/img/{image_path.name}"
                (test_entries if page_index == 3 else train_entries).append(entry)

        (self.split_root / "train.txt").write_text(
            "\n".join(train_entries) + "\n", encoding="utf-8"
        )
        (self.split_root / "test.txt").write_text(
            "\n".join(test_entries) + "\n", encoding="utf-8"
        )

    def args(self, output_root: Path) -> object:
        return prepare.parse_args(
            [
                "--raw-root",
                str(self.raw_root),
                "--train-list",
                str(self.split_root / "train.txt"),
                "--test-list",
                str(self.split_root / "test.txt"),
                "--output-root",
                str(output_root),
                "--validation-ratio",
                "0.34",
                "--seed",
                "17",
                "--copy-images",
            ]
        )

    def test_converts_all_subsets_and_preserves_official_test(self) -> None:
        output_root = self.tmp_path / "converted"
        summary = prepare.build_dataset(self.args(output_root))

        self.assertEqual(summary["total_pages"], 12)
        self.assertEqual(summary["official_split_counts"], {"test": 3, "train": 9})
        self.assertEqual(summary["splits"]["train"]["pages"], 6)
        self.assertEqual(summary["splits"]["validation"]["pages"], 3)
        self.assertEqual(summary["splits"]["test"]["pages"], 3)
        self.assertEqual(summary["image_storage"], "copied_from_raw")

        all_records = {
            split: load_jsonl(output_root / split / "manifest.jsonl")
            for split in prepare.OUTPUT_SPLITS
        }
        self.assertTrue(all(record["official_split"] == "test" for record in all_records["test"]))
        self.assertTrue(
            all(record["official_split"] == "train" for record in all_records["validation"])
        )
        self.assertEqual(
            {record["subset"] for records in all_records.values() for record in records},
            set(prepare.SUBSETS),
        )

        record = all_records["test"][0]
        self.assertEqual(record["page_text"], "甲,乙丙")
        self.assertEqual([region["reading_order"] for region in record["regions"]], [0, 1])
        self.assertEqual(record["regions"][0]["polygon_px"][0], [90.0, 10.0])
        self.assertEqual(record["regions"][0]["bbox_px"], [90.0, 10.0, 99.0, 190.0])
        self.assertEqual(record["regions"][0]["bbox"], [0.9, 0.05, 0.99, 0.95])
        self.assertTrue((output_root / "test" / record["image"]).is_file())

        sidecar = json.loads(
            (output_root / "test" / record["annotation_file"]).read_text(encoding="utf-8")
        )
        self.assertEqual(len(sidecar["characters"]), 2)
        self.assertEqual(sidecar["characters"][0]["bbox_xyxy_px"], [90.0, 10.0, 99.0, 20.0])
        self.assertEqual(sidecar["boundary_lines"][0]["start_px"], [50.0, 0.0])
        self.assertEqual(sidecar["boundary_lines"][0]["end_px"], [50.0, 200.0])

    def test_split_coverage_error_removes_new_output(self) -> None:
        test_list = self.split_root / "test.txt"
        test_list.write_text("/official/dzj/MTH1000/img/missing.png\n", encoding="utf-8")
        output_root = self.tmp_path / "failed_output"

        with self.assertRaisesRegex(ValueError, "coverage differs"):
            prepare.build_dataset(self.args(output_root))
        self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
