from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "preprocessing"
    / "split_mthv2_layout_chunks.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("mthv2_chunks_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chunks = load_module()


class MTHv2LayoutChunkTests(unittest.TestCase):
    def test_33_regions_become_16_16_1_and_coordinates_are_rebased(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        input_root = root / "input"
        output_root = root / "output"
        for split in chunks.SPLITS:
            split_root = input_root / split
            (split_root / "images").mkdir(parents=True)
            image_path = split_root / "images" / f"{split}.png"
            Image.new("RGB", (660, 100), color="white").save(image_path)
            regions = []
            for index in range(33):
                x1 = 660 - index * 20
                x0 = x1 - 12
                regions.append(
                    {
                        "region_id": f"{split}_region_{index}",
                        "content_id": f"{split}_content_{index}",
                        "source_group_id": f"{split}_group",
                        "text": str(index),
                        "polygon_px": [[x0, 5], [x0, 95], [x1, 95], [x1, 5]],
                        "bbox_px": [x0, 5, x1, 95],
                        "bbox": [x0 / 660, 0.05, x1 / 660, 0.95],
                        "reading_order": index,
                        "writing_direction": "vertical_rtl",
                    }
                )
            record = {
                "schema_version": 2,
                "input_level": "page",
                "layout_source": "real_mthv2_official",
                "layout_annotation_status": "complete",
                "layout_level": "textline",
                "page_id": f"{split}_page",
                "split": split,
                "source_group_id": f"{split}_group",
                "source_group_ids": [f"{split}_group"],
                "content_id": f"{split}_page",
                "image": f"images/{split}.png",
                "annotation_file": f"annotations/{split}.json",
                "page_size": [660, 100],
                "page_text": "".join(str(index) for index in range(33)),
                "regions": regions,
                "conversations": [],
            }
            (split_root / "manifest.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

        summary = chunks.build_dataset(
            chunks.parse_args(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--max-regions",
                    "16",
                    "--margin-pixels",
                    "0",
                ]
            )
        )
        self.assertEqual(summary["source_pages"], 3)
        self.assertEqual(summary["output_chunks"], 9)
        self.assertEqual(summary["chunk_region_count_distribution"], {"1": 3, "16": 6})

        records = [
            json.loads(line)
            for line in (output_root / "train" / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual([len(record["regions"]) for record in records], [16, 16, 1])
        self.assertEqual(records[0]["source_region_indices"], list(range(16)))
        self.assertEqual(records[1]["source_region_indices"], list(range(16, 32)))
        self.assertEqual(records[2]["source_region_indices"], [32])
        self.assertEqual(records[0]["regions"][0]["reading_order"], 0)
        self.assertEqual(records[0]["regions"][0]["bbox_px"], [300.0, 5.0, 312.0, 95.0])
        self.assertEqual(records[0]["source_crop_box_px"], [348, 0, 660, 100])
        self.assertTrue((output_root / "train" / records[0]["image"]).is_file())


if __name__ == "__main__":
    unittest.main()
