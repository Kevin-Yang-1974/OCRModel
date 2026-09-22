import importlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest


def prepare(root, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools/evaluation"))
    evaluate = importlib.import_module("evaluate_window_mask_routing")
    merge = importlib.import_module("merge_window_mask_shards")
    ids = [str(i) for i in range(149)]
    sizes = []
    for index in range(5):
        pages = evaluate.shard_records(ids, 5, index)
        sizes.append(len(pages))
        folder = root / "shards" / str(index)
        folder.mkdir(parents=True)
        summary = {
            "status": "complete",
            "profile": asdict(evaluate.PROFILE),
            "mode": "gt",
            "model_path": "model",
            "backbone_checkpoint": "checkpoint",
            "backbone_lora_sha256": "weights",
            "validation_sha256": evaluate.PROFILE.validation_sha256,
            "test_manifest_read": False,
            "test_used_for_selection": False,
            "reads_ground_truth_for_routing": True,
            "processor": "fast",
            "attention_backend": "math-sdpa",
            "precision": "bfloat16",
            "layout_branch_present": False,
            "shard_count": 5,
            "shard_index": index,
            "full_page_ids": ids,
            "page_ids": pages,
            "limited": False,
            "validation": {"pages": len(pages)},
        }
        (folder / "summary.json").write_text(json.dumps(summary))
        rows = [
            {
                "page_id": p,
                "reference": "a" * (int(p) + 1),
                "prediction": "a" * int(p),
                "generation_tokens": int(p),
                "generation_limit_hit": False,
            }
            for p in pages
        ]
        (folder / "validation_predictions.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return merge, sizes


def test_five_shards_cover_149_once_and_merge_by_edit_counts(tmp_path, monkeypatch):
    merge, sizes = prepare(tmp_path, monkeypatch)
    summary, rows = merge.merge_shards(tmp_path)
    assert sizes == [30, 30, 30, 30, 29]
    assert [r["page_id"] for r in rows] == [str(i) for i in range(149)]
    assert summary["validation"]["deletions"] == 149
    assert summary["validation"]["cer"] == 149 / sum(range(1, 150))
    assert summary["acceptance"]["passed"] is True


@pytest.mark.parametrize("fault", ["missing", "duplicate", "protocol"])
def test_merge_rejects_incomplete_or_mixed_results(tmp_path, monkeypatch, fault):
    merge, _ = prepare(tmp_path, monkeypatch)
    folder = tmp_path / "shards/2"
    if fault == "protocol":
        path = folder / "summary.json"
        data = json.loads(path.read_text())
        data["backbone_lora_sha256"] = "different"
        path.write_text(json.dumps(data))
    else:
        path = folder / "validation_predictions.jsonl"
        rows = path.read_text().splitlines()
        rows = rows[:-1] if fault == "missing" else rows + rows[:1]
        path.write_text("\n".join(rows))
    with pytest.raises(ValueError):
        merge.merge_shards(tmp_path)
