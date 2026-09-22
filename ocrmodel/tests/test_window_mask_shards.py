import importlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest


def prepare(root, monkeypatch, mode="gt", target_mode="window", bias=1.0):
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
            "mode": mode,
            "target_mode": target_mode,
            "bias": bias,
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
            "layout_branch_present": mode == "legacy-line",
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


def test_merge_refuses_a_mode_other_than_the_one_the_caller_launched(tmp_path, monkeypatch):
    merge, _ = prepare(tmp_path, monkeypatch, mode="legacy-line")
    with pytest.raises(ValueError, match="legacy-line"):
        merge.merge_shards(tmp_path, expected_mode="gt")


def test_main_writes_a_verdict_only_for_the_acceptance_configuration(tmp_path, monkeypatch):
    # The diagnostic modes must produce numbers without producing anything that
    # reads as this fusion's acceptance result.
    merge, _ = prepare(tmp_path, monkeypatch, mode="legacy-line")
    monkeypatch.setattr(
        "sys.argv", ["merge_window_mask_shards.py", "--run-root", str(tmp_path), "--mode", "legacy-line"]
    )
    merge.main()
    out = tmp_path / "results"
    assert not (out / "summary.json").exists()
    assert json.loads((out / "merged.json").read_text())["validation"]["deletions"] == 149


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


def test_acceptance_is_not_inherited_by_a_non_window_target(tmp_path, monkeypatch):
    """mode=gt with an anchored target is a diagnostic, not the acceptance run.

    The acceptance criterion was fixed for the 3-5 character window.  A different
    spatial target measured under the same mode must not come back as a verdict,
    or a stronger target could be reported as the recorded configuration passing.
    """

    merge, _ = prepare(tmp_path, monkeypatch, mode="gt", target_mode="anchored")
    summary, _ = merge.merge_shards(tmp_path)
    assert summary["acceptance"]["eligible"] is False
    assert summary["acceptance"]["passed"] is None
    # The window target under the same mode still scores.
    other = tmp_path / "window"
    other.mkdir()
    merge2, _ = prepare(other, monkeypatch, mode="gt", target_mode="window")
    summary2, _ = merge2.merge_shards(other)
    assert summary2["acceptance"]["eligible"] is True


def test_acceptance_is_not_inherited_by_a_swept_bias(tmp_path, monkeypatch):
    """A beta sweep is a diagnostic; only the recorded B=1.0 can pass.

    The window arm inherited B=1.0 from the whole-line arm and it was never
    swept for this target, so the sweep has to be free to find a better value
    without any of its points coming back as the recorded configuration passing.
    """

    merge, _ = prepare(tmp_path, monkeypatch, mode="gt", target_mode="window", bias=2.0)
    summary, _ = merge.merge_shards(tmp_path)
    assert summary["acceptance"]["eligible"] is False
    assert summary["acceptance"]["passed"] is None
    assert summary["bias"] == 2.0

    other = tmp_path / "recorded"
    other.mkdir()
    merge2, _ = prepare(other, monkeypatch, mode="gt", target_mode="window", bias=1.0)
    summary2, _ = merge2.merge_shards(other)
    assert summary2["acceptance"]["eligible"] is True
