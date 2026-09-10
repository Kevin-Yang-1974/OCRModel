import importlib.util
import json
from argparse import Namespace
from pathlib import Path

from PIL import Image


def _module(name: str, relative_path: str):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, split: str, image_dir: Path, count: int) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        image = image_dir / f"{split}-{index}.png"
        Image.new("RGB", (8, 8), "white").save(image)
        rows.append(
            {
                "page_id": f"{split}-{index}",
                "split": split,
                "official_split": split,
                "image": f"images/{image.name}",
                "page_text": "甲乙",
                "regions": [
                    {
                        "bbox": [0, 0, 8, 8],
                        "reading_order": 0,
                        "writing_direction": "horizontal_ltr",
                    }
                ],
            }
        )
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_full_manifest_audit_resolves_relative_images(tmp_path: Path) -> None:
    module = _module("audit_mthv2_manifest", "tools/audit_mthv2_manifest.py")
    paths = {}
    for split in ("train", "validation", "test"):
        split_dir = tmp_path / split
        manifest = split_dir / "manifest.jsonl"
        _write_manifest(manifest, split, split_dir / "images", 2)
        paths[split] = manifest
    protocol = module.build_protocol(
        Namespace(
            train_manifest=paths["train"],
            validation_manifest=paths["validation"],
            test_manifest=paths["test"],
            num_queries=512,
            require_official_counts=False,
        )
    )
    assert protocol["split_pages"] == {"train": 2, "validation": 2, "test": 2}
    assert protocol["max_regions"] == 1
    assert len(protocol["manifest_stats"]["train"]["image_sha256"]) == 2
    assert protocol["test_used_for_selection"] is False


def test_manifest_audit_can_omit_test_without_opening_it(tmp_path: Path) -> None:
    module = _module("audit_mthv2_manifest_without_test", "tools/audit_mthv2_manifest.py")
    paths = {}
    for split in ("train", "validation"):
        split_dir = tmp_path / split
        manifest = split_dir / "manifest.jsonl"
        _write_manifest(manifest, split, split_dir / "images", 2)
        paths[split] = manifest
    protocol = module.build_protocol(
        Namespace(
            train_manifest=paths["train"],
            validation_manifest=paths["validation"],
            num_queries=512,
            require_official_counts=False,
            without_test=True,
        )
    )
    assert protocol["split_pages"] == {"train": 2, "validation": 2}
    assert protocol["test_manifest_read"] is False


def test_validation_subset_is_deterministic_and_validation_only(tmp_path: Path) -> None:
    module = _module("subset_mthv2_manifest", "tools/subset_mthv2_manifest.py")
    source = tmp_path / "validation.jsonl"
    _write_manifest(source, "validation", tmp_path / "images", 8)
    first = tmp_path / "subset-a.jsonl"
    second = tmp_path / "subset-b.jsonl"

    first_records = sorted(module.load_records(source), key=lambda record: record["page_id"])
    selected_a = module.random.Random(42).sample(first_records, 3)
    selected_a.sort(key=lambda record: record["page_id"])
    module.write_subset(first, selected_a)
    selected_b = module.random.Random(42).sample(first_records, 3)
    selected_b.sort(key=lambda record: record["page_id"])
    module.write_subset(second, selected_b)

    assert first.read_bytes() == second.read_bytes()
    assert len(first.read_text(encoding="utf-8").splitlines()) == 3
    assert all(
        json.loads(line)["split"] == "validation"
        for line in first.read_text(encoding="utf-8").splitlines()
    )


def test_full_manifest_audit_rejects_query_overflow(tmp_path: Path) -> None:
    module = _module("audit_mthv2_manifest_overflow", "tools/audit_mthv2_manifest.py")
    manifest = tmp_path / "train.jsonl"
    image = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(image)
    row = {
        "page_id": "overflow",
        "split": "train",
        "image": str(image),
        "page_text": "甲",
        "regions": [
            {"bbox": [0, 0, 1, 1], "reading_order": index}
            for index in range(3)
        ],
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    try:
        module.load_split(manifest, "train", num_queries=2)
    except ValueError as exc:
        assert "exceeding num_queries=2" in str(exc)
    else:
        raise AssertionError("query overflow was not rejected")


def test_full_summary_selects_common_validation_step(tmp_path: Path, monkeypatch) -> None:
    module = _module("summarize_mthv2_full", "tools/summarize_mthv2_full.py")
    for seed in (42, 43, 44):
        run_dir = tmp_path / f"seed{seed}"
        run_dir.mkdir()
        candidates = []
        for step, cer in ((432, 0.20), (864, 0.18)):
            candidates.append(
                {
                    "step": step,
                    "cer": cer + (seed - 42) * 0.001,
                    "exact_page_rate": 0.1,
                    "generation_limit_hit_rate": 0.05,
                    "residual_relative_norm": 0.008,
                    "writeback_residual_relative_norm": 0.008,
                    "checkpoint_health": {
                        "parameters_finite": True,
                        "checkpoint_finite": True,
                    },
                    "test_used_for_selection": False,
                }
            )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "test_used_for_selection": False,
                    "selection_candidates": candidates,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mode": "geometry",
                    "adapter_precision": "fp32",
                    "layout_loss_profile": "full",
                    "query_assignment": "hungarian",
                    "processor_mode": "fast",
                    "world_size": 5,
                    "num_queries": 512,
                    "distributed_strategy": "ddp",
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "sys.argv",
        ["summarize_mthv2_full.py", "--run-root", str(tmp_path), "--steps", "432,864"],
    )
    module.main()
    selection = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert selection["selected_step"] == 864
    assert selection["test_used_for_selection"] is False
    assert selection["stability"]["passed"] is True
