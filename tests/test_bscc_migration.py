from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_bscc_bundle_uses_only_train_whole_pages(tmp_path: Path) -> None:
    module = load_module(
        ROOT / "tools" / "preprocessing" / "prepare_bscc_mthv2_smoke_bundle.py",
        "prepare_bscc_bundle_under_test",
    )
    image_root = tmp_path / "source"
    (image_root / "images").mkdir(parents=True)
    records = []
    for index in range(3):
        image = f"images/page-{index}.jpg"
        (image_root / image).write_bytes(bytes([index + 1]))
        records.append({
            "page_id": f"page-{index}", "split": "train", "input_level": "page",
            "image": image, "regions": [],
            "conversations": [{"from": "human", "value": "<image>\nOCR: "}],
        })
    manifest = image_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    output = tmp_path / "bundle"
    old_argv = sys.argv
    try:
        sys.argv = ["prepare", "--manifest", str(manifest), "--image-root", str(image_root),
                    "--output-dir", str(output), "--pages", "2"]
        module.main()
    finally:
        sys.argv = old_argv
    metadata = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    assert metadata["pages"] == 2
    assert metadata["source_split"] == "train"
    assert metadata["model_inputs"] == ["whole_page_image", "ocr_prompt"]
    assert metadata["layout_metadata_as_model_input"] is False
    assert metadata["test_read"] is False


def test_bscc_slurm_smoke_uses_allocated_devices_and_bounds_steps() -> None:
    source = (ROOT / "tools" / "training" / "run_bscc_pvld_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "BSCC_PVLD_STEPS must be 1..10" in source
    assert "--layout_split train" in source
    assert "--layout_image_root" in source
    assert "--layout_boundary_loss_weight 1.0" in source
    assert "--layout_count_condition_strength 1.0" in source
    assert 'export TMPDIR="${workspace}/tmp"' in source
    assert 'export TMPDIR="/tmp"' not in source
    assert "--test" not in source
    assert '"test_read": False' in source


def test_bscc_environment_pins_expected_stack() -> None:
    source = (ROOT / "tools" / "environment" / "setup_bscc_got2_env.sh").read_text(
        encoding="utf-8"
    )
    assert "torch-2.0.1+cu118" in source
    assert "deepspeed==0.12.3" in source
    assert "DS_BUILD_OPS=0" in source


def test_bscc_multigpu_entry_requests_two_slurm_gpus() -> None:
    source = (ROOT / "tools" / "training" / "run_bscc_multigpu_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:2" in source
    assert "torchrun --standalone --nproc_per_node=2" in source
    assert "deepspeed --num_gpus 2" in source
    assert "BSCC_PVLD_STEPS must be 1..10" in source


def test_bscc_nccl_smoke_binds_local_rank_device() -> None:
    source = (ROOT / "tools" / "training" / "run_bscc_smoke_component.py").read_text(
        encoding="utf-8"
    )
    assert 'torch.cuda.set_device(local_rank)' in source
    assert 'torch.device("cuda", local_rank)' in source
