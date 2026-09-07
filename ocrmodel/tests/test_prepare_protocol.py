import importlib.util
from pathlib import Path

from PIL import Image


def _module():
    path = Path(__file__).parents[1] / "tools" / "prepare_bscc_protocol.py"
    spec = importlib.util.spec_from_file_location("prepare_bscc_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_group_derivation() -> None:
    module = _module()
    assert module.derive_version_group(
        {"subset": "MTH1000", "original_image": "MTH1000/img/01-V007P0009.png"}
    ) == "MTH1000:01-V007"
    assert module.derive_version_group(
        {"subset": "TKH", "original_image": "TKH/img/005.jpg"}
    ) == "TKH:UNVERSIONED_NUMERIC"


def test_dhash_is_deterministic(tmp_path: Path) -> None:
    module = _module()
    image = tmp_path / "page.png"
    Image.new("RGB", (20, 20), "white").save(image)
    assert module.image_dhash(image) == module.image_dhash(image)
