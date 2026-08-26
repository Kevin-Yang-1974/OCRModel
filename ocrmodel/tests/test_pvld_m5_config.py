from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools" / "training" / "pvld_m5_config.py"
SPEC = importlib.util.spec_from_file_location("pvld_m5_config", MODULE)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_m5_registry_is_configuration_only_and_queries_are_not_region_slots() -> None:
    registry = module.registry()
    assert list(registry) == ["K16", "K32", "K64"]
    assert [registry[key]["num_prompt_queries"] for key in registry] == [16, 32, 64]
    assert all(registry[key]["parameter_estimate"] > 0 for key in registry)
    assert all(registry[key]["attention_flops_estimate"] > 0 for key in registry)
