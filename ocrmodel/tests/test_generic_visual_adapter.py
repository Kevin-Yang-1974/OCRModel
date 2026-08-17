from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1] / "src" / "GOT-OCR-2.0" / "GOT" / "model"

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

@unittest.skipIf(torch is None, "PyTorch is not installed in the local static-test environment.")
class GenericVisualAdapterTests(unittest.TestCase):
    def test_forward_shape_gate_and_parameter_match(self) -> None:
        generic_module = load_module("generic_adapter_under_test", ROOT / "generic_adapter.py")
        layout_module = load_module("layout_query_for_capacity_test", ROOT / "layout_query.py")
        generic = generic_module.GenericVisualTransformerAdapter()
        vlqa = layout_module.VisualLayoutQueryAdapter()
        visual = torch.randn(2, 256, 1024)
        output = generic(visual)
        self.assertEqual(tuple(output.shape), tuple(visual.shape))
        self.assertTrue(torch.equal(output, visual))
        generic_count = sum(parameter.numel() for parameter in generic.parameters())
        auxiliary = ("prediction_norm.", "object_head.", "box_head.", "direction_head.")
        vlqa_ocr_count = sum(
            parameter.numel() for name, parameter in vlqa.named_parameters()
            if not name.startswith(auxiliary)
        )
        self.assertLess(abs(generic_count - vlqa_ocr_count) / vlqa_ocr_count, 0.01)

if __name__ == "__main__":
    unittest.main()
