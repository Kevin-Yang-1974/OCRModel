"""Versioned registry for internal and external MTHv2 comparison controls.

The registry deliberately records the official identifiers that were verified
against the provider APIs on 2026-08-23. It is metadata, not a claim that a
model has already been run on MTHv2.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    official_repo: str
    checkpoint_id: str
    revision: str
    license: str
    parameter_count: int | None
    parameter_count_source: str
    input_format: str
    output_format: str
    zero_shot_supported: bool
    finetune_status: str
    finetune_source: str
    adapter: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EXTERNAL_MODELS: dict[str, ModelSpec] = {
    "paddleocr_vl_1_6": ModelSpec(
        name="PaddleOCR-VL-1.6",
        family="external_sota",
        official_repo="https://github.com/PaddlePaddle/PaddleOCR",
        checkpoint_id="PaddlePaddle/PaddleOCR-VL-1.6",
        revision="c5630abae1d940eafe0697512a0325494b02ab42",
        license="Apache-2.0",
        parameter_count=900_000_000,
        parameter_count_source="official model naming/OmniDocBench model registry; verify runtime config",
        input_format="whole_page_image",
        output_format="text_or_markdown_or_structured",
        zero_shot_supported=True,
        finetune_status="official_entrypoint_available",
        finetune_source="PaddleOCR official PaddleOCR-VL tutorial section 5; ERNIEKit SFT (layout analysis/ranking fine-tuning remains unsupported)",
        adapter="paddleocr_vl",
        notes="Do not substitute PaddleOCR-VL-1.5 when 1.6 is requested.",
    ),
    "mineru2_5_pro": ModelSpec(
        name="MinerU2.5-Pro",
        family="external_sota",
        official_repo="https://github.com/opendatalab/MinerU",
        checkpoint_id="opendatalab/MinerU2.5-Pro-2605-1.2B",
        revision="bff20d4ae2bf202df9f45284b4d43681555a97ed",
        license="Apache-2.0",
        parameter_count=1_200_000_000,
        parameter_count_source="official checkpoint name/model card; verify runtime config",
        input_format="whole_page_image",
        output_format="markdown_or_structured",
        zero_shot_supported=True,
        finetune_status="official_finetuning_unavailable",
        finetune_source="MinerU official repository/model card; no verified public fine-tuning entry",
        adapter="mineru",
    ),
    "glm_ocr": ModelSpec(
        name="GLM-OCR",
        family="external_sota",
        official_repo="https://github.com/zai-org/GLM-OCR",
        checkpoint_id="zai-org/GLM-OCR",
        revision="ca5d8b3e287e52589e37c28385d9655ee4372f9d",
        license="MIT",
        parameter_count=900_000_000,
        parameter_count_source="official README (0.9B); verify runtime config",
        input_format="whole_page_image",
        output_format="markdown_and_json_layout",
        zero_shot_supported=True,
        finetune_status="official_entrypoint_available",
        finetune_source="GLM-OCR examples/finetune via LLaMA-Factory",
        adapter="glm_ocr",
    ),
    "opendoc_0_1b": ModelSpec(
        name="OpenDoc-0.1B",
        family="external_sota",
        official_repo="https://github.com/Topdu/OpenOCR",
        checkpoint_id="topdu/unirec-0.1b",
        revision="a377e00d62c01b6544603e2a90f2cffe2a0388e1",
        license="Apache-2.0",
        parameter_count=100_000_000,
        parameter_count_source="OpenOCR docs/opendoc.md (0.1B); system also uses PP-DocLayoutV2",
        input_format="whole_page_image",
        output_format="markdown_and_json_layout",
        zero_shot_supported=True,
        finetune_status="official_finetuning_unavailable",
        finetune_source="OpenOCR OpenDoc documentation; inference/download only verified",
        adapter="opendoc",
        notes="The official OpenDoc system checkpoint is UniRec, not a guessed OpenDoc HF model ID.",
    ),
}


INTERNAL_BASELINES: dict[str, dict[str, Any]] = {
    "B0": {"name": "official GOT2", "mode": "zero_shot", "architecture": "got2", "training_allowed": False},
    "B1": {"name": "GOT2 OCR-only", "mode": "mthv2_adapted", "architecture": "got2", "layout_loss": "none"},
    "B2": {"name": "GOT2 equal-parameter generic adaptor", "mode": "mthv2_adapted", "architecture": "generic_adapter", "trainable_parameter_target": 5_280_536},
    "B3": {"name": "Fixed-Slot VLQA-K32", "mode": "mthv2_adapted", "architecture": "fixed_slot_k32", "capacity_limit": 32},
    "B4": {"name": "causal PVLD C3", "mode": "mthv2_adapted", "architecture": "pvld", "run_reuse_required": True},
    "B5": {"name": "causal PVLD C4", "mode": "mthv2_adapted", "architecture": "pvld", "run_reuse_required": True},
    "B6": {"name": "causal PVLD C5", "mode": "mthv2_adapted", "architecture": "pvld", "extra_p1_budget": True, "run_reuse_required": True},
}


def get_model(name: str) -> ModelSpec:
    try:
        return EXTERNAL_MODELS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown external SOTA model: {name!r}") from exc
