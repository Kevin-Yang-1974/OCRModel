from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

ABLATION_IDS = ("got2_zero_shot", "projector_only", "generic_adapter_projector",
                "vlqa_ocr_only", "vlqa_layout_direct", "vlqa_layout_p1_p2")

LOSS_PRESETS: dict[str, dict[str, float]] = {
    "layout_none": {"object": 0.0, "bbox_l1": 0.0, "bbox_giou": 0.0, "direction_order": 0.0, "layout": 0.0},
    "object_only": {"object": 1.0, "bbox_l1": 0.0, "bbox_giou": 0.0, "direction_order": 0.0, "layout": 1.0},
    "object_bbox": {"object": 1.0, "bbox_l1": 5.0, "bbox_giou": 2.0, "direction_order": 0.0, "layout": 1.0},
    "object_direction_order": {"object": 1.0, "bbox_l1": 0.0, "bbox_giou": 0.0, "direction_order": 1.0, "layout": 1.0},
    "layout_full": {"object": 1.0, "bbox_l1": 5.0, "bbox_giou": 2.0, "direction_order": 1.0, "layout": 1.0},
}

@dataclass(frozen=True)
class AblationSpec:
    use_vlqa: bool
    use_generic_adapter: bool
    train_projector: bool
    p1_required: bool
    training_allowed: bool = True

ABLATIONS = {
    "got2_zero_shot": AblationSpec(False, False, False, False, False),
    "projector_only": AblationSpec(False, False, True, False),
    "generic_adapter_projector": AblationSpec(False, True, True, False),
    "vlqa_ocr_only": AblationSpec(True, False, True, False),
    "vlqa_layout_direct": AblationSpec(True, False, True, False),
    "vlqa_layout_p1_p2": AblationSpec(True, False, True, True),
}

def expected_trainable_modules(ablation: str, stage: str) -> set[str]:
    if ablation == "got2_zero_shot":
        return set()
    if ablation == "projector_only":
        return {"mm_projector_vary"}
    if ablation == "generic_adapter_projector":
        return {"mm_projector_vary", "generic_adapter"}
    if ablation in {"vlqa_ocr_only", "vlqa_layout_direct"}:
        return {"mm_projector_vary", "vlqa"}
    if ablation == "vlqa_layout_p1_p2":
        return {"vlqa"} if stage == "p1" else {"mm_projector_vary", "vlqa"}
    raise ValueError(f"Unknown ablation: {ablation!r}.")

def assert_parameter_report(ablation: str, stage: str,
                            report: Mapping[str, Mapping[str, object]]) -> None:
    if int(report["vary_vit"]["trainable"]) != 0 or int(report["qwen"]["trainable"]) != 0:
        raise ValueError("Formal ablations require frozen Vary ViT and Qwen.")
    actual = {
        name for name in ("mm_projector_vary", "generic_adapter", "vlqa")
        if int(report[name]["trainable"]) > 0
    }
    expected = expected_trainable_modules(ablation, stage)
    if actual != expected:
        raise ValueError(
            f"{ablation} {stage.upper()} trainable modules mismatch: "
            f"actual={sorted(actual)}, expected={sorted(expected)}."
        )

def loss_weights_for(ablation: str, stage: str, preset: str,
                     ocr_weight: float) -> dict[str, float]:
    if ablation not in ABLATIONS or preset not in LOSS_PRESETS:
        raise ValueError(f"Unknown ablation or layout loss preset: {ablation!r}, {preset!r}.")
    if stage not in {"p1", "p2"}:
        raise ValueError(f"Unknown training stage: {stage!r}.")
    if ablation == "got2_zero_shot":
        raise ValueError("got2_zero_shot does not create a training optimizer.")
    if stage == "p1" and ablation != "vlqa_layout_p1_p2":
        raise ValueError(f"{ablation} must enter P2 directly and cannot run P1.")
    layout = dict(LOSS_PRESETS[preset])
    if ablation in {"projector_only", "generic_adapter_projector", "vlqa_ocr_only"}:
        if preset != "layout_none":
            raise ValueError(f"{ablation} requires --layout-loss-preset layout_none.")
        layout = dict(LOSS_PRESETS["layout_none"])
    if ablation in {"vlqa_layout_direct", "vlqa_layout_p1_p2"} and preset == "layout_none":
        raise ValueError(f"{ablation} requires at least one enabled layout loss.")
    if stage == "p1" and ocr_weight != 0.0:
        raise ValueError("P1 requires OCR loss weight 0.")
    if stage == "p2" and ocr_weight <= 0.0:
        raise ValueError("P2 requires a positive OCR loss weight.")
    return {**layout, "ocr": float(ocr_weight)}

def assert_source_protocol(ablation: str, stage: str,
                           source_config: Mapping[str, object],
                           source_metrics: Mapping[str, object] | None) -> None:
    source_vlqa = source_config.get("use_vlqa") is True
    if source_config.get("use_generic_adapter") is True:
        raise ValueError("Formal ablations cannot initialize from a generic-adapter checkpoint.")
    if ablation == "vlqa_layout_p1_p2" and stage == "p2":
        if not source_vlqa or not source_metrics or source_metrics.get("layout_stage") != "p1":
            raise ValueError("A5 P2 must load the completed P1 checkpoint from the same A5 run.")
        if source_metrics.get("ablation_id") != ablation:
            raise ValueError("A5 P2 source checkpoint has the wrong ablation_id.")
    elif source_vlqa:
        raise ValueError(f"{ablation} {stage.upper()} must start from original GOT2 without VLQA.")
