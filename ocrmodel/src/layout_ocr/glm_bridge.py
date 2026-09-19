from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor, nn

from .adapter import LayoutAdapterOutput, PreMergeLayoutAdapter
from .config import AdapterPrecision, LayoutAdapterConfig, ValidityGatingMode


def patch_grid_positions(grid_thw: Tensor, spatial_merge_size: int) -> Tensor:
    """Return normalized centers in GLM-OCR's post-downsample token order."""

    if grid_thw.shape != (1, 3):
        raise ValueError("the mechanism screen requires exactly one whole-page image per step")
    temporal, height, width = (int(value) for value in grid_thw[0].tolist())
    if temporal != 1:
        raise ValueError("the whole-page protocol does not accept video/multi-frame input")
    merged_height = height // spatial_merge_size
    merged_width = width // spatial_merge_size
    rows = (torch.arange(merged_height, device=grid_thw.device, dtype=torch.float32) + 0.5)
    cols = (torch.arange(merged_width, device=grid_thw.device, dtype=torch.float32) + 0.5)
    rows = rows / merged_height
    cols = cols / merged_width
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1).unsqueeze(0)


def _probe_merger_attenuation(
    path: str,
    hidden_state: Tensor,
    merged_tokens: Tensor,
    base_merger: nn.Module,
) -> None:
    """Measure how much of the write-back survives the visual merger.

    The adapter writes into the *pre-merger* hidden state, so whatever the layout
    branch contributes still has to pass GLM-OCR's patch merger (neighbour
    concatenation plus a learned projection) before the language model can see
    it.  A perturbation of relative size ``in_rel`` can leave the merger as one of
    relative size ``out_rel``; ``attenuation = out_rel / in_rel`` is the transfer
    factor of the fusion seam.

    If the merger attenuates the write-back by orders of magnitude then the
    injection point, not the layout content, is what bounds the decoder's access
    to layout information -- and no re-parameterisation of the context at the same
    seam can fix it.  This is a forward-only, eval-only measurement; it runs an
    extra no-grad merger call, so it stays behind its own env var.
    """

    import json

    with torch.no_grad():
        plain = base_merger(hidden_state)
        perturbed = base_merger(merged_tokens)
        plain = plain.float()
        perturbed = perturbed.float()
        in_rel = float(
            (merged_tokens.float() - hidden_state.float()).norm(dim=-1).mean()
            / hidden_state.float().norm(dim=-1).mean().clamp_min(1e-12)
        )
        out_rel = float(
            (perturbed - plain).norm(dim=-1).mean()
            / plain.norm(dim=-1).mean().clamp_min(1e-12)
        )
        record = {
            "in_rel": in_rel,
            "out_rel": out_rel,
            "attenuation": None if in_rel <= 0 else out_rel / in_rel,
            "pre_tokens": int(hidden_state.shape[0]),
            "post_tokens": int(plain.shape[0]),
        }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + chr(10))


class LayoutAwarePatchMerger(nn.Module):
    """Insert the project adapter between GLM-OCR downsampling and patch merging."""

    def __init__(
        self,
        base_merger: nn.Module,
        adapter: PreMergeLayoutAdapter,
        spatial_merge_size: int,
        adapter_precision: AdapterPrecision = "mixed_bf16",
    ) -> None:
        super().__init__()
        if adapter_precision not in {"mixed_bf16", "fp32"}:
            raise ValueError(f"unsupported adapter precision: {adapter_precision}")
        self.base_merger = base_merger
        self.adapter = adapter
        self.spatial_merge_size = spatial_merge_size
        self.adapter_precision = adapter_precision
        self._grid_thw: Tensor | None = None
        self.last_output: LayoutAdapterOutput | None = None
        self.last_patch_positions: Tensor | None = None
        self.last_visual_tokens: Tensor | None = None
        self.last_residual: Tensor | None = None
        self.last_writeback_residual: Tensor | None = None
        self.last_input_dtype: torch.dtype | None = None
        self.last_adapter_input_dtype: torch.dtype | None = None
        self.last_adapter_output_dtype: torch.dtype | None = None
        self.last_merged_dtype: torch.dtype | None = None
        self._region_targets: dict[str, Tensor] | None = None
        self._region_pointer_mask: bool | None = None
        self._region_spatial_penalty: bool | None = None
        # Set by ``install_prefix_injection``.  When present the branch output is
        # also published as decoder prefix tokens; the write-back is left alone,
        # so the two routes can be run together or one at a time.
        self.prefix_splice: Any | None = None
        # Ground-truth region boxes for the oracle arm, or None for the normal
        # run.  Set per page by the caller; see ``set_oracle_boxes``.
        self._oracle_boxes: tuple[Tensor, Tensor] | None = None

    def set_grid_thw(self, grid_thw: Tensor) -> None:
        self._grid_thw = grid_thw

    def set_region_targets(self, targets: dict[str, Tensor] | None) -> None:
        """Set training-only ordered region targets for the AR decoder."""

        self._region_targets = targets

    def set_oracle_boxes(self, boxes: Tensor | None, mask: Tensor | None) -> None:
        """Substitute ground-truth regions for the predicted ones in the context.

        Eval-only and deliberately narrow: only the box values that feed the
        geometry term change, so an oracle run differs from a normal one in the
        *accuracy* of the layout and nothing else.  Without this the branch's own
        output cannot distinguish "the decoder ignores layout" from "the decoder
        is handed layout too inaccurate to be worth using".
        """

        self._oracle_boxes = None if boxes is None else (boxes, mask)

    def set_region_decode_controls(
        self, *, pointer_mask: bool | None = None, spatial_penalty: bool | None = None
    ) -> None:
        """Override inference-time duplicate guards without changing weights."""

        self._region_pointer_mask = pointer_mask
        self._region_spatial_penalty = spatial_penalty

    def forward(self, hidden_state: Tensor) -> Tensor:
        if self._grid_thw is None:
            raise RuntimeError("image_grid_thw must be set before the GLM-OCR visual forward")
        positions = patch_grid_positions(self._grid_thw, self.spatial_merge_size)
        expected_tokens = positions.shape[1]
        if hidden_state.shape[0] != expected_tokens:
            raise RuntimeError(
                f"pre-merge token mismatch: got {hidden_state.shape[0]}, expected {expected_tokens}"
            )
        adapter_input = hidden_state.unsqueeze(0).float()
        if self.adapter_precision == "fp32":
            with torch.autocast(device_type=hidden_state.device.type, enabled=False):
                output = self.adapter(
                    adapter_input,
                    positions,
                    region_targets=self._region_targets,
                    region_pointer_mask=self._region_pointer_mask,
                    region_spatial_penalty=self._region_spatial_penalty,
                    oracle_boxes=None if self._oracle_boxes is None else self._oracle_boxes[0],
                    oracle_mask=None if self._oracle_boxes is None else self._oracle_boxes[1],
                )
        else:
            output = self.adapter(
                adapter_input,
                positions,
                region_targets=self._region_targets,
                region_pointer_mask=self._region_pointer_mask,
                region_spatial_penalty=self._region_spatial_penalty,
                oracle_boxes=None if self._oracle_boxes is None else self._oracle_boxes[0],
                oracle_mask=None if self._oracle_boxes is None else self._oracle_boxes[1],
            )
        self.last_output = output
        self.last_patch_positions = positions
        if self.prefix_splice is not None:
            from .prefix_injection import publish_prefix_payload

            # Published here rather than read by the text-model hook because this
            # is the only place the branch output exists, and ``GlmOcrModel``
            # reaches the text model in the same forward -- see the module
            # docstring of ``prefix_injection`` for why that ordering is what
            # makes a single vision pass sufficient.
            publish_prefix_payload(self.prefix_splice, output)
        # Keep the un-anchored write-back for the attenuation probe: the graph
        # anchor added below is a DDP bookkeeping term with no forward meaning.
        raw_merged = output.merged_tokens.squeeze(0).to(hidden_state.dtype)
        adapted = raw_merged
        self.last_visual_tokens = hidden_state.detach()
        self.last_residual = (output.merged_tokens.squeeze(0).float() - hidden_state.float()).detach()
        self.last_writeback_residual = (adapted.float() - hidden_state.float()).detach()
        self.last_input_dtype = hidden_state.dtype
        self.last_adapter_input_dtype = adapter_input.dtype
        self.last_adapter_output_dtype = output.merged_tokens.dtype
        self.last_merged_dtype = adapted.dtype
        # Layout heads are consumed by the training loss through the bridge's
        # side-channel ``last_output`` rather than by the base model's return
        # object.  Keep a zero-valued dependency in the model graph so DDP
        # sees those parameters as used during the decoder forward.  Without
        # this anchor, full-model LoRA DDP can either hang on an unused head or
        # double-mark it when ``find_unused_parameters`` is enabled.
        auxiliary_tensors = [
            output.boxes,
            output.order_scores,
            output.direction_logits,
            output.validity_logits,
            output.validity_probs,
            output.gated_transport,
            output.valid_coverage,
        ]
        region_output = output.region_output
        if region_output is not None:
            # The AR branch replaces the legacy ``content_norm`` fusion path
            # with its region context.  Keep the normalization parameters in
            # the graph as a zero-valued dependency so five-process DDP sees
            # the same trainable parameter set on every step without changing
            # the forward value.
            auxiliary_tensors.append(self.adapter.content_norm(output.layout_queries))
            auxiliary_tensors.extend(
                [
                    region_output.pointer_logits,
                    region_output.boxes,
                    region_output.direction_logits,
                    region_output.eos_logits,
                    region_output.region_features,
                    region_output.candidate_objectness,
                    region_output.candidate_boxes,
                    region_output.count_logits,
                ]
            )
        graph_anchor = None
        for tensor in auxiliary_tensors:
            if tensor is None or tensor.numel() == 0 or not tensor.requires_grad:
                continue
            term = tensor.reshape(-1)[0] * 0.0
            graph_anchor = term if graph_anchor is None else graph_anchor + term
        if graph_anchor is not None:
            adapted = adapted + graph_anchor.to(dtype=adapted.dtype)
        probe_path = os.environ.get("GLMOCR_MERGER_ATTENUATION")
        if probe_path:
            _probe_merger_attenuation(
                probe_path, hidden_state, raw_merged, self.base_merger
            )
        return self.base_merger(adapted)


def install_layout_adapter(
    model: nn.Module,
    mode: str,
    num_queries: int = 32,
    max_residual_scale: float | None = None,
    initial_residual_scale: float = 0.0,
    use_validity_head: bool = False,
    initial_valid_probability: float = 0.05,
    validity_gating_mode: ValidityGatingMode = "legacy_normalized",
    validity_use_transport_evidence: bool = False,
    adapter_precision: AdapterPrecision = "mixed_bf16",
    region_autoregressive: bool = False,
    region_decoder_hidden_size: int = 256,
    region_decoder_layers: int = 2,
    region_decoder_num_heads: int = 8,
    region_pointer_mask: bool = True,
    region_spatial_penalty: float = 4.0,
    region_spatial_iou_threshold: float = 0.8,
    box_head_mlp: bool = False,
    box_head_hidden: int = 0,
    sem_adapter_mlp: bool = False,
    sem_adapter_hidden: int = 0,
    query_refine_layers: int = 0,
) -> LayoutAwarePatchMerger:
    """Install the adapter at the verified Transformers GLM-OCR pre-merger seam."""

    visual: Any = model.model.visual
    config = LayoutAdapterConfig(
        hidden_size=int(visual.config.out_hidden_size),
        num_queries=num_queries,
        num_heads=8,
        mode=mode,
        ot_epsilon=0.1,
        ot_relaxation=0.5,
        ot_iterations=20,
        max_residual_scale=max_residual_scale,
        initial_residual_scale=initial_residual_scale,
        use_validity_head=use_validity_head,
        initial_valid_probability=initial_valid_probability,
        validity_gating_mode=validity_gating_mode,
        validity_use_transport_evidence=validity_use_transport_evidence,
        region_autoregressive=region_autoregressive,
        region_decoder_hidden_size=region_decoder_hidden_size,
        region_decoder_layers=region_decoder_layers,
        region_decoder_num_heads=region_decoder_num_heads,
        region_pointer_mask=region_pointer_mask,
        region_spatial_penalty=region_spatial_penalty,
        region_spatial_iou_threshold=region_spatial_iou_threshold,
        box_head_mlp=box_head_mlp,
        box_head_hidden=box_head_hidden,
        sem_adapter_mlp=sem_adapter_mlp,
        sem_adapter_hidden=sem_adapter_hidden,
        query_refine_layers=query_refine_layers,
    )
    adapter = PreMergeLayoutAdapter(config).to(next(model.parameters()).device)
    bridge = LayoutAwarePatchMerger(
        base_merger=visual.merger,
        adapter=adapter,
        spatial_merge_size=int(visual.spatial_merge_size),
        adapter_precision=adapter_precision,
    )
    visual.merger = bridge
    return bridge
