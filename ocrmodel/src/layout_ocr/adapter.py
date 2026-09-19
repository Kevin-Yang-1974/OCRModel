from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .autoregressive_region import (
    AutoregressiveRegionDecoder,
    RegionDecoderOutput,
    RegionDecoderConfig,
)
from .config import LayoutAdapterConfig, ValidityGatingMode
from .transport import SemiRelaxedTransport


@dataclass
class LayoutAdapterOutput:
    merged_tokens: Tensor
    layout_queries: Tensor
    boxes: Tensor
    order_scores: Tensor
    direction_logits: Tensor
    transport: Tensor | None
    validity_logits: Tensor | None = None
    validity_probs: Tensor | None = None
    gated_transport: Tensor | None = None
    valid_coverage: Tensor | None = None
    validity_gating_mode: ValidityGatingMode | None = None
    region_output: RegionDecoderOutput | None = None


def _probe_layout_writeback(path: str, token_weights: Tensor | None, layout_context: Tensor,
                            visual_tokens: Tensor, residual_scale: Tensor | None = None,
                            intervention: str = "full", oracle: bool = False) -> None:
    """Diagnostic hook: how much spatial information the write-back actually carries.

    The residual reaches the decoder only as ``layout_context[p]``, a per-patch
    combination of query features.  If that vector barely varies across patches it
    degenerates into a global offset, and no amount of layout-branch quality can
    put spatial or reading-order information into it.  ``lc_flat`` <= ~0.1 is that
    degenerate regime; ``vt_flat`` is the same statistic for the visual tokens, so a
    comparison is against a representation that is known to be patch-specific.

    ``lc_norm_over_vt`` is the norm of the *context*, not of the injected residual.
    The write-back is ``alpha * layout_context`` with ``alpha`` clipped to
    ``max_residual_scale`` (0.03 in every current entry point), so the perturbation
    that actually reaches the merger is ``inj_over_vt = alpha * lc_norm_over_vt`` --
    roughly 30x smaller than ``lc_norm_over_vt`` at the cap.  Reporting both keeps
    "the amplitude is sufficient" from being read off the wrong number.
    """

    import json, os

    from .writeback_intervention import flatness as _flatness

    with torch.no_grad():
        def flatness(x: Tensor) -> float:
            return float(_flatness(x))

        visual_norm = visual_tokens.float().norm(dim=-1).mean().clamp_min(1e-12)
        context_norm = layout_context.float().norm(dim=-1).mean()
        mean = layout_context.float().mean(dim=1, keepdim=True)
        scale = None if residual_scale is None else float(residual_scale.detach())
        record = {
            "patches": int(visual_tokens.shape[1]),
            "lc_flat": flatness(layout_context),
            "vt_flat": flatness(visual_tokens),
            "tw_flat": float(token_weights.float().std(dim=1).mean()) if token_weights is not None else None,
            "lc_norm_over_vt": float(context_norm / visual_norm),
            "lc_global_share": float(mean.norm(dim=-1).mean() / context_norm.clamp_min(1e-12)),
            "lc_spatial_share": float(
                (layout_context.float() - mean).norm(dim=-1).mean() / context_norm.clamp_min(1e-12)
            ),
            "residual_scale": scale,
            # The number the "amplitude is sufficient" claim depends on.
            "inj_over_vt": None if scale is None else float(scale * context_norm / visual_norm),
            "intervention": intervention,
            "oracle_boxes": bool(oracle),
        }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + chr(10))


class PreMergeLayoutAdapter(nn.Module):
    """Generate layout queries from whole-page visual tokens before VLM merging.

    ``patch_positions`` are deterministic vision-grid coordinates, not layout
    annotations. Ground-truth layout is deliberately absent from this interface.
    """

    def __init__(self, config: LayoutAdapterConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.query_seed = nn.Parameter(torch.empty(config.num_queries, d))
        nn.init.normal_(self.query_seed, std=0.02)
        self.query_attention = nn.MultiheadAttention(
            d, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(d)
        self.content_norm = nn.LayerNorm(d)
        self.box_head = nn.Linear(d, 4)
        self.geom_adapter: nn.Sequential | None = None
        if config.box_head_mlp:
            hidden = config.box_head_hidden or d
            self.geom_adapter = nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Linear(hidden, d),
            )
            nn.init.zeros_(self.geom_adapter[-1].weight)
            nn.init.zeros_(self.geom_adapter[-1].bias)
        self.sem_adapter: nn.Sequential | None = None
        if config.sem_adapter_mlp:
            hidden = config.sem_adapter_hidden or d
            self.sem_adapter = nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Linear(hidden, d),
            )
            nn.init.zeros_(self.sem_adapter[-1].weight)
            nn.init.zeros_(self.sem_adapter[-1].bias)
        self.query_refine: nn.ModuleList | None = None
        if config.query_refine_layers > 0:
            self.query_refine = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=config.num_heads,
                    dim_feedforward=4 * d,
                    dropout=config.dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.query_refine_layers)
            )
        self.order_head = nn.Linear(d, 1)
        self.direction_head = nn.Linear(d, config.num_directions)
        self.validity_head: nn.Linear | None = None
        if config.use_validity_head:
            self.validity_head = nn.Linear(d, 1)
            nn.init.zeros_(self.validity_head.weight)
            initial_probability = config.initial_valid_probability
            initial_logit = math.log(initial_probability / (1.0 - initial_probability))
            nn.init.constant_(self.validity_head.bias, initial_logit)
        initial_gate = math.atanh(config.initial_residual_scale)
        self.content_gate = nn.Parameter(torch.tensor(initial_gate, dtype=torch.float32))
        self.ot = SemiRelaxedTransport(
            config.ot_epsilon, config.ot_relaxation, config.ot_iterations
        )
        self.region_decoder: AutoregressiveRegionDecoder | None = None
        self.region_context_projection: nn.Linear | None = None
        if config.region_autoregressive:
            self.region_decoder = AutoregressiveRegionDecoder(
                RegionDecoderConfig(
                    input_hidden_size=d,
                    decoder_hidden_size=config.region_decoder_hidden_size,
                    num_heads=config.region_decoder_num_heads,
                    num_layers=config.region_decoder_layers,
                    candidate_count=config.num_queries,
                    max_regions=config.num_queries,
                    num_directions=config.num_directions,
                    pointer_mask=config.region_pointer_mask,
                    spatial_penalty=config.region_spatial_penalty,
                    spatial_iou_threshold=config.region_spatial_iou_threshold,
                )
            )
            self.region_context_projection = nn.Linear(
                config.region_decoder_hidden_size, d
            )

    def effective_residual_scale(self) -> Tensor:
        """Return the checkpoint-configured residual scale.

        ``None`` preserves the original unbounded ``tanh(gate)`` behavior for
        legacy checkpoints.  New stabilization runs clamp the effective scale
        without changing the stored/raw gate parameter.
        """

        scale = self.content_gate.tanh()
        if self.config.max_residual_scale is not None:
            scale = scale.clamp(
                min=-self.config.max_residual_scale,
                max=self.config.max_residual_scale,
            )
        return scale

    def _queries(self, visual_tokens: Tensor) -> Tensor:
        seed = self.query_seed.unsqueeze(0).expand(visual_tokens.shape[0], -1, -1)
        update, _ = self.query_attention(seed, visual_tokens, visual_tokens, need_weights=False)
        queries = self.query_norm(seed + update)
        if self.query_refine is not None:
            for layer in self.query_refine:
                queries = layer(queries)
        return queries

    def _scores(
        self, queries: Tensor, visual_tokens: Tensor, boxes: Tensor, patch_positions: Tensor | None
    ) -> Tensor:
        scores = torch.matmul(queries, visual_tokens.transpose(-1, -2))
        scores = scores / self.config.hidden_size**0.5
        if self.config.mode in {"geometry", "layout_ot"}:
            if patch_positions is None:
                raise ValueError(f"patch_positions are required for mode={self.config.mode}")
            centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
            distance = torch.cdist(centers.float(), patch_positions.float(), p=2)
            scores = scores - distance.to(scores.dtype) / self.config.geometry_temperature
        return scores

    def _raw_mass_context(
        self,
        transport: Tensor,
        sem: Tensor,
        validity_probs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fuse queries while retaining raw token mass for validity gating."""

        raw_token_mass = transport.sum(dim=1)
        raw_token_weights = transport.transpose(1, 2) / raw_token_mass.unsqueeze(-1).clamp_min(
            1e-12
        )
        weighted_queries = raw_token_weights * validity_probs.unsqueeze(1)
        valid_coverage = weighted_queries.sum(dim=-1)
        layout_context = self.content_norm(torch.matmul(weighted_queries, sem))
        return layout_context * valid_coverage.unsqueeze(-1), valid_coverage

    def _region_context(
        self,
        region_output: RegionDecoderOutput,
        patch_positions: Tensor | None,
        token_count: int,
    ) -> Tensor:
        if self.region_context_projection is None:
            raise RuntimeError("region context projection is not initialized")
        features = self.region_context_projection(region_output.region_features)
        mask = region_output.selected_mask.to(features.dtype)
        if patch_positions is None:
            weights = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            context = torch.einsum("bs,bsd->bd", weights, features).unsqueeze(1)
            return context.expand(-1, token_count, -1)
        centers = (region_output.boxes[..., :2] + region_output.boxes[..., 2:]) * 0.5
        distances = torch.cdist(centers.float(), patch_positions.float(), p=2)
        weights = torch.exp(-distances / 0.15) * mask.unsqueeze(-1)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bts,bsd->btd", weights.transpose(1, 2), features)

    def forward(
        self,
        visual_tokens: Tensor,
        patch_positions: Tensor | None = None,
        *,
        region_targets: dict[str, Tensor] | None = None,
        region_pointer_mask: bool | None = None,
        region_spatial_penalty: bool | None = None,
        oracle_boxes: Tensor | None = None,
        oracle_mask: Tensor | None = None,
    ) -> LayoutAdapterOutput:
        """``oracle_boxes`` substitutes ground-truth regions for the predicted ones.

        Everything else in the context path is left alone -- the queries, the
        geometry temperature, the transport, ``content_norm``, the residual scale --
        so the difference between an oracle run and a normal one is the accuracy of
        the boxes and nothing else.  That is the only way to separate "the decoder
        does not use layout information" from "the decoder is handed boxes too
        inaccurate to use", which the branch's own output cannot distinguish.

        Unused queries are pushed to a very negative score rather than zeroed after
        the fact, so the transport keeps its own normalisation and the substitution
        stays inside the path the decoder was trained on.
        """
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [batch, tokens, hidden]")
        queries = self._queries(visual_tokens)
        sem = queries
        if self.sem_adapter is not None:
            sem = queries + self.sem_adapter(queries)
        geom = queries
        if self.geom_adapter is not None:
            geom = queries + self.geom_adapter(queries)
        raw_boxes = self.box_head(geom).sigmoid()
        xy_min = torch.minimum(raw_boxes[..., :2], raw_boxes[..., 2:])
        xy_max = torch.maximum(raw_boxes[..., :2], raw_boxes[..., 2:])
        boxes = torch.cat((xy_min, xy_max), dim=-1)
        order_scores = self.order_head(geom).squeeze(-1)
        direction_logits = self.direction_head(geom)
        region_output = None
        if self.region_decoder is not None:
            region_output = self.region_decoder(
                queries,
                boxes,
                targets=region_targets,
                enable_pointer_mask=region_pointer_mask,
                enable_spatial_penalty=region_spatial_penalty,
            )
        validity_logits = None
        validity_probs = None
        # ``raw_mass`` gating fuses inside ``_raw_mass_context`` and never builds
        # ``token_weights``.  Bind it here so the write-back probe below cannot
        # raise NameError in that mode.
        token_weights: Tensor | None = None

        if self.config.mode == "content_only":
            transport = None
            gated_transport = None
            valid_coverage = None
            if self.validity_head is not None:
                validity_logits = self.validity_head(queries).squeeze(-1)
                validity_probs = validity_logits.sigmoid()
            merged = visual_tokens
        else:
            scores = self._scores(
                queries,
                visual_tokens,
                boxes if oracle_boxes is None else oracle_boxes,
                patch_positions,
            )
            if oracle_mask is not None:
                # A query with no region behind it must not attract transport mass.
                scores = scores.masked_fill(
                    ~oracle_mask.unsqueeze(-1).to(torch.bool), -1e4
                )
            if self.config.mode == "layout_ot":
                transport = self.ot(scores)
            else:
                transport = scores.softmax(dim=-1) / self.config.num_queries

            if self.validity_head is not None:
                validity_input = queries
                if self.config.validity_use_transport_evidence:
                    # Each raw transport row sums to 1 / Q.  Multiplying by Q
                    # gives the query's actual visual evidence without adding
                    # a second learned projection or allowing validity to use
                    # the target mask.  Stop the evidence branch so the object
                    # head cannot alter the transport it is supposed to read.
                    query_transport = transport.detach() * self.config.num_queries
                    visual_evidence = torch.matmul(
                        query_transport, visual_tokens.detach()
                    )
                    visual_evidence = F.layer_norm(
                        visual_evidence,
                        (visual_evidence.shape[-1],),
                    )
                    validity_input = F.layer_norm(
                        queries + visual_evidence.detach(),
                        (queries.shape[-1],),
                    )
                validity_logits = self.validity_head(validity_input).squeeze(-1)
                validity_probs = validity_logits.sigmoid()

            gated_transport = None
            valid_coverage = None
            if validity_probs is not None:
                gated_transport = transport * validity_probs.unsqueeze(-1)
                raw_token_mass = transport.sum(dim=1)
                gated_token_mass = gated_transport.sum(dim=1)
                valid_coverage = gated_token_mass / raw_token_mass.clamp_min(1e-12)
                if self.config.validity_gating_mode == "raw_mass":
                    # Preserve raw token mass.  The validity probability is a
                    # true contribution gate, not a token-wise mixture
                    # re-normalizer: p_q == 0 removes query q exactly.
                    layout_context, valid_coverage = self._raw_mass_context(
                        transport,
                        sem,
                        validity_probs,
                    )
                else:
                    fusion_transport = gated_transport
                    token_weights = fusion_transport.transpose(1, 2)
                    token_weights = token_weights / token_weights.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-12)
                    layout_context = self.content_norm(torch.matmul(token_weights, sem))
                    layout_context = layout_context * valid_coverage.unsqueeze(-1)
            else:
                token_weights = transport.transpose(1, 2)
                token_weights = token_weights / token_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                layout_context = self.content_norm(torch.matmul(token_weights, sem))
            # Keep the zero-initialized gate an exact identity path.  This makes
            # the attention/geometry comparison attributable to the learned
            # layout context instead of an unconditional extra LayerNorm.
            if region_output is not None:
                layout_context = self._region_context(
                    region_output, patch_positions, visual_tokens.shape[1]
                )
            import os as _os
            from .writeback_intervention import apply_intervention, intervention_mode

            # Attribution arms replace the context before it is scaled, so every
            # arm shares one code path and one residual magnitude.  Eval-only:
            # applying an arm during training would train against a corrupted
            # signal, and the mode is recorded in the probe so an artifact
            # cannot be mistaken for a ``full`` run after the fact.
            _intervention = intervention_mode()
            if _intervention != "full":
                layout_context = apply_intervention(layout_context, _intervention)
            _scale = self.effective_residual_scale()
            _probe_path = _os.environ.get("GLMOCR_ADAPTER_PROBE")
            if _probe_path:
                _probe_layout_writeback(
                    _probe_path, token_weights,
                    layout_context, visual_tokens, _scale, _intervention,
                )
            merged = visual_tokens + _scale * layout_context

        return LayoutAdapterOutput(
            merged_tokens=merged,
            layout_queries=queries,
            boxes=boxes,
            order_scores=order_scores,
            direction_logits=direction_logits,
            transport=transport,
            validity_logits=validity_logits,
            validity_probs=validity_probs,
            gated_transport=gated_transport,
            valid_coverage=valid_coverage,
            validity_gating_mode=(
                self.config.validity_gating_mode if validity_probs is not None else None
            ),
            region_output=region_output,
        )
