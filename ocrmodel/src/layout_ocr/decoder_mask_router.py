"""A learnable per-token spatial mask head for the GLM-OCR decoder.

The plan for this module is ``plans/DECODER_LEARNED_MASK_ROUTING_PLAN.md``.  In
short: a light head reads the decoder's intermediate hidden state ``h_t`` (the
output of layers ``0..split_layer-1``), the whole-page visual tokens ``V`` and
the previous predicted mask ``M_{t-1}``, and predicts the soft spatial mask
``M_t`` that the next token should attend to.  ``M_t`` is then added as an
additive bias to the attention logits of layers ``split_layer..`` on the visual
keys, so the decoder learns to *look where it is reading*.

The head is trained with the *predicted* mask fed back into the next step (no GT
mask teacher forcing).  The only GT is the rasterised character box, used to
supervise ``M_t`` through ``L_mask``; the feedback path is corrupted with noise
during training so the recurrence stays stable when the model runs free (see
plan section 5.1).

Nothing here touches the tokenizer, the layout adapter, or any predicted-box
branch: the input is a plain tensor ``V`` and a grid of normalised patch
centres, and the output is a soft mask over those ``N`` visual keys.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class DecoderMaskConfig:
    """Hyper-parameters of the mask head.  Defaults are the plan's first cut,
    explicitly marked as to-be-validated rather than tuned."""

    hidden_size: int = 1536  # text hidden size, read from the real checkpoint
    router_dim: int = 256  # internal width ``d``
    split_layer: int = 8  # layers 0..split-1 feed the head; split.. get the bias
    # beta upper bound during warmup.  Anchored to the attention-routing sweep:
    # a +2.0-logit bias on a binary box gave the best CER (-20%) while +4.0 went
    # catastrophic (repetition / 114 length-cap hits).  The learned mask is soft
    # in [0,1], so the effective boost is beta * M and 2.0 lets a sharp mask reach
    # the oracle's sweet spot without the +4 failure regime.
    bias_max: float = 2.0
    initial_mask_bias: float = -4.0  # z bias, keeps an untrained head near zero
    initial_stop_bias: float = -4.0  # stop-head bias, near "do not stop"
    spatial_kernel: int = 3  # 3x3 conv over the previous mask
    use_prev_mask: bool = True  # feed M_{t-1} back into the next step
    mask_feedback_noise: float = 0.15  # Bernoulli flip prob on the fed-back mask
    input_noise: float = 0.05  # Gaussian noise on the projected h_t, relative to h RMS
    detach_every: int = 64  # truncated-BPTT boundary for the mask recurrence (plan 5)

    def __post_init__(self) -> None:
        if self.router_dim <= 0:
            raise ValueError("router_dim must be positive")
        if self.split_layer <= 0:
            raise ValueError("split_layer must be positive")
        if not 0.0 <= self.mask_feedback_noise < 1.0:
            raise ValueError("mask_feedback_noise must be in [0, 1)")
        if self.input_noise < 0.0:
            raise ValueError("input_noise must be non-negative")
        if self.detach_every <= 0:
            raise ValueError("detach_every must be positive")


def _normalized_grid_xywh(grid_thw: Tensor, spatial_merge_size: int) -> tuple[Tensor, tuple[int, int]]:
    """Return per-token normalised centres ``[x, y, w, h]`` and ``(H', W')``.

    The order matches ``glm_bridge.patch_grid_positions``: row-major over the
    ``(merged_height, merged_width)`` grid, so index ``k`` is the ``k``-th visual
    token in the decoder's key sequence.  ``w``/``h`` are the constant cell size.
    """

    if grid_thw.shape != (1, 3):
        raise ValueError("the mask head expects exactly one whole-page image per step")
    temporal, height, width = (int(value) for value in grid_thw[0].tolist())
    if temporal != 1:
        raise ValueError("the mask head does not accept video/multi-frame input")
    merged_height = height // spatial_merge_size
    merged_width = width // spatial_merge_size
    if merged_height <= 0 or merged_width <= 0:
        raise ValueError("image grid is too small for the merge size")
    rows = (torch.arange(merged_height, device=grid_thw.device, dtype=torch.float32) + 0.5) / merged_height
    cols = (torch.arange(merged_width, device=grid_thw.device, dtype=torch.float32) + 0.5) / merged_width
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    cell_w = 1.0 / merged_width
    cell_h = 1.0 / merged_height
    xyw = torch.stack(
        (
            xx.reshape(-1),
            yy.reshape(-1),
            torch.full((merged_height * merged_width,), cell_w, device=grid_thw.device, dtype=torch.float32),
            torch.full((merged_height * merged_width,), cell_h, device=grid_thw.device, dtype=torch.float32),
        ),
        dim=-1,
    )
    return xyw.unsqueeze(0), (merged_height, merged_width)


class DecoderMaskRouter(nn.Module):
    """The spatial mask head.  All internal arithmetic runs in float32."""

    def __init__(self, config: DecoderMaskConfig) -> None:
        super().__init__()
        self.config = config
        d = config.router_dim
        hidden = config.hidden_size
        # Project the whole-page visual tokens once per page.
        self.visual_norm = nn.LayerNorm(hidden, eps=1e-5)
        self.visual_proj = nn.Linear(hidden, d, bias=False)
        self.pos_proj = nn.Linear(4, d, bias=False)
        # Project the query hidden state.
        self.query_proj = nn.Linear(hidden, d, bias=False)
        # A 3x3, 1->1 conv over the previous mask, reshaped to the patch grid.
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=config.spatial_kernel, padding=config.spatial_kernel // 2, bias=False)
        # q_t MLP: [projected h; c_prev; centre; variance; mean_mask] -> d.
        mlp_in = d + d + 2 + 2 + 1
        self.q_mlp = nn.Sequential(
            nn.Linear(mlp_in, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        # Stop head reads the query projection and the previous-mask summary.
        self.stop_head = nn.Linear(d + d, 1, bias=False)
        self.register_buffer("mask_bias", torch.tensor(float(config.initial_mask_bias)))
        self.register_buffer("stop_bias", torch.tensor(float(config.initial_stop_bias)))
        # Mutable noise levels, seeded from config and annealed by the trainer.
        # Kept as plain attributes (not buffers) because they are schedule state
        # that must never be saved into a checkpoint.
        self.feedback_noise = float(config.mask_feedback_noise)
        self.input_noise = float(config.input_noise)

    def _summarize(self, prev_mask: Tensor, keys: Tensor) -> Tensor:
        """Weighted summary of the previous mask over the visual keys ``[B, d]``.

        An all-zero mask yields zeros, so a first step and a blank-but-valid mask
        both present a zero context to the head (the head must learn to continue
        from both, per plan 5.1).  The centre/variance/mean are computed in
        ``_step`` because they need the grid ``xywh``.
        """

        mass = prev_mask.sum(dim=-1, keepdim=True) + 1e-6
        return (prev_mask.unsqueeze(-1) * keys).sum(dim=1) / mass  # [B, d]

    def _feedback(self, prev_mask: Tensor) -> Tensor:
        """Return the fed-back context the recurrence sees (plan 5.1, item 1).

        When ``use_prev_mask`` is off (B2), the recurrence sees a zero context, so
        the head predicts the next mask from the query alone.  When it is on, the
        fed mask is corrupted with Bernoulli noise during training; the clean
        ``prev_mask`` is still used for the attention bias and for ``L_mask`` by
        the caller.
        """

        if not self.config.use_prev_mask:
            return torch.zeros_like(prev_mask)
        if not self.training or self.feedback_noise <= 0.0:
            return prev_mask
        p = self.feedback_noise
        flip = (torch.rand_like(prev_mask) < p).to(prev_mask.dtype)
        return prev_mask * (1.0 - flip) + flip * (1.0 - prev_mask)

    def _noisy_query(self, query_proj: Tensor) -> Tensor:
        """Add input noise to the projected query (plan 5.1, item 2)."""

        if not self.training or self.input_noise <= 0.0:
            return query_proj
        rms = query_proj.norm(dim=-1, keepdim=True).clamp_min(1e-8) / (query_proj.shape[-1] ** 0.5)
        noise = torch.randn_like(query_proj) * (self.input_noise * rms)
        return query_proj + noise

    def project_visual(self, visual: Tensor, grid_thw: Tensor, spatial_merge_size: int) -> tuple[Tensor, Tensor, tuple[int, int]]:
        """Project the merged visual tokens ``visual`` (``[B, N, hidden]``) into
        per-key features ``K`` (``[B, N, d]``), returning ``K``, the grid and the
        spatial shape.  Called once per page."""

        xywh, spatial_shape = _normalized_grid_xywh(grid_thw, spatial_merge_size)
        visual = visual.float()
        keys = self.visual_proj(self.visual_norm(visual)) + self.pos_proj(xywh.to(visual.dtype))
        return keys, xywh, spatial_shape

    def _step(
        self,
        query: Tensor,
        prev_mask: Tensor,
        keys: Tensor,
        xywh: Tensor,
        spatial_shape: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One head step for a batch of queries ``query`` (``[B, hidden]``)."""

        d = self.config.router_dim
        q_proj = self._noisy_query(self.query_proj(query.float()))
        c_prev = self._summarize(prev_mask, keys)
        # Centre / variance / mean of the previous mask over the grid.
        mass = prev_mask.sum(dim=-1, keepdim=True) + 1e-6
        centre = (prev_mask.unsqueeze(-1) * xywh[..., :2]).sum(dim=1) / mass  # [B, 2]
        sq = (xywh[..., :2].square() * prev_mask.unsqueeze(-1)).sum(dim=1) / mass
        variance = (sq - centre.square()).clamp_min(0.0)  # [B, 2]
        mean_mask = prev_mask.mean(dim=-1, keepdim=True)  # [B, 1]
        q_t = self.q_mlp(torch.cat((q_proj, c_prev, centre, variance, mean_mask), dim=-1))  # [B, d]
        # Local spatial feature from the previous mask: 3x3 conv on the grid.
        merged_h, merged_w = spatial_shape
        prev_grid = prev_mask.reshape(prev_mask.shape[0], 1, merged_h, merged_w)
        r_prev = self.spatial_conv(prev_grid).reshape(prev_mask.shape[0], -1)  # [B, N]
        # Single-line score with the fixed mask bias (plan 3.1).
        z = torch.einsum("bd,bnd->bn", q_t, keys) / (d**0.5) + r_prev + self.mask_bias  # [B, N]
        stop_logit = self.stop_head(torch.cat((q_proj, c_prev), dim=-1)) + self.stop_bias  # [B, 1]
        e = torch.sigmoid(stop_logit)
        mask = (1.0 - e) * torch.sigmoid(z)  # [B, N]
        return mask, e, z

    def scan(
        self,
        hidden_states: Tensor,
        keys: Tensor,
        xywh: Tensor,
        spatial_shape: tuple[int, int],
        detach_every: int = 64,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the head over ``hidden_states`` (``[B, T, hidden]``) in order.

        ``hidden_states`` is the decoder's intermediate state at the target query
        positions (layer ``split_layer`` input, i.e. the output of layer
        ``split_layer - 1``).  The recurrence is detached every ``detach_every``
        steps to bound BPTT (plan 5).  Returns ``(M, e, z)`` as
        ``[B, T, N]``, ``[B, T, 1]`` and ``[B, T, N]``.
        """

        b, t, _ = hidden_states.shape
        n = keys.shape[1]
        device = hidden_states.device
        dtype = keys.dtype
        masks: list[Tensor] = []
        stops: list[Tensor] = []
        logits: list[Tensor] = []
        prev_mask = torch.zeros(b, n, device=device, dtype=dtype)
        for step in range(t):
            if step > 0 and step % detach_every == 0:
                prev_mask = prev_mask.detach()
            feedback = self._feedback(prev_mask)
            mask, stop, z = self._step(hidden_states[:, step], feedback, keys, xywh, spatial_shape)
            masks.append(mask)
            stops.append(stop)
            logits.append(z)
            prev_mask = mask
        return (
            torch.stack(masks, dim=1),
            torch.stack(stops, dim=1),
            torch.stack(logits, dim=1),
        )

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
