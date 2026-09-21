"""Conditional-VAE variant of the mask head (experiment group G3).

The MLP head predicts the mask *deterministically*: a bilinear score against the
visual keys, smoothed by a conv, squashed by a sigmoid.  This module replaces
that generator with a latent-variable one, and nothing else -- it subclasses
:class:`~layout_ocr.decoder_mask_router.DecoderMaskRouter`, so the visual
projection, the query MLP, the feedback noise, the stop head and the recurrence
are byte-identical to G1/G2.  The only difference between the arms is how the
mask is produced.

The conditioning invariant (the reason a conditional VAE is correct here):

    q(z | mask_gt, h_t, keys, prev)   -- posterior, training only
    p(z |          h_t, keys, prev)   -- prior, used for the KL and at inference
    decoder(z,     h_t, keys, prev)   -- both

``q`` and ``p`` must see exactly the same conditioning, with ``q`` additionally
seeing the ground truth; the decoder must see exactly what the prior can supply.
Break either direction and the KL either destroys information the prior cannot
represent, or the decoder depends on something unavailable at inference.  The
inference path therefore never builds ``q`` and never reads ``mask_gt``, and a
test asserts it.

Two details that decide whether this trains at all:

* the KL is a **mean** over latent dimensions and tokens, not a sum.  A spatial
  latent has thousands of dimensions; a summed KL would dwarf the reconstruction
  term by orders of magnitude and collapse the posterior immediately.
* the decoder's readout is **small-random**, not zero, initialised.  A zero
  readout gives ``z`` no gradient at step 0, so the KL wins before the decoder
  ever learns to use the latent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoder_mask_router import DecoderMaskConfig, DecoderMaskRouter

LOGVAR_MIN, LOGVAR_MAX = -8.0, 4.0


class DecoderMaskVaeRouter(DecoderMaskRouter):
    """The mask head with a conditional-VAE mask generator."""

    def __init__(self, config: DecoderMaskConfig) -> None:
        super().__init__(config)
        d = config.router_dim
        channels = config.vae_latent_channels
        self.latent_size = int(config.vae_latent_size)
        # The conditioning map is the query vector broadcast over the grid, plus
        # the visual keys pooled to the latent resolution -- this is how spatial
        # information reaches the prior, so it does not have to be memorised by
        # the decoder.
        self.cond_proj = nn.Linear(2 * d, d)
        self.prior_conv = nn.Sequential(
            nn.Conv2d(2 * d, d, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d, 2 * channels, kernel_size=3, padding=1),
        )
        self.posterior_conv = nn.Sequential(
            nn.Conv2d(2 * d + 1, d, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d, 2 * channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels + 2 * d, d, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d, d, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d, 1, kernel_size=1),
        )
        nn.init.normal_(self.decoder[-1].weight, std=0.02)
        nn.init.zeros_(self.decoder[-1].bias)
        self._step_kl: list[Tensor] = []
        self._diagnostics: dict[str, float] = {}

    # --- latent plumbing -----------------------------------------------------

    def _keys_map(self, keys: Tensor, spatial_shape: tuple[int, int], size: int) -> Tensor:
        """Pool the visual keys ``[B, N, d]`` to a ``[B, d, size, size]`` map."""

        b, n, d = keys.shape
        height, width = spatial_shape
        if n != height * width:
            raise ValueError(f"keys have {n} cells but the router grid is {height}x{width}")
        grid = keys.transpose(1, 2).reshape(b, d, height, width)
        return F.adaptive_avg_pool2d(grid, (size, size))

    @staticmethod
    def _split_latent(parameters: Tensor) -> tuple[Tensor, Tensor]:
        mean, logvar = parameters.chunk(2, dim=1)
        return mean, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def _posterior(self, cond: Tensor, target: Tensor, spatial_shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """``q(z | mask_gt, ...)`` -- the ground truth enters only here."""

        b, n = target.shape
        height, width = spatial_shape
        truth = target.reshape(b, 1, height, width).float()
        truth = F.adaptive_avg_pool2d(truth, (self.latent_size, self.latent_size))
        return self._split_latent(self.posterior_conv(torch.cat((cond, truth), dim=1)))

    # --- head step -----------------------------------------------------------

    def _step(
        self,
        query: Tensor,
        prev_mask: Tensor,
        keys: Tensor,
        xywh: Tensor,
        spatial_shape: tuple[int, int],
        target: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One VAE head step.  Returns ``(mask, stop, logits)`` like the MLP head."""

        d = self.config.router_dim
        size = self.latent_size
        height, width = spatial_shape
        q_proj = self._noisy_query(self.query_proj(query.float()))
        c_prev = self._summarize(prev_mask, keys)
        mass = prev_mask.sum(dim=-1, keepdim=True) + 1e-6
        centre = (prev_mask.unsqueeze(-1) * xywh[..., :2]).sum(dim=1) / mass
        sq = (xywh[..., :2].square() * prev_mask.unsqueeze(-1)).sum(dim=1) / mass
        variance = (sq - centre.square()).clamp_min(0.0)
        mean_mask = prev_mask.mean(dim=-1, keepdim=True)
        q_t = self.q_mlp(torch.cat((q_proj, c_prev, centre, variance, mean_mask), dim=-1))  # [B, d]

        # Conditioning map: the query broadcast over the latent grid, plus the
        # pooled visual keys.  Shared by the prior, the posterior and the decoder.
        cond_vec = self.cond_proj(torch.cat((q_t, c_prev), dim=-1))  # [B, d]
        cond_map = cond_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, size, size)
        keys_map = self._keys_map(keys, spatial_shape, size)
        cond = torch.cat((cond_map, keys_map), dim=1)  # [B, 2d, size, size]

        mu_p, logvar_p = self._split_latent(self.prior_conv(cond))
        if target is not None:
            mu_q, logvar_q = self._posterior(cond, target, spatial_shape)
            std_q = torch.exp(0.5 * logvar_q)
            z = mu_q + std_q * torch.randn_like(std_q)
            self._step_kl.append(self._kl_per_dim(mu_q, logvar_q, mu_p, logvar_p))
        else:
            # Inference: the posterior is never constructed.
            mu_q = logvar_q = None
            if self.config.vae_inference == "sample" and self.training is False:
                std_p = torch.exp(0.5 * logvar_p)
                z = mu_p + std_p * torch.randn_like(std_p)
            else:
                z = mu_p
        self._last_posterior = None if mu_q is None else (mu_q, logvar_q)

        decoded = self.decoder(torch.cat((z, cond), dim=1))
        logits = F.interpolate(decoded, size=(height, width), mode="bilinear", align_corners=False)
        logits = logits.reshape(logits.shape[0], -1) + self.mask_bias  # [B, N_router]
        stop_logit = self.stop_head(torch.cat((q_proj, c_prev), dim=-1)) + self.stop_bias
        e = torch.sigmoid(stop_logit)
        mask = (1.0 - e) * torch.sigmoid(logits)
        return mask, e, logits

    @staticmethod
    def _kl_per_dim(mu_q: Tensor, logvar_q: Tensor, mu_p: Tensor, logvar_p: Tensor) -> Tensor:
        """KL(q || p) per latent dimension, in nats."""

        var_q, var_p = logvar_q.exp(), logvar_p.exp()
        return 0.5 * ((mu_q - mu_p).square() / var_p + var_q / var_p - 1.0 + logvar_p - logvar_q)

    # --- recurrence ----------------------------------------------------------

    def scan(
        self,
        hidden_states: Tensor,
        keys: Tensor,
        xywh: Tensor,
        spatial_shape: tuple[int, int],
        detach_every: int = 64,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Same recurrence as the MLP head; ``targets`` drives the posterior.

        ``targets`` is ``[B, T, N]`` ground-truth masks and is passed **only** by
        the training path.  Generation never supplies it, which is what keeps the
        posterior out of inference.
        """

        self._step_kl = []
        b, t, _ = hidden_states.shape
        n = keys.shape[1]
        device, dtype = hidden_states.device, keys.dtype
        masks: list[Tensor] = []
        stops: list[Tensor] = []
        logits: list[Tensor] = []
        prev_mask = torch.zeros(b, n, device=device, dtype=dtype)
        for step in range(t):
            if step > 0 and step % detach_every == 0:
                prev_mask = prev_mask.detach()
            feedback = self._feedback(prev_mask)
            step_target = None if targets is None else targets[:, step]
            mask, stop, z = self._step(hidden_states[:, step], feedback, keys, xywh, spatial_shape, step_target)
            masks.append(mask)
            stops.append(stop)
            logits.append(z)
            prev_mask = mask
        self._record_diagnostics()
        return (
            torch.stack(masks, dim=1),
            torch.stack(stops, dim=1),
            torch.stack(logits, dim=1),
        )

    def _record_diagnostics(self) -> None:
        if not self._step_kl:
            self._diagnostics = {}
            return
        stacked = torch.stack(self._step_kl)  # [T, B, C, L, L]
        free = self.config.vae_kl_free_bits
        self._diagnostics = {
            "kl_raw": float(stacked.mean().detach().item()),
            "kl_active_dims": float((stacked > free).float().mean().detach().item()),
        }

    def kl_loss(self) -> Tensor:
        """Mean free-bits KL over the last ``scan``; zero when no posterior ran."""

        if not self._step_kl:
            return torch.zeros((), device=self.mask_bias.device)
        stacked = torch.stack(self._step_kl)
        return stacked.clamp_min(self.config.vae_kl_free_bits).mean()

    def kl_diagnostics(self) -> dict[str, float]:
        return dict(self._diagnostics)
