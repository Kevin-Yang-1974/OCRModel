"""Eval-only interventions on the layout write-back, for attribution.

The layout branch reaches the decoder through exactly one tensor: the residual
``alpha * layout_context`` added to the pre-merger visual tokens
(``adapter.py``, the ``merged`` assignment).  Every claim of the form "the
decoder does not use the layout information" is therefore a claim about that one
tensor, and to test it the tensor has to be varied while everything else is held
fixed.

The arms are chosen so that amplitude and content can be separated:

``full``     identity.  The measured baseline.
``global``   patch-mean broadcast across patches.  Keeps the patch-common
             offset, removes every per-patch difference.  A random-query
             mechanism degenerates to exactly this shape (the write-back probe
             measures ``lc_flat = 0.008`` for it against ``0.3895`` for a trained
             branch), so this arm is the control for "is the measured gain just a
             patch-common bias rather than layout information?".
``spatial``  patch-centred only (``H - mean_p H``).  Removes the patch-common
             offset and keeps the per-patch structure that actually carries
             layout.  Not amplitude-matched to ``full`` by construction: the two
             arms ``global``/``spatial`` are the orthogonal split of ``full``.
``spatial_scaled``
             ``spatial`` rescaled per page so its norm matches ``full``.  Without
             this arm the ``spatial``-vs-``full`` comparison is amplitude
             confounded: the 2026-09-19 v2 matrix measured ``spatial`` at
             ``inj/vt = 0.0072`` against ``full`` at ``0.0206``, a factor of 2.9, so
             "the per-patch component does nothing" could not be told apart from
             "the per-patch component was too quiet to matter".  The v5 matrix
             settled it in the direction that closes the route: at an amplitude
             bit-identical to ``full`` this arm scored 0.116270 against ``zero``'s
             0.112988 -- the per-patch component is not inert, it is adversarial.
``shuffle``  a fixed per-page permutation of the patch axis.  The multiset of
             vectors is bit-identical to ``full``, so norm, flatness and every
             other marginal statistic match exactly; only the correspondence
             between a patch and its own layout context is destroyed.  This is
             the sharpest content control available.
``noise``    zero-mean Gaussian matched to ``full``'s flatness *and* norm.  The
             control proposed in ``plans/LAYOUT_FUSION_REDESIGN.md`` section 3.1,
             kept because it is cheap and it is the one already written down.
``zero``     all-zero context, i.e. a closed gate.  The reference floor.
``global_perm``
             ``global`` with a fixed permutation of the hidden axis.  Permuting a
             vector preserves its norm exactly, and the permutation is fixed across
             pages, so the arm stays a page-specific function of the page while the
             learned direction is destroyed.  This separates "the page-level
             component carries learned content" from "any page-specific
             page-level vector of the right size does the same thing".  The v5
             matrix measured it at 0.112876 against ``full``'s 0.111608 and
             ``zero``'s 0.112988: destroying the direction while preserving norm and
             page-specificity gives the gain back, so the branch's usable
             contribution is a page-level vector with a *learned direction* -- not
             generic page conditioning.

The 2026-09-19 matrices showed ``global`` ~= ``full`` and ``spatial`` ~= ``zero``, so
the write-back's whole effect lives in the patch-mean component.

``shuffle`` is strictly stronger than ``noise``: matching a random draw to a
measured flatness is easy to get subtly wrong, whereas permuting the patch axis
cannot change any statistic but the one under test.

Determinism matters here because the arms are compared page by page under a
paired bootstrap: ``shuffle`` and ``noise`` are seeded from ``seed`` rather than
drawn afresh per forward, so an eval-only rerun reproduces the same arms.
"""

from __future__ import annotations

import os
from typing import Literal, get_args

import torch
from torch import Tensor

InterventionMode = Literal[
    "full", "global", "spatial", "spatial_scaled", "shuffle", "noise", "zero", "global_perm"
]

INTERVENTION_MODES: tuple[str, ...] = get_args(InterventionMode)
ENV_VAR = "GLMOCR_LAYOUT_INTERVENE"


def intervention_mode() -> InterventionMode:
    """Return the configured arm; the unset default is the no-op ``full``."""

    raw = os.environ.get(ENV_VAR, "").strip().lower()
    if not raw:
        return "full"
    if raw not in INTERVENTION_MODES:
        raise ValueError(
            f"{ENV_VAR} must be one of {INTERVENTION_MODES}, got {raw!r}"
        )
    return raw  # type: ignore[return-value]


def flatness(x: Tensor) -> Tensor:
    """Patch-to-patch variation of ``[batch, patch, hidden]`` relative to its own norm.

    Matches the statistic reported by the write-back probe in ``adapter.py``:
    ``mean_p ||x_p - mean_p x|| / mean_p ||x_p||``.
    """

    x = x.float()
    centred = x - x.mean(dim=1, keepdim=True)
    return centred.norm(dim=-1).mean() / x.norm(dim=-1).mean().clamp_min(1e-12)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _page_seed(layout_context: Tensor, seed: int) -> int:
    """Mix the page's own content into the seed so a random arm is page-specific.

    A fixed seed is a defect, not a neutral choice: ``torch.randn`` is driven by the
    shape and the seed only, so a fixed seed injects *the same tensor on every page*
    and the arm becomes a control for "a fixed random offset" rather than for
    "random content".  The 2026-09-19 v2 matrix had exactly this bug -- its ``noise``
    arm came out bit-identical to ``zero``, which is what a page-invariant offset
    would do, and it could not answer whether the page-level component's content
    matters.  Folding a digest of the context into the seed keeps the arm
    reproducible for the same page while making it vary across pages.
    """

    digest = int(layout_context.float().abs().sum().item() * 1e6) & 0x7FFFFFFF
    return (int(seed) * 0x9E3779B1 + digest) & 0x7FFFFFFF


def apply_intervention(
    layout_context: Tensor, mode: InterventionMode, *, seed: int = 0
) -> Tensor:
    """Return the context the given arm would write back.

    ``layout_context`` is ``[batch, patch, hidden]``; every arm preserves that
    shape so the call site is unchanged.
    """

    if mode == "full":
        return layout_context
    if mode == "zero":
        return torch.zeros_like(layout_context)
    if mode == "global":
        return layout_context.mean(dim=1, keepdim=True).expand_as(layout_context)
    if mode == "global_perm":
        mean = layout_context.mean(dim=1, keepdim=True)
        hidden = mean.shape[-1]
        order = torch.randperm(hidden, generator=_generator(seed)).to(mean.device)
        return mean.index_select(-1, order).expand_as(layout_context)
    if mode == "spatial":
        return layout_context - layout_context.mean(dim=1, keepdim=True)
    if mode == "spatial_scaled":
        return _scale_to_full_norm(
            layout_context - layout_context.mean(dim=1, keepdim=True), layout_context
        )
    if mode == "shuffle":
        patches = layout_context.shape[1]
        if patches < 2:
            return layout_context
        order = torch.randperm(patches, generator=_generator(seed)).to(
            layout_context.device
        )
        return layout_context.index_select(1, order)
    if mode == "noise":
        return _flatness_matched_noise(layout_context, seed=seed)
    raise ValueError(f"unsupported intervention mode: {mode}")


def _scale_to_full_norm(arm: Tensor, reference: Tensor) -> Tensor:
    """Rescale ``arm`` per page so each page's mean token norm matches ``reference``.

    Scaling is per page rather than global because the global/spatial split varies a
    little from page to page; a single factor would leave the amplitude confound in
    place for exactly the pages that need it corrected most.
    """

    arm_norm = arm.float().norm(dim=-1).mean(dim=1, keepdim=True).clamp_min(1e-12)
    reference_norm = reference.float().norm(dim=-1).mean(dim=1, keepdim=True)
    factor = (reference_norm / arm_norm).unsqueeze(-1).to(dtype=arm.dtype)
    return arm * factor


def _flatness_matched_noise(
    layout_context: Tensor, *, seed: int, iterations: int = 32
) -> Tensor:
    """Zero-mean-free Gaussian surrogate with ``full``'s flatness and norm.

    A patch-wise zero-mean draw cannot be used for this.  ``flatness`` of a tensor
    whose patch mean is exactly zero is exactly ``1.0``, and flatness is invariant
    under scaling, so the rescaling in ``plans/LAYOUT_FUSION_REDESIGN.md`` section
    3.1 (``noise *= target_flat / current_flat``) leaves flatness at ``1.0`` rather
    than the ``0.39`` it is meant to match -- the arm it produces is 2.5x more
    patch-varying than the context it replaces, confounding content with the scale
    of the variation.

    Reaching a *lower* flatness needs a patch-common component, so the surrogate is
    ``n_p = r * g + e_p`` with a single shared draw ``g`` and per-patch draws
    ``e_p``.  Flatness falls monotonically from the pure-per-patch value to 0 as
    ``r`` grows, so ``r`` is solved by bisection against the measured target.  Norm
    is matched afterwards, which is safe because scaling does not move flatness.

    The construction can only reach flatness *at or below* the pure-per-patch value
    of its own draws, which fluctuates around 1 by O(1/sqrt(hidden)).  When the
    target sits above that ceiling -- only possible for very small hidden sizes,
    where the fluctuation is large -- the solve clamps to ``r = 0`` and reports the
    closest achievable value.  At the hidden size this project runs at (1536) the
    match is exact; at a toy size like 16 the ceiling is a few 1e-3 away, which is a
    property of the tensor size and not of the arm.
    """

    shape = layout_context.shape
    dtype = layout_context.dtype
    device = layout_context.device
    generator = _generator(_page_seed(layout_context, seed))
    shared = torch.randn(shape[0], 1, shape[2], generator=generator, dtype=torch.float32)
    per_patch = torch.randn(shape, generator=generator, dtype=torch.float32)

    target = flatness(layout_context).to(torch.float32)

    def at(ratio: torch.Tensor) -> Tensor:
        return ratio * shared + per_patch

    low = torch.zeros((), dtype=torch.float32)
    high = torch.ones((), dtype=torch.float32)
    for _ in range(iterations):
        if float(flatness(at(high))) <= float(target):
            break
        high = high * 2.0
    else:
        high = high
    for _ in range(iterations):
        middle = (low + high) * 0.5
        if float(flatness(at(middle))) > float(target):
            low = middle
        else:
            high = middle

    surrogate = at((low + high) * 0.5).to(device=device, dtype=dtype)
    target_norm = layout_context.float().norm(dim=-1).mean()
    current_norm = surrogate.float().norm(dim=-1).mean().clamp_min(1e-12)
    return surrogate * (target_norm / current_norm).to(dtype=dtype)


def describe_intervention(layout_context: Tensor, mode: InterventionMode, *, seed: int = 0) -> dict:
    """Report what an arm changed, so a run's artifacts are self-documenting.

    ``global_share`` and ``spatial_share`` are the orthogonal split of the
    context norm; ``relative_norm`` is the arm's norm against the untouched
    context, which is how an amplitude confound becomes visible instead of
    assumed away.
    """

    with torch.no_grad():
        original = layout_context.float()
        applied = apply_intervention(layout_context, mode, seed=seed).float()
        mean = original.mean(dim=1, keepdim=True)
        base = original.norm(dim=-1).mean().clamp_min(1e-12)
        return {
            "mode": mode,
            "flatness_full": float(flatness(original)),
            "flatness_arm": float(flatness(applied)),
            "global_share": float(mean.norm(dim=-1).mean() / base),
            "spatial_share": float((original - mean).norm(dim=-1).mean() / base),
            "relative_norm": float(
                applied.norm(dim=-1).mean() / base
            ),
        }
