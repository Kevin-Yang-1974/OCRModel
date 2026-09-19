"""Tests for the write-back attribution arms."""

from __future__ import annotations

import math

import pytest
import torch

from layout_ocr.writeback_intervention import (
    ENV_VAR,
    INTERVENTION_MODES,
    apply_intervention,
    describe_intervention,
    flatness,
    intervention_mode,
)


def _context(batch: int = 2, patches: int = 12, hidden: int = 16, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, patches, hidden, generator=generator)


def test_unset_env_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert intervention_mode() == "full"
    context = _context()
    assert apply_intervention(context, "full") is context


def test_unknown_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "sector")
    with pytest.raises(ValueError, match="must be one of"):
        intervention_mode()


def test_every_mode_preserves_shape_and_dtype() -> None:
    context = _context().to(torch.bfloat16)
    for mode in INTERVENTION_MODES:
        applied = apply_intervention(context, mode, seed=3)  # type: ignore[arg-type]
        assert applied.shape == context.shape
        assert applied.dtype == context.dtype, mode


def test_global_and_spatial_reconstruct_the_full_context() -> None:
    """``global`` + ``spatial`` must be an exact orthogonal split of ``full``."""

    context = _context()
    global_part = apply_intervention(context, "global")
    spatial_part = apply_intervention(context, "spatial")
    assert torch.allclose(global_part + spatial_part, context, atol=1e-5)
    # Orthogonal in the patch-mean sense: the spatial arm has zero patch mean.
    assert spatial_part.mean(dim=1).abs().max() < 1e-5


def test_global_arm_collapses_flatness_to_zero() -> None:
    context = _context()
    assert flatness(apply_intervention(context, "global")) < 1e-6
    assert flatness(apply_intervention(context, "zero")) < 1e-6
    assert flatness(context) > 0.1


def test_shuffle_preserves_norm_and_flatness_exactly() -> None:
    """The whole point of the shuffle arm: only spatial correspondence changes."""

    context = _context(patches=24, seed=5)
    shuffled = apply_intervention(context, "shuffle", seed=11)
    assert not torch.allclose(shuffled, context)
    assert torch.allclose(
        shuffled.norm(dim=-1).mean(), context.norm(dim=-1).mean(), atol=1e-6
    )
    assert math.isclose(
        float(flatness(shuffled)), float(flatness(context)), rel_tol=1e-5
    )
    # A permutation, so the multiset of patch vectors is identical.
    assert torch.allclose(
        shuffled.sort(dim=1).values, context.sort(dim=1).values, atol=1e-6
    )


def test_shuffle_is_seeded_not_fresh_per_call() -> None:
    """Arms are compared page-by-page under a paired bootstrap, so they must be stable."""

    context = _context(patches=16, seed=7)
    first = apply_intervention(context, "shuffle", seed=42)
    second = apply_intervention(context, "shuffle", seed=42)
    assert torch.equal(first, second)
    assert not torch.equal(first, apply_intervention(context, "shuffle", seed=43))


def test_noise_arm_matches_flatness_and_norm_of_full() -> None:
    """The plan's 3.1 control.  Its snippet cannot reach the target flatness.

    A patch-wise zero-mean draw has flatness exactly ``1.0`` and flatness is
    scale-invariant, so ``noise *= target / current`` is a no-op on flatness.  The
    arm has to bend flatness *down* with a patch-common component to match.

    The context here mimics the measured layout context (92% patch-common, 35%
    per-patch) rather than an iid draw: an iid draw sits at the surrogate's own
    ceiling, where matching is impossible for any construction of this family and
    the test would be measuring tensor size, not the arm.
    """

    generator = torch.Generator().manual_seed(9)
    shared = torch.randn(2, 1, 512, generator=generator)
    per_patch = torch.randn(2, 32, 512, generator=generator)
    context = 2.6 * shared + per_patch
    assert float(flatness(context)) < 0.6  # a genuine patch-common component
    noise = apply_intervention(context, "noise", seed=17)
    assert math.isclose(
        float(flatness(noise)), float(flatness(context)), rel_tol=1e-3
    )
    assert math.isclose(
        float(noise.float().norm(dim=-1).mean()),
        float(context.float().norm(dim=-1).mean()),
        rel_tol=1e-4,
    )
    # Not patch-wise zero-mean -- that is what reaching the target requires.
    assert noise.mean(dim=1).abs().max() > 0


def test_noise_arm_varies_across_pages() -> None:
    """A fixed seed would inject the same tensor on every page (the v2 defect)."""

    first = apply_intervention(_context(patches=16, seed=71), "noise", seed=0)
    second = apply_intervention(_context(patches=16, seed=72), "noise", seed=0)
    assert not torch.allclose(first, second)
    # Reproducible for the same page, so a rerun reproduces the same arm.
    same_page = apply_intervention(_context(patches=16, seed=71), "noise", seed=0)
    assert torch.allclose(first, same_page)


def test_plan_section_3_1_snippet_cannot_hit_its_target() -> None:
    """Pin the arithmetic the plan relies on, so the failure is not rediscovered."""

    context = _context(patches=32, seed=9)
    target_flat = float(flatness(context))
    noise = torch.randn_like(context)
    noise = noise - noise.mean(dim=1, keepdim=True)
    current_flat = float(flatness(noise))
    rescaled = noise * (target_flat / current_flat)
    assert math.isclose(current_flat, 1.0, rel_tol=1e-6)
    assert math.isclose(float(flatness(rescaled)), 1.0, rel_tol=1e-6)
    assert not math.isclose(float(flatness(rescaled)), target_flat, rel_tol=1e-2)


def test_noise_is_finite_in_low_precision() -> None:
    """The task plan's rescaling by a computed flatness could divide by zero."""

    context = _context().to(torch.bfloat16)
    noise = apply_intervention(context, "noise", seed=1)
    assert torch.isfinite(noise).all()


def test_single_patch_shuffle_degrades_to_identity() -> None:
    context = _context(patches=1)
    assert torch.equal(apply_intervention(context, "shuffle", seed=0), context)


def test_spatial_scaled_matches_full_amplitude_but_keeps_only_perpatch_content() -> None:
    """Removes the amplitude confound between the ``spatial`` and ``full`` arms."""

    context = _context(patches=32, hidden=64, seed=41)
    plain = apply_intervention(context, "spatial")
    scaled = apply_intervention(context, "spatial_scaled")
    # Same direction per patch, only the magnitude changed.
    assert torch.allclose(
        scaled / scaled.norm(dim=-1, keepdim=True),
        plain / plain.norm(dim=-1, keepdim=True),
        atol=1e-5,
    )
    # Amplitude now matches the untouched context, which is the whole point.
    assert torch.allclose(
        scaled.float().norm(dim=-1).mean(),
        context.float().norm(dim=-1).mean(),
        rtol=1e-4,
    )
    # Still no patch-common component.
    assert scaled.mean(dim=1).abs().max() < 1e-4
    described = describe_intervention(context, "spatial_scaled")
    assert math.isclose(described["relative_norm"], 1.0, rel_tol=1e-4)
    assert math.isclose(described["flatness_arm"], 1.0, rel_tol=1e-4)


def test_recorded_spatial_arm_amplitude_was_confounded() -> None:
    """Pin the measured numbers the confound was diagnosed from.

    The v2 matrix reported inj/vt 0.0206 for ``full`` and 0.0072 for ``spatial``;
    those are the medians in docs/LAYOUT_WRITEBACK_INTERVENTION_RESULT.md.  The
    ratio is what ``spatial_scaled`` corrects.
    """

    full_inj, spatial_inj = 0.0206, 0.0072
    assert math.isclose(full_inj / spatial_inj, 2.86, rel_tol=0.02)
    global_share, spatial_share = 0.918, 0.353
    assert math.isclose(global_share * full_inj, 0.0189, abs_tol=5e-4)
    assert math.isclose(spatial_share * full_inj, spatial_inj, abs_tol=5e-4)
    assert math.isclose(global_share**2 + spatial_share**2, 1.0, rel_tol=0.05)


def test_global_perm_preserves_norm_and_page_specificity() -> None:
    """The direction control: same magnitude, same per-page variation, no learned axis."""

    context = _context(patches=32, hidden=64, seed=31)
    plain = apply_intervention(context, "global")
    permuted = apply_intervention(context, "global_perm", seed=5)
    assert not torch.allclose(permuted, plain)
    # Norm preserved exactly: a permutation moves entries, it does not scale them.
    assert torch.allclose(
        permuted.norm(dim=-1), plain.norm(dim=-1), atol=1e-5
    )
    # Still patch-constant within a page, and still a different function per page.
    assert permuted.std(dim=1).max() < 1e-6
    other = apply_intervention(_context(patches=32, hidden=64, seed=32), "global_perm", seed=5)
    assert not torch.allclose(permuted[0, 0], other[0, 0])
    # Same multiset of coordinates, so it is a genuine permutation.
    assert torch.allclose(
        permuted[0, 0].sort().values, plain[0, 0].sort().values, atol=1e-6
    )


def test_describe_intervention_reports_the_amplitude_confound() -> None:
    context = _context(patches=32, hidden=64, seed=21)
    full = describe_intervention(context, "full")
    assert math.isclose(full["relative_norm"], 1.0, rel_tol=1e-5)
    # The two halves of the orthogonal split.
    global_arm = describe_intervention(context, "global")
    spatial_arm = describe_intervention(context, "spatial")
    assert math.isclose(
        global_arm["relative_norm"], global_arm["global_share"], rel_tol=1e-5
    )
    assert math.isclose(
        spatial_arm["relative_norm"], spatial_arm["spatial_share"], rel_tol=1e-5
    )
    assert math.isclose(
        full["global_share"] ** 2 + full["spatial_share"] ** 2, 1.0, rel_tol=1e-3
    )
    # Amplitude-matched controls must not change the norm.
    for mode in ("shuffle", "noise"):
        assert math.isclose(
            describe_intervention(context, mode, seed=2)["relative_norm"],
            1.0,
            rel_tol=1e-3,
        )
