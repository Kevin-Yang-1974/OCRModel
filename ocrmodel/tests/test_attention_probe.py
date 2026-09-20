"""Tests for the probe-only attention tracker.

The probe's whole output is a measurement, so the properties that matter are the
ones that would make a wrong number look like a real one.  Three of them are pinned
here because they cannot be checked by reading the file once:

* the GQA head mapping, which must be ``repeat_kv``'s kv-major layout -- the other
  order is a silent permutation that mislabels every per-head number;
* the rotary transform, which the text attention receives as ``position_embeddings``
  and *not* through a ``rotary_emb`` attribute or ``position_ids``, so a transform
  written the obvious way reports logits from un-rotated states;
* the ``-inf`` paths, where a fully masked row used to turn into ``nan``.

The reduction is checked against an independent expectation computed in plain Python
rather than against a re-run of the same torch code.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from layout_ocr import attention_probe as probe_module
from layout_ocr.attention_probe import (
    AttentionProbe,
    _logsumexp,
    _region_owners,
    install_attention_probe,
    probe_heads,
    probe_layers,
    probe_path,
)

IMAGE_TOKEN_ID = 3
VISUAL_COUNT = 4
PROMPT_LENGTH = 6


def _bridge(positions=None):
    if positions is None:
        positions = torch.tensor(
            [[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]], dtype=torch.float32
        )
    return SimpleNamespace(last_patch_positions=positions)


def _softmax(values):
    top = max(values)
    weights = [math.exp(value - top) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def _log_sum_exp(values):
    top = max(values)
    return math.log(sum(math.exp(value - top) for value in values)) + top


# -- the pure helpers ----------------------------------------------------


def test_region_owners_follows_reading_order_on_overlap():
    """Overlap resolves to the earlier region, which is what ``layout_targets`` does."""

    grid = torch.tensor([[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]])
    regions = [
        {"bbox": [0.0, 0.0, 1.0, 0.5]},  # the whole top half: tokens 0 and 1
        {"bbox": [0.0, 0.0, 0.5, 1.0]},  # the left column: tokens 0 and 2, token 0 overlaps
    ]
    assert _region_owners(regions, grid).tolist() == [0, 0, 1, -1]


def test_region_owners_leaves_everything_off_box_as_background():
    grid = torch.tensor([[[0.5, 0.5], [0.9, 0.9]]])
    assert _region_owners([{"bbox": [0.0, 0.0, 0.1, 0.1]}], grid).tolist() == [-1, -1]
    # No regions at all is not an error, it is a page with nothing to localize.
    assert _region_owners([], grid).tolist() == [-1, -1]


def test_logsumexp_matches_plain_python():
    values = torch.tensor([[0.5, -1.0, 2.0], [1e4, 1e4, 1e4]])
    result = _logsumexp(values)
    assert result[0].item() == pytest.approx(_log_sum_exp([0.5, -1.0, 2.0]), rel=1e-6)
    # The large row is where a naive implementation overflows to ``inf``.  The
    # tolerance is float32's, which is what the logits are computed in.
    assert result[1].item() == pytest.approx(math.log(3) + 1e4, rel=1e-6)


def test_logsumexp_of_nothing_is_minus_infinity_not_nan():
    assert _logsumexp(torch.empty(2, 0)).tolist() == [float("-inf")] * 2
    # A row that a padding or routing mask removed entirely.
    masked = torch.tensor([[float("-inf"), float("-inf")], [0.0, 0.0]])
    result = _logsumexp(masked)
    assert result[0].item() == float("-inf")
    assert not torch.isnan(result).any()


# -- the reduction -------------------------------------------------------


def _armed(
    *,
    layer: int = 0,
    visual: torch.Tensor,
    query: torch.Tensor,
    text: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    owners: list[int] | None = None,
    directions: list[str] | None = None,
    num_heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 1,
    scale: float = 1.0,
    heads: tuple[int, ...] | None = None,
    grid: torch.Tensor | None = None,
) -> AttentionProbe:
    """A probe holding hand-built keys, so the reduction can be checked by hand.

    ``head_dim`` is 1 and ``scale`` is 1 so the logits are a table someone can read;
    the real geometry is exercised by the transform tests below.
    """

    runtime = AttentionProbe(_bridge(grid), IMAGE_TOKEN_ID, layers=(layer,), heads=heads)
    runtime.visual_count = visual.shape[2]
    runtime.visual_start = 1
    runtime._geometry[layer] = (num_heads, kv_heads, head_dim, scale)
    runtime._visual_keys[layer] = visual
    if text is not None:
        runtime._prompt_text_keys[layer] = text
    if owners is not None:
        runtime.owners = torch.tensor(owners, dtype=torch.long)
        runtime.num_regions = len({owner for owner in owners if owner >= 0})
    runtime._line_direction = list(directions or [])
    runtime.steps = 1
    runtime._observe_step(layer, 1, query, mask)
    return runtime


# Query head 0 and 1 share KV head 0 (which points only at visual token 0); heads 2
# and 3 share KV head 1 (which points only at visual token 3).  The four queries are
# 1, 2, 4, 8 so a mis-mapped head is visible in the mass, not just in the ordering.
def _visual_keys():
    return torch.tensor([[[[1.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [1.0]]]])


def _queries():
    return torch.tensor([[[[1.0]], [[2.0]], [[4.0]], [[8.0]]]])


def test_each_query_head_reads_the_kv_head_repeat_kv_gives_it():
    """``repeat_kv`` makes query head ``kv * groups + g`` read KV head ``kv``.

    The groups-major order would pair heads 0 and 2 with KV head 0 instead -- a
    silent permutation that leaves the aggregate numbers looking entirely normal.
    The two KV heads point at opposite ends of the line, so which end each head
    landed on says which KV head it read.
    """

    # All queries equal, so the only thing deciding where a head looks is its KV head.
    runtime = _armed(
        visual=_visual_keys(),
        query=torch.ones(1, 4, 1, 1),
        owners=[0, 0, 1, 1],
        directions=["horizontal_ltr", "horizontal_ltr"],
    )
    heads = runtime.report()["steps"][0]["heads"]
    # KV head 0 points at visual token 0 (line 0); KV head 1 at token 3 (line 1).
    assert [row["argmax_line"] for row in heads] == [0, 0, 1, 1]
    # And the pairing is by KV head, not by parity: 0 and 1 agree, 2 and 3 agree.
    assert heads[0]["top_line_mass"] == pytest.approx(heads[1]["top_line_mass"])
    assert heads[2]["top_line_mass"] == pytest.approx(heads[3]["top_line_mass"])


def test_the_visual_mass_is_checked_against_plain_python():
    """m_t is measured against an independently computed log-sum-exp."""

    # Query head 0 has q = 1 and reads KV head 0, whose keys are [1, 0, 0, 0], so the
    # scaled logits over the visual span are exactly that.
    visual_logits = [1.0, 0.0, 0.0, 0.0]
    text_logits = [0.0, 0.0]
    runtime = _armed(
        visual=_visual_keys(),
        query=_queries(),
        text=torch.zeros(1, 2, 2, 1),
        heads=(0,),
    )
    row = runtime.report()["steps"][0]["heads"][0]
    lse_vis = _log_sum_exp(visual_logits)
    lse_text = _log_sum_exp(text_logits)
    assert row["lse_vis"] == pytest.approx(lse_vis, rel=1e-6)
    assert row["lse_text"] == pytest.approx(lse_text, rel=1e-6)
    assert row["m_t"] == pytest.approx(
        math.exp(lse_vis - _log_sum_exp([lse_vis, lse_text])), rel=1e-6
    )
    # Both log-sums are recorded separately because m_t alone cannot separate "looks
    # at the image" from "generated a lot": the visual keys are fixed and the text
    # keys grow, so m_t falls with generation length for reasons that have nothing to
    # do with looking.
    assert row["lse_vis"] > row["lse_text"]


def test_the_visual_distribution_is_normalized_over_visual_keys_only():
    """``a_vis`` sums to one even when the model barely looks at the image."""

    # A text key with a huge logit would take almost all the mass, and must not
    # change the shape of the visual-conditional distribution at all.
    text = torch.zeros(1, 2, 2, 1)
    text[:, :, 0, :] = 50.0
    runtime = _armed(
        visual=_visual_keys(),
        query=_queries(),
        text=text,
        owners=[0, 0, 1, -1],
        heads=(0,),
    )
    row = runtime.report()["steps"][0]["heads"][0]
    assert sum(row["line_probs"]) == pytest.approx(1.0, rel=1e-6)
    assert row["m_t"] < 1e-6  # the text keys dominate the total mass
    assert row["argmax_line"] == 0
    expected = _softmax([1.0, 0.0, 0.0, 0.0])
    assert row["line_probs"][0] == pytest.approx(expected[0] + expected[1], rel=1e-6)
    assert row["top_line_mass"] == pytest.approx(expected[0] + expected[1], rel=1e-6)


def test_the_line_readout_separates_background_from_a_line():
    """A high background share means the line readout is reading noise."""

    runtime = _armed(
        visual=_visual_keys(),
        query=_queries(),
        owners=[0, -1, -1, 1],
        heads=(0,),
    )
    row = runtime.report()["steps"][0]["heads"][0]
    # Query head 0 has scaled logits [1, 0, 0, 0], so its distribution is the softmax
    # of that -- not a point mass: line 0 keeps token 0's share, and both background
    # tokens are counted in the last slot rather than dropped.
    dist = _softmax([1.0, 0.0, 0.0, 0.0])
    assert row["line_probs"][0] == pytest.approx(dist[0], rel=1e-6)
    assert row["line_probs"][1] == pytest.approx(dist[3], rel=1e-6)
    assert row["background_mass"] == pytest.approx(dist[1] + dist[2], rel=1e-6)
    assert sum(row["line_probs"]) == pytest.approx(1.0, rel=1e-6)


def test_the_entropy_is_normalized_by_the_visual_token_count():
    """The 1M and 4M arms see 630 and 2496 tokens; raw entropy is not comparable."""

    runtime = _armed(visual=_visual_keys(), query=_queries(), heads=(0,))
    entropy_norm = runtime.report()["steps"][0]["heads"][0]["entropy_norm"]
    dist = _softmax([1.0, 0.0, 0.0, 0.0])
    raw = -sum(p * math.log(p) for p in dist)
    assert entropy_norm == pytest.approx(raw / math.log(VISUAL_COUNT), rel=1e-6)
    assert 0.0 <= entropy_norm <= 1.0


def test_an_entirely_masked_visual_row_is_zero_not_nan():
    """A padding or routing mask can remove every visual key from a row."""

    mask = torch.zeros(1, 1, 1, VISUAL_COUNT + 2)
    mask[0, 0, 0, 1:5] = float("-inf")  # the visual span, all of it
    runtime = _armed(
        visual=_visual_keys(),
        query=_queries(),
        text=torch.zeros(1, 2, 2, 1),
        mask=mask,
        owners=[0, 0, 1, -1],
        heads=(0,),
    )
    for row in runtime.report()["steps"][0]["heads"]:
        assert row["m_t"] == 0.0
        assert not math.isnan(row["m_t"])
        assert row["line_probs"] == [0.0, 0.0, 0.0]
        assert not math.isnan(row["entropy_norm"])


def test_a_routing_bias_is_added_to_the_scaled_logits():
    """``sdpa_attention_forward`` scales first, so the mask is not scaled.

    Scaling the mask would divide the routing arm's B by sqrt(head_dim) and report a
    different arm than the one that ran.
    """

    # A bias of +10 on visual token 3 should dominate, but by a bounded amount: it is
    # 10 logits, not 10/sqrt(head_dim).
    scale = 0.5
    mask = torch.zeros(1, 1, 1, VISUAL_COUNT + 2)
    mask[0, 0, 0, 4] = 10.0
    runtime = _armed(
        visual=_visual_keys(),
        query=_queries(),
        text=torch.zeros(1, 2, 2, 1),
        mask=mask,
        owners=[0, 0, 1, -1],
        scale=scale,
        heads=(0,),
    )
    row = runtime.report()["steps"][0]["heads"][0]
    # Query head 0 scaled logits are [0.5, 0, 0, 0]; token 3 gets +10.
    dist = _softmax([0.5, 0.0, 0.0, 10.0])
    assert row["line_probs"][0] == pytest.approx(dist[0] + dist[1], rel=1e-6)
    assert row["line_probs"][1] == pytest.approx(dist[2], rel=1e-6)
    assert row["line_probs"][-1] == pytest.approx(dist[3], rel=1e-6)


def test_a_boolean_mask_excludes_rather_than_adds():
    """A bool mask means "attend / do not attend"; adding it would invert its meaning."""

    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.visual_count = VISUAL_COUNT
    runtime.visual_start = 1
    runtime._geometry[0] = (4, 2, 1, 1.0)
    runtime._visual_keys[0] = _visual_keys()
    runtime._prompt_text_keys[0] = torch.zeros(1, 2, 2, 1)
    # One row of kv_len flags, with the visual span allowed except its last token.
    mask = torch.ones(1, 1, 1, VISUAL_COUNT + 2, dtype=torch.bool)
    mask[0, 0, 0, 4] = False
    visual, text = runtime._split_mask(mask, 0)
    assert visual[:3].tolist() == [0.0, 0.0, 0.0]
    assert visual[3].item() == float("-inf")  # excluded, not boosted by +1
    assert text.tolist() == [0.0, 0.0]


def test_a_mask_that_does_not_match_the_captured_keys_is_refused():
    """A silently misaligned mask would attribute one token's bias to another.

    Raising is deliberate: this runs inside ``generate``, so the alternative is an
    expensive evaluation whose every number is quietly shifted by one key.
    """

    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.visual_count = VISUAL_COUNT
    runtime.visual_start = 1
    runtime._geometry[0] = (4, 2, 1, 1.0)
    runtime._visual_keys[0] = _visual_keys()
    runtime._prompt_text_keys[0] = torch.zeros(1, 2, 2, 1)
    with pytest.raises(RuntimeError, match="cannot align the mask"):
        runtime._observe_step(0, 1, _queries(), torch.zeros(1, 1, 1, 99))


def test_a_layer_without_visual_keys_records_nothing():
    """A failed transform must leave a gap, not a fabricated row."""

    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.visual_count = VISUAL_COUNT
    runtime.visual_start = 1
    runtime._geometry[0] = (4, 2, 1, 1.0)
    runtime.steps = 1
    # No ``_visual_keys`` for this layer: the prefill's transform failed.
    runtime._observe_step(0, 1, _queries(), None)
    assert runtime.report()["steps"] == []


def test_observing_a_prefill_query_is_refused():
    """A multi-token query must go to ``_store_prompt``, not through the reduction."""

    runtime = _armed(visual=_visual_keys(), query=_queries(), heads=(0,))
    with pytest.raises(RuntimeError, match="one decode step at a time"):
        runtime._observe_step(0, 2, torch.zeros(1, 4, 5, 1), None)


# -- span splitting and the prefill --------------------------------------


def test_the_prompt_key_span_is_split_around_the_visual_span():
    """Visual tokens are contiguous, and the text keeps its sequence order."""

    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.visual_start = 2
    runtime.visual_count = 3
    # Two KV heads, head_dim 1, six keys: two text, three visual, one text.  The
    # two heads carry different values so a slice taken on the wrong axis -- dim 1 is
    # the KV head, not the sequence -- shows up as the wrong numbers rather than the
    # right shape.
    key = torch.tensor(
        [
            [
                [[1.0], [2.0], [10.0], [11.0], [12.0], [3.0]],
                [[4.0], [5.0], [20.0], [21.0], [22.0], [6.0]],
            ]
        ]
    )
    runtime._store_prompt(0, key)
    assert runtime._visual_keys[0].squeeze(0).flatten().tolist() == [
        10.0, 11.0, 12.0, 20.0, 21.0, 22.0,
    ]
    assert runtime._visual_keys[0].shape == (1, 2, 3, 1)
    # The trailing text key follows the leading ones, so the text span is in sequence
    # order and a mask split the same way lines up.
    assert runtime._prompt_text_keys[0].shape == (1, 2, 3, 1)
    assert runtime._prompt_text_keys[0].squeeze(0).flatten().tolist() == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    ]
    assert runtime._text_key_count(0) == 3


def test_decoding_keys_are_folded_in_sequence_order():
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.visual_start = 0
    runtime.visual_count = 1
    runtime._store_prompt(0, torch.tensor([[[[10.0], [1.0]]]]))
    runtime._gen_keys[0] = [torch.tensor([[[[2.0]]]]), torch.tensor([[[[3.0]]]])]
    assert runtime._text_key_count(0) == 3
    assert runtime._text_keys(0).flatten().tolist() == [1.0, 2.0, 3.0]


def test_the_mask_is_split_the_way_the_keys_were():
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.visual_start = 2
    runtime.visual_count = 3
    runtime._prompt_text_keys[0] = torch.zeros(1, 1, 3, 1)
    # One row of ``kv_len`` entries, which is what the routing hook builds.
    mask = torch.tensor([[[[1.0, 2.0, 10.0, 11.0, 12.0, 3.0]]]])
    visual, text = runtime._split_mask(mask, 0)
    assert visual.tolist() == [10.0, 11.0, 12.0]
    assert text.tolist() == [1.0, 2.0, 3.0]


# -- the transform -------------------------------------------------------


class _FakeAttention(nn.Module):
    """The structural shape ``_find_attention_modules`` looks for."""

    def __init__(self, hidden: int = 8, num_heads: int = 4, kv_heads: int = 2, head_dim: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)


def _identity_rotary(monkeypatch):
    """Stands in for the model's own helper, recording what it was handed."""

    calls = []

    def fake(query, key, cos, sin):
        calls.append((query.shape, key.shape, cos, sin))
        return query, key

    monkeypatch.setattr(probe_module, "_apply_rotary_pos_emb", fake)
    return calls


def test_the_transform_shapes_heads_and_applies_rotary(monkeypatch):
    calls = _identity_rotary(monkeypatch)
    module = _FakeAttention()
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime._read_shape(module, 0)
    cos, sin = torch.ones(1, 5, 2), torch.zeros(1, 5, 2)
    hidden = torch.randn(1, 5, 8)
    query, key = runtime._project(module, 0, hidden, (cos, sin))
    assert query.shape == (1, 4, 5, 2)
    assert key.shape == (1, 2, 5, 2)
    # The rotary has to be applied with the model's own cos/sin, and it is the only
    # place the rotation happens.
    assert len(calls) == 1
    assert torch.equal(calls[0][2], cos) and torch.equal(calls[0][3], sin)


def test_a_forward_without_position_embeddings_fails_loudly(monkeypatch):
    """The text attention has no ``rotary_emb`` and never sees ``position_ids``.

    A transform written against either would apply no rotation at all and still
    produce a plausible-looking distribution, so the absence has to be an error.
    """

    _identity_rotary(monkeypatch)
    module = _FakeAttention()
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime._read_shape(module, 0)
    assert runtime._project(module, 0, torch.randn(1, 5, 8), None) is None
    failure = runtime.report()["transform_failed"]
    assert "position_embeddings" in failure["0"]
    # And the layer stays failed rather than being retried into a wrong answer.
    later = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))
    assert runtime._project(module, 0, torch.randn(1, 5, 8), later) is None


def test_a_single_return_rotary_helper_is_refused(monkeypatch):
    """The two-tensor signature is what rotates the keys as well as the queries.

    A helper that returned only the queries would leave the keys un-rotated, which
    every logit downstream depends on -- so it is refused rather than unpacked.
    """

    monkeypatch.setattr(probe_module, "_rotary_helper", lambda: (lambda q, k, cos, sin: q))
    with pytest.raises(RuntimeError, match="single tensor"):
        probe_module._apply_rotary_pos_emb(
            torch.zeros(1, 2, 3, 4),
            torch.zeros(1, 2, 3, 4),
            torch.ones(1, 3, 4),
            torch.zeros(1, 3, 4),
        )


def test_the_geometry_is_read_from_the_module_not_recomputed():
    module = _FakeAttention(head_dim=4)
    module.scaling = 0.25  # a config that changed the scale must move the probe with it
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime._read_shape(module, 0)
    assert runtime._geometry[0] == (4, 2, 4, 0.25)


def test_a_head_beyond_the_layer_is_refused():
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,), heads=(9,))
    with pytest.raises(RuntimeError, match="query heads"):
        runtime._read_shape(_FakeAttention(), 0)


# -- hooks ---------------------------------------------------------------


def test_the_hooks_leave_the_forward_untouched(monkeypatch):
    """The probe must not write into kwargs: that is what keeps the run bit-identical."""

    _identity_rotary(monkeypatch)
    module = _FakeAttention()
    module._probe_layer_idx = 0
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.set_page("p0", [], PROMPT_LENGTH, _prompt_ids())
    runtime.visual_count = VISUAL_COUNT
    runtime.visual_start = 1

    hidden = torch.randn(1, 1, 8)
    kwargs = {
        "hidden_states": hidden,
        "position_embeddings": (torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
        "attention_mask": None,
    }
    keys_before = list(kwargs)
    values_before = dict(kwargs)
    runtime._pre(module, (), kwargs)
    assert list(kwargs) == keys_before
    runtime._post(module, (), kwargs, None)
    # Same keys, same objects: the probe observed and wrote nothing back.
    assert list(kwargs) == keys_before
    for key, value in values_before.items():
        assert kwargs[key] is value


def test_a_prefill_is_stored_and_a_decode_step_is_observed(monkeypatch):
    _identity_rotary(monkeypatch)
    module = _FakeAttention()
    module._probe_layer_idx = 0
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.set_page("p0", [], PROMPT_LENGTH, _prompt_ids())

    prefill = {
        "hidden_states": torch.randn(1, PROMPT_LENGTH, 8),
        "position_embeddings": (torch.ones(1, PROMPT_LENGTH, 2), torch.zeros(1, PROMPT_LENGTH, 2)),
    }
    runtime._pre(module, (), prefill)
    runtime._post(module, (), prefill, None)
    assert runtime.steps == 0
    assert runtime._visual_keys[0].shape == (1, 2, VISUAL_COUNT, 2)

    for step in range(3):
        decode = {
            "hidden_states": torch.randn(1, 1, 8),
            "position_embeddings": (torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
        }
        runtime._pre(module, (), decode)
        runtime._post(module, (), decode, None)
        assert runtime.steps == step + 1
    # One generated key per step, folded onto the prompt's text keys.
    assert runtime._text_key_count(0) == PROMPT_LENGTH - VISUAL_COUNT + 3
    assert runtime.report()["decoding_steps"] == 3


def test_position_ids_are_not_a_substitute_for_position_embeddings(monkeypatch):
    """The trap: ``position_ids`` is not an argument of the attention forward."""

    _identity_rotary(monkeypatch)
    module = _FakeAttention()
    module._probe_layer_idx = 0
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.set_page("p0", [], PROMPT_LENGTH, _prompt_ids())
    forward = {"hidden_states": torch.randn(1, 1, 8), "position_ids": torch.tensor([[5]])}
    runtime._pre(module, (), forward)
    runtime._post(module, (), forward, None)
    report = runtime.report()
    # Nothing was observed, which is the honest count -- and ``transform_failed`` says
    # why, so a zero here cannot be mistaken for a page that never generated.
    assert report["decoding_steps"] == 0
    assert "position_embeddings" in report["transform_failed"]["0"]


def test_a_page_without_a_grid_counts_the_gap(monkeypatch):
    """A report of "no localization" and one of "never measured it" must differ."""

    _identity_rotary(monkeypatch)
    module = _FakeAttention()
    module._probe_layer_idx = 0
    runtime = AttentionProbe(
        SimpleNamespace(last_patch_positions=None), IMAGE_TOKEN_ID, layers=(0,)
    )
    runtime.set_page(
        "p0",
        [{"bbox": [0.0, 0.0, 1.0, 1.0], "reading_order": 0}],
        PROMPT_LENGTH,
        _prompt_ids(),
    )
    prefill = {
        "hidden_states": torch.randn(1, PROMPT_LENGTH, 8),
        "position_embeddings": (torch.ones(1, PROMPT_LENGTH, 2), torch.zeros(1, PROMPT_LENGTH, 2)),
    }
    runtime._pre(module, (), prefill)
    runtime._post(module, (), prefill, None)
    decode = {
        "hidden_states": torch.randn(1, 1, 8),
        "position_embeddings": (torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
    }
    runtime._pre(module, (), decode)
    runtime._post(module, (), decode, None)
    report = runtime.report()
    assert report["grid_missing_steps"] == 1
    assert "line_probs" not in report["steps"][0]["heads"][0]


def test_the_owners_are_resolved_once_the_grid_exists(monkeypatch):
    _identity_rotary(monkeypatch)
    module = _FakeAttention()
    module._probe_layer_idx = 0
    grid = torch.tensor([[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]])
    runtime = AttentionProbe(_bridge(grid), IMAGE_TOKEN_ID, layers=(0,))
    runtime.set_page(
        "p0",
        [{"bbox": [0.0, 0.0, 0.5, 1.0], "reading_order": 0, "writing_direction": "vertical_rtl"}],
        PROMPT_LENGTH,
        _prompt_ids(),
    )
    assert runtime.owners is None  # the visual tower has not run yet
    prefill = {
        "hidden_states": torch.randn(1, PROMPT_LENGTH, 8),
        "position_embeddings": (torch.ones(1, PROMPT_LENGTH, 2), torch.zeros(1, PROMPT_LENGTH, 2)),
    }
    runtime._pre(module, (), prefill)
    runtime._post(module, (), prefill, None)
    decode = {
        "hidden_states": torch.randn(1, 1, 8),
        "position_embeddings": (torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
    }
    runtime._pre(module, (), decode)
    runtime._post(module, (), decode, None)
    assert runtime.owners.tolist() == [0, -1, 0, -1]
    # A vertical column is read top-to-bottom, so the position axis is y.
    row = runtime.report()["steps"][0]["heads"][0]
    assert 0.0 <= row["in_line_pos"] <= 1.0


# -- installation and the environment -----------------------------------


class _FakeTextModel(nn.Module):
    def __init__(self, hidden_size: int = 8, layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden_size)
        self.layers = nn.ModuleList([nn.Module() for _ in range(layers)])
        for index, layer in enumerate(self.layers):
            attention = _FakeAttention(hidden=hidden_size)
            # transformers sets this for the KV cache; the probe must not clobber it.
            attention.layer_idx = index
            layer.self_attn = attention


class _FakeModel(nn.Module):
    def __init__(self, layers: int = 4):
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = nn.Module()
        self.model.language_model = _FakeTextModel(layers=layers)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens


def _prompt_ids():
    return torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2]])


def test_installation_hooks_only_the_selected_layers():
    model = _FakeModel(layers=6)
    runtime, handles = install_attention_probe(model, _bridge(), layers=(1, 3))
    assert len(handles) == 4  # a pre and a post hook for each of two layers
    assert runtime.layers == (1, 3)
    assert runtime.image_token_id == IMAGE_TOKEN_ID
    for handle in handles:
        handle.remove()


def test_installation_does_not_clobber_the_kv_cache_layer_index():
    """``layer_idx`` is how the cache finds its slot; overwriting it corrupts updates."""

    model = _FakeModel(layers=3)
    install_attention_probe(model, _bridge(), layers=(0, 2))
    for index, layer in enumerate(model.model.language_model.layers):
        assert layer.self_attn.layer_idx == index
        assert layer.self_attn._probe_layer_idx == index


def test_installation_refuses_a_layer_the_model_does_not_have():
    model = _FakeModel(layers=2)
    with pytest.raises(RuntimeError, match="does not have"):
        install_attention_probe(model, _bridge(), layers=(0, 7))


def test_installation_refuses_a_model_without_decoder_layers():
    model = _FakeModel()
    model.model.language_model.layers = nn.ModuleList()
    with pytest.raises(RuntimeError, match="text decoder layers"):
        install_attention_probe(model, _bridge(), layers=(0,))


def test_installation_refuses_a_model_without_attention_modules():
    model = _FakeModel()
    model.model.language_model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
    with pytest.raises(RuntimeError, match="attention modules"):
        install_attention_probe(model, _bridge(), layers=(0,))


def test_the_probe_reads_its_environment(monkeypatch):
    monkeypatch.delenv("GLMOCR_ATTENTION_PROBE", raising=False)
    monkeypatch.delenv("GLMOCR_ATTENTION_PROBE_LAYERS", raising=False)
    monkeypatch.delenv("GLMOCR_ATTENTION_PROBE_HEADS", raising=False)
    assert probe_path() is None
    assert probe_layers() == (0, 4, 8, 12)
    assert probe_heads() is None  # every head, which the cross-head readout needs

    monkeypatch.setenv("GLMOCR_ATTENTION_PROBE", "probe.jsonl")
    monkeypatch.setenv("GLMOCR_ATTENTION_PROBE_LAYERS", "0, 2 ,4")
    monkeypatch.setenv("GLMOCR_ATTENTION_PROBE_HEADS", "1,3")
    assert probe_path() == "probe.jsonl"
    assert probe_layers() == (0, 2, 4)
    assert probe_heads() == (1, 3)

    monkeypatch.setenv("GLMOCR_ATTENTION_PROBE_LAYERS", "0,x")
    with pytest.raises(ValueError, match="comma-separated"):
        probe_layers()
    monkeypatch.setenv("GLMOCR_ATTENTION_PROBE_HEADS", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        probe_heads()


def test_write_probe_appends_one_line_per_page(tmp_path, monkeypatch):
    monkeypatch.setenv("GLMOCR_ATTENTION_PROBE", str(tmp_path / "probe.jsonl"))
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.set_page("p0", [], PROMPT_LENGTH, _prompt_ids())
    runtime.write_probe()
    runtime.set_page("p1", [], PROMPT_LENGTH, _prompt_ids())
    runtime.write_probe()
    lines = (tmp_path / "probe.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [__import__("json").loads(line)["page_id"] for line in lines] == ["p0", "p1"]


class _CumulativeDecoder:
    """The tokenizer behaviour the alignment depends on: cumulative decode."""

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[int(index)] for index in ids)


def test_emitted_text_is_stamped_with_the_one_step_offset(monkeypatch):
    """Token 0 comes from the prefill, so step ``s`` predicts token ``s``.

    Getting this wrong shifts every step's text by one character, which would move
    the whole localization readout by one line-break's worth of error and still look
    plausible.
    """

    _identity_rotary(monkeypatch)
    decoder = _CumulativeDecoder(["甲", "乙", "丙", "丁"])
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.set_page("p0", [], PROMPT_LENGTH, _prompt_ids())
    for step in range(3):
        runtime.steps = step + 1
        runtime._records.append({"step": step + 1, "text_keys": 0, "heads": []})
    runtime.attach_emitted(torch.tensor([0, 1, 2, 3]), decoder)
    # Steps 1..3 take tokens 1..3; token 0 belongs to the prefill and to no step.
    assert [record["emitted"] for record in runtime._records] == ["乙", "丙", "丁"]
    assert runtime.emitted_missing == 0
    assert runtime.report()["emitted_missing_steps"] == 0


def test_a_multi_character_token_lands_whole_on_its_step():
    """One token can carry several characters; the step gets all of them."""

    decoder = _CumulativeDecoder(["甲", "乙丙", "丁"])
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.set_page("p0", [], PROMPT_LENGTH, _prompt_ids())
    runtime._records.append({"step": 1, "text_keys": 0, "heads": []})
    runtime._records.append({"step": 2, "text_keys": 0, "heads": []})
    runtime.attach_emitted(torch.tensor([0, 1, 2]), decoder)
    assert [record["emitted"] for record in runtime._records] == ["乙丙", "丁"]


def test_a_step_without_text_is_counted_not_guessed():
    """A step past the generated span is reported, never given a fabricated string."""

    decoder = _CumulativeDecoder(["甲", "乙"])
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.set_page("p0", [], PROMPT_LENGTH, _prompt_ids())
    runtime._records.append({"step": 1, "text_keys": 0, "heads": []})
    runtime._records.append({"step": 9, "text_keys": 0, "heads": []})
    assert runtime.attach_emitted(torch.tensor([0, 1]), decoder) == 1
    assert "emitted" not in runtime._records[1]
    assert runtime.report()["emitted_missing_steps"] == 1


def test_clearing_a_page_drops_the_captured_evidence():
    runtime = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(0,))
    runtime.set_page("p0", [], PROMPT_LENGTH, _prompt_ids())
    runtime._visual_keys[0] = torch.zeros(1, 2, VISUAL_COUNT, 1)
    runtime.clear_page()
    assert runtime.report()["visual_tokens"] is None
    assert runtime._visual_keys == {}


def test_a_probe_without_an_image_token_id_is_an_error():
    runtime = AttentionProbe(_bridge(), None, layers=(0,))
    with pytest.raises(RuntimeError, match="image token id"):
        runtime.set_page("p0", [], PROMPT_LENGTH, _prompt_ids())
