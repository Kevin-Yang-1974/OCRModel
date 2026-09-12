"""Training-only recovery from detached, naturally generated OCR loops."""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .stabilization import generated_cycle_window


CONFIG = {
    "mode": "aligned_recovery_v1", "interval": 4, "max_new_tokens": 768,
    "min_cycle_length": 8, "max_cycle_length": 32, "cycle_repeats": 3,
    "horizon": 16, "max_alignment_error": 0.2, "tail_matches": 8,
    "ul_weight": 0.01, "recover_weight": 0.05, "end_weight": 0.05,
    "ramp_steps": 102, "inference_intervention": False,
}


def align_prefix(predicted: Sequence[int], target: Sequence[int]) -> dict[str, Any]:
    """Unit-cost edit alignment to any GT prefix; trailing GT is unconsumed."""
    n, m = len(predicted), len(target)
    costs = [list(range(m + 1))]
    for i, token in enumerate(predicted, 1):
        previous = costs[-1]
        row = [i]
        for j, gold in enumerate(target, 1):
            row.append(min(previous[j] + 1, row[-1] + 1,
                           previous[j - 1] + (token != gold)))
        costs.append(row)
    best = min(costs[-1])
    endpoints = [j for j, cost in enumerate(costs[-1]) if cost == best]
    end = endpoints[0]
    i, j = n, end
    deletions = insertions = substitutions = tail = 0
    at_tail = True
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (predicted[i - 1] != target[j - 1]):
            if predicted[i - 1] == target[j - 1]:
                if at_tail:
                    tail += 1
                else:
                    at_tail = False
            else:
                substitutions += 1
                at_tail = False
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            i -= 1
            insertions += 1
        else:
            j -= 1
            deletions += 1
            at_tail = False
    return {"cost": best, "endpoint": end, "unique": len(endpoints) == 1,
            "endpoints": endpoints,
            "error": (substitutions + deletions) / max(1, end),
            "raw_error": best / max(1, n, end),
            "tail_matches": tail, "deletions": deletions,
            "substitutions": substitutions, "insertions": insertions}


def recovery_target(generated: Sequence[int], gold: Sequence[int], eos_ids: set[int]) -> dict[str, Any]:
    generated, gold = list(generated), list(gold)
    eos_position = next((i for i, t in enumerate(gold) if t in eos_ids), len(gold))
    terminal = gold[eos_position:eos_position + 1]
    gold = gold[:eos_position]
    stop = next((i for i, t in enumerate(generated) if t in eos_ids), len(generated))
    cycle = generated_cycle_window(generated[:stop], min_cycle_length=8,
                                  max_cycle_length=32, cycle_repeats=3,
                                  recent_window=None)
    result = {"accepted": False, "detected": bool(cycle["detected"]),
              "reason": "no_cycle", "cycle": cycle}
    if not cycle["detected"]:
        return result
    prefix = generated[:cycle["end"]]
    collapsed = prefix[:cycle["start"]] + list(cycle["cycle"])
    original, aligned = align_prefix(prefix, gold), align_prefix(collapsed, gold)
    endpoint = aligned["endpoint"]
    gold_next = gold[endpoint] if endpoint < len(gold) else (terminal[0] if terminal else None)
    result.update(alignment=aligned, original_alignment=original, prefix=prefix,
                  negative=int(cycle["cycle"][0]), endpoint=endpoint, gold_next=gold_next)
    endpoints = aligned["endpoints"]
    unique_endpoint = (
        len(endpoints) == 1 or max(endpoints) - min(endpoints) <= cycle["length"]
    )
    checks = [
        (unique_endpoint, "ambiguous_endpoint"),
        (aligned["error"] <= 0.2, "alignment_error"),
        (aligned["tail_matches"] >= 8, "weak_tail"),
        (original["cost"] - aligned["cost"] >= math.ceil(cycle["length"] / 2), "legal_or_ambiguous_repeat"),
    ]
    for valid, reason in checks:
        if not valid:
            result["reason"] = reason
            return result
    suffix = gold[endpoint:endpoint + 16]
    if not suffix:
        if aligned["deletions"] or not terminal:
            result["reason"] = "unsafe_end"
            return result
        suffix = terminal
    elif endpoint + len(suffix) == len(gold) and len(suffix) < 16 and not aligned["deletions"]:
        suffix += terminal
    result.update(accepted=True, reason="accepted", suffix=suffix)
    return result


def rebuild_inputs(inputs: dict[str, Any], prompt_length: int,
                   prefix: Sequence[int], suffix: Sequence[int]) -> dict[str, Any]:
    """Rebuild all sequence-aligned processor fields; preserve image tensors."""
    ids = inputs["input_ids"]
    tail = torch.tensor([list(prefix) + list(suffix)], dtype=ids.dtype, device=ids.device)
    rebuilt = {k: v for k, v in inputs.items()
               if k not in {"labels", "position_ids", "cache_position", "past_key_values", "rope_deltas"}}
    rebuilt["input_ids"] = torch.cat((ids[:, :prompt_length], tail), dim=1)
    for key in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
        if key in inputs:
            value = inputs[key]
            extension = torch.full_like(tail, 1 if key == "attention_mask" else 0,
                                        dtype=value.dtype)
            rebuilt[key] = torch.cat((value[:, :prompt_length], extension), dim=1)
    labels = torch.full_like(rebuilt["input_ids"], -100)
    if suffix:
        labels[:, -len(suffix):] = tail[:, -len(suffix):]
    rebuilt["labels"] = labels
    return rebuilt


def collect_rollout(model: Any, inputs: dict[str, Any], eos_ids: set[int]) -> dict[str, Any]:
    positions = (inputs["labels"][0] != -100).nonzero().flatten()
    prompt_length = int(positions[0])
    prompt = rebuild_inputs(inputs, prompt_length, [], [])
    prompt.pop("labels")
    # Preserve the individual module modes: the frozen backbone may be eval
    # while only its LoRA modules are train. Also restore multimodal RoPE state.
    modes = [(module, module.training) for module in model.modules()]
    rope_state = [(module, module.rope_deltas) for module in model.modules() if hasattr(module, "rope_deltas")]
    previous_cache = getattr(model.config, "use_cache", None)
    model.eval()
    try:
        model.config.use_cache = True
        with torch.no_grad():
            output = model.generate(**prompt, do_sample=False, use_cache=True,
                                    max_new_tokens=768, repetition_penalty=1.0,
                                    no_repeat_ngram_size=0, forced_eos_token_id=None,
                                    eos_token_id=sorted(eos_ids))
    finally:
        model.config.use_cache = previous_cache
        for module, training in modes:
            module.training = training
        for module, value in rope_state:
            module.rope_deltas = value
    tokens = output[0, prompt_length:].detach().cpu().tolist()
    gold = inputs["labels"][0, positions].detach().cpu().tolist()
    target = recovery_target(tokens, gold, eos_ids)
    target["rollout_tokens"] = len(tokens)
    target["boundary"] = None
    target["prefix_inputs"] = inputs
    if target["detected"]:
        # A detected loop always rebuilds the real error prefix so the
        # unlikelihood term can fire even when recovery alignment is rejected;
        # rejected samples keep all-ignored labels and still run the forward.
        suffix = target.get("suffix") or []
        rebuilt = rebuild_inputs(inputs, prompt_length, target["prefix"], suffix)
        text_config = getattr(model.config, "text_config", model.config)
        capacity = getattr(text_config, "max_position_embeddings", None)
        if capacity and rebuilt["input_ids"].shape[1] > capacity:
            target.update(accepted=False, reason="context_capacity")
        else:
            target["prefix_inputs"] = rebuilt
            target["boundary"] = prompt_length + len(target["prefix"]) - 1
    return target


def _global_token_mean(total: torch.Tensor, count: int) -> torch.Tensor:
    denominator = total.new_tensor(float(count))
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(denominator)
        world_size = dist.get_world_size()
    # DDP averages rank gradients; undo that average for a global token mean.
    return total * world_size / denominator.clamp_min(1)


def recovery_losses(logits: torch.Tensor, rollout: dict[str, Any], eos_ids: set[int]) -> dict[str, torch.Tensor]:
    zero = logits[..., :1].float().sum() * 0.0
    labels = rollout["prefix_inputs"]["labels"][:, 1:]
    valid = labels != -100 if rollout["accepted"] else torch.zeros_like(labels, dtype=torch.bool)
    ends = valid & torch.isin(labels, labels.new_tensor(sorted(eos_ids)))
    content = valid & ~ends
    counts = {"recover": int(content.sum()), "end": int(ends.sum()), "ul": 0}
    sums = {"recover": zero, "end": zero, "ul": zero}
    for key, mask in (("recover", content), ("end", ends)):
        if counts[key]:
            sums[key] = F.cross_entropy(logits[:, :-1][mask].float(), labels[mask], reduction="sum")
    boundary = rollout.get("boundary")
    negative = rollout.get("negative")
    if (bool(rollout.get("detected")) and boundary is not None
            and negative is not None and negative != rollout.get("gold_next")):
        boundary_logits = logits[0, boundary].float()
        probability = boundary_logits.log_softmax(-1)[negative].exp().clamp(max=1 - 1e-6)
        sums["ul"] = -torch.log1p(-probability)
        counts["ul"] = 1
    means = {key: _global_token_mean(sums[key], counts[key]) for key in ("ul", "recover", "end")}
    scalar = lambda value: zero.detach().new_tensor(float(value))
    return {"loss": 0.01 * means["ul"] + 0.05 * means["recover"] + 0.05 * means["end"],
            "unlikelihood": means["ul"], "continuation": means["recover"], "end": means["end"],
            "active_tokens": scalar(counts["ul"]), "candidate_tokens": scalar(counts["ul"]),
            "continuation_tokens": scalar(counts["recover"]), "end_tokens": scalar(counts["end"]),
            "active_pages": scalar(rollout["accepted"])}
