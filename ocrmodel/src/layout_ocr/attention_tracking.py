"""Drive the routing bias from the model's own attention estimate of which line it is on.

``gtmap_track``: the boxes are annotation (so detection error is out of the picture) and the
*state* -- which line is being read -- comes from the attention readout rather than from the
reference text.

## Why this is the arm that decides the route

The oracle line arm biases the true line at every step and gains 26.6% CER. The static predicted
map biases every line and is an exact null, because it has no per-step state: it points at the right
line for one line in twenty-four. So the deployable form needs a per-step line estimate, and the
only signal available for one is the model's own attention.

That is the plan's stated root difficulty. Observation and intervention act on the same attention:
once a bias is applied, the distribution is no longer spontaneous, so reading the line off it is
reading a quantity the previous step's bias has moved. This module is where that is measured rather
than argued about.

## The lag is structural, not a design choice

The routing hook is on the decoder layers and fires before any attention runs; the probe's estimate
is produced by a post-hook on layer 8, later in the same forward. So at step ``t`` the routing reads
what the probe wrote at step ``t - 1`` -- automatically, and without either module knowing about the
other's timing. That is why this file is small: the one-step lag the plan calls for falls out of
where the two hooks already are.

## The gate

A step's estimate is used only when it clears the confidence bar that stage 1 pre-registered and
then validated on held-out pages: ``top_line_mass * num_regions >= 6.0``, the scale-free unit that
stops the bar from meaning different things on a page with ten lines and one with forty.

Below the bar the estimate is *dropped*, not carried forward. The plan asks for exactly that --
"when confidence is low, set the gate to zero" -- and carrying a stale line would keep aiming at a
line the readout has just said it is unsure about, which is the failure the gate exists to prevent.
"""

from __future__ import annotations

from typing import Any

# Stage 1's registered bar, in the scale-free unit.  Kept as a named default so a run that changes
# it has to say so rather than passing a bare number.
DEFAULT_CONFIDENCE = 6.0


class TrackedLineState:
    """The lagged line estimate, written by the probe and read by the routing bias.

    Deliberately a plain shared object rather than the two modules knowing each other: whoever
    produces estimates calls ``observe``, whoever consumes them reads ``line``, and the ordering is
    whatever the hooks already do.
    """

    def __init__(self, confidence: float = DEFAULT_CONFIDENCE, switch_bar: float = 1.0) -> None:
        if switch_bar < 1.0:
            # Below one a different line would win while holding less mass than the current one,
            # which is not a policy this module has been measured on; one is the recorded
            # behaviour and is the floor.
            raise ValueError("switch_bar must be at least 1.0")
        self.threshold = float(confidence)
        # How much more mass a different line has to hold, relative to what the line in use still
        # holds, before the tracker moves to it.  One is the recorded behaviour.
        #
        # The margin is against the mass on the current line, not against the bar, and that is
        # the whole design.  A penalty on the bar was tried first and does nothing: at a bar of 6
        # the accepted steps mostly sit far above it, and the switches that cost are not
        # low-confidence ones -- they are steps where a second line briefly out-peaks the one in
        # use.  Against the current line's own mass, 1.5 takes the change rate from 0.121 to
        # 0.078 per accepted step while keeping the corrected readout's accuracy.
        #
        # Why it is wanted at all: dividing the bias out of the readout raises instantaneous
        # accuracy but took the change rate from 0.060 to 0.123 and deletions from 771 to 1066.
        # The bias inflates the mass of the line already in use, so it was supplying hysteresis
        # by accident.  This makes the stability deliberate rather than borrowed.
        self.switch_bar = float(switch_bar)
        self.line = -1
        self.confidence = 0.0
        self.observations = 0
        self.accepted = 0
        self.gated = 0
        self.held = 0

    def set_page(self) -> None:
        """Forget the previous page's estimate: it says nothing about this one."""

        self.line = -1
        self.confidence = 0.0
        self.observations = 0
        self.accepted = 0
        self.gated = 0
        self.held = 0

    def observe(
        self,
        line: int | None,
        confidence: float,
        *,
        held_mass: float | None = None,
        num_regions: int | None = None,
    ) -> None:
        """Take one step's estimate, keeping it only if it clears the bar.

        ``confidence`` is expected in the scale-free unit the caller has already applied. A rejected
        estimate clears the target rather than leaving the previous line in place -- see the module
        docstring.

        ``held_mass`` is what the line currently in use still holds in the same readout, which is
        what the switch margin compares against. A caller that does not supply it gets the
        recorded behaviour -- any accepted estimate wins -- because without the distribution the
        margin cannot be evaluated and inventing one would be worse than not applying it.
        """

        self.observations += 1
        if line is None or line < 0 or confidence < self.threshold:
            self.gated += 1
            self.line = -1
            self.confidence = 0.0
            return
        candidate = int(line)
        if (
            self.switch_bar > 1.0
            and self.line >= 0
            and candidate != self.line
            and held_mass is not None
            and num_regions
            and confidence / num_regions < held_mass * self.switch_bar
        ):
            # A different line, but not by enough to be worth moving.  Counted apart from a gated
            # step: the estimate cleared the bar and was overruled, which is a decision this
            # policy made rather than one the data forced.
            self.held += 1
            self.confidence = float(confidence)
            return
        self.accepted += 1
        self.line = candidate
        self.confidence = float(confidence)

    def report(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "confidence": self.confidence,
            "confidence_bar": self.threshold,
            "observations": self.observations,
            "accepted": self.accepted,
            "gated": self.gated,
            "held": self.held,
            "switch_bar": self.switch_bar,
            "accepted_fraction": (
                self.accepted / self.observations if self.observations else None
            ),
        }


def aggregate_line_distribution(
    rows: list[dict[str, Any]], num_regions: int, heads: tuple[int, ...] | None = None
) -> list[float] | None:
    """The averaged line distribution itself, for a policy that needs more than its argmax.

    Kept beside ``aggregate_line_estimate`` rather than replacing it: the estimate is what the
    confidence unit is defined on, and the distribution is only needed by a switch margin.
    """

    if not rows or num_regions <= 0:
        return None
    chosen = [row for row in rows if heads is None or row["head"] in heads]
    if not chosen:
        return None
    width = len(chosen[0].get("line_probs") or [])
    if width < num_regions:
        return None
    total = [0.0] * width
    for row in chosen:
        for index, value in enumerate(row["line_probs"]):
            total[index] += float(value)
    return [value / len(chosen) for value in total]


def aggregate_line_estimate(
    rows: list[dict[str, Any]], num_regions: int, heads: tuple[int, ...] | None = None
) -> tuple[int | None, float]:
    """One line estimate from the probed heads, in the scale-free confidence unit.

    Averages the heads' line distributions before taking the argmax, matching the aggregation stage
    1 registered: averaging the argmaxes would discard the spread that says how sure the step is,
    and choosing the most confident head per step would make the confidence a property of a
    selector.
    """

    if not rows or num_regions <= 0:
        return None, 0.0
    chosen = [row for row in rows if heads is None or row["head"] in heads]
    if not chosen:
        return None, 0.0
    width = len(chosen[0].get("line_probs") or [])
    if width < num_regions:
        return None, 0.0
    total = [0.0] * width
    for row in chosen:
        for index, value in enumerate(row["line_probs"]):
            total[index] += float(value)
    mean = [value / len(chosen) for value in total]
    # The last slot is everything off every line; it can win the argmax but is not a line.
    best = max(range(num_regions), key=lambda index: mean[index])
    return best, mean[best] * num_regions
