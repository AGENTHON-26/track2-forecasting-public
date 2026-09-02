"""Reference-scale normalization: a rankability invariant, not a convenience (T2-7).

## Executive summary (read this first)

The Track 2 composite is `0.5*marginal + 0.3*joint + 0.2*tail`, and the three components live on
different natural scales — a CPI-index CRPS and a 30-year-yield CRPS differ by orders of
magnitude. Without normalization the mean over the roster is a weighted average dominated by
whichever units happen to carry the largest numbers, so a participant's rank depends on which
instruments the organizers picked. `ref_scale.json` divides each component by the official M0
baseline's value for that unit, which puts every unit on one scale where **the baseline is 1.0 by
construction**. That is what makes `W = 4.0` mean something ("four times worse than a text-blind
random walk") and what makes clipping at 4.0 a real bound rather than an arbitrary one.

Three faults are closed here, all armed and not yet live:

* `hydrate_ctx` built the scale from **whichever keys were present**, and `crps.crps_composite`
  then indexed `ref_scale["marginal"]` unconditionally whenever the dict was truthy — so
  `{"tail": 1.0}` raised an uncaught `KeyError` out of the scorer.
* `hydrate_ctx` ended with `ctx.setdefault("ref_scale", None)`, so a **missing scale file silently
  produced a raw composite**, and the driver then averaged raw and normalized units together with
  nothing refusing the mix.
* "complete" meant all three components positive, but the joint component does not exist on a
  1-cell variogram grid, so the correct scale (`joint: 0.0`) was unloadable — 60 of 104 public
  cards. Latent only because the generator writes a placeholder `1.0` there; it arms the moment
  the scales are regenerated honestly. `load_ref_scale` now takes the grid shape and the joint
  statistic. See `_OPTIONAL_COMPONENT`.

All three are now impossible by construction: `load_ref_scale` returns a complete scale or raises,
and `NormalizationMode` has no third value that means "whatever we found on disk".

### The firewall note that matters more than the arithmetic

`ref_scale.json` is **answer-equivalent**. It is derived from the sealed realized outcome — it is
the baseline's error against that outcome — so given the baseline's forecast it inverts to the
target. It looks innocuous (three floats, no dates, no identifiers) and it is *not* the answer
file, which is precisely why a tool classifying unit files by name will ship it as configuration.
C6 has `answer_equivalent: bool` for exactly this artifact. `assert_reference_only()` below is the
scorer-side restatement: the loader refuses to read a scale out of anything but the reference root.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .failures import organizer_fault
from .limits import DEFAULT_LIMITS, ParseLimits, read_json_bounded

__all__ = [
    "REF_SCALE_ALWAYS_REQUIRED",
    "REF_SCALE_COMPONENTS",
    "REF_SCALE_FILENAME",
    "REF_SCALE_PROVENANCE_KEYS",
    "NormalizationMode",
    "RefScale",
    "assert_reference_only",
    "load_ref_scale",
]

#: Every component the composite weights.
REF_SCALE_COMPONENTS: tuple[str, ...] = ("marginal", "joint", "tail")

#: The only component that can be structurally absent. The VARIOGRAM is a between-cells statistic,
#: so on a 1-cell grid it is 0 by construction, making `0.0` the correct scale — which the
#: positivity rule below refuses. That made the correct scale unloadable for 60 of 104 public
#: cards, and a comparable share on each sealed split, and `load_ref_scale` raises outside the
#: participant try/except, aborting the whole evaluation.
#:
#: This is a property of the STATISTIC, not of the grid: `energy_score` on a 1-cell grid equals the
#: marginal CRPS, not 0. `load_ref_scale` gates on both.
_OPTIONAL_COMPONENT = "joint"

#: Required on every grid shape. Derived, so it cannot drift out of `REF_SCALE_COMPONENTS`.
REF_SCALE_ALWAYS_REQUIRED: tuple[str, ...] = tuple(
    c for c in REF_SCALE_COMPONENTS if c != _OPTIONAL_COMPONENT
)

#: Stand-in for the `joint` slot when the component does not exist, needed because
#: `crps_composite` indexes `ref_scale["joint"]` unconditionally — unlike `"tail"`, which it guards
#: with a membership test. It never reaches a score because `scoring.py` assigns a LITERAL `0.0`
#: joint weight on a 1-cell grid, and `0.0 * anything finite` is `0.0`. Note the numerator is not
#: necessarily zero — see `_OPTIONAL_COMPONENT` — so the weight is the whole of the safety, and
#: `as_mapping` refuses to emit this when the weight is live.
_JOINT_PLACEHOLDER: float = 1.0

#: Keys the generator writes for provenance and the metric never reads. Named as a CLOSED set
#: rather than tolerated by a wildcard: every scale file the generator has written carries all
#: three, so refusing them outright would fail every legitimate unit — and a gate that
#: rejects the legitimate case makes every rejection beside it uninterpretable. Anything outside
#: the union of these and `REF_SCALE_COMPONENTS` is still refused.
REF_SCALE_PROVENANCE_KEYS: tuple[str, ...] = ("method", "seed", "generated")

REF_SCALE_FILENAME = "ref_scale.json"


class NormalizationMode(StrEnum):
    """How a unit's composite was produced. There is no `auto` and no `whatever_was_on_disk`.

    `RAW_UNRANKABLE` exists for the participant smoke path, where no evaluation plan and no sealed
    reference exist and a raw composite is the only thing computable. It is named for what it
    costs: a raw score can be *displayed*, and it can never enter a ranked aggregate. The
    aggregator refuses a mixed set, so the name is load-bearing rather than decorative.
    """

    REF_SCALE = "ref_scale"
    RAW_UNRANKABLE = "raw_unrankable"


@dataclass(frozen=True, slots=True)
class RefScale:
    """A finite, positive normalization scale for every component the grid HAS.

    `joint` is `None` exactly when the component does not exist — see `_OPTIONAL_COMPONENT`.

    Constructing one validates every component that is present. It does NOT validate that an
    absent `joint` is legitimate: that pairs the scale with the composite's joint weight, which is
    card material and not in scope here. `as_mapping` settles it, as an `OrganizerFault` that
    `official.py` does not catch and that therefore aborts the evaluation rather than being charged
    to a participant.
    """

    marginal: float
    joint: float | None
    tail: float

    def __post_init__(self) -> None:
        for name in REF_SCALE_COMPONENTS:
            value = getattr(self, name)
            if value is None and name == _OPTIONAL_COMPONENT:
                continue
            if not isinstance(value, float):  # pragma: no cover - constructor coerces
                raise organizer_fault(f"ref_scale.{name} must be a float")
            if value != value or value in (float("inf"), float("-inf")):
                raise organizer_fault(
                    f"ref_scale.{name} is non-finite. A non-finite intermediate statistic is an "
                    "organizer failure, and a scale that is not a number cannot normalize anything."
                )
            if value <= 0.0:
                raise organizer_fault(
                    f"ref_scale.{name}={value} is not positive. Dividing by zero or by a negative "
                    "baseline inverts the direction of the metric, which would make a worse "
                    "forecast rank better."
                )

    def as_mapping(self, *, joint_weight: float) -> dict[str, float]:
        """The `ref_scale` argument `crps.crps_composite` expects: all three keys, always.

        `joint_weight` is the composite's live weight on the joint term. A `None` joint is only
        representable when that weight is zero; asking for it otherwise is an organizer fault
        rather than a silent placeholder.
        """
        if self.joint is None:
            if joint_weight != 0.0:
                raise organizer_fault(
                    "ref_scale.joint does not exist for this unit but the composite weights the "
                    f"joint term at {joint_weight}. A scale that normalizes some of the sum and "
                    "not the rest is not a defined metric."
                )
            joint = _JOINT_PLACEHOLDER
        else:
            joint = self.joint
        return {"marginal": self.marginal, "joint": joint, "tail": self.tail}


def assert_reference_only(path: pathlib.Path, reference_root: pathlib.Path) -> None:
    """Refuse to load a scale from anywhere but the organizer's reference root.

    `ref_scale.json` inverts to the sealed target (C6 `answer_equivalent`). A scorer that would
    read it out of the participant's own output directory, or out of the mounted unit tree, is one
    misconfigured mount away from letting a submission supply its own denominator — which sets the
    composite to whatever the participant chooses.
    """
    resolved = path.resolve()
    root = reference_root.resolve()
    if not resolved.is_relative_to(root):
        raise organizer_fault(
            "refusing to load ref_scale.json from outside the reference root: the file is "
            "answer-equivalent (C6 answer_equivalent=true) and a participant-reachable copy "
            "would let the submission choose its own normalization denominator"
        )


def load_ref_scale(
    reference_root: pathlib.Path,
    *,
    cell_count: int | None = None,
    joint_statistic: str = "variogram",
    limits: ParseLimits = DEFAULT_LIMITS,
) -> RefScale:
    """Load and validate the unit's frozen scale. Any defect is an organizer fault.

    There is no `None` return and no partial dict. The pre-freeze loader built the scale from the
    keys it happened to find; this one requires every component the grid HAS, refuses an unknown
    key, and refuses a non-positive or non-finite value. Every one of those refusals is an
    `OrganizerFault`, because a scale is organizer material and a participant cannot cause, detect
    or repair a missing one.

    The joint component is dropped — loaded as `None` — only on a 1-cell VARIOGRAM grid, where it
    does not exist. It is dropped whatever the file says, including the `1.0` the generator writes
    today: carrying that through would make `joint is None` mean "the file was honest" rather than
    "the grid has no joint", and `as_mapping`'s guard would then cover only the honest minority.
    A non-numeric joint is still refused rather than silently dropped, and the emitted mapping is
    unchanged either way, because the placeholder it substitutes is that same `1.0`.

    Any other shape keeps the strict rule, including `cell_count=None` (the caller did not say) and
    any `joint_statistic` other than the variogram — `energy_score` on a 1-cell grid equals the
    marginal CRPS, so its scale genuinely exists and must be positive.
    """
    path = reference_root / REF_SCALE_FILENAME
    assert_reference_only(path, reference_root)
    if not path.is_file():
        raise organizer_fault(
            "this unit is rankable under normalization mode 'ref_scale' and carries no "
            f"reference/{REF_SCALE_FILENAME}. A rankable unit without a complete scale is an "
            "organizer failure, not a fallback to raw components: raw and normalized composites "
            "are not comparable and averaging them produces a leaderboard nobody can interpret."
        )
    try:
        raw: Mapping[str, Any] = read_json_bounded(
            path, what=REF_SCALE_FILENAME, max_bytes=limits.max_meta_bytes
        )
    except Exception as exc:  # noqa: BLE001 - a participant refusal here is a category error
        raise organizer_fault(
            f"reference/{REF_SCALE_FILENAME} is unreadable: {type(exc).__name__}: {exc}"
        ) from None

    allowed = set(REF_SCALE_COMPONENTS) | set(REF_SCALE_PROVENANCE_KEYS)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise organizer_fault(
            f"reference/{REF_SCALE_FILENAME} carries unknown key(s) {unknown}; the scale is "
            f"{list(REF_SCALE_COMPONENTS)} plus the provenance keys "
            f"{list(REF_SCALE_PROVENANCE_KEYS)}, and nothing else"
        )
    single_cell = cell_count == 1 and joint_statistic == "variogram"
    required = REF_SCALE_ALWAYS_REQUIRED if single_cell else REF_SCALE_COMPONENTS
    absent = [k for k in required if k not in raw]
    nulled = [k for k in required if k in raw and raw[k] is None]
    if absent or nulled:
        raise organizer_fault(
            f"reference/{REF_SCALE_FILENAME} is missing {absent} and null in {nulled}. The "
            "composite weights every component the grid has, so a partial scale normalizes some "
            "of the sum and not the rest — which is how an uncaught KeyError reached the scorer "
            "before the freeze."
        )

    def _number(key: str) -> float:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise organizer_fault(
                f"reference/{REF_SCALE_FILENAME}.{key} must be a number, got {type(value).__name__}"
            )
        return float(value)

    joint: float | None
    if single_cell:
        # Whatever the file says about a component that does not exist is not a scale, so it is
        # dropped -- but it is still validated first. A negative or non-finite joint is a corrupt
        # file whatever the grid, and dropping it silently would lose that detector on 58% of the
        # roster. Only `0.0` (honest), absent, and a positive placeholder are tolerated.
        if raw.get("joint") is not None:
            probe = _number("joint")
            if not (probe >= 0.0) or probe == float("inf"):
                raise organizer_fault(
                    f"reference/{REF_SCALE_FILENAME}.joint={probe} is not a scale. The component "
                    "does not exist on a 1-cell variogram grid, so absent, `0.0` or the "
                    "generator's positive placeholder are all expected -- a negative or "
                    "non-finite value is a corrupt file, not a grid property."
                )
        joint = None
    else:
        joint = _number("joint")
    return RefScale(marginal=_number("marginal"), joint=joint, tail=_number("tail"))
