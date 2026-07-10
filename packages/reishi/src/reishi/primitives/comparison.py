"""Comparison: one raw A-vs-B judgment between two trials.

Pairwise measurement (LLM-as-judge, Elo, Bradley-Terry, human preference) is
a property of a pair -- or the whole population -- not of one trial, and it
is non-stationary: it changes as new comparisons arrive. Storing a rating on
a Trial manifest would go stale the instant another comparison lands, so the
raw judgment is its own record kind and ratings are computed at the Board
layer instead (see docs/design/claims-under-review.md section C and
reishi.primitives.ratings).
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NotRequired, TypedDict

from reishi import store
from reishi.primitives.trial import ScoringInfo

WINNERS = ("a", "b", "tie")


class ComparisonManifest(TypedDict):
    id: str
    created: str
    trial_a: str
    trial_b: str
    winner: str  # one of WINNERS
    judge: str  # model name that judged, or "human"
    scoring: NotRequired[ScoringInfo]


@dataclass
class Comparison:
    id: str
    trial_a: str
    trial_b: str
    winner: str
    judge: str
    created: str = ""
    # Measurement-key pinning for the judgment itself (task/dataset/split --
    # same K fields as Trial.scoring), so a board can tell whether comparisons
    # being rated together were judged under the same conditions.
    scoring: ScoringInfo = field(default_factory=dict)  # type: ignore[assignment]
    # Unknown top-level manifest keys, carried verbatim -- same losslessness
    # guarantee as Trial.extra (see trial.py).
    extra: dict = field(default_factory=dict)

    def to_manifest(self) -> ComparisonManifest:
        m = {
            "id": self.id,
            "created": self.created,
            "trial_a": self.trial_a,
            "trial_b": self.trial_b,
            "winner": self.winner,
            "judge": self.judge,
            "scoring": self.scoring,
        }
        # Known keys win, so a stale carried-over value never shadows the live one.
        return {**self.extra, **m}  # type: ignore[typeddict-item]

    @classmethod
    def from_manifest(cls, m: Mapping[str, object]) -> "Comparison":
        # Tolerant reader, routed off `known` (which excludes `extra` itself)
        # on both sides -- mirrors Trial.from_manifest so a manifest key
        # literally named "extra" round-trips instead of being swallowed.
        known = {k for k in cls.__dataclass_fields__ if k != "extra"}
        fields: dict = {}
        extra: dict = {}
        for k, v in m.items():
            (fields if k in known else extra)[k] = v
        return cls(**fields, extra=extra)  # type: ignore[arg-type]


def record(
    trial_a: str,
    trial_b: str,
    winner: str,
    judge: str,
    scoring: ScoringInfo | None = None,
) -> Comparison:
    if winner not in WINNERS:
        raise ValueError(f"winner must be one of {WINNERS}, got {winner!r}")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return Comparison(
        id=f"cmp-{uuid.uuid4().hex[:8]}",
        trial_a=trial_a,
        trial_b=trial_b,
        winner=winner,
        judge=judge,
        created=now,
        scoring=scoring or {},
    )


def save(c: Comparison) -> None:
    store.save("comparisons", c.id, c.to_manifest())


def load(comparison_id: str) -> Comparison:
    return Comparison.from_manifest(store.load("comparisons", comparison_id))


def load_all() -> list[Comparison]:
    return [Comparison.from_manifest(m) for m in store.load_all("comparisons")]


def resolve(ref: str) -> Comparison:
    matches = [c for c in load_all() if c.id == ref or c.id.startswith(ref)]
    if not matches:
        raise FileNotFoundError(f"no comparison matching '{ref}'")
    if len(matches) > 1:
        ids = ", ".join(c.id for c in matches)
        raise ValueError(f"'{ref}' is ambiguous: {ids}")
    return matches[0]
