"""Bounded DISC controller (Ye et al., 2025, Algorithm 1).

The weight of a candidate is the legal model mass of every step: branching
steps, forced tokens and the unwritten rest of an early-resolved phrase. Forced
tokens skip their own forward but are still scored from the next one.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class DiscDecodingError(RuntimeError):
    """A bounded decoding pass did not reach a valid terminal state."""


@dataclass
class Candidate(Generic[T]):
    value: T
    log_weight: float
    details: dict[str, object]


def select_disc(
    sample: Callable[[], Candidate[T]],
    budget: int = 4,
    rng: random.Random | None = None,
) -> tuple[T, dict[str, object]]:
    if not 1 <= budget <= 8:
        raise ValueError("DISC budget must be between 1 and 8")
    rng = rng or random.Random()
    attempts: list[dict[str, object]] = []

    def draw(phase: str) -> Candidate[T] | None:
        try:
            candidate = sample()
            if not math.isfinite(candidate.log_weight) or candidate.log_weight > 1e-6:
                raise DiscDecodingError("Invalid DISC weight")
        except DiscDecodingError as error:
            attempts.append({"phase": phase, "status": "invalid", "reason": str(error)})
            return None
        attempts.append({
            **candidate.details, "phase": phase, "status": "sampled",
            "log_weight": candidate.log_weight,
            "weight": math.exp(candidate.log_weight),
        })
        return candidate

    def result(candidate: Candidate[T], index: int, mode: str):
        attempts[index]["status"] = "selected"
        return candidate.value, {
            "enabled": True, "budget": budget, "mode": mode,
            "selected_attempt": index + 1, "attempt_count": len(attempts),
            "log_weight": candidate.log_weight,
            "weight": math.exp(candidate.log_weight), "attempts": attempts,
            "weight_scope": "ambiguous_decisions",
        }

    for _ in range(budget):
        candidate = draw("acceptance")
        if candidate is not None:
            if math.log(max(rng.random(), 1e-300)) < candidate.log_weight:
                return result(candidate, len(attempts) - 1, "accepted")
            attempts[-1]["status"] = "rejected"

    # Algorithm 1 uses a fresh independent pool, not the rejected candidates.
    pool: list[tuple[int, Candidate[T]]] = []
    for _ in range(budget):
        candidate = draw("fallback")
        if candidate is not None:
            pool.append((len(attempts) - 1, candidate))
    if not pool:
        raise DiscDecodingError("DISC: all bounded passes failed to reach a terminal state")
    peak = max(candidate.log_weight for _, candidate in pool)
    weights = [math.exp(candidate.log_weight - peak) for _, candidate in pool]
    index, candidate = rng.choices(pool, weights=weights, k=1)[0]
    return result(candidate, index, "weighted_fallback")
