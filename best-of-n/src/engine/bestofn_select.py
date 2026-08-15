"""Selectors: choosing the final answer among N sampled trajectories.

Performance of a Best-of-N system decomposes into two quantities:

    coverage  = P(at least one of N trajectories is correct) = 1 - (1-p)^N
    selection = P(the selector actually returns it)

Coverage saturates logarithmically in N; the gap between the two does not close
on its own. That gap -- the *selection gap* -- is what these selectors attack.

    majority         modal answer. Free, but structurally cannot recover a
                     correct answer held by a minority of trajectories.
    self_certainty   weights each vote by exp(mean token log-probability).
                     Measures fluency, which correlates with -- but is not --
                     correctness.
    verifier         weights each vote by an external P(correct) score.
                     The only selector that can promote a minority answer.
    verifier_argmax  returns the single highest-scored trajectory. Included for
                     completeness; ``verifier`` measured 8.9 points better.
    oracle           returns the correct answer if any trajectory found it.
                     Requires the gold answer, so it is a diagnostic ceiling
                     only -- never a deployable selector.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from .extract import normalise

__all__ = ["Sample", "select", "agreement", "coverage", "SELECTORS"]

SELECTORS = ("majority", "self_certainty", "verifier", "verifier_argmax", "oracle")


class Sample:
    """One sampled reasoning trajectory.

    Args:
        answer: extracted answer string.
        text: full trajectory text (optional; kept for inspection).
        logprob: mean token log-probability. Used by ``self_certainty``.
        score: external P(correct) in [0, 1]. Used by the verifier selectors.
    """

    __slots__ = ("answer", "text", "logprob", "score")

    def __init__(self, answer: str, text: str = "",
                 logprob: Optional[float] = None,
                 score: Optional[float] = None):
        self.answer = answer
        self.text = text
        self.logprob = logprob
        self.score = score

    @property
    def key(self) -> str:
        """Normalised answer used for grouping votes."""
        return normalise(self.answer)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Sample(answer={self.answer!r}, logprob={self.logprob}, "
                f"score={self.score})")


def _as_samples(samples: Sequence) -> List[Sample]:
    """Accept Sample objects, dicts, or bare answer strings."""
    out = []
    for s in samples:
        if isinstance(s, Sample):
            out.append(s)
        elif isinstance(s, dict):
            out.append(Sample(
                answer=s.get("answer", s.get("respuesta", "")),
                text=s.get("text", s.get("texto", "")),
                logprob=s.get("logprob", s.get("certeza")),
                score=s.get("score", s.get("prm_score")),
            ))
        else:
            out.append(Sample(answer=str(s)))
    return out


def _tally(samples: List[Sample], weight_fn) -> Dict[str, float]:
    """Accumulate weight per normalised answer, skipping unparseable ones."""
    weights: Dict[str, float] = defaultdict(float)
    for s in samples:
        k = s.key
        if k:
            weights[k] += weight_fn(s)
    return weights


def _winner(weights: Dict[str, float], samples: List[Sample]) -> str:
    """Highest-weight answer, returned in its original readable form.

    Ties break towards the answer that appears first, which keeps the function
    deterministic -- important for reproducible evaluation.
    """
    if not weights:
        return ""
    best = max(weights.values())
    tied = {k for k, v in weights.items() if v == best}
    for s in samples:               # first occurrence wins
        if s.key in tied:
            return s.answer
    return next(iter(tied))


def select(samples: Sequence, method: str = "majority",
           gold: Optional[str] = None) -> str:
    """Return the final answer chosen among ``samples``.

    Args:
        samples: sequence of :class:`Sample`, dicts, or answer strings.
        method: one of :data:`SELECTORS`.
        gold: reference answer. Only used by ``method="oracle"``.

    Returns:
        The selected answer as a string, or ``""`` if nothing was selectable.

    Raises:
        ValueError: if ``method`` is unknown, or ``oracle`` is used without
            ``gold``, or a verifier method is used with no scores present.
    """
    items = _as_samples(samples)
    if not items:
        return ""

    if method == "oracle":
        if gold is None:
            raise ValueError("method='oracle' requires gold=...")
        g = normalise(gold)
        return next((s.answer for s in items if s.key == g), "")

    if method == "verifier_argmax":
        scored = [s for s in items if s.score is not None and s.key]
        if not scored:
            raise ValueError(
                "method='verifier_argmax' requires Sample.score on the inputs"
            )
        return max(scored, key=lambda s: s.score).answer

    if method == "majority":
        weights = _tally(items, lambda s: 1.0)

    elif method == "self_certainty":
        if all(s.logprob is None for s in items):
            raise ValueError(
                "method='self_certainty' requires Sample.logprob on the inputs"
            )
        # exp(mean logprob) in (0, 1]; missing values contribute nothing.
        weights = _tally(
            items,
            lambda s: math.exp(s.logprob) if s.logprob is not None else 0.0,
        )

    elif method == "verifier":
        if all(s.score is None for s in items):
            raise ValueError(
                "method='verifier' requires Sample.score on the inputs"
            )
        weights = _tally(
            items, lambda s: float(s.score) if s.score is not None else 0.0
        )

    else:
        raise ValueError(
            f"unknown method {method!r}; choose from {SELECTORS}"
        )

    # Degenerate weighting (all zero) would make the result arbitrary;
    # fall back to the unweighted vote instead of returning noise.
    if not weights or max(weights.values()) <= 0.0:
        if method != "majority":
            return select(items, "majority")
        return ""

    return _winner(weights, items)


def agreement(samples: Sequence) -> float:
    """Fraction of samples agreeing with the modal answer, in ``[0, 1]``.

    Useful for early stopping: high agreement after few samples means extra
    compute is unlikely to change the outcome.
    """
    items = _as_samples(samples)
    if not items:
        return 0.0
    counts = _tally(items, lambda s: 1.0)
    if not counts:
        return 0.0
    return max(counts.values()) / len(items)


def coverage(samples: Sequence, gold: str) -> bool:
    """Whether any sample reached ``gold`` -- the pass@N indicator.

    This is the ceiling of every selector at this sample budget. Comparing it
    against a deployed selector measures the remaining selection gap.
    """
    g = normalise(gold)
    return any(s.key == g for s in _as_samples(samples))
