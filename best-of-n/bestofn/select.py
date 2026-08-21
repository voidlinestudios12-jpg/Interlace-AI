"""Selectors: choosing the final answer among N sampled trajectories.

Performance of a Best-of-N system decomposes into two quantities:

    coverage  = P(at least one of N trajectories is correct) = 1 - (1-p)^N
    selection = P(the selector actually returns it)

Coverage saturates logarithmically in N; the gap between the two does not close
on its own. That gap is what these selectors attack.

    random           uniformly random trajectory. **The baseline every other
                     selector must beat.** A selector below this is
                     anti-correlated with correctness and is doing harm.
    majority         modal answer. Free, but structurally cannot recover a
                     correct answer held by a minority of trajectories.
    self_certainty   weights each vote by exp(mean token log-probability).
                     Measures fluency, which correlates with -- but is not --
                     correctness. Frequently loses to ``random``; measure it.
    verifier         weights each vote by an external P(correct) score.
                     The only selector that can promote a minority answer.
    verifier_argmax  returns the single highest-scored trajectory.
    oracle           returns the correct answer if any trajectory found it.
                     Requires the gold answer, so it is a diagnostic ceiling
                     only -- never a deployable selector, and never a bound on
                     what a realisable selector can achieve.

Two invariants this module now guarantees, and previously did not:

* **One normalisation everywhere.** ``select`` returns the same canonical form
  that ``coverage`` compares against. Mixing raw and canonical forms inflates
  the measured selection gap, which is exactly the quantity being studied.
* **No silent degradation.** A verifier whose scores fall outside ``[0, 1]``
  raises rather than quietly reverting to majority voting.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
from __future__ import annotations

import math
import random as _random
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from .extract import equivalent, have_math_verify, normalise

__all__ = ["Sample", "select", "agreement", "coverage", "SELECTORS",
           "effective_n", "abstentions"]

SELECTORS = ("random", "majority", "self_certainty", "verifier",
             "verifier_argmax", "oracle")


class Sample:
    """One sampled reasoning trajectory.

    Args:
        answer: extracted answer string. ``""`` means the trajectory abstains.
        text: full trajectory text (optional; kept for inspection and replay).
        logprob: mean token log-probability. Used by ``self_certainty``.
        score: external P(correct) in ``[0, 1]``. Used by the verifier
            selectors. Reward models usually emit unbounded logits -- apply a
            sigmoid before passing them in.
        finish_reason: ``"stop"`` if the model ended on its own, ``"length"``
            if it hit the token limit. A truncated trajectory rarely contains
            its answer, and knowing which is which is the difference between a
            vote and a guess.
        n_tokens: generated token count, for compute accounting.
    """

    __slots__ = ("answer", "text", "logprob", "score", "finish_reason",
                 "n_tokens")

    def __init__(self, answer: str, text: str = "",
                 logprob: Optional[float] = None,
                 score: Optional[float] = None,
                 finish_reason: Optional[str] = None,
                 n_tokens: Optional[int] = None):
        self.answer = answer
        self.text = text
        self.logprob = logprob
        self.score = score
        self.finish_reason = finish_reason
        self.n_tokens = n_tokens

    @property
    def key(self) -> str:
        """Normalised answer used for grouping votes. ``""`` means abstain."""
        return normalise(self.answer)

    @property
    def truncated(self) -> bool:
        """Whether generation stopped because it ran out of tokens."""
        return self.finish_reason == "length"

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Sample(answer={self.answer!r}, logprob={self.logprob}, "
                f"score={self.score}, finish_reason={self.finish_reason!r})")


def _finite(x) -> bool:
    """True only for a real, usable number.

    Backends can emit NaN or -inf for degenerate sequences. Those must never
    reach a comparison: NaN compares false against everything, so a single one
    can silently decide the vote.
    """
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


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
                finish_reason=s.get("finish_reason"),
                n_tokens=s.get("n_tokens"),
            ))
        else:
            out.append(Sample(answer=str(s)))
    return out


def _voting(items: List[Sample]) -> List[Sample]:
    """The samples that actually cast a vote -- those with an answer."""
    return [s for s in items if s.key]


def _merge_map(keys: Sequence[str]) -> Dict[str, str]:
    """Map each distinct answer onto a representative of its equivalence class.

    Textually different answers are often the same answer: ``1/2``, ``0.5`` and
    ``\\frac{1}{2}`` are one result written three ways, and counting them as
    three votes splits a bloc that should have won together. When
    ``math-verify`` is installed they are merged here.

    Deterministic by construction: classes are formed over sorted keys and
    represented by the first member, so the same pool always votes the same way.
    """
    distinct = sorted(set(keys))
    rep = {k: k for k in distinct}
    if len(distinct) < 2 or not have_math_verify():
        return rep
    for i, k in enumerate(distinct):
        if rep[k] != k:                 # already folded into an earlier class
            continue
        for other in distinct[i + 1:]:
            if rep[other] == other and equivalent(k, other):
                rep[other] = k
    return rep


def _tally(samples: List[Sample], weight_fn) -> Dict[str, float]:
    """Accumulate weight per answer, skipping abstentions.

    Equivalent answers are pooled: see :func:`_merge_map`.
    """
    merge = _merge_map([s.key for s in samples if s.key])
    weights: Dict[str, float] = defaultdict(float)
    for s in samples:
        k = s.key
        if k:
            weights[merge.get(k, k)] += weight_fn(s)
    return weights


def _winner(weights: Dict[str, float], samples: List[Sample]) -> str:
    """Highest-weight answer, in canonical form.

    Returning the canonical key rather than the raw string matters: the caller
    compares this against a gold answer that is itself normalised, and mixing
    the two forms penalises the numerator while leaving the denominator alone.

    Ties break towards the answer that appears first, which keeps the function
    deterministic -- important for reproducible evaluation.
    """
    if not weights:
        return ""
    best = max(weights.values())
    tied = {k for k, v in weights.items() if v == best}
    merge = _merge_map([s.key for s in samples if s.key])
    for s in samples:               # first occurrence wins
        if merge.get(s.key, s.key) in tied:
            return merge.get(s.key, s.key)
    return sorted(tied)[0]


def _check_scores(items: List[Sample]) -> None:
    """Reject reward-model logits before they silently become a majority vote.

    Most open reward models emit unbounded logits, not probabilities. Weighting
    a vote by a negative number is meaningless, and the previous behaviour --
    falling back to majority voting without a word -- meant users believed they
    were running their verifier when they were not.
    """
    vals = [float(s.score) for s in items if _finite(s.score)]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    if lo < 0.0 or hi > 1.0:
        raise ValueError(
            f"verifier scores must be probabilities in [0, 1], got "
            f"[{lo:.4g}, {hi:.4g}]. Reward models usually return unbounded "
            f"logits: apply a sigmoid first, e.g. "
            f"score = 1 / (1 + math.exp(-logit)). Passing logits straight "
            f"through would silently reduce the verifier to a majority vote."
        )


def select(samples: Sequence, method: str = "majority",
           gold: Optional[str] = None, seed: Optional[int] = None) -> str:
    """Return the final answer chosen among ``samples``, in canonical form.

    Args:
        samples: sequence of :class:`Sample`, dicts, or answer strings.
        method: one of :data:`SELECTORS`.
        gold: reference answer. Only used by ``method="oracle"``.
        seed: seed for ``method="random"``, so evaluation stays reproducible.

    Returns:
        The selected answer, normalised, or ``""`` if nothing was selectable.

    Raises:
        ValueError: if ``method`` is unknown, ``oracle`` is used without
            ``gold``, a verifier method is used with no scores, or verifier
            scores fall outside ``[0, 1]``.
    """
    items = _as_samples(samples)
    if not items:
        return ""

    if method == "oracle":
        if gold is None:
            raise ValueError("method='oracle' requires gold=...")
        g = normalise(gold)
        if not g:
            return ""
        return next((s.key for s in items if s.key == g), "")

    if method == "random":
        pool = _voting(items)
        if not pool:
            return ""
        rng = _random.Random(seed)
        return rng.choice(pool).key

    if method == "verifier_argmax":
        scored = [s for s in items if _finite(s.score) and s.key]
        if not scored:
            raise ValueError(
                "method='verifier_argmax' requires a finite Sample.score on at "
                "least one sample with an extractable answer"
            )
        _check_scores(scored)
        return max(scored, key=lambda s: float(s.score)).key

    if method == "majority":
        weights = _tally(items, lambda s: 1.0)

    elif method == "self_certainty":
        if not any(_finite(s.logprob) for s in items):
            raise ValueError(
                "method='self_certainty' requires Sample.logprob on the inputs"
            )
        # exp(mean logprob) in (0, 1]. Missing or non-finite values contribute
        # nothing: a single NaN would otherwise make every comparison false and
        # silently randomise the result.
        weights = _tally(
            items,
            lambda s: math.exp(s.logprob) if _finite(s.logprob) else 0.0,
        )

    elif method == "verifier":
        if not any(_finite(s.score) for s in items):
            raise ValueError(
                "method='verifier' requires Sample.score on the inputs"
            )
        _check_scores(items)
        weights = _tally(
            items, lambda s: float(s.score) if _finite(s.score) else 0.0
        )

    else:
        raise ValueError(
            f"unknown method {method!r}; choose from {SELECTORS}"
        )

    if not weights:
        return ""

    if max(weights.values()) <= 0.0:
        # Every candidate scored exactly zero. Falling back to an unweighted
        # vote is the least surprising behaviour, but it must be audible: the
        # caller asked for one selector and is getting another.
        if method != "majority":
            warnings.warn(
                f"bestofn: every {method} weight was zero, so this call fell "
                f"back to a majority vote. The returned answer was NOT chosen "
                f"by your {method}.",
                RuntimeWarning, stacklevel=2,
            )
            return select(items, "majority")
        return ""

    return _winner(weights, items)


def agreement(samples: Sequence) -> float:
    """Fraction of *voting* samples agreeing with the modal answer, in [0, 1].

    Abstentions are excluded from both sides of the ratio. Counting them only
    in the denominator understates agreement precisely when extraction is
    failing -- which is when the number is most likely to be looked at.

    Useful for early stopping: high agreement after few samples means extra
    compute is unlikely to change the outcome.
    """
    items = _as_samples(samples)
    voting = _voting(items)
    if not voting:
        return 0.0
    counts = _tally(voting, lambda s: 1.0)
    if not counts:
        return 0.0
    return max(counts.values()) / len(voting)


def coverage(samples: Sequence, gold: str) -> bool:
    """Whether any sample reached ``gold`` -- the pass@N indicator.

    A gold answer that does not normalise to anything (empty, ``inf``, ``nan``)
    returns ``False`` rather than matching every abstention.

    This is a diagnostic ceiling, not a bound on realisable selectors: the
    problems that sustain its tail are single hits among N samples, which by
    definition are never the mode and are indistinguishable from noise without
    the label.
    """
    g = normalise(gold)
    if not g:
        return False
    items = _as_samples(samples)
    if any(s.key == g for s in items):
        return True
    if not have_math_verify():
        return False
    return any(equivalent(s.answer, gold) for s in items if s.key)


def abstentions(samples: Sequence) -> int:
    """How many trajectories produced no usable answer."""
    items = _as_samples(samples)
    return sum(1 for s in items if not s.key)


def effective_n(samples: Sequence) -> int:
    """How many trajectories actually voted.

    The N you paid for is ``len(samples)``. This is the N you got. When the
    two diverge, the compute budget is being spent on trajectories that
    contribute nothing, and every per-sample statistic should use this one.
    """
    return len(_voting(_as_samples(samples)))
