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
                     correctness. On our GSM8K run it scores 66.5%, level with
                     majority; on your task it may do better or worse, so
                     print ``random`` beside it and see.
    verifier         weights each vote by an external P(correct) score.
                     The only selector that can promote a minority answer.
    verifier_argmax  returns the single highest-scored trajectory.
    oracle           returns the correct answer if any trajectory found it.
                     Requires the gold answer, so it is a diagnostic ceiling
                     only -- never a deployable selector, and never a bound on
                     what a realisable selector can achieve.

Three invariants this module now guarantees, and previously did not:

* **Selection does not depend on the order of the pool.** Shuffle the samples
  and every selector except ``random`` -- which is order-dependent by
  definition -- returns the same answer. This needs three separate things to
  be true: the equivalence partition is a transitive closure rather than a
  greedy grouping (see :func:`_merge_map`), weights are summed with
  ``math.fsum`` over a sorted list rather than accumulated in arrival order,
  and every tie breaks on the canonical key rather than on which trajectory
  came first.

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
           "effective_n", "abstentions", "merge_cap_hits"]

#: How many times :func:`_merge_map` has fallen back to exact matching because
#: the pool held more distinct answers than ``_MERGE_LIMIT``.
#:
#: Counted here rather than by intercepting the warning. Python's default
#: filter shows a given warning once per location, so counting warnings counts
#: *distinct messages*, not calls -- and because the message embeds the
#: distinct-answer count, near-identical pools collapsed into one. The
#: published summary reported 28 where the true figure was 2,040.
_MERGE_CAP_HITS = [0]


def merge_cap_hits(reset: bool = False) -> int:
    """Number of merge-cap fallbacks so far; pass ``reset`` to zero it.

    Exposed so a benchmark can report how often symbolic merging was skipped,
    which is the only honest way to say whether the cap cost anything.
    """
    n = _MERGE_CAP_HITS[0]
    if reset:
        _MERGE_CAP_HITS[0] = 0
    return n

#: Above this many distinct answers, fall back to exact matching. Grouping is
#: quadratic in distinct answers and each comparison can invoke a symbolic
#: parser, so the cap exists to stop a pathological pool stalling the caller.
#:
#: It was 24 through 1.1.1, chosen when equivalent() was uncached. It is
#: memoised now, so the same budget buys far more: raising the cap to 64 costs
#: no measurable time on a 200-problem GSM8K sweep at N=128 and silences the
#: warning on the 37%% of pools that sat between the two. Measured on that
#: sweep, 24, 64 and 160 all return 66.5%% -- the cap was not costing accuracy
#: on this data. That is a property of these pools, not a general guarantee,
#: which is why analyse.py reports how often the cap fires.
_MERGE_LIMIT = 64

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


def _safe_exp(x) -> float:
    """``exp`` that saturates instead of raising.

    A log-probability should never be positive, but a mis-signed backend would
    otherwise take down the whole call with ``OverflowError`` -- an exception
    this function does not document and the caller cannot anticipate.
    """
    try:
        return math.exp(float(x))
    except OverflowError:
        return math.inf


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
    ``\frac{1}{2}`` are one result written three ways, and counting them as
    three votes splits a bloc that should have won together. When
    ``math-verify`` is installed they are merged here.

    The representative is the **most frequent** member of the class, breaking
    ties towards the one seen first. Under an exact tie the label therefore
    depends on input order -- harmlessly, since the members are equivalent by
    construction and :meth:`Result.is_correct` scores them identically.

    **The partition itself does not depend on order.** That is not free.
    ``equivalent`` is not transitive: ``math-verify`` accepts ``0.3333333333``
    against ``1/3`` and ``1/3`` against ``0.33333333333333``, but rejects the
    two decimals against each other. Greedy grouping -- comparing each new
    answer against one representative per class and stopping at the first hit
    -- therefore returns a different partition depending on which answer the
    model happened to emit first, so the same pool of trajectories in a
    different order could return a different winner. We instead take the
    **transitive closure** over all pairs with a union-find, which is unique
    and independent of input order by construction. It costs the full
    quadratic comparison that ``_MERGE_LIMIT`` already budgets for.

    Choosing the representative alphabetically instead -- as
    an earlier version did -- meant a pool of ``["2*3", "6", "6"]`` returned the
    unevaluated ``2*3``, and a class containing the gold answer could be
    represented by something that did not match it. The winner has to be a
    string the caller can compare against their reference.
    """
    order: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for k in keys:
        if k not in order:
            order[k] = len(order)
        counts[k] = counts.get(k, 0) + 1

    distinct = sorted(order, key=lambda k: order[k])
    rep = {k: k for k in distinct}
    if len(distinct) < 2 or not have_math_verify():
        return rep
    if len(distinct) > _MERGE_LIMIT:
        _MERGE_CAP_HITS[0] += 1
        # Merging is quadratic in distinct answers and each comparison invokes
        # a symbolic parser. Past this point the cost is worse than the benefit,
        # so fall back to exact matching rather than stall the caller.
        warnings.warn(
            f"bestofn: {len(distinct)} distinct answers exceeds the symbolic "
            f"merge limit of {_MERGE_LIMIT}; comparing them textually instead. "
            f"Equivalent answers written differently will vote separately.",
            RuntimeWarning, stacklevel=3,
        )
        return rep

    # Union-find over every pair. Comparing against a single representative
    # per class would make the result order-dependent (see the docstring);
    # taking the transitive closure does not.
    parent = {k: k for k in distinct}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    for i, a in enumerate(distinct):
        for b in distinct[i + 1:]:
            if find(a) != find(b) and equivalent(a, b):
                parent[find(b)] = find(a)

    grouped: Dict[str, List[str]] = {}
    for k in distinct:                     # distinct is in first-seen order,
        grouped.setdefault(find(k), []).append(k)   # so members are too

    for members in grouped.values():
        # Most frequent member, ties to the lexicographically smallest.
        # Not "-order[m]": first-seen is order-dependent, and the whole point
        # of the union-find above is that this partition is not.
        best = min(members, key=lambda m: (-counts[m], m))
        for m in members:
            rep[m] = best
    return rep


def _tally(samples: List[Sample], weight_fn,
           merge: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Accumulate weight per answer, skipping abstentions.

    Equivalent answers are pooled: see :func:`_merge_map`. The caller passes
    ``merge`` in so the quadratic symbolic comparison happens once per
    ``select`` call rather than once per helper that needs it.
    """
    if merge is None:
        merge = _merge_map([s.key for s in samples if s.key])

    # Collect first, sum second. Accumulating with ``+=`` in list order makes
    # the total depend on the order the samples arrived in, because float
    # addition is not associative: two permutations of the same weights can
    # differ in the last bit, which is enough to flip a near-tie in
    # ``self_certainty`` or ``verifier``. math.fsum over a sorted list is
    # exactly rounded and gives the same total for every permutation.
    #
    # ``majority`` is unaffected either way -- it sums 1.0 -- but it costs
    # nothing to make the guarantee hold for all of them rather than for the
    # default only.
    grouped: Dict[str, List[float]] = defaultdict(list)
    for s in samples:
        k = s.key
        if k:
            grouped[merge.get(k, k)].append(weight_fn(s))
    return {k: math.fsum(sorted(v)) for k, v in grouped.items()}


def _winner(weights: Dict[str, float], samples: List[Sample],
            merge: Optional[Dict[str, str]] = None) -> str:
    """Highest-weight answer, in canonical form.

    Returning the canonical key rather than the raw string matters: the caller
    compares this against a gold answer that is itself normalised, and mixing
    the two forms penalises the numerator while leaving the denominator alone.

    An exact tie is broken towards the lexicographically smallest canonical
    key. That is arbitrary, as any tie-break must be -- when two answers carry
    identical weight there is nothing in the data preferring one. What matters
    is that it does not depend on input order.

    Breaking towards "whichever the model emitted first" is also deterministic
    for a fixed pool, and that is what this did until 1.1.2, but it means the
    same trajectories shuffled can return a different answer. It showed up as
    a non-zero spread at exhausted N in the published curve, where every draw
    is a permutation of one pool and the spread has to be zero. Together with
    the order-independent partition in :func:`_merge_map`, selection is now
    invariant under permutation of the pool.
    """
    if not weights:
        return ""
    best = max(weights.values())
    return min(k for k, v in weights.items() if v == best)


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
        if any(s.key == g for s in items):
            return g
        # Same equivalence rule coverage() uses. Having the two disagree meant
        # covered() said yes while the oracle selector said no.
        #
        # ``min``, not ``next``: several distinct canonical keys can each be
        # equivalent to the gold answer -- a pool holding both ``2/4`` and
        # ``0.5`` against a gold of ``\frac{1}{2}`` -- and taking the first one
        # encountered made the result depend on the order of the pool. Every
        # other selector here is permutation-invariant; this one was the
        # exception, and the exception was undocumented.
        return min((s.key for s in items
                    if s.key and equivalent(s.answer, gold)), default="")

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
        # Validate every score, not only the ones attached to a usable answer:
        # otherwise the same pool raises under 'verifier' and passes here.
        _check_scores(items)
        # 'verifier' warns when every score is zero, because the weighted vote
        # it would run is indistinguishable from an unweighted one. argmax has
        # the same problem and used to say nothing: it returned a trajectory
        # picked by tie-break, with no signal behind it at all.
        if all(float(s.score) == 0.0 for s in scored):
            warnings.warn(
                "bestofn: every verifier score is zero, so 'verifier_argmax' "
                "is choosing on the tie-break alone and the verifier is "
                "contributing nothing. Check that it is wired up and that its "
                "outputs are probabilities, not logits.",
                RuntimeWarning, stacklevel=3,
            )
        # min on (-score, key), not max on score: max keeps the first
        # maximal element, so an exact tie between two trajectories was
        # resolved by whichever the model happened to emit first. Breaking on
        # the canonical key instead makes this independent of pool order, the
        # same way _winner does for the voting selectors.
        # Through the merge map, like every other selector. Returning the
        # raw key meant the same pool could answer "1/2" under argmax and
        # "0.5" under verifier, and the caller compares against one gold.
        argmax_merge = _merge_map([s.key for s in items if s.key])
        best = min(scored, key=lambda s: (-float(s.score), s.key))
        return argmax_merge.get(best.key, best.key)

    # Computed once and threaded through: it is quadratic in distinct answers
    # and each comparison runs a symbolic parser.
    merge = _merge_map([s.key for s in items if s.key])

    if method == "majority":
        weights = _tally(items, lambda s: 1.0, merge)

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
            lambda s: _safe_exp(s.logprob) if _finite(s.logprob) else 0.0,
            merge,
        )

    elif method == "verifier":
        if not any(_finite(s.score) for s in items):
            raise ValueError(
                "method='verifier' requires Sample.score on the inputs"
            )
        _check_scores(items)
        weights = _tally(
            items, lambda s: float(s.score) if _finite(s.score) else 0.0, merge
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

    return _winner(weights, items, merge)


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
