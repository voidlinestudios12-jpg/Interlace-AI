"""Tests for the parts a model is not needed to exercise.

Run: python tests/test_engine_and_verifiers.py

The 1.1.0 audit found that `_merge_map`, the engine and the verifier adapters
had no coverage at all -- which is exactly where its three worst findings were.
This file closes that gap. No GPU, no model download, no network.
"""
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bestofn                                                  # noqa: E402
from bestofn import BestOfN, Result, Sample, select             # noqa: E402
from bestofn.extract import equivalent, have_math_verify        # noqa: E402
from bestofn.select import _merge_map, _safe_exp                # noqa: E402
from bestofn.verifiers import (KNOWN_VERIFIERS, RewardModelVerifier,  # noqa: E402
                               from_callable, from_hub, license_of,
                               sigmoid)

passed = failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  ok    {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}: got {got!r}, want {want!r}")


def check_raises(name, fn, exc=ValueError):
    global passed, failed
    try:
        fn()
    except exc:
        passed += 1
        print(f"  ok    {name}")
    except Exception as other:
        failed += 1
        print(f"  FAIL  {name}: raised {type(other).__name__}, want {exc.__name__}")
    else:
        failed += 1
        print(f"  FAIL  {name}: expected {exc.__name__}, nothing raised")


print("\nEQUIVALENCE MERGING")
check("identical keys are one class", _merge_map(["7", "7"]), {"7": "7"})
check("distinct numbers stay separate",
      len(set(_merge_map(["3", "7", "9"]).values())), 3)
check("empty input", _merge_map([]), {})
check("merging is deterministic",
      len({tuple(sorted(_merge_map(["1/2", "0.5", "3"]).items()))
           for _ in range(20)}), 1)
# The grouping must not depend on input order. Which member of a tied class is
# used as its label may, and that is harmless: the members are equivalent by
# construction, so is_correct() gives the same verdict either way.
_a = _merge_map(["1/2", "0.5", "3"])
_b = _merge_map(["3", "0.5", "1/2"])
check("the partition is order-independent",
      sorted(len({k for k in _a if _a[k] == v}) for v in set(_a.values())),
      sorted(len({k for k in _b if _b[k] == v}) for v in set(_b.values())))
check("either representative scores the same",
      Result(answer=_a["1/2"]).is_correct("0.5"),
      Result(answer=_b["1/2"]).is_correct("0.5"))

print("\nREGRESSION A3 - the class representative is the most-voted member")
check("majority wins over alphabetical order",
      select(["2*3", "6", "6"], "majority"), "6")
check("and the other way round too",
      select(["6", "6", "2*3"], "majority"), "6")
check("a lone unevaluated expression still wins when it is the majority",
      select(["2*3", "2*3", "6"], "majority"), "2*3")

print("\nREGRESSION A2 - covered() and the oracle must agree")
one = [Sample("0.5")]
check("oracle matches whatever coverage matches",
      bool(select(one, "oracle", gold="1/2")),
      bestofn.coverage(one, "1/2"))
check("Result.is_correct uses the same rule",
      Result(answer="0.5", samples=one).is_correct("1/2"),
      bestofn.coverage(one, "1/2"))
check("is_correct on an empty answer", Result(answer="").is_correct("1"), False)
check("is_correct plain case", Result(answer="204").is_correct("204.0"), True)

print("\nREGRESSION A4 - merging must not be quadratic without a bound")
many = [Sample(f"{i}/7") for i in range(1, 61)]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    got = select(many, "majority")
    capped = any("merge limit" in str(w.message) for w in caught)
check("large pools fall back instead of stalling",
      capped or not have_math_verify(), True)
check("and still return an answer", bool(got), True)

print("\nREGRESSION M5 - a positive logprob must not raise")
check("safe exp saturates", math.isinf(_safe_exp(10000.0)), True)
check("safe exp normal case", round(_safe_exp(0.0), 6), 1.0)
check("selector survives a mis-signed backend",
      select([Sample("7", logprob=1000.0), Sample("3", logprob=-1.0)],
             "self_certainty"), "7")

print("\nREGRESSION M6 - both verifier methods validate the same way")
mixed = [Sample("7", score=0.5), Sample("", score=12345.0)]
check_raises("verifier rejects", lambda: select(mixed, "verifier"))
check_raises("verifier_argmax rejects the same pool",
             lambda: select(mixed, "verifier_argmax"))

print("\nREGRESSION M7 - the sigmoid must stay inside (0, 1)")
check("very negative does not reach zero", sigmoid(-800.0) > 0.0, True)
check("very positive does not reach one", sigmoid(800.0) < 1.0, True)
check("midpoint", sigmoid(0.0), 0.5)
check("monotonic", sigmoid(-1.0) < sigmoid(0.0) < sigmoid(1.0), True)
check("nan is neutral, not poison", sigmoid(float("nan")), 0.5)
check("inf is clamped", sigmoid(float("inf")) < 1.0, True)
check("a saturated pool still weights",
      select([Sample("7", score=sigmoid(-900)), Sample("3", score=sigmoid(-900)),
              Sample("3", score=sigmoid(900))], "verifier"), "3")

print("\nVERIFIER ADAPTERS")
check("from_callable squashes logits",
      round(from_callable(lambda p, t: 0.0)("p", "t"), 6), 0.5)
check("from_callable passes probabilities through",
      from_callable(lambda p, t: 0.7, already_probability=True)("p", "t"), 0.7)
check_raises("probability out of range is rejected",
             lambda: from_callable(lambda p, t: 1.7,
                                   already_probability=True)("p", "t"))
check("non-finite becomes zero weight",
      from_callable(lambda p, t: float("nan"))("p", "t"), 0.0)
check("adapter construction is lazy and needs no model",
      isinstance(from_hub("openbmb/Eurus-RM-7b"), RewardModelVerifier), True)
check("known licences are recorded", len(KNOWN_VERIFIERS) >= 5, True)
check("every entry declares a licence",
      all("licence" in v for v in KNOWN_VERIFIERS.values()), True)
check("license_of degrades to None on a nonexistent model",
      license_of("definitely/not-a-real-model-xyz", timeout=5), None)

print("\nENGINE CONSTRUCTION (no model is loaded)")
check_raises("n must be positive", lambda: BestOfN("m", n=0))
check_raises("n cannot be negative", lambda: BestOfN("m", n=-1))
check_raises("temperature 0 with n>1", lambda: BestOfN("m", n=8, temperature=0))
check("temperature 0 with n=1 is fine",
      BestOfN("m", n=1, temperature=0).n, 1)
check_raises("unknown backend", lambda: BestOfN("m", backend="nope"))
check_raises("unknown extractor", lambda: BestOfN("m", extractor="nope"))
check_raises("regex extractor without a pattern",
             lambda: BestOfN("m", extractor="regex"))
check("backend resolves without vllm installed",
      BestOfN("m").backend in ("vllm", "transformers"), True)
check("logprobs default off", BestOfN("m").logprobs, False)

print("\nRESULT ACCOUNTING")
r = Result(answer="7", method="majority", samples=[
    Sample("7", finish_reason="stop", n_tokens=100),
    Sample("7", finish_reason="stop", n_tokens=120),
    Sample("", finish_reason="length", n_tokens=400),
    Sample("3", finish_reason="stop", n_tokens=90),
])
check("n counts what you paid for", r.n, 4)
check("effective_n counts what voted", r.effective_n, 3)
check("abstentions", r.n_abstained, 1)
check("truncated", r.n_truncated, 1)
check("total tokens", r.total_tokens, 710)
check("agreement excludes abstentions", round(r.agreement, 4), round(2 / 3, 4))
check("answers list is complete", len(r.answers), 4)
check("covered", r.covered("7"), True)
check("not covered", r.covered("99"), False)
check("select_with is free and works", r.select_with("random", seed=0) in
      {"7", "3"}, True)
check("empty result", Result(answer="").effective_n, 0)
check("empty result agreement", Result(answer="").agreement, 0.0)
check("empty result tokens", Result(answer="").total_tokens, None)

print("\nENGINE GUARDS ON solve_batch")
eng = BestOfN("m", n=1, temperature=0.0)
check_raises("temperature guard is repeated per call",
             lambda: eng.solve_batch(["p"], n=32))
check_raises("verifier method without a verifier",
             lambda: BestOfN("m").solve_batch(["p"], method="verifier"))
check_raises("verifier_argmax without a verifier",
             lambda: BestOfN("m").solve_batch(["p"], method="verifier_argmax"))
check_raises("self_certainty without logprobs",
             lambda: BestOfN("m").solve_batch(["p"], method="self_certainty"))

print("\nBATCHED SCORING")


class _Counting:
    """A verifier that records how it was called."""

    def __init__(self):
        self.batches = 0
        self.singles = 0

    def score_batch(self, problem, texts):
        self.batches += 1
        return [0.5] * len(texts)

    def __call__(self, problem, text):
        self.singles += 1
        return 0.5


v = _Counting()
eng = BestOfN("m", n=4, verifier=v)
check("batched path is preferred",
      (lambda out: (v.batches, v.singles))(eng._score_all("p", ["a", "b", "c"])),
      (1, 0))
check("returns one score per trajectory",
      len(eng._score_all("p", ["a", "b", "c"])), 3)
check("no verifier means no scores",
      BestOfN("m")._score_all("p", ["a", "b"]), [None, None])

print("\nPACKAGE SURFACE")
check("version is exported", isinstance(bestofn.__version__, str), True)
check("everything in __all__ exists",
      all(hasattr(bestofn, name) for name in bestofn.__all__), True)
check("SELECTORS matches what select accepts",
      all(s in bestofn.SELECTORS
          for s in ("random", "majority", "verifier", "oracle")), True)

print(f"\n{'='*46}\n{passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
