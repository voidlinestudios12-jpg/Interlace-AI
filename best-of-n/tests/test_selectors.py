"""Unit tests for extraction and selection. Run: python tests/test_selectors.py

No GPU or model download required.

Every test under REGRESSIONS corresponds to a defect that shipped in 1.0.0 and
was found by an adversarial audit of the published package. They are labelled
with the behaviour that was wrong so the bug cannot come back unnoticed.
"""
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bestofn import (Sample, abstentions, agreement, coverage,  # noqa: E402
                     effective_n, extract_boxed, extract_letter,
                     extract_number, normalise, select)

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
    else:
        failed += 1
        print(f"  FAIL  {name}: expected {exc.__name__}, nothing raised")


print("\nEXTRACTION")
check("boxed simple", extract_boxed(r"so the answer is \boxed{42}."), "42")
check("boxed last of several", extract_boxed(r"\boxed{1} then \boxed{99}"), "99")
check("boxed negative", extract_boxed(r"\boxed{-17}"), "-17")
check("boxed units stripped", extract_boxed(r"\boxed{204\text{ km}}"), "204")
check("boxed empty input", extract_boxed(""), "")
check("letter from box", extract_letter(r"therefore \boxed{C}"), "C")
check("letter standalone", extract_letter("the answer is B"), "B")

print("\nREGRESSION E1 - fractions and radicals must not collapse to a digit")
check("half", extract_boxed(r"\boxed{\frac{1}{2}}"), "1/2")
check("third", extract_boxed(r"\boxed{\frac{1}{3}}"), "1/3")
check("negative half keeps its sign", extract_boxed(r"\boxed{-\frac{1}{2}}"), "-1/2")
check("dfrac", extract_boxed(r"\boxed{\dfrac{3}{4}}"), "3/4")
check("pi survives", extract_boxed(r"\boxed{2\pi}"), "2pi")
check("radical survives", extract_boxed(r"\boxed{\sqrt{2}}"), "sqrt(2)")
check("coefficient and radical", extract_boxed(r"\boxed{2\sqrt{5}}"), "2sqrt(5)")
check("power", extract_boxed(r"\boxed{10^{3}}"), "10**3")
check("nested fraction", extract_boxed(r"\boxed{\frac{\sqrt{3}}{2}}"), "sqrt(3)/2")
check("sum of fractions", extract_boxed(r"\boxed{\frac{1}{2}+\frac{1}{3}}"),
      "1/2+1/3")
check("parens kept where they matter",
      extract_boxed(r"\boxed{\frac{a+b}{2}}"), "(a+b)/2")
check("symbolic answer", extract_boxed(r"\boxed{x+1}"), "x+1")
check("word answer", extract_boxed(r"\boxed{\text{Canberra}}"), "Canberra")
check("four distinct answers stay distinct",
      len({extract_boxed(r"\boxed{\frac{1}{2}}"),
           extract_boxed(r"\boxed{\frac{1}{3}}"),
           extract_boxed(r"\boxed{-\frac{1}{2}}"),
           extract_boxed(r"\boxed{1}")}), 4)

print("\nREGRESSION E2 - commas must not be deleted blindly")
check("ordered pair", extract_boxed(r"\boxed{(3,4)}"), "(3,4)")
check("interval", extract_boxed(r"\boxed{[0, 1]}"), "[0,1]")
check("latex thousands", extract_boxed(r"\boxed{1{,}000}"), "1000")
check("plain thousands", extract_boxed(r"\boxed{1,000}"), "1000")
check("pair is not an integer", extract_boxed(r"\boxed{(3,4)}") == "34", False)
check("number stops at a comma", extract_number("the point is (3,4)"), "4")
check("number still ungroups thousands",
      extract_number("that gives 1,234 apples"), "1234")

print("\nREGRESSION E8 - a truncated trajectory must abstain, not guess")
cut = "...the sum is 161. Then I add 12 and get 173. Maybe I should use 204 and"
check("no box means no vote", extract_boxed(cut), "")
check("guessing is opt-in", extract_boxed(cut, allow_fallback=True), "204")
check("unclosed box abstains", extract_boxed(r"the answer is \boxed{20"), "")
check("abstention is not a vote", select([Sample(""), Sample("7")], "majority"), "7")

print("\nNORMALISATION")
check("int vs float", normalise("204.0"), "204")
check("whitespace", normalise("  42 "), "42")
check("trailing period", normalise("42."), "42")
check("equivalent fractions reduce", normalise("2/4"), "1/2")
check("integer fraction reduces", normalise("4/2"), "2")
check("zero denominator rejected", normalise("1/0"), "")
check("infinity rejected", normalise("inf"), "")
check("nan rejected", normalise("nan"), "")
check("empty", normalise(""), "")
check("none", normalise(None), "")
check("text upper-cased", normalise("canberra"), "CANBERRA")

print("\nREGRESSION E9 - one normalisation on both sides of the gap")
mixed = [Sample("204.0"), Sample("204.0"), Sample("3")]
check("select returns canonical form", select(mixed, "majority"), "204")
check("and it matches the gold directly", select(mixed, "majority") == "204", True)
check("coverage agrees with select", coverage(mixed, "204"), True)

print("\nSELECTORS")
check("majority", select(["7", "7", "3"], "majority"), "7")
check("majority single", select(["7"], "majority"), "7")
check("majority empty", select([], "majority"), "")
check("all abstain", select([Sample(""), Sample("")], "majority"), "")
check_raises("unknown method", lambda: select(["1"], "nope"))

print("\nREGRESSION E10 - random is the baseline every selector must beat")
pool = [Sample("A"), Sample("B"), Sample("C"), Sample("")]
check("random never returns an abstention",
      all(select(pool, "random", seed=i) for i in range(20)), True)
check("random is reproducible",
      select(pool, "random", seed=7), select(pool, "random", seed=7))
check("random can pick a minority",
      len({select(pool, "random", seed=i) for i in range(40)}) > 1, True)

print("\nREGRESSION E7 - reward-model logits must not become a majority vote")
logits = [Sample("100", score=-0.5), Sample("100", score=-0.6),
          Sample("100", score=-0.7), Sample("42", score=-0.01)]
check_raises("verifier rejects logits", lambda: select(logits, "verifier"))
check_raises("verifier_argmax rejects logits too",
             lambda: select(logits, "verifier_argmax"))
probs = [Sample("100", score=0.30), Sample("100", score=0.28),
         Sample("100", score=0.25), Sample("42", score=0.99)]
check("verifier promotes a confident minority", select(probs, "verifier"), "42")
check("majority would have said otherwise", select(probs, "majority"), "100")
check("verifier_argmax takes the top score",
      select(probs, "verifier_argmax"), "42")
check_raises("verifier without scores", lambda: select(["1", "2"], "verifier"))

print("\nREGRESSION E21 - NaN must not decide the vote by list order")
nan = float("nan")
first = [Sample("99", score=nan), Sample("7", score=0.9)]
second = [Sample("7", score=0.9), Sample("99", score=nan)]
check("nan first", select(first, "verifier_argmax"), "7")
check("nan second", select(second, "verifier_argmax"), "7")
check("order does not matter",
      select(first, "verifier_argmax"), select(second, "verifier_argmax"))

print("\nSELF-CERTAINTY")
lp = [Sample("7", logprob=-0.1), Sample("3", logprob=-5.0),
      Sample("3", logprob=-5.0)]
check("confident single beats unconfident pair",
      select(lp, "self_certainty"), "7")
check_raises("self_certainty without logprobs",
             lambda: select(["1", "2"], "self_certainty"))
check("nan logprob is ignored, not fatal",
      select([Sample("7", logprob=nan), Sample("3", logprob=-0.2)],
             "self_certainty"), "3")

print("\nREGRESSION E20 - agreement must not count abstentions in the denominator")
three_and_two_blank = [Sample("7"), Sample("7"), Sample("7"),
                       Sample(""), Sample(" ")]
check("unanimous among voters", agreement(three_and_two_blank), 1.0)
check("abstentions counted", abstentions(three_and_two_blank), 2)
check("effective n", effective_n(three_and_two_blank), 3)
check("agreement 3 of 4", round(agreement(["7", "7", "7", "3"]), 3), 0.75)
check("agreement empty", agreement([]), 0.0)
check("agreement all abstain", agreement([Sample(""), Sample("")]), 0.0)

print("\nORACLE AND COVERAGE")
check("oracle finds it", select(["1", "2", "42"], "oracle", gold="42"), "42")
check("oracle misses", select(["1", "2"], "oracle", gold="42"), "")
check_raises("oracle without gold", lambda: select(["1"], "oracle"))
check("coverage true", coverage(["1", "42"], "42"), True)
check("coverage false", coverage(["1", "2"], "42"), False)
check("coverage rejects a non-finite gold", coverage([Sample("")], "inf"), False)
check("coverage rejects an empty gold", coverage([Sample("")], ""), False)
check("coverage rejects a nan gold", coverage([Sample("")], "nan"), False)

print("\nTRUNCATION BOOKKEEPING")
mix = [Sample("7", finish_reason="stop"), Sample("", finish_reason="length"),
       Sample("3", finish_reason="stop")]
check("truncated flag", [s.truncated for s in mix], [False, True, False])
check("effective n excludes the truncated one", effective_n(mix), 2)

print("\nINPUT FORMATS")
check("dict input", select([{"answer": "5"}, {"answer": "5"}], "majority"), "5")
check("spanish dict keys",
      select([{"respuesta": "5"}, {"respuesta": "5"}], "majority"), "5")
check("dict carries finish_reason",
      select([{"answer": "5", "finish_reason": "stop"}], "majority"), "5")
check("bare strings", select(["5", "5", "6"], "majority"), "5")

print("\nDETERMINISM")
same = [Sample("7"), Sample("3"), Sample("7"), Sample("3")]
check("ties break deterministically",
      len({select(same, "majority") for _ in range(50)}), 1)
check("ties break to the smallest canonical key", select(same, "majority"), "3")
check("tie-breaking does not depend on order",
      select(list(reversed(same)), "majority"), select(same, "majority"))

import random as _rnd                                          # noqa: E402
_pool = [Sample(x) for x in ["7", "3", "7", "3", "11", "11"]]
_shuffled = list(_pool)
_seen = set()
for _i in range(60):
    _rnd.Random(_i).shuffle(_shuffled)
    _seen.add(select(list(_shuffled), "majority"))
check("60 shuffles of one pool give one answer", len(_seen), 1)



print("\nREGRESSION GUARDS  (these encode defects that shipped once)")

# --- A1: the equivalence partition must not depend on input order ----------
# equivalent() is not transitive: math-verify accepts 0.3333333333 ~ 1/3 and
# 1/3 ~ 0.33333333333333 but rejects the two decimals against each other.
# Greedy grouping against one representative per class therefore returned a
# different partition per input order. Comparing class *sizes* -- which the
# earlier test did -- does not catch it, because the sizes can coincide.
from bestofn.select import _merge_map          # noqa: E402
import itertools                               # noqa: E402


def _partition(keys):
    m = _merge_map(list(keys))
    return frozenset(frozenset(k for k in m if m[k] == r)
                     for r in set(m.values()))


_tri = ["0.3333333333", "1/3", "0.33333333333333"]
check("partition is identical under all input orders",
      len({_partition(p) for p in itertools.permutations(_tri)}), 1)

_mix = ["1/2", "0.5", "2/4", "7", "7.0"]
check("partition is order-independent on a mixed pool",
      len({_partition(p) for p in itertools.permutations(_mix)}), 1)

check("the winner does not depend on pool order",
      len({select(list(p), "majority")
           for p in itertools.permutations(["1/3", "1/3", "0.3333333333"])}), 1)

# --- A2: case folding must not destroy LaTeX control words -----------------
# normalise() used to end in .upper(), turning FRAC into a token no symbolic
# parser accepts, so is_correct() returned False on answers covered() called
# True. Structure has to survive canonicalisation.
for _cmd in ("frac{1}{2}", "sqrt{2}", "pi", "dfrac{3}{4}"):
    _raw = "\\" + _cmd
    check("normalise preserves " + "\\" + _cmd,
          "\\" + _cmd.split("{")[0] in normalise(_raw), True)

check("normalise still folds plain text", normalise("abc"), "ABC")
check("normalise still folds variables", normalise("x=5"), "X=5")
# Symbolic equivalence is what merges "\\frac{1}{2}" with "0.5" and what
# makes the mixed-number form compare equal to a decimal. Without math-verify
# these cannot pass, and asserting them anyway turned a default
# `pip install bestofn` into a red test suite.
from bestofn.extract import have_math_verify   # noqa: E402
_SKIPPED = 0

if have_math_verify():
    check("LaTeX and decimal still merge",
          len(_partition(["\\frac{1}{2}", "0.5"])), 1)
else:
    _SKIPPED += 1

# --- M7: say so when the symbolic layer is absent --------------------------
# A bare print is swallowed by pytest, so the previous version of this notice
# was invisible in exactly the run that needed it. Go through warnings, which
# pytest collects and displays in its summary.
if not have_math_verify():
    import warnings as _w                       # noqa: E402
    _w.warn(
        "math-verify is not installed, so %d symbolic-equivalence checks were "
        "skipped and the remaining ones did not exercise the symbolic path. "
        "Install with: pip install \"bestofn[math]\"" % _SKIPPED,
        RuntimeWarning, stacklevel=2)
    print("\n  NOTE: %d checks skipped (math-verify not installed)." % _SKIPPED)
    print('  Install with:  pip install "bestofn[math]"')


# --- M5: a mixed number is a sum, not a product ----------------------------
# Juxtaposition means multiplication everywhere else in _latex_to_text, so the
# generic rule read "2\frac{1}{2}" -- two and a half -- as 2*(1/2) = 1. Not a
# lost answer but a fabricated one, and a plausible enough value to match some
# other problem's reference by accident.
from bestofn.extract import extract_boxed, equivalent   # noqa: E402

_BOX = "\\boxed"
_FRAC = "\\frac"

# These two hold with or without the symbolic layer: they are about what the
# extractor emits, not about what compares equal to it.
check("mixed number extracts as a sum, not a product",
      extract_boxed(_BOX + "{2" + _FRAC + "{1}{2}}"), "(2+1/2)")
check("negative mixed number negates the whole thing",
      extract_boxed(_BOX + "{-3" + _FRAC + "{1}{3}}"), "-(3+1/3)")

if have_math_verify():
    check("mixed number equals two and a half",
          equivalent(extract_boxed(_BOX + "{2" + _FRAC + "{1}{2}}"), "2.5"), True)
    check("mixed number is not the product",
          equivalent(extract_boxed(_BOX + "{2" + _FRAC + "{1}{2}}"), "1"), False)
    check("negative mixed number equals -10/3",
          equivalent(extract_boxed(_BOX + "{-3" + _FRAC + "{1}{3}}"), "-10/3"), True)
    check("negative mixed number is not -8/3",
          equivalent(extract_boxed(_BOX + "{-3" + _FRAC + "{1}{3}}"), "-8/3"), False)
    check("a plain fraction is left alone",
          equivalent(extract_boxed(_BOX + "{" + _FRAC + "{5}{2}}"), "2.5"), True)
else:
    _SKIPPED += 5


print("\nPERMUTATION INVARIANCE  (every selector, fuzzed)")

# The claim is that a fixed pool returns a fixed answer whatever order it
# arrives in. It has three separate failure modes and each one shipped once:
# a greedy equivalence partition, weights accumulated in arrival order (float
# addition is not associative), and ties broken on first-seen. `oracle` had a
# fourth -- it returned the first key equivalent to gold -- and no test could
# see it, because the only coverage here was 60 shuffles of `majority`.
import random as _rng_mod                                      # noqa: E402

_ANSWERS = ["1/2", "0.5", "2/4", "7", "7.0", "1/3", "0.3333333333",
            "0.33333333333333", "3", "11", "-2", "2.50", "5/2"]
# A gold whose canonical form matches NO key in the pool, so the exact-match
# branch cannot fire and the equivalence fallback is the code under test.
# "2/4" would not do: normalise already reduces it to "1/2".
_GOLD = "\\frac{1}{2}"


def _fuzz_selector(name, trials=400, shuffles=6, kw=None):
    """Return how many pools changed their answer when reordered."""
    kw = kw or {}
    rng = _rng_mod.Random(90210 + len(name))
    moved = 0
    for _ in range(trials):
        pool = [Sample(rng.choice(_ANSWERS),
                       logprob=-rng.random() * 3,
                       score=round(rng.random(), 2))
                for _ in range(rng.randint(2, 14))]
        try:
            base = select(pool, name, **kw)
        except Exception:
            continue
        for k in range(shuffles):
            shuffled = list(pool)
            _rng_mod.Random(k).shuffle(shuffled)
            try:
                if select(shuffled, name, **kw) != base:
                    moved += 1
                    break
            except Exception:
                pass
    return moved


for _name, _kw in (("majority", None), ("self_certainty", None),
                   ("verifier", None), ("verifier_argmax", None),
                   ("oracle", {"gold": _GOLD})):
    check("%s is invariant under permutation" % _name,
          _fuzz_selector(_name, kw=_kw), 0)

# The specific pool that falsified `oracle`: two distinct canonical keys both
# equivalent to the gold, so whichever arrived first was returned.
if have_math_verify():
    # Two DISTINCT canonical keys, each equivalent to the gold, and the gold
    # itself canonicalising to neither. Returning the first one encountered
    # made the answer depend on pool order; this is the pool that showed it.
    _both = [Sample("1/2"), Sample("0.5")]
    _g = "\\frac{1}{2}"
    check("oracle's two equivalent keys really are distinct",
          normalise("1/2") != normalise("0.5") and normalise(_g) not in ("1/2", "0.5"),
          True)
    check("oracle does not depend on which equivalent key came first",
          select(_both, "oracle", gold=_g),
          select(list(reversed(_both)), "oracle", gold=_g))

# Exact ties are the case max() and += both get wrong.
_tied_scores = [Sample("7", score=0.5), Sample("3", score=0.5)]
check("verifier_argmax breaks an exact tie the same way both ways",
      select(_tied_scores, "verifier_argmax"),
      select(list(reversed(_tied_scores)), "verifier_argmax"))

_tied_lp = [Sample("7", logprob=-1.0), Sample("3", logprob=-1.0)]
check("self_certainty breaks an exact tie the same way both ways",
      select(_tied_lp, "self_certainty"),
      select(list(reversed(_tied_lp)), "self_certainty"))

# A pool past the merge cap falls back to exact matching; that path has to be
# order-independent too.
_big = [Sample("%d/17" % i) for i in range(1, 90)] + [Sample("7"), Sample("7")]
check("a pool past the merge cap is still invariant",
      len({select(_big[i:] + _big[:i], "majority") for i in (0, 17, 43, 88)}), 1)

print(f"\n{'='*46}\n{passed} passed, {failed} failed\n{'='*46}")
if __name__ == "__main__":
    sys.exit(1 if failed else 0)
else:
    # Imported rather than run: every check above has already executed at
    # import time, so expose the verdict as a single test pytest can collect.
    # Without this the bare sys.exit above aborts collection with an
    # INTERNALERROR -- and `pytest` is the first thing anyone who clones the
    # repository types.
    def test_all_checks_passed():
        assert failed == 0, "%d checks failed" % failed
