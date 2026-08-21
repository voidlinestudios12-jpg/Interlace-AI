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
check("ties favour the first seen", select(same, "majority"), "7")

print(f"\n{'='*46}\n{passed} passed, {failed} failed\n{'='*46}")
sys.exit(1 if failed else 0)
