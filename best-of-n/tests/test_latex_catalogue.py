# -*- coding: utf-8 -*-
"""The LaTeX command catalogue, as a committed test rather than a claim.

Three releases in a row shipped a changelog entry certifying that some
verification had been run -- "a catalogue of 89 LaTeX commands", "24,000
shuffles" -- where the verification existed only in a terminal that had since
been closed. Each time, the next audit found the defect still present one code
block away from where it had been fixed.

So the catalogue lives here. Every command below is real LaTeX. None of them
has a value this library can canonicalise, so every one of them must make the
extractor **abstain**. The failure mode being guarded against is not a missing
answer: it is a *fabricated* one, where a command loses its backslash, stops
looking like a command, and votes as a plausible string. ``\\rightarrow``
becoming the vote ``ARROW`` is the exact shape of it.

Run directly, or under pytest.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bestofn import normalise                                  # noqa: E402
from bestofn.extract import extract_boxed                      # noqa: E402

passed = 0
failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print("  FAIL  %s: got %r, want %r" % (name, got, want))


# Commands with no canonicalisable value. Deliberately dense in prefixes of the
# tokens this module deletes or rewrites: left/le, right, rm, quad, circ, ne,
# ge, pi, mu, div, times, phi, bf, it.
NO_VALUE = """
leftarrow leftrightarrow leftthreetimes leftharpoonup leftharpoondown
leftleftarrows leftrightsquigarrow Leftarrow Leftrightarrow lefteqn
rightarrow rightleftharpoons rightharpoonup rightharpoondown rightsquigarrow
rightrightarrows Rightarrow rightthreetimes
rmoustache rmfamily
quadrature
circlearrowleft circlearrowright circledast circledcirc circleddash circledS
negthinspace negmedspace negthickspace nearrow neg newline nexists ni nmid
nparallel nprec nsim nsubseteq notin
geqslant gets gg ggg gnapprox gvertneqq grave
piecewise pitchfork
mumble
divideontimes
timesnewroman
phantom
bfseries bigcap bigcup bigoplus bigotimes bigsqcup biguplus bigvee bigwedge
itshape iiint iiiint idotsint imath jmath
binom dbinom tbinom choose atop over above
begin end array matrix pmatrix bmatrix vmatrix Vmatrix cases aligned
overline underline overbrace underbrace widehat widetilde overrightarrow
mathbb mathcal mathfrak mathscr mathsf mathtt boldsymbol
substack overset underset stackrel
prod coprod oint iint intop smallint
sup limsup liminf varliminf varlimsup varprojlim varinjlim
ker hom Pr deg det dim exp arg
arcsin arccos arctan sinh cosh tanh coth sec csc cot
partial nabla forall exists emptyset varnothing aleph beth gimel daleth
subset supset subseteq supseteq subsetneq supsetneq
cup cap setminus smallsetminus uplus sqcup sqcap
oplus ominus otimes oslash odot bigcirc bullet star ast
equiv approx approxeq cong simeq asymp propto models vdash dashv
ldots cdots vdots ddots dotsb dotsc dotsi dotsm
hbar ell wp Re Im mho angle measuredangle sphericalangle
top bot vert Vert lVert rVert lvert rvert lceil rceil lfloor rfloor
langle rangle llbracket rrbracket
label ref eqref cite footnote newcommand renewcommand
""".split()

print("LATEX CATALOGUE  (%d commands, none has a canonicalisable value)"
      % len(NO_VALUE))

fabricated = []
for cmd in NO_VALUE:
    for shape in ("\\boxed{\\%s}", "\\boxed{\\%s{1}{2}}", "\\boxed{5\\%s 3}"):
        got = extract_boxed(shape % cmd)
        if got:
            fabricated.append(("\\" + cmd, shape % cmd, got, normalise(got)))

check("no command in the catalogue fabricates a vote", len(fabricated), 0)
for cmd, text, got, key in fabricated[:12]:
    print("        %-18s %-32s -> %r (key %r)" % (cmd, text, got, key))

# Spacing must not fuse two numbers into a third that was never written.
print("\n  spacing commands do not fuse digits")
for gap in ("\\quad", "\\qquad", "\\;", "\\,", "\\!", "\\:", "\\ ",
            "\\displaystyle", "\\left", "\\right"):
    check("7 %s 3 does not become 73" % gap,
          extract_boxed("\\boxed{7%s 3}" % gap) in ("", None), True)

# And the legitimate forms must all survive. A guard that abstains on
# everything would pass the block above and be useless.
print("\n  legitimate answers still vote")
LEGIT = [
    ("\\boxed{204}", "204"),
    ("\\boxed{1,000}", "1000"),
    ("\\boxed{-42}", "-42"),
    ("\\boxed{3.75}", "3.75"),
    ("\\boxed{\\frac{5}{2}}", "5/2"),
    ("\\boxed{\\dfrac{5}{2}}", "5/2"),
    ("\\boxed{2\\frac{1}{2}}", "(2+1/2)"),
    # Case is folded outside LaTeX control words, so these are the canonical
    # keys, not the extracted text. Gold answers go through the same function,
    # so they still compare equal.
    ("\\boxed{\\sqrt{2}}", "SQRT(2)"),
    ("\\boxed{2\\pi}", "2PI"),
    ("\\boxed{90^\\circ}", "90"),
    ("\\boxed{45^{\\circ}}", "45"),
    ("\\boxed{x \\geq 5}", "X>=5"),
    ("\\boxed{x \\ge 5}", "X>=5"),
    ("\\boxed{x \\neq 5}", "X!=5"),
    ("\\boxed{3 \\times 4}", "3*4"),
    ("\\boxed{3 \\cdot 4}", "3*4"),
    ("\\boxed{\\left(5\\right)}", "5"),
    ("\\boxed{\\left(3,4\\right)}", "(3,4)"),
    ("\\boxed{2^{10}}", "2**10"),
    ("\\boxed{x^2}", "X**2"),
]
for text, want_key in LEGIT:
    check("%s votes" % text, normalise(extract_boxed(text)), want_key)

# normalise must never raise, whatever reaches it.
print("\n  normalise never raises")
HOSTILE = [
    None, 42, 3.5, "", "   ", "\x00\x01", "٠١٢",
    "9" * 10000, "-" + "9" * 10000, ("9" * 5000) + "/" + ("3" * 5000),
    "1e999999", "-1e999999", "0" * 5000, "x" * 100000,
    "\\frac{" + "9" * 5000 + "}{" + "3" * 5000 + "}",
    "1/0", "-0", "inf", "-inf", "nan",
]
raised = []
for h in HOSTILE:
    try:
        normalise(h)
    except Exception as exc:                                   # noqa: BLE001
        raised.append((repr(h)[:40], type(exc).__name__))
check("normalise survives every hostile input", raised, [])
for what, exc in raised:
    print("        %s -> %s" % (what, exc))

# and a pool containing one of them must still be selectable
from bestofn import Sample, select, agreement, effective_n     # noqa: E402
poison = ("9" * 5000) + "/" + ("3" * 5000)
pool = [Sample("42")] * 127 + [Sample(poison)]
for fn_name, fn in (("select", lambda: select(pool, "majority")),
                    ("agreement", lambda: agreement(pool)),
                    ("effective_n", lambda: effective_n(pool))):
    try:
        fn()
        ok = True
    except Exception as exc:                                   # noqa: BLE001
        ok = False
        print("        %s raised %s" % (fn_name, type(exc).__name__))
    check("%s survives a poisoned pool" % fn_name, ok, True)

print("\n%s\n%d passed, %d failed\n%s" % ("=" * 46, passed, failed, "=" * 46))

if __name__ == "__main__":
    sys.exit(1 if failed else 0)
else:
    def test_latex_catalogue():
        assert failed == 0, "%d checks failed" % failed
