"""Answer extraction and normalisation.

Turning a free-form reasoning trajectory into a comparable answer token is the
step where naive implementations lose most of their accuracy: an extraction
failure is indistinguishable from a reasoning failure at the metric level.

Extractors provided, plus support for any user-supplied callable:

    boxed   -- last \\boxed{...}, bracket-balanced, LaTeX canonicalised
    number  -- last number in the text
    letter  -- last standalone A/B/C/D (multiple choice)
    regex   -- first capture group of a user pattern

Three rules, all learned from defects that shipped:

1. **Never invent an answer.** A trajectory with no recoverable answer returns
   ``""`` and abstains. Guessing at the last number in an unfinished chain of
   thought casts a phantom vote indistinguishable from a real one. The same
   applies to LaTeX this module cannot fully resolve: it abstains rather than
   emit a plausible-looking fragment.
2. **Never destroy structure while cleaning.** Commas separate tuple elements
   as often as they group thousands. Parentheses mark products as often as
   they are decoration. Braces delimit sets. All are handled with context.
3. **Never merge two different answers.** ``(2)(3)`` is six, not twenty-three;
   ``{1}`` is a set, not the number one.

Equivalence is decided in :func:`normalise` and :func:`equivalent`. With
``math-verify`` installed (``pip install "bestofn[math]"``) comparison is
symbolic, so ``1/2``, ``0.5`` and ``\\frac{2}{4}`` count as one answer.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
from __future__ import annotations

import functools
import logging
import math
import re
import warnings
from typing import Callable, Optional, Union

__all__ = ["extract_boxed", "extract_number", "extract_letter", "normalise",
           "equivalent", "get_extractor", "Extractor", "have_math_verify"]

Extractor = Callable[[str], str]


# --------------------------------------------------------------- math-verify

def _load_math_verify():
    """Return math-verify's ``(parse, verify)``, or ``None`` if unavailable.

    Optional on purpose: the selectors must run with zero dependencies so the
    package installs anywhere, but symbolic equivalence beats string matching
    whenever it is available.

    Its timeout machinery is disabled at the call site -- it spawns worker
    processes, which fail on some platforms -- and the resulting per-call
    notice is silenced here so it does not flood the caller's logs.
    """
    try:
        from math_verify import parse, verify        # type: ignore
    except Exception:
        return None
    for name in ("math_verify", "math_verify.parser", "math_verify.grader",
                 "math_verify.utils"):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass
    return parse, verify


_MATH_VERIFY = _load_math_verify()


def have_math_verify() -> bool:
    """Whether symbolic answer comparison is available in this environment."""
    return _MATH_VERIFY is not None


# ------------------------------------------------------------ LaTeX cleaning

#: Purely decorative tokens. Note what is NOT here: commas, spaces, braces and
#: parentheses, all of which carry meaning that deleting them would destroy.
_LATEX_NOISE = (
    r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ",
    r"\$", r"\%", r"\quad", r"\qquad", r"\displaystyle", r"\rm",
    "$", "%",
)

#: 1,000 or 1{,}000 -> 1000, but only when the commas really do group digits.
_THOUSANDS = re.compile(r"^(-?\d{1,3})((?:\{?,\}?\d{3})+)(\.\d+)?$")

_TEXTUAL = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|operatorname|bm)\s*\{([^{}]*)\}"
)

#: A whole answer written as prose -- \boxed{\text{Canberra}} -- as opposed to
#: a unit trailing a value -- \boxed{204\text{ km}}. The first is the answer;
#: the second is decoration, and keeping it splits the vote between "204" and
#: "204km".
_ONLY_TEXT = re.compile(
    r"^\s*\\(?:text|mathrm|mathbf|mathit|operatorname|bm)\s*\{([^{}]*)\}\s*$"
)

#: A unit can carry digits, slashes and exponents -- "m/s", "cm^2", "km2" -- so
#: matching letters alone left those attached and split the vote.
_UNIT_LIKE = re.compile(r"^[A-Za-z0-9\s.,\-/^*()\u00b0\u00b2\u00b3]+$")
_HAS_LETTER = re.compile(r"[A-Za-z]")

#: Private-use placeholders standing in for set braces while the grouping
#: braces are removed. No real answer contains them.
_LB, _RB = "\ue000", "\ue001"

#: Commands this module rewrites. Any still present after resolution means the
#: expression was nested too deeply to canonicalise, and emitting the partial
#: result would produce a plausible-looking wrong answer.
_UNRESOLVED = re.compile(r"\\(?:[dt]?frac|sqrt)|(?:frac|sqrt)(?=[^(a-zA-Z]|$)")

_MAX_NESTING = 24

#: A LaTeX command still present after every rewrite above has run. It cannot
#: be canonicalised, and stripping the backslash produces a plausible-looking
#: wrong answer rather than a missing one, so the extractor abstains instead.
#: ``\begin``/``\end`` are caught by the same rule.
_LEFTOVER_COMMAND = re.compile(r"\\[A-Za-z]")


def _is_unit(content: str) -> bool:
    """Whether a ``\\text{...}`` group is a unit rather than the answer."""
    c = (content or "").strip()
    return bool(c) and bool(_HAS_LETTER.search(c)) and bool(_UNIT_LIKE.match(c))


def _strip_thousands(s: str) -> str:
    """Remove digit-grouping commas, leaving every other comma intact."""
    m = _THOUSANDS.match(s.replace(" ", ""))
    if not m:
        return s
    head, groups, frac = m.group(1), m.group(2), m.group(3) or ""
    return head + re.sub(r"[{},]", "", groups) + frac


def _latex_to_text(s: str) -> Optional[str]:
    """Canonicalise LaTeX into a compact, comparable text form.

    ``\\frac{1}{2}`` becomes ``1/2`` rather than ``1``; ``2\\sqrt{5}`` becomes
    ``2sqrt(5)`` rather than ``2``. Structure is preserved, which is the point:
    collapsing answers onto their first digit makes different answers compare
    equal and corrupts the vote.

    Returns ``None`` when the expression could not be fully resolved, so the
    caller abstains instead of voting with a mangled fragment.
    """
    if not s:
        return None

    whole = _ONLY_TEXT.match(s)
    if whole:                               # the prose IS the answer
        return re.sub(r"\s+", " ", whole.group(1)).strip()

    s = _TEXTUAL.sub(
        lambda m: "" if _is_unit(m.group(1)) else (m.group(1) or ""), s)

    # A mixed number has to be resolved before the general \frac rule sees it.
    # "2\frac{1}{2}" is two and a half, but juxtaposition means multiplication
    # everywhere else in this function, so the generic rule turns it into
    # "2(1)/(2)" -- which every downstream parser reads as 2*(1/2) = 1. A
    # wrong answer, silently, and a plausible-looking one. Only an integer
    # written directly against the \frac counts; "x\frac{1}{2}" really is a
    # product and is left alone.
    s = re.sub(r"(?<![\w.)}])(-?\d+)\s*\\([dt]?frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
               lambda m: "{0}(({1})+({2})/({3}))".format(
                   "-" if m.group(1).startswith("-") else "",
                   m.group(1).lstrip("-"), m.group(3), m.group(4)),
               s)

    for _ in range(_MAX_NESTING):
        new = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
                     r"(\1)/(\2)", s)
        new = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}",
                     r"(\2)**(1/(\1))", new)
        new = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", new)
        if new == s:
            break
        s = new

    if re.search(r"\\[dt]?frac|\\sqrt", s):
        return None                     # too deeply nested; abstain

    # Degrees are a unit, like the "km" in "204 km", and the whole construct
    # has to go before the exponent rule sees it. Left to that rule,
    # "45^{\circ}" becomes "45**(\circ)" and then "45**()", which is not a
    # number; dropping only the marker leaves "90^", which no longer compares
    # equal to a gold of "90".
    s = re.sub(r"\^\s*\{\s*\\(?:circ|degree)\s*\}", "", s)
    s = re.sub(r"\^\s*\\(?:circ|degree)", "", s)
    s = re.sub(r"\\(?:circ|degree)", "", s)

    s = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^(-?\w)", r"**\1", s)
    s = re.sub(r"_\s*\{([^{}]*)\}", r"_\1", s)

    # Commands with an unambiguous plain form. Order matters: "geq" has to be
    # tried before "ge", or the shorter name eats the start of the longer one.
    for name, plain in (
            ("geq", ">="), ("leq", "<="), ("neq", "!="), ("ge", ">="),
            ("le", "<="), ("ne", "!="), ("pm", "+-"), ("circ", ""),
            ("degree", ""), ("cdot", "*"), ("times", "*"), ("div", "/"),
            ("infty", "inf"), ("pi", "pi"), ("theta", "theta"),
            ("alpha", "alpha"), ("beta", "beta"), ("gamma", "gamma"),
            ("phi", "phi"), ("lambda", "lambda"), ("mu", "mu"),
            ("sigma", "sigma"), ("omega", "omega")):
        s = s.replace("\\" + name, plain)

    for tok in _LATEX_NOISE:
        s = s.replace(tok, "")

    # Set braces are part of the answer: {1,2,3} is not (1,2,3), and {1} is not
    # 1. Protect them before the grouping braces go.
    s = s.replace("\\{", _LB).replace("\\}", _RB)

    # Anything still carrying a backslash is a command this module cannot
    # resolve. Deleting the backslash and the braces -- which is what happened
    # until now -- turns "\binom{5}{2}", which is 10, into the vote
    # "binom52", and "\begin{pmatrix}1\\2\end{pmatrix}" into a run of digits.
    # That is exactly the fabricated-answer failure rule 1 in the module
    # docstring promises not to commit, and the promise was only being kept
    # for \frac and \sqrt. Abstaining costs a vote; guessing corrupts the
    # tally, and a fabricated number can match another problem's reference by
    # accident.
    if _LEFTOVER_COMMAND.search(s):
        return None

    s = s.replace("\\", "").replace("{", "").replace("}", "")
    s = s.replace(_LB, "{").replace(_RB, "}")
    s = re.sub(r"\s+", "", s)

    return _strip_thousands(s.strip())


#: A parenthesis touching an identifier is a call -- sqrt(5), f(x) -- and one
#: touching another parenthesis is a product -- (2)(3). Both must survive:
#: stripping them concatenates the digits and manufactures an answer, exactly
#: as deleting commas once turned (3,4) into 34.
_ADJACENT = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_)]"
)
_REDUNDANT_PARENS = re.compile(
    r"(?<![A-Za-z0-9_)\]])\((-?\d+(?:\.\d+)?)\)(?![\(\[])"
)


def _atomic(inner: str) -> bool:
    """Whether ``inner`` needs no parentheses to keep its meaning.

    True for a single term such as ``5`` or ``sqrt(3)``; false as soon as a
    top-level operator or comma appears, because ``(1+2)/3`` is not ``1+2/3``.
    """
    if not inner:
        return False
    depth = 0
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return False
        elif depth == 0 and ch in "+-*/,^":
            return False
    return depth == 0


def _strip_atomic_parens(s: str) -> str:
    """Remove one layer of parentheses around any single, complete term."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "(" and (i == 0 or s[i - 1] not in _ADJACENT):
            depth, j = 1, i + 1
            while j < len(s) and depth:
                depth += (s[j] == "(") - (s[j] == ")")
                j += 1
            # A closing parenthesis immediately followed by an opening one is
            # multiplication by juxtaposition, not decoration.
            if depth == 0 and not (j < len(s) and s[j] in "(["):
                inner = s[i + 1:j - 1]
                if _atomic(inner):
                    out.append(inner)
                    i = j
                    continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _tidy_parens(s: str) -> str:
    """Drop the redundant parentheses ``_latex_to_text`` introduces.

    Deliberately conservative: tuple parentheses in ``(3,4)``, call parentheses
    in ``sqrt(5)`` and product parentheses in ``(2)(3)`` all carry meaning, and
    removing any of them would change the answer.
    """
    prev = None
    while prev != s:
        prev = s
        s = _REDUNDANT_PARENS.sub(r"\1", s)
        s = _strip_atomic_parens(s)
    return s


# ----------------------------------------------------------------- extractors

def extract_boxed(text: str, allow_fallback: bool = False) -> str:
    """Content of the LAST ``\\boxed{...}``, handling nested braces.

    A regex cannot do this correctly: ``\\boxed{\\frac{1}{2}}`` needs brace
    counting.

    Args:
        text: the trajectory.
        allow_fallback: when no box is present, guess at the last number in the
            text. **Off by default.** A truncated trajectory almost never
            contains its answer, so guessing manufactures a phantom vote. Only
            enable it for generators that reliably answer without ``\\boxed{}``.

    Returns:
        The canonicalised answer, or ``""`` to abstain.
    """
    if not text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return extract_number(text) if allow_fallback else ""

    i, depth, out = idx + len("\\boxed{"), 1, []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1

    if depth > 0:                      # unclosed brace: the box was truncated
        return extract_number(text) if allow_fallback else ""

    canonical = _latex_to_text("".join(out))
    if canonical is None:
        return extract_number(text) if allow_fallback else ""
    inner = _tidy_parens(canonical)
    if inner:
        return inner
    return extract_number(text) if allow_fallback else ""


def extract_number(text: str) -> str:
    """Last number appearing in the text.

    Digit-grouping commas are stripped (``1,000`` -> ``1000``); other commas
    terminate the match, so ``(3,4)`` yields ``4`` rather than ``34``.
    """
    if not text:
        return ""
    nums = re.findall(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else ""


def extract_letter(text: str, options: str = "ABCD") -> str:
    """Last standalone option letter, preferring one inside ``\\boxed{}``."""
    if not text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx != -1:
        # Start *after* the token: the 'b' of "boxed" would otherwise match.
        inner = text[idx + len("\\boxed{"):idx + len("\\boxed{") + 60]
        m = re.search(rf"[{options}{options.lower()}]", inner)
        if m:
            return m.group(0).upper()
    found = re.findall(rf"\b([{options}])\b", text)
    return found[-1] if found else ""


def _regex_extractor(pattern: str) -> Extractor:
    compiled = re.compile(pattern)

    def _extract(text: str) -> str:
        m = compiled.search(text or "")
        if not m:
            return ""
        return (m.group(1) if m.groups() else m.group(0)).strip()

    return _extract


def get_extractor(kind: Union[str, Extractor], **kwargs) -> Extractor:
    """Resolve ``kind`` to a callable.

    ``kind`` may be ``"boxed"``, ``"number"``, ``"letter"``, ``"regex"``
    (requires ``pattern=``), or any callable ``str -> str``.

    ``boxed`` accepts ``allow_fallback=True`` to guess at the last number when
    no box is present. It defaults to ``False``; see :func:`extract_boxed`.
    """
    if callable(kind):
        return kind
    kind = str(kind).lower()
    if kind == "boxed":
        fallback = bool(kwargs.get("allow_fallback", False))
        return lambda t: extract_boxed(t, allow_fallback=fallback)
    if kind == "number":
        return extract_number
    if kind == "letter":
        options = kwargs.get("options", "ABCD")
        return lambda t: extract_letter(t, options)
    if kind == "regex":
        pattern = kwargs.get("pattern")
        if not pattern:
            raise ValueError("extractor 'regex' requires pattern=...")
        return _regex_extractor(pattern)
    raise ValueError(
        f"unknown extractor {kind!r}; use 'boxed', 'number', 'letter', "
        "'regex', or pass a callable"
    )


# --------------------------------------------------------------- normalisation

_FRACTION = re.compile(r"^(-?\d+)\s*/\s*(\d+)$")


def _numeric_key(s: str) -> Optional[str]:
    """Canonical key for anything that is a plain number, else ``None``."""
    try:
        f = float(s)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(f):
        return ""                       # inf/nan can never win a vote
    return str(int(f)) if f == int(f) else repr(f)


#: A LaTeX control word. Case matters inside one -- ``\frac`` is a command and
#: ``\FRAC`` is nothing -- so case folding has to step over these.
_CONTROL_WORD = re.compile(r"\\[A-Za-z]+")


def _fold_case(s: str) -> str:
    r"""Upper-case everything except LaTeX control words.

    Folding case lets ``x=5`` and ``X=5``, or a multiple-choice ``a`` and
    ``A``, count as one vote. Applying it to the whole string instead turns
    ``\frac{1}{2}`` into ``\FRAC{1}{2}``, which no symbolic parser accepts:
    :func:`normalise` and :func:`equivalent` then disagree about the same
    answer, and ``is_correct`` reports False on a problem ``covered`` reports
    True on. That regression shipped once; this is the guard against it.
    """
    out, last = [], 0
    for m in _CONTROL_WORD.finditer(s):
        out.append(s[last:m.start()].upper())
        out.append(m.group(0))
        last = m.end()
    out.append(s[last:].upper())
    return "".join(out)


def normalise(answer: str) -> str:
    """Canonical form so equivalent answers are counted as the same vote.

    ``"204"``, ``"204.0"`` and ``" 204 "`` all become ``"204"``. Digit-grouping
    commas are removed, so a gold written ``1,000`` matches an answer of
    ``1000``. Exact integer fractions are reduced, so ``2/4`` and ``1/2`` agree.
    Non-finite values are rejected: a trajectory that produced ``inf`` cannot
    win a vote, and neither can an empty one.
    """
    if answer is None:
        return ""
    s = str(answer).strip().rstrip(".")
    if not s:
        return ""

    s = _strip_thousands(s)

    # float() overflows to inf past about 1e308 and _numeric_key rejects
    # non-finite values, so a 400-digit integer normalised to "" and abstained
    # on an answer that was perfectly well formed. Digits are already exact.
    digits = s[1:] if s[:1] == "-" else s
    if digits.isdigit():
        return str(int(s))

    key = _numeric_key(s)
    if key is not None:
        return key

    m = _FRACTION.match(s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return ""
        g = math.gcd(abs(num), den)
        num, den = num // g, den // g
        return str(num) if den == 1 else f"{num}/{den}"

    return _fold_case(re.sub(r"\s+", "", s))


def _equivalent_uncached(a: str, b: str) -> bool:
    """Whether two answers should count as the same vote.

    Uses ``math-verify`` when available -- which recognises that ``0.5``,
    ``1/2`` and ``\\frac{1}{2}`` are one answer, not three -- and falls back to
    comparing :func:`normalise` keys otherwise.

    Note:
        math-verify's timeouts spawn worker processes, which fail noisily on
        some platforms, so they are disabled. Answers come out of a
        ``\\boxed{}`` and are short, so the pathological-input risk the timeout
        guards against does not really arise; if you are feeding this arbitrary
        text, that assumption no longer holds.
    """
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if _MATH_VERIFY is None:
        return False
    # Two distinct finite numbers are never the same answer, and this is by far
    # the common case. Short-circuiting here keeps a 200-problem sweep from
    # spending its whole budget in a symbolic parser that can only say "no".
    if _numeric_key(na) is not None and _numeric_key(nb) is not None:
        return False
    parse, verify = _MATH_VERIFY
    try:
        pa = parse(str(a), parsing_timeout=None)
        pb = parse(str(b), parsing_timeout=None)
        if not pa or not pb:            # nothing recognisable to compare
            return False
        return bool(verify(pa, pb, timeout_seconds=None))
    except _TransientVerifyError:       # never memoise a transient fault
        raise
    except Exception:                   # a malformed answer is simply not equal
        return False


class _TransientVerifyError(Exception):
    """A symbolic-backend fault that says nothing about the two answers.

    Kept distinct from a malformed answer so the memoising wrapper below never
    records "not equal" for the life of the process on the strength of one
    transient failure. Nothing raises it today; it exists so that marking a
    recoverable failure class is a one-line change rather than an argument
    about cache correctness.
    """


@functools.lru_cache(maxsize=100_000)
def _equivalent_cached(a: str, b: str) -> bool:
    return _equivalent_uncached(a, b)


def equivalent(a: str, b: str) -> bool:
    """Whether two answers should count as the same vote.

    Thin memoising wrapper over :func:`_equivalent_uncached`. Building the
    equivalence partition takes the transitive closure over every pair, and a
    resampled accuracy curve rebuilds it thousands of times over the same
    small set of answer strings, so without a cache the symbolic parser is
    called again and again on pairs it has already decided. The relation is
    symmetric, so the key is ordered to let ``(a, b)`` and ``(b, a)`` share an
    entry.

    ``equivalent`` is a pure function of its arguments -- it consults no state
    that can change between calls -- so caching cannot alter a result. Call
    ``equivalent.cache_clear()`` if you swap the symbolic backend at runtime.
    """
    if a is None or b is None:
        return False
    a, b = str(a), str(b)
    if a > b:
        a, b = b, a
    return _equivalent_cached(a, b)


equivalent.cache_clear = _equivalent_cached.cache_clear      # type: ignore[attr-defined]
equivalent.cache_info = _equivalent_cached.cache_info        # type: ignore[attr-defined]


def warn_if_no_math_verify() -> None:
    """Emit a one-line notice when symbolic comparison is unavailable."""
    if _MATH_VERIFY is None:
        warnings.warn(
            "bestofn: math-verify is not installed, so answers are compared "
            "textually. Fractions, radicals and symbolic answers that are "
            "written differently will be counted as different votes. "
            'Install with: pip install "bestofn[math]"',
            RuntimeWarning, stacklevel=2,
        )
