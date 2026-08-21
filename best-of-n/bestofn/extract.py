"""Answer extraction and normalisation.

Turning a free-form reasoning trajectory into a comparable answer token is the
step where naive implementations lose most of their accuracy: an extraction
failure is indistinguishable from a reasoning failure at the metric level.

Extractors provided, plus support for any user-supplied callable:

    boxed   -- last \\boxed{...}, bracket-balanced, LaTeX canonicalised
    number  -- last number in the text
    letter  -- last standalone A/B/C/D (multiple choice)
    regex   -- first capture group of a user pattern

Two design rules, both learned the hard way:

1. **Never invent an answer.** A trajectory with no recoverable answer returns
   ``""`` and abstains from the vote. Guessing at the last number in an
   unfinished chain of thought produces a phantom vote that is worse than no
   vote, because it is indistinguishable from a real one.
2. **Never destroy structure while cleaning.** Commas separate tuple elements
   as often as they group thousands, so they are handled with context rather
   than deleted.

Equivalence is decided in :func:`normalise`. If ``math-verify`` is installed
(``pip install "bestofn[math]"``) it is used for symbolic comparison, so
``1/2``, ``0.5`` and ``\\frac{2}{4}`` count as the same vote. Without it, a
conservative textual canonicalisation is used.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
from __future__ import annotations

import logging
import math
import re
import warnings
from typing import Callable, Optional, Union

__all__ = ["extract_boxed", "extract_number", "extract_letter",
           "normalise", "get_extractor", "Extractor", "have_math_verify"]

Extractor = Callable[[str], str]


# --------------------------------------------------------------- math-verify

def _load_math_verify():
    """Return math-verify's (parse, verify) pair, or ``None`` if unavailable.

    Optional on purpose: the selectors must run with zero dependencies so the
    package stays installable anywhere, but symbolic equivalence is strictly
    better than string matching when it is available.

    Its own timeout machinery is disabled at the call site (it spawns worker
    processes, which fail on some platforms), and the resulting per-call notice
    is silenced here so it does not flood the caller's logs.
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

# Purely decorative tokens. Note what is NOT here: commas, spaces and braces,
# all of which carry meaning that deleting them would destroy.
_LATEX_NOISE = (
    r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ",
    r"\$", r"\%", r"\quad", r"\qquad", r"\displaystyle", r"\rm",
    "$", "%",
)

# 1,000 or 1{,}000 -> 1000, but only when the commas really do group digits.
_THOUSANDS = re.compile(r"^(-?\d{1,3})((?:\{?,\}?\d{3})+)(\.\d+)?$")

_TEXTUAL = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|operatorname|bm)\s*\{([^{}]*)\}"
)
# A whole answer written as prose -- \boxed{\text{Canberra}} -- versus a unit
# trailing a value -- \boxed{204\text{ km}}. The first is the answer; the
# second is decoration, and keeping it would split the vote between "204" and
# "204km".
_ONLY_TEXT = re.compile(
    r"^\s*\\(?:text|mathrm|mathbf|mathit|operatorname|bm)\s*\{([^{}]*)\}\s*$"
)
_WORDS_ONLY = re.compile(r"^[A-Za-z\s.\-]+$")


def _strip_thousands(s: str) -> str:
    """Remove digit-grouping commas, leaving every other comma intact."""
    m = _THOUSANDS.match(s.replace(" ", ""))
    if not m:
        return s
    head, groups, frac = m.group(1), m.group(2), m.group(3) or ""
    return head + re.sub(r"[{},]", "", groups) + frac


def _latex_to_text(s: str) -> str:
    """Canonicalise LaTeX into a compact, comparable text form.

    ``\\frac{1}{2}`` becomes ``1/2`` rather than ``1``; ``2\\sqrt{5}`` becomes
    ``2*sqrt(5)`` rather than ``2``. Structure is preserved, which is the whole
    point -- collapsing them onto their first digit makes different answers
    compare equal and corrupts the vote.
    """
    if not s:
        return ""

    whole = _ONLY_TEXT.match(s)
    if whole:                           # the prose IS the answer
        return re.sub(r"\s+", " ", whole.group(1)).strip()

    # Otherwise any all-alphabetic \text{...} is a unit riding along with a
    # value, and gets dropped; anything else is unwrapped and kept.
    s = _TEXTUAL.sub(
        lambda m: "" if _WORDS_ONLY.match(m.group(1) or " ") else m.group(1), s
    )

    for _ in range(4):                      # resolve nesting, bounded
        new = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
                     r"(\1)/(\2)", s)
        new = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}",
                     r"(\2)**(1/(\1))", new)
        new = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", new)
        if new == s:
            break
        s = new

    s = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^(-?\w)", r"**\1", s)
    s = re.sub(r"_\s*\{([^{}]*)\}", r"_\1", s)

    for name in ("pi", "theta", "alpha", "beta", "gamma", "phi", "lambda",
                 "mu", "sigma", "omega", "infty", "cdot", "times", "div"):
        s = s.replace("\\" + name, {"cdot": "*", "times": "*", "div": "/",
                                    "infty": "inf"}.get(name, name))

    for tok in _LATEX_NOISE:
        s = s.replace(tok, "")

    s = s.replace("\\", "").replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)
    return _strip_thousands(s.strip())


# A parenthesis that follows an identifier is a function call -- sqrt(5), f(x)
# -- and must survive. Only the ones this module itself introduced when
# rewriting \frac and \sqrt are redundant.
_REDUNDANT_PARENS = re.compile(r"(?<![A-Za-z0-9_])\((-?\d+(?:\.\d+)?)\)")


def _atomic(inner: str) -> bool:
    """Whether ``inner`` needs no parentheses to keep its meaning.

    True for a single term such as ``5`` or ``sqrt(3)``; false as soon as a
    top-level operator or comma appears, because ``(1+2)/3`` is not ``1+2/3``.
    """
    if not inner:
        return False
    depth = 0
    for ch in inner:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth < 0:
                return False
        elif depth == 0 and ch in "+-*/,^":
            return False
    return depth == 0


def _tidy_parens(s: str) -> str:
    """Drop the redundant parentheses ``_latex_to_text`` introduces.

    Deliberately conservative: tuple parentheses in ``(3,4)`` and call
    parentheses in ``sqrt(5)`` carry meaning, and removing them would make
    different answers compare equal.
    """
    prev = None
    while prev != s:
        prev = s
        s = _REDUNDANT_PARENS.sub(r"\1", s)
        s = _strip_atomic_parens(s)
    return s


def _strip_atomic_parens(s: str) -> str:
    """Remove one layer of parentheses around any single, complete term."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "(" and (i == 0 or s[i - 1] not in "abcdefghijklmnopqrstuvwxyz"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"):
            depth, j = 1, i + 1
            while j < len(s) and depth:
                depth += (s[j] == "(") - (s[j] == ")")
                j += 1
            if depth == 0:
                inner = s[i + 1:j - 1]
                if _atomic(inner):
                    out.append(inner)
                    i = j
                    continue
        out.append(s[i])
        i += 1
    return "".join(out)


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

    inner = _tidy_parens(_latex_to_text("".join(out)))
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
    if not nums:
        return ""
    return nums[-1].replace(",", "")


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


def normalise(answer: str) -> str:
    """Canonical form so equivalent answers are counted as the same vote.

    ``"204"``, ``"204.0"`` and ``" 204 "`` all become ``"204"``. Exact integer
    fractions are reduced, so ``2/4`` and ``1/2`` agree. Non-finite values are
    rejected: a trajectory that produced ``inf`` cannot win a vote, and neither
    can an empty one.

    With ``math-verify`` installed, symbolic equivalence is used instead, which
    additionally makes ``0.5``, ``\\frac{1}{2}`` and ``1/2`` agree.
    """
    if answer is None:
        return ""
    s = str(answer).strip().rstrip(".")
    if not s:
        return ""

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

    return re.sub(r"\s+", "", s).upper()


def equivalent(a: str, b: str) -> bool:
    """Whether two answers should count as the same vote.

    Uses ``math-verify`` when available -- which recognises that ``0.5``,
    ``1/2`` and ``\\frac{1}{2}`` are one answer, not three -- and falls back to
    comparing :func:`normalise` keys otherwise.

    Note:
        math-verify's timeouts are implemented with worker processes, which
        fail noisily on some platforms. They are disabled here. Answers come
        out of a ``\\boxed{}`` and are short, so the pathological-input risk
        the timeout guards against does not really arise; if you are feeding
        this arbitrary text, that assumption no longer holds.
    """
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if _MATH_VERIFY is None:
        return False
    parse, verify = _MATH_VERIFY
    try:
        pa = parse(str(a), parsing_timeout=None)
        pb = parse(str(b), parsing_timeout=None)
        if not pa or not pb:            # nothing recognisable to compare
            return False
        return bool(verify(pa, pb, timeout_seconds=None))
    except Exception:                   # a malformed answer is simply not equal
        return False


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
