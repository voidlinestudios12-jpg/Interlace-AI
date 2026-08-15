"""Answer extraction and normalisation.

Turning a free-form reasoning trajectory into a comparable answer token is the
step where naive implementations lose most of their accuracy: an extraction
failure is indistinguishable from a reasoning failure at the metric level.

Four extractors are provided, plus support for any user-supplied callable.

    boxed   -- last \\boxed{...}, bracket-balanced, then numeric canonicalisation
    number  -- last number in the text
    letter  -- last standalone A/B/C/D (multiple choice)
    regex   -- first capture group of a user pattern

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional, Union

__all__ = ["extract_boxed", "extract_number", "extract_letter",
           "normalise", "get_extractor", "Extractor"]

Extractor = Callable[[str], str]

# LaTeX wrappers stripped before numeric parsing.
_LATEX_NOISE = (
    r"\text", r"\mathrm", r"\mathbf", r"\rm", r"\displaystyle",
    r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ",
    r"\$", r"\%", r"$", "%", ",", " ",
)


def _strip_latex(s: str) -> str:
    """Remove LaTeX decoration, keeping the mathematical content."""
    s = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^}]*)\}", r"\1", s)
    for tok in _LATEX_NOISE:
        s = s.replace(tok, "")
    return s.replace("\\", "").strip()


def extract_boxed(text: str) -> str:
    """Content of the LAST ``\\boxed{...}``, handling nested braces.

    A regex cannot do this correctly: ``\\boxed{\\frac{1}{2}}`` needs brace
    counting. Falls back to :func:`extract_number` when no box is present.
    """
    if not text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return extract_number(text)

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

    inner = _strip_latex("".join(out))
    m = re.search(r"-?\d+\.?\d*", inner)
    if m:
        return m.group(0)
    return inner or extract_number(text)


def extract_number(text: str) -> str:
    """Last number appearing in the text."""
    if not text:
        return ""
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
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
    """
    if callable(kind):
        return kind
    kind = str(kind).lower()
    if kind == "boxed":
        return extract_boxed
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


def normalise(answer: str) -> str:
    """Canonical form so equivalent answers are counted as the same vote.

    ``"204"``, ``"204.0"`` and ``" 204 "`` all become ``"204"``. Non-numeric
    answers are upper-cased and stripped. Non-finite values are rejected, so a
    trajectory that produced ``inf`` cannot win a vote.
    """
    if answer is None:
        return ""
    s = str(answer).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if not math.isfinite(f):
            return ""
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError, OverflowError):
        return s.upper()
