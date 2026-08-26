"""Best-of-N: inference-time compute for language models.

Sample N reasoning trajectories from a frozen model and select among them.
No weights are modified; all gains come from how the model is used.

    >>> from bestofn import BestOfN
    >>> engine = BestOfN("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", n=32)
    >>> r = engine.solve("What is 17 * 23?")
    >>> r.answer, r.effective_n, r.agreement
    ('391', 32, 0.94)

The one habit worth forming: before believing any selector, compare it against
``"random"``. Re-running a selector over an existing result is free.

    >>> r.select_with("random", seed=0)     # the baseline to beat
    >>> r.select_with("majority")

Technical report: https://doi.org/10.5281/zenodo.21936832

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""

from .engine import BestOfN, Result
from .extract import (equivalent, extract_boxed, extract_letter,
                      extract_number, get_extractor, have_math_verify,
                      normalise)
from .select import (SELECTORS, Sample, abstentions, agreement, coverage,
                     effective_n, select)

__version__ = "1.1.6"
__author__ = "Alejandro Areces Rivera"
__license__ = "Apache-2.0"

__all__ = [
    "BestOfN", "Result", "Sample",
    "select", "agreement", "coverage", "SELECTORS",
    "abstentions", "effective_n",
    "extract_boxed", "extract_number", "extract_letter",
    "get_extractor", "normalise", "equivalent", "have_math_verify",
    "__version__",
]
