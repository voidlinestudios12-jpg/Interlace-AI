"""Adapters for using published reward models as Best-of-N verifiers.

Interlace AI does not ship a reward model. This module lets you plug in
somebody else's.

The contract a verifier must satisfy is deliberately narrow::

    verifier(problem: str, trajectory: str) -> float in [0, 1]

Almost every published reward model violates it out of the box, because they
emit **unbounded logits** rather than probabilities. Passing those straight to
``method="verifier"`` used to silently reduce it to a majority vote; it now
raises. The adapters here apply the sigmoid for you, so the value that reaches
the selector really is a probability.

    >>> from bestofn import BestOfN
    >>> from bestofn.verifiers import from_hub
    >>> v = from_hub("openbmb/Eurus-RM-7b")          # Apache-2.0
    >>> engine = BestOfN("your/model", n=16, verifier=v)
    >>> engine.solve(problem, method="verifier")

**Licensing is your responsibility, and it is not uniform.** A reward model's
licence governs how you may use its outputs, and several popular ones are more
restrictive than they look. :data:`KNOWN_VERIFIERS` records what was verified
on the Hub in August 2026, and :func:`license_of` queries the Hub live so you
never have to rely on a table going stale.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
import warnings
from typing import Callable, Dict, List, Optional, Sequence

__all__ = ["sigmoid", "RewardModelVerifier", "from_hub", "from_callable",
           "license_of", "KNOWN_VERIFIERS"]


#: Floats saturate: sigmoid(-800) rounds to exactly 0.0, and a pool of zeros
#: makes the weighted vote degenerate back into a majority vote -- the very
#: failure the score-range check exists to prevent. Clamping keeps the result
#: inside the open interval the contract promises.
_EPS = 1e-12


def sigmoid(x: float) -> float:
    """Map an unbounded reward-model logit onto ``(0, 1)``.

    Saturating tails are clamped away from the endpoints, so an extremely
    negative logit yields a very small weight rather than no weight at all.
    Non-finite input yields the neutral ``0.5`` rather than propagating a NaN
    into the vote.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(v):
        return 0.5 if math.isnan(v) else (1.0 - _EPS if v > 0 else _EPS)
    if v >= 0:
        s = 1.0 / (1.0 + math.exp(-v)) if v < 700 else 1.0
    else:
        e = math.exp(v) if v > -700 else 0.0
        s = e / (1.0 + e)
    return min(1.0 - _EPS, max(_EPS, s))


# --------------------------------------------------------------- known models

#: Licences read from the Hugging Face Hub on 16 August 2026. Verify before
#: relying on any of them: model cards change, and ``other`` always means the
#: terms live in the repository rather than in a standard identifier.
KNOWN_VERIFIERS: Dict[str, Dict[str, str]] = {
    "openbmb/Eurus-RM-7b": {
        "licence": "apache-2.0",
        "kind": "orm",
        "note": "Cleanest licence of the group. Free for commercial use.",
    },
    "OpenAssistant/reward-model-deberta-v3-large-v2": {
        "licence": "mit",
        "kind": "orm",
        "note": "Small and permissive. General preference model, not maths.",
    },
    "internlm/internlm2-1_8b-reward": {
        "licence": "other",
        "kind": "orm",
        "note": "1.8B, the smallest usable ORM here. Read the repository terms.",
    },
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B": {
        "licence": "llama3.1",
        "kind": "orm",
        "note": "Meta Llama 3.1 licence: acceptable-use policy and attribution "
                "requirements apply.",
    },
    "nvidia/Llama-3.1-Nemotron-70B-Reward-HF": {
        "licence": "llama3.1",
        "kind": "orm",
        "note": "Same Llama restrictions, and far too large for one consumer GPU.",
    },
    "Qwen/Qwen2.5-Math-PRM-7B": {
        "licence": "other",
        "kind": "prm",
        "note": "Step-level process reward model, not a drop-in ORM. Widely "
                "used for maths. Read the repository terms.",
    },
    "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B": {
        "licence": "other",
        "kind": "prm",
        "note": "1.5B PRM. Published benchmarks put it below chance on "
                "PRMBench, so measure it against `random` before trusting it.",
    },
}


def license_of(model_id: str, timeout: float = 15.0) -> Optional[str]:
    """Fetch a model's licence identifier from the Hub, or ``None``.

    Always prefer this over :data:`KNOWN_VERIFIERS`: it reflects the model card
    as it stands today rather than when this file was written.
    """
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bestofn"})
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            data = json.loads(fh.read().decode("utf-8"))
    except Exception:
        return None
    for tag in data.get("tags", []):
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


#: Seconds to wait on the Hub when checking a licence at construction time.
#: Building an adapter is documented as cheap, and a firewalled or offline box
#: used to stall 15 seconds per call before finding that out. Set
#: ``BESTOFN_NO_LICENCE_CHECK=1`` to skip the lookup entirely.
_LICENCE_TIMEOUT = 3.0


def _warn_licence(model_id: str) -> None:
    """Tell the user what they are agreeing to, once, at load time.

    The Hub lookup is best-effort and short. Building an adapter is documented
    as cheap and GPU-free, and a 15-second call to a host that may be
    unreachable does not fit that description: on a firewalled or offline
    machine it stalled the constructor before falling back to the local table
    anyway. Set ``BESTOFN_NO_LICENCE_CHECK=1`` to skip the network entirely and
    rely on :data:`KNOWN_VERIFIERS`.
    """
    live = (None if os.environ.get("BESTOFN_NO_LICENCE_CHECK")
            else license_of(model_id, timeout=_LICENCE_TIMEOUT))
    known = KNOWN_VERIFIERS.get(model_id, {})
    licence = live or known.get("licence") or "unknown"
    note = known.get("note", "")
    if licence in ("apache-2.0", "mit", "bsd-3-clause", "cc0-1.0"):
        return                              # permissive: nothing to flag
    warnings.warn(
        f"bestofn: reward model {model_id!r} is licensed as {licence!r}. "
        f"Its terms govern how you may use its scores, including "
        f"commercially. {note} Check the model card before deploying.",
        UserWarning, stacklevel=3,
    )


# ------------------------------------------------------------------ adapters

class RewardModelVerifier:
    """Wrap a Hugging Face sequence-classification reward model.

    Covers the standard outcome reward model interface: the model reads
    ``(problem, trajectory)`` as a chat exchange and emits one scalar. The
    scalar is a logit, so it is passed through a sigmoid before it reaches the
    selector.

    Args:
        model_id: Hub id of the reward model.
        device: ``"cuda"``, ``"cpu"`` or ``None`` to choose automatically.
        max_length: truncation length for the scored text.
        batch_size: trajectories scored per forward pass.
        dtype: torch dtype; defaults to bfloat16 on GPU, float32 on CPU.
        trust_remote_code: forwarded to ``from_pretrained``. Required by some
            reward models, and it does mean executing code from the repository.

    Note:
        Loading the model is deferred to the first call, so constructing an
        adapter is cheap and does not require a GPU.
    """

    def __init__(self, model_id: str, device: Optional[str] = None,
                 max_length: int = 4096, batch_size: int = 4,
                 dtype=None, trust_remote_code: bool = False):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.batch_size = max(1, int(batch_size))
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None
        _warn_licence(model_id)

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dtype is None:
            self.dtype = (torch.bfloat16 if self.device == "cuda"
                          else torch.float32)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code,
            num_labels=1,
        ).to(self.device).to(self.dtype)
        self._model.eval()

    def _encode(self, problem: str, trajectory: str) -> str:
        """Render one exchange the way reward models expect to see it."""
        messages = [{"role": "user", "content": problem},
                    {"role": "assistant", "content": trajectory}]
        try:
            return self._tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            return f"{problem}\n\n{trajectory}"

    def score_batch(self, problem: str,
                    trajectories: Sequence[str]) -> List[float]:
        """Score several trajectories at once. Returns probabilities."""
        import torch

        self._load()
        texts = [self._encode(problem, t) for t in trajectories]
        out: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            enc = self._tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            for v in logits.reshape(logits.shape[0], -1)[:, 0].float().tolist():
                out.append(sigmoid(v) if math.isfinite(v) else 0.0)
        return out

    def __call__(self, problem: str, trajectory: str) -> float:
        """Score one trajectory. Returns P(correct) in ``[0, 1]``."""
        return self.score_batch(problem, [trajectory])[0]

    def __repr__(self) -> str:  # pragma: no cover
        return f"RewardModelVerifier({self.model_id!r})"


def from_hub(model_id: str, **kwargs) -> RewardModelVerifier:
    """Build a verifier from any sequence-classification reward model.

    >>> v = from_hub("openbmb/Eurus-RM-7b")
    >>> engine = BestOfN("your/model", n=16, verifier=v)

    Emits a warning if the model's licence is anything other than permissive,
    because that licence governs what you may do with its scores.
    """
    return RewardModelVerifier(model_id, **kwargs)


def from_callable(fn: Callable[[str, str], float],
                  already_probability: bool = False
                  ) -> Callable[[str, str], float]:
    """Adapt your own scoring function to the verifier contract.

    Args:
        fn: any ``(problem, trajectory) -> float``.
        already_probability: set ``True`` if ``fn`` already returns ``[0, 1]``.
            Otherwise the output is treated as a logit and squashed, which is
            almost always what you want with a reward model.

    >>> verifier = from_callable(my_reward_model)        # logits -> sigmoid
    >>> verifier = from_callable(my_prob_fn, already_probability=True)
    """
    if already_probability:
        def _checked(problem: str, trajectory: str) -> float:
            v = float(fn(problem, trajectory))
            if not math.isfinite(v):
                return 0.0
            if not 0.0 <= v <= 1.0:
                raise ValueError(
                    f"verifier returned {v!r}, which is outside [0, 1]. Either "
                    f"clamp it, or drop already_probability=True so it is "
                    f"passed through a sigmoid."
                )
            return v
        return _checked

    def _squashed(problem: str, trajectory: str) -> float:
        v = float(fn(problem, trajectory))
        return sigmoid(v) if math.isfinite(v) else 0.0
    return _squashed
