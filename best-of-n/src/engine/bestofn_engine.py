"""End-to-end Best-of-N engine: sample N trajectories, then select.

Works with any causal language model on the Hugging Face Hub. Two backends:

    vllm          fast, batched, shares the prompt KV-cache across the N
                  samples. Strongly recommended, and required for large N.
    transformers  universal fallback; runs anywhere torch runs.

The backend is chosen automatically unless pinned with ``backend=``.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union

from .extract import get_extractor
from .select import Sample, agreement, coverage, select

__all__ = ["BestOfN", "Result"]

DEFAULT_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


@dataclass
class Result:
    """Outcome of one Best-of-N call."""

    answer: str
    samples: List[Sample] = field(default_factory=list)
    method: str = "majority"

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def agreement(self) -> float:
        """Fraction of samples agreeing with the modal answer."""
        return agreement(self.samples)

    @property
    def answers(self) -> List[str]:
        return [s.answer for s in self.samples]

    def covered(self, gold: str) -> bool:
        """Whether any sample reached ``gold`` (pass@N)."""
        return coverage(self.samples, gold)

    def select_with(self, method: str, gold: Optional[str] = None) -> str:
        """Re-run a different selector over the same samples, for free."""
        return select(self.samples, method, gold=gold)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Result(answer={self.answer!r}, n={self.n}, "
                f"method={self.method!r}, agreement={self.agreement:.2f})")


class BestOfN:
    """Best-of-N inference wrapper around a frozen language model.

    The model is never modified. All gains come from sampling N reasoning
    trajectories and selecting among them.

    Args:
        model: Hub id or local path of any causal LM.
        n: default number of trajectories. Override per call.
        temperature: sampling temperature. Above 0 is required -- at 0 all N
            samples are identical and Best-of-N reduces to a single sample.
        top_p: nucleus sampling threshold.
        max_tokens: generation cap per trajectory. Reasoning models need room;
            too low a cap truncates trajectories and silently costs accuracy.
        extractor: ``"boxed"``, ``"number"``, ``"letter"``, ``"regex"``, or a
            callable ``str -> str``.
        prompt_suffix: appended to every problem. Set ``""`` to disable.
        backend: ``"auto"``, ``"vllm"`` or ``"transformers"``.
        verifier: optional callable ``(problem, text) -> float`` returning
            P(correct). Enables ``method="verifier"``.
        **backend_kwargs: forwarded to the backend constructor
            (e.g. ``dtype``, ``gpu_memory_utilization``, ``max_model_len``).

    Example:
        >>> from bestofn import BestOfN
        >>> engine = BestOfN("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", n=32)
        >>> result = engine.solve("What is 17 * 23?")
        >>> result.answer
        '391'
    """

    def __init__(
        self,
        model: str,
        n: int = 8,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        extractor: Union[str, Callable[[str], str]] = "boxed",
        prompt_suffix: str = DEFAULT_SUFFIX,
        backend: str = "auto",
        verifier: Optional[Callable[[str, str], float]] = None,
        **backend_kwargs,
    ):
        if n < 1:
            raise ValueError("n must be >= 1")
        if temperature <= 0 and n > 1:
            raise ValueError(
                "temperature must be > 0 for Best-of-N: at temperature 0 all "
                "samples are identical, so sampling N of them gains nothing."
            )

        self.model_name = model
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.extract = get_extractor(extractor)
        self.prompt_suffix = prompt_suffix
        self.verifier = verifier
        self._backend_kwargs = backend_kwargs

        self.backend = self._resolve_backend(backend)
        self._llm = None
        self._tokenizer = None

    # -------------------------------------------------------------- backend

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        backend = (backend or "auto").lower()
        if backend not in ("auto", "vllm", "transformers"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend != "auto":
            return backend
        try:
            import vllm  # noqa: F401
            return "vllm"
        except ImportError:
            return "transformers"

    def _load(self) -> None:
        """Load the model on first use, so construction stays cheap."""
        if self._llm is not None:
            return

        if self.backend == "vllm":
            from vllm import LLM
            kwargs = {"model": self.model_name, "trust_remote_code": True}
            kwargs.update(self._backend_kwargs)
            self._llm = LLM(**kwargs)
        else:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # float16 is unsupported or pathologically slow for many CPU ops,
            # so only request it when there is a GPU to run it on.
            default_dtype = (torch.float16 if torch.cuda.is_available()
                             else torch.float32)
            # transformers >= 5 renamed torch_dtype -> dtype. The signature is
            # **kwargs in both, so dispatch on the version rather than inspect.
            major = int(str(transformers.__version__).split(".")[0] or 0)
            dtype_key = "dtype" if major >= 5 else "torch_dtype"

            kwargs = {dtype_key: default_dtype, "device_map": "auto"}
            kwargs.update(self._backend_kwargs)
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.model_name, **kwargs
            )
            self._llm.eval()

    # ------------------------------------------------------------ prompting

    def _build_prompt(self, problem: str) -> str:
        return problem + self.prompt_suffix

    # ----------------------------------------------------------- generation

    def _generate_vllm(self, problems: Sequence[str], n: int
                       ) -> List[List[tuple]]:
        """Return, per problem, a list of (text, mean_logprob)."""
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            logprobs=1,
        )
        conversations = [
            [{"role": "user", "content": self._build_prompt(p)}]
            for p in problems
        ]
        outputs = self._llm.chat(conversations, params)

        batch = []
        for out in outputs:
            per_problem = []
            for o in out.outputs:
                n_tok = len(o.token_ids)
                mean_lp = (
                    o.cumulative_logprob / n_tok
                    if o.cumulative_logprob is not None and n_tok
                    else None
                )
                per_problem.append((o.text, mean_lp))
            batch.append(per_problem)
        return batch

    def _generate_transformers(self, problems: Sequence[str], n: int
                               ) -> List[List[tuple]]:
        import torch

        batch = []
        for problem in problems:
            messages = [{"role": "user", "content": self._build_prompt(problem)}]
            try:
                inputs = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                )
            except Exception:                      # model without chat template
                inputs = self._tokenizer(
                    self._build_prompt(problem), return_tensors="pt"
                )
            inputs = {k: v.to(self._llm.device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = self._llm.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_tokens,
                    num_return_sequences=n,
                    pad_token_id=self._tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = out.sequences
            # Mean log-probability of the sampled continuation.
            logprobs = [None] * n
            if getattr(out, "scores", None):
                stacked = torch.stack(out.scores, dim=1)          # [n*b, T, V]
                logits = torch.log_softmax(stacked.float(), dim=-1)
                gen = sequences[:, prompt_len:]
                steps = min(gen.shape[1], logits.shape[1])
                if steps > 0:
                    picked = logits[:, :steps, :].gather(
                        2, gen[:, :steps].unsqueeze(-1)
                    ).squeeze(-1)
                    mask = (gen[:, :steps] != self._tokenizer.eos_token_id).float()
                    denom = mask.sum(dim=1).clamp(min=1)
                    logprobs = ((picked * mask).sum(dim=1) / denom).tolist()

            texts = self._tokenizer.batch_decode(
                sequences[:, prompt_len:], skip_special_tokens=True
            )
            batch.append(list(zip(texts, logprobs)))
        return batch

    # ---------------------------------------------------------------- solve

    def solve(self, problem: str, n: Optional[int] = None,
              method: str = "majority", gold: Optional[str] = None) -> Result:
        """Solve one problem with Best-of-N.

        Args:
            problem: the problem statement.
            n: number of trajectories; defaults to the constructor value.
            method: selector -- ``"majority"``, ``"self_certainty"``,
                ``"verifier"``, ``"verifier_argmax"`` or ``"oracle"``.
            gold: reference answer, only for ``method="oracle"``.

        Returns:
            A :class:`Result`. Every sample is retained, so other selectors can
            be applied afterwards with :meth:`Result.select_with` at no cost.
        """
        return self.solve_batch([problem], n=n, method=method,
                                golds=[gold] if gold is not None else None)[0]

    def solve_batch(self, problems: Sequence[str], n: Optional[int] = None,
                    method: str = "majority",
                    golds: Optional[Sequence[str]] = None) -> List[Result]:
        """Solve several problems. Much faster than looping :meth:`solve`."""
        n = int(n or self.n)
        if n < 1:
            raise ValueError("n must be >= 1")
        if method == "verifier" and self.verifier is None:
            raise ValueError(
                "method='verifier' requires a verifier=... callable at "
                "construction time"
            )

        self._load()
        generate = (self._generate_vllm if self.backend == "vllm"
                    else self._generate_transformers)
        raw = generate(problems, n)

        results = []
        for i, (problem, per_problem) in enumerate(zip(problems, raw)):
            samples = []
            for text, mean_lp in per_problem:
                score = self.verifier(problem, text) if self.verifier else None
                samples.append(Sample(
                    answer=self.extract(text),
                    text=text,
                    logprob=mean_lp,
                    score=score,
                ))
            gold = golds[i] if golds else None
            results.append(Result(
                answer=select(samples, method, gold=gold),
                samples=samples,
                method=method,
            ))
        return results

    def __repr__(self) -> str:  # pragma: no cover
        return (f"BestOfN(model={self.model_name!r}, n={self.n}, "
                f"backend={self.backend!r}, temperature={self.temperature})")
