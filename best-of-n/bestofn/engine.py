"""End-to-end Best-of-N engine: sample N trajectories, then select.

Works with any causal language model on the Hugging Face Hub. Two backends:

    vllm          fast, batched, shares the prompt KV-cache across the N
                  samples. Strongly recommended, and required for large N.
    transformers  universal fallback; runs anywhere torch runs.

The backend is chosen automatically unless pinned with ``backend=``.

A note on log-probabilities, because it is the one place the two backends can
disagree. Asking ``transformers`` for per-token scores materialises a tensor of
shape ``[n, tokens, vocab]``; at n=8, 8192 tokens and a 152k vocabulary that is
roughly 20 GB before any arithmetic. This engine therefore:

* does not request them unless ``logprobs=True``;
* generates in sub-batches of ``max_parallel`` and reduces each one to a scalar
  per sequence before the next, so peak memory is bounded by the sub-batch;
* uses raw logits rather than the temperature- and top-p-processed scores, so
  the number means the same thing as vLLM's.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union

from .extract import get_extractor, warn_if_no_math_verify
from .select import (Sample, abstentions, agreement, coverage, effective_n,
                     select)

__all__ = ["BestOfN", "Result"]

DEFAULT_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


@dataclass
class Result:
    """Outcome of one Best-of-N call.

    Beyond the chosen answer, this carries the accounting needed to tell a
    genuine result from an artefact: how many trajectories actually voted, how
    many were cut off, and how many tokens it cost.
    """

    answer: str
    samples: List[Sample] = field(default_factory=list)
    method: str = "majority"

    @property
    def n(self) -> int:
        """Trajectories requested and paid for."""
        return len(self.samples)

    @property
    def effective_n(self) -> int:
        """Trajectories that actually cast a vote.

        Below :attr:`n` whenever extraction failed or a trajectory was cut off.
        The difference is compute spent for nothing, and every per-sample
        statistic should be computed against this number rather than ``n``.
        """
        return effective_n(self.samples)

    @property
    def n_abstained(self) -> int:
        """Trajectories with no usable answer."""
        return abstentions(self.samples)

    @property
    def n_truncated(self) -> int:
        """Trajectories that hit the token limit instead of finishing."""
        return sum(1 for s in self.samples if s.truncated)

    @property
    def total_tokens(self) -> Optional[int]:
        """Generated tokens across all trajectories, if the backend reports it."""
        counts = [s.n_tokens for s in self.samples if s.n_tokens is not None]
        return sum(counts) if counts else None

    @property
    def agreement(self) -> float:
        """Fraction of *voting* samples agreeing with the modal answer."""
        return agreement(self.samples)

    @property
    def answers(self) -> List[str]:
        return [s.answer for s in self.samples]

    def is_correct(self, gold: str) -> bool:
        """Whether the selected answer matches ``gold``.

        Use this rather than ``result.answer == gold``. With symbolic
        comparison available, a pool that agreed on ``0.5`` is a correct answer
        to a gold of ``1/2``, and a plain string comparison would score it
        wrong while :meth:`covered` scored it right -- inflating exactly the
        coverage-to-accuracy gap this library exists to measure.
        """
        from .extract import equivalent
        return bool(self.answer) and equivalent(self.answer, gold)

    def covered(self, gold: str) -> bool:
        """Whether any sample reached ``gold`` (pass@N).

        A diagnostic ceiling, not a bound on what a selector can reach.
        """
        return coverage(self.samples, gold)

    def select_with(self, method: str, gold: Optional[str] = None,
                    seed: Optional[int] = None) -> str:
        """Re-run a different selector over the same samples, for free.

        Generation is the expensive part; every selector in
        :data:`~bestofn.select.SELECTORS` can be applied to an existing result
        at no additional cost. Comparing against ``"random"`` is the cheapest
        sanity check there is.
        """
        return select(self.samples, method, gold=gold, seed=seed)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Result(answer={self.answer!r}, n={self.n}, "
                f"effective_n={self.effective_n}, method={self.method!r}, "
                f"agreement={self.agreement:.2f})")


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
            too low a cap truncates trajectories, which costs accuracy twice --
            once by losing the answer, and again by depressing the baseline
            you are comparing against. Check :attr:`Result.n_truncated`.
        extractor: ``"boxed"``, ``"number"``, ``"letter"``, ``"regex"``, or a
            callable ``str -> str``.
        prompt_suffix: appended to every problem. Set ``""`` to disable.
        backend: ``"auto"``, ``"vllm"`` or ``"transformers"``.
        verifier: optional callable ``(problem, text) -> float`` returning
            P(correct) in ``[0, 1]``. Enables ``method="verifier"``. See
            :mod:`bestofn.verifiers` for ready-made adapters around published
            reward models.
        logprobs: collect per-token log-probabilities. Required by
            ``self_certainty``, off by default because on the ``transformers``
            backend it is the single most memory-hungry part of generation.
        max_parallel: how many trajectories to generate at once on the
            ``transformers`` backend. Lower it if you run out of memory;
            ``None`` picks a value from the model's vocabulary size.
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
        logprobs: bool = False,
        max_parallel: Optional[int] = None,
        **backend_kwargs,
    ):
        _check_sampling(n, temperature)
        # Without math-verify, equivalent answers written differently vote as
        # separate blocs and the merge is a no-op. That degrades quietly -- the
        # run still finishes and still reports a number -- so say it out loud
        # once, here, rather than let someone publish the weaker result
        # believing the symbolic layer was on.
        warn_if_no_math_verify()

        self.model_name = model
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.extract = get_extractor(extractor)
        self.prompt_suffix = prompt_suffix
        self.verifier = verifier
        self.logprobs = bool(logprobs)
        self.max_parallel = max_parallel
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

    def _chunk_size(self, n: int) -> int:
        """Trajectories to generate at once without exhausting memory.

        Only the log-probability path is expensive: ``generate`` retains one
        ``[chunk, vocab]`` tensor per decoding step, so the allocation grows as
        ``chunk * max_tokens * vocab``. Sizing this from the memory actually
        free -- rather than a fixed constant -- matters in both directions: too
        large and it dies, too small and the GPU sits idle between launches.
        """
        if self.max_parallel:
            return max(1, int(self.max_parallel))
        if not self.logprobs:
            return n
        vocab = getattr(getattr(self._llm, "config", None), "vocab_size", 0)
        if not vocab:
            return max(1, min(n, 4))

        budget = 2 * 1024 ** 3                      # conservative default
        try:
            import torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                budget = max(budget, int(free * 0.45))
        except Exception:
            pass

        # bf16 during generation, float32 in the reduction, plus the block
        # this engine stacks: about six bytes per logit end to end.
        per_seq = max(1, self.max_tokens * vocab * 6)
        return max(1, min(n, budget // per_seq))

    # ------------------------------------------------------------ prompting

    def _build_prompt(self, problem: str) -> str:
        return problem + self.prompt_suffix

    # ----------------------------------------------------------- generation

    def _generate_vllm(self, problems: Sequence[str], n: int
                       ) -> List[List[dict]]:
        """Return, per problem, a list of per-trajectory dicts."""
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            logprobs=1 if self.logprobs else None,
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
                # Exclude the terminal EOS so this counts the same tokens as
                # the transformers path, which masks it out of both the sum and
                # the denominator. Without this the two backends report
                # different quantities under the same name.
                n_tok = len(o.token_ids)
                cum = o.cumulative_logprob
                if o.finish_reason == "stop" and n_tok:
                    n_tok -= 1
                    # cumulative_logprob still carries the EOS term. Dropping
                    # it from the denominator alone inflates the magnitude of
                    # the mean by roughly one token in n -- 12.5% at the
                    # 8-token answers this model produces -- and the two
                    # backends stop being comparable.
                    if cum is not None:
                        cum -= _eos_logprob(o)
                mean_lp = None
                if self.logprobs and cum is not None and n_tok:
                    mean_lp = cum / n_tok
                per_problem.append({
                    "text": o.text,
                    "logprob": mean_lp,
                    "finish_reason": o.finish_reason,
                    "n_tokens": n_tok,
                })
            batch.append(per_problem)
        return batch

    def _generate_transformers(self, problems: Sequence[str], n: int
                               ) -> List[List[dict]]:
        import torch

        eos = self._tokenizer.eos_token_id
        chunk = self._chunk_size(n)
        batch = []

        for problem in problems:
            inputs = self._encode(problem)
            prompt_len = inputs["input_ids"].shape[1]
            per_problem: List[dict] = []

            remaining = n
            while remaining > 0:
                k = min(chunk, remaining)
                remaining -= k

                with torch.no_grad():
                    out = self._llm.generate(
                        **inputs,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_new_tokens=self.max_tokens,
                        num_return_sequences=k,
                        pad_token_id=eos,
                        return_dict_in_generate=True,
                        # Raw, pre-processor logits. output_scores would give
                        # the temperature/top-p-truncated distribution, which
                        # is a different quantity from vLLM's and not
                        # comparable across backends.
                        output_logits=self.logprobs,
                    )

                gen = out.sequences[:, prompt_len:]
                lps = self._mean_logprobs(out, gen, eos) if self.logprobs \
                    else [None] * k
                texts = self._tokenizer.batch_decode(
                    gen, skip_special_tokens=True
                )

                for row, (text, lp) in enumerate(zip(texts, lps)):
                    ids = gen[row]
                    live = (ids != eos).sum().item() if eos is not None \
                        else ids.numel()
                    hit_cap = ids.numel() >= self.max_tokens and (
                        eos is None or ids[-1].item() != eos
                    )
                    per_problem.append({
                        "text": text,
                        "logprob": lp,
                        "finish_reason": "length" if hit_cap else "stop",
                        "n_tokens": int(live),
                    })

                del out                       # release before the next chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            batch.append(per_problem)
        return batch

    def _encode(self, problem: str):
        """Tokenise, preferring the model's chat template when it has one."""
        messages = [{"role": "user", "content": self._build_prompt(problem)}]
        try:
            inputs = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
        except Exception as exc:
            # Losing the chat template on an instruction-tuned model degrades
            # output badly, and the user would blame the model. Say so.
            warnings.warn(
                f"bestofn: could not apply the chat template for "
                f"{self.model_name!r} ({type(exc).__name__}: {exc}). Falling "
                f"back to the raw prompt. If this is an instruction-tuned "
                f"model, quality will be noticeably worse.",
                RuntimeWarning, stacklevel=3,
            )
            inputs = self._tokenizer(
                self._build_prompt(problem), return_tensors="pt"
            )
        return {k: v.to(self._llm.device) for k, v in inputs.items()}

    #: Decoding steps reduced per pass. Bounds peak memory at roughly
    #: ``block * k * vocab * 4`` bytes while keeping the number of kernel
    #: launches low enough that the GPU stays busy. Stepping one token at a
    #: time is correct but leaves the device idle waiting on Python.
    _LOGPROB_BLOCK = 64

    def _mean_logprobs(self, out, gen, eos) -> List[Optional[float]]:
        """Mean log-probability of each generated continuation.

        Reduced in blocks of decoding steps, so the full ``[k, T, V]`` tensor
        is never materialised and memory stays flat in ``max_tokens``.
        """
        import torch

        logits = getattr(out, "logits", None)
        if not logits:
            return [None] * gen.shape[0]

        steps = min(len(logits), gen.shape[1])
        total = torch.zeros(gen.shape[0], dtype=torch.float32,
                            device=gen.device)
        count = torch.zeros_like(total)

        for start in range(0, steps, self._LOGPROB_BLOCK):
            stop = min(start + self._LOGPROB_BLOCK, steps)
            block = torch.stack(logits[start:stop], dim=1).float()
            block = torch.log_softmax(block, dim=-1)
            tok = gen[:, start:stop]
            picked = block.gather(2, tok.unsqueeze(-1)).squeeze(-1)
            keep = (tok != eos) if eos is not None \
                else torch.ones_like(tok, dtype=torch.bool)
            # Padding positions carry -inf. Multiplying by a 0/1 mask gives
            # -inf * 0 = NaN, which poisons the mean; select instead.
            total += torch.where(keep, picked, torch.zeros_like(picked)).sum(1)
            count += keep.float().sum(1)
            del block, picked

        means = total / count.clamp(min=1)
        return [(float(v) if math.isfinite(float(v)) else None) for v in means]

    # ---------------------------------------------------------------- solve

    def solve(self, problem: str, n: Optional[int] = None,
              method: str = "majority", gold: Optional[str] = None,
              seed: Optional[int] = None) -> Result:
        """Solve one problem with Best-of-N.

        Args:
            problem: the problem statement.
            n: number of trajectories; defaults to the constructor value.
            method: any selector in :data:`~bestofn.select.SELECTORS`.
            gold: reference answer, only for ``method="oracle"``.
            seed: only for ``method="random"``.

        Returns:
            A :class:`Result`. Every sample is retained, so other selectors can
            be applied afterwards with :meth:`Result.select_with` at no cost.
        """
        return self.solve_batch([problem], n=n, method=method, seed=seed,
                                golds=[gold] if gold is not None else None)[0]

    def solve_batch(self, problems: Sequence[str], n: Optional[int] = None,
                    method: str = "majority",
                    golds: Optional[Sequence[str]] = None,
                    seed: Optional[int] = None) -> List[Result]:
        """Solve several problems. Much faster than looping :meth:`solve`."""
        n = int(n if n is not None else self.n)
        # Same guard as the constructor: n and temperature can both be
        # overridden here, and a per-call n>1 at temperature 0 would silently
        # generate N identical samples at N times the cost.
        _check_sampling(n, self.temperature)

        if method in ("verifier", "verifier_argmax") and self.verifier is None:
            raise ValueError(
                f"method={method!r} requires a verifier=... callable at "
                f"construction time. See bestofn.verifiers for adapters "
                f"around published reward models."
            )
        if method == "self_certainty" and not self.logprobs:
            raise ValueError(
                "method='self_certainty' requires log-probabilities. "
                "Construct with BestOfN(..., logprobs=True); they are off by "
                "default because collecting them is memory-hungry on the "
                "transformers backend."
            )

        self._load()
        generate = (self._generate_vllm if self.backend == "vllm"
                    else self._generate_transformers)
        raw = generate(problems, n)

        results = []
        for i, (problem, per_problem) in enumerate(zip(problems, raw)):
            texts = [row["text"] for row in per_problem]
            scores = self._score_all(problem, texts)
            samples = []
            for row, score in zip(per_problem, scores):
                text = row["text"]
                samples.append(Sample(
                    answer=self.extract(text),
                    text=text,
                    logprob=row.get("logprob"),
                    score=score,
                    finish_reason=row.get("finish_reason"),
                    n_tokens=row.get("n_tokens"),
                ))
            gold = golds[i] if golds else None
            results.append(Result(
                answer=select(samples, method, gold=gold, seed=seed),
                samples=samples,
                method=method,
            ))
        return results

    def _score_all(self, problem: str, texts: Sequence[str]) -> List[Optional[float]]:
        """Score every trajectory, in one batch when the verifier supports it.

        Calling the verifier once per trajectory means one forward pass each,
        which on a 7B reward model at N=16 is sixteen times slower than it
        needs to be.
        """
        if self.verifier is None:
            return [None] * len(texts)
        batch = getattr(self.verifier, "score_batch", None)
        if callable(batch):
            try:
                scores = list(batch(problem, list(texts)))
                if len(scores) != len(texts):
                    # The caller zips scores against samples. A short list
                    # would silently drop the tail of the pool -- the vote
                    # would run, return a plausible answer, and never say that
                    # it ignored trajectories. Refuse instead.
                    raise ValueError(
                        f"score_batch returned {len(scores)} scores for "
                        f"{len(texts)} trajectories; they must correspond "
                        f"one-to-one and in order."
                    )
                return scores
            except Exception as exc:
                warnings.warn(
                    f"bestofn: batched scoring failed ({type(exc).__name__}: "
                    f"{exc}); falling back to one call per trajectory.",
                    RuntimeWarning, stacklevel=3,
                )
        return [self.verifier(problem, t) for t in texts]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"BestOfN(model={self.model_name!r}, n={self.n}, "
                f"backend={self.backend!r}, temperature={self.temperature})")


def _eos_logprob(o) -> float:
    """Log-probability vLLM assigned to the terminal token of one output.

    ``cumulative_logprob`` sums every token the model emitted, the closing EOS
    included. The transformers path masks that token out of both the sum and
    the count, so to report the same quantity under the same name we have to
    take it back out here.

    Returns ``0.0`` when per-token log-probabilities were not requested, which
    leaves the previous behaviour rather than inventing a correction we cannot
    compute. The caller only reaches this line when ``logprobs`` is on, so in
    practice the fallback is unused.
    """
    lps = getattr(o, "logprobs", None)
    if not lps:
        return 0.0
    try:
        last = lps[-1]
        entry = last[o.token_ids[-1]]
        return float(getattr(entry, "logprob", entry))
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


def _check_sampling(n: int, temperature: float) -> None:
    """Validate the two parameters that make Best-of-N meaningful."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if temperature <= 0 and n > 1:
        raise ValueError(
            "temperature must be > 0 for Best-of-N: at temperature 0 all "
            "samples are identical, so sampling N of them costs N times as "
            "much and gains nothing."
        )
