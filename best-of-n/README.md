<div align="center">

<img src="https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/interlace-logo.png" width="96" alt="Interlace AI">

**INTERLACE&nbsp;AI**

# Best-of-N

### Sample a frozen model N times, and find out whether selection is your problem

[![PyPI](https://img.shields.io/pypi/v/bestofn?color=blue)](https://pypi.org/project/bestofn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936832.svg)](https://doi.org/10.5281/zenodo.21936832)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-155%20passing-brightgreen)](tests/test_selectors.py)

```bash
pip install bestofn
```

<img src="https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/terminal_bestofn.gif" width="760" alt="Best-of-N in the terminal">

</div>

---

## The question this answers

Your small model gets a problem wrong. There are two possible reasons, and they
need opposite fixes:

- **It never knew the answer.** Sampling more will not help. You need a
  different model.
- **It found the answer and you didn't return it.** Sampling more *will* help,
  and so will a better way of choosing.

Most tooling cannot tell you which one you are looking at. This library
measures both halves separately, on your own task, and tells you which:

```python
from bestofn import BestOfN

engine = BestOfN("Qwen/Qwen2.5-0.5B-Instruct", n=16)
r = engine.solve(problem)

r.answer            # what the system returns
r.is_correct(gold)  # did it match? (equivalence-aware, unlike ==)
r.covered(gold)     # whether any of the 16 trajectories found it
r.effective_n       # how many actually voted -- often fewer than you paid for
```

| returned | reachable | Diagnosis | Fix |
|:---:|:---:|---|---|
| ✗ | ✗ | Generation problem | Better model or prompt. More samples will not help |
| ✗ | ✓ | **Selection problem** | A verifier can recover it |
| ✓ | ✓ | Working | Ask whether you need this much N |

Everything else in this library exists to make those three numbers trustworthy.

---

## Quickstart

```python
from bestofn import BestOfN

engine = BestOfN("Qwen/Qwen2.5-0.5B-Instruct", n=16)
r = engine.solve("A train travels at 60 km/h for 3 hours. How far does it go?")

r.answer          # '180'
r.agreement       # 0.81   how much the voters agreed
r.effective_n     # 13     how many of the 16 produced a usable answer
r.n_truncated     # 3      how many ran out of tokens
```

Works with any causal LM, at any N:

```python
BestOfN("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", n=32)
BestOfN("meta-llama/Llama-3.1-8B-Instruct", n=16)
BestOfN("/path/to/your/local/model", n=64)
```

---

## Measured results

`Qwen2.5-0.5B-Instruct` on **GSM8K**, 200 problems, 16 trajectories each,
weights frozen, one RTX 3060. Every figure below is recomputed from the
published trajectories by `scripts/analyse.py`, which re-runs extraction over
the raw reasoning text.

| N | random | majority | 95% CI | coverage |
|---:|---:|---:|---:|---:|
| 1 | 36.1% | 36.1% | [31.0, 44.0] | 36.1% |
| 2 | 40.4% | 40.6% | [30.5, 44.0] | 47.2% |
| 4 | 42.4% | 47.1% | [39.0, 52.5] | 57.3% |
| 8 | 43.2% | 52.8% | [48.5, 62.0] | 66.8% |
| **16** | 43.5% | **54.8%** | [49.0, 62.5] | 74.5% |

![GSM8K, and how much of the gain is actually selection](https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/06_gsm8k_11.png)

Majority voting at N=16: **54.8%**. Against random
selection at the same N, exact McNemar gives **p = 4.2 × 10⁻⁷** — majority wins
on 32 problems where random loses, and loses on 3.

### Where the gain actually comes from

This is the part most reports leave out, and it changes how the headline should
be read:

| | | |
|---|---:|---:|
| N=1, a single sample | 36.1% | |
| N=16, **random** among the trajectories that answered | 43.5% | **+7.3** |
| N=16, **majority vote** | 54.8% | **+11.3** |
| | | **+18.7 total** |

Random selection improves with N *without selecting anything*, because with
more trajectories there is almost always one that did not abstain. **Of the
18.7 points, 7.4 are simply avoiding abstention and only 11.3 come from
voting.**

Comparing a single-sample baseline against an N-sample system counts the first
part as if it were method. It is not, and this library reports the two
separately.

### The accounting nobody publishes

| | |
|---|---|
| Trajectories generated | 3,200 |
| Cast a vote | 2,281 — **71.3%** |
| Abstained | 919 — 28.7% |
| Truncated at the token limit | 768 — 24.0% |
| Tokens generated | 985,545 |
| Re-extraction drift on replay | **0** |

**Effective N is 11.4, not 16.** Nearly a third of the compute produced nothing
usable. In version 1.0.0 those 919 trajectories voted anyway, with whatever
number happened to appear last in an unfinished chain of thought.

Coverage at N=16 is 74.5% against 54.8% returned. Some of that 19.7-point
distance is a selection problem a verifier could attack; some of it is single
correct answers among sixteen, which no selector can identify without the
label.

---

## The habit this library tries to build

**Compare every selector against random.**

```python
r = engine.solve(problem, n=16)

r.select_with("random", seed=0)     # the baseline
r.select_with("majority")           # is it actually better?
```

Re-running a selector over an existing result costs nothing — generation is the
expensive part. And a selector that cannot beat picking a trajectory at random
is not selecting: it is adding noise, and it should be removed rather than
tuned.

This matters more than it sounds. Published implementations of confidence-based
selection routinely land *below* the random baseline on small models, because
mean token log-probability rewards fluent, repetitive text — which correlates
with being cut off mid-reasoning, not with being right. `random` is one line
and it catches that immediately.

| Method | Needs | What it is for |
|---|---|---|
| **`random`** | nothing | **The baseline. Print it every time** |
| `majority` | nothing | The sensible default |
| `self_certainty` | `logprobs=True` | Fluency, not correctness. Verify before trusting |
| `verifier` | a verifier callable | The only one that can promote a minority answer |
| `verifier_argmax` | a verifier callable | Single best trajectory, no vote |
| `oracle` | the gold answer | Diagnostic ceiling only |

---

## Using a reward model

**This project does not ship a reward model.** It works with published ones,
and it makes them work correctly:

```python
from bestofn import BestOfN
from bestofn.verifiers import from_hub

verifier = from_hub("openbmb/Eurus-RM-7b")          # Apache-2.0
engine = BestOfN("your/model", n=16, verifier=verifier)
engine.solve(problem, method="verifier")
```

Reward models emit **unbounded logits**, not probabilities, and weighting a vote
by `-3.7` is meaningless. Passing raw logits raises a clear error instead of
silently degrading to a majority vote — which is what a naive implementation
does, leaving you convinced you are running your verifier when you are not.

<img src="https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/verifier_bestofn.gif" width="760" alt="Plugging a reward model into Best-of-N">

Licences are not uniform and they govern how you may use the scores. `from_hub`
warns on anything non-permissive, and `verifiers.license_of(model_id)` checks
the Hub live. See [USAGE.md](USAGE.md#using-someone-elses-reward-model) for the
table.

> Worth knowing before you invest in one: **no small discriminative reward model
> is known to work well.** `Skywork-o1-Open-PRM-1.5B` scores below chance on
> PRMBench in its own published evaluation. Measure any verifier against
> `random` on your task before trusting it.

---

## What it does not do

- **It cannot invent knowledge.** If the model never reaches the answer,
  `1 − (1−p)^N` is zero for every N. Check `covered()` before spending compute.
- **Coverage is not headroom.** The gap between coverage and accuracy looks like
  available improvement, but the problems sustaining its tail are single
  correct answers among N — never the mode, and indistinguishable from noise
  without the label. Some of that gap is structurally unreachable.
- **Cost is linear in N.** N=128 means 128 generations. This is compute traded
  for accuracy, deliberately, and the trade is only worth it when your GPU
  would otherwise be idle.
- **It is not for open-ended text.** There is no well-defined vote over an
  essay.
- **`n=2` is a waste.** No majority exists with two trajectories; ties break
  arbitrarily. Start at 8.

---

## Reproducing every number here

```bash
python scripts/analyse.py     # no GPU needed
```

The published trajectory file contains the **complete reasoning text** of every
sample, with `finish_reason`, log-probabilities and token counts — not answers
that were extracted earlier. `analyse.py` re-runs extraction over that raw text,
so if extraction breaks, the numbers move and the check fails.

That distinction is the point. A replay that re-counts pre-extracted answers
cannot detect an extraction bug, and extraction is where these systems actually
go wrong.

To regenerate from scratch:

```bash
python scripts/run_gsm8k.py   # ~3 hours on an RTX 3060
```

---

## Built on existing work

Inference-time compute is one of the most active areas in the field, and this
sits inside it rather than beside it:

- Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021 — verifier reranking of N samples
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*, 2022 — majority voting
- Lightman et al., *Let's Verify Step by Step*, 2023 — process vs outcome supervision
- Snell et al., *Scaling LLM Test-Time Compute Optimally*, 2024 — compute versus parameters
- Brown et al., *Large Language Monkeys*, 2024 — coverage scaling with repeated sampling

Related tooling worth knowing about, because you may want it instead:
[`lighteval`](https://github.com/huggingface/lighteval) has `pass@k`, `maj@n`
and `avg@n` as first-class metrics if you need evaluation rather than
inference; [`its_hub`](https://github.com/Red-Hat-AI-Innovation-Team/its_hub)
implements a wider set of search strategies including particle filtering and
beam search.

What this adds: the selection layer the serving stacks deliberately leave out
(vLLM removed `best_of` in 2025, SGLang discourages `n>1`, LMDeploy supports
only 1), a single small API over six interchangeable selectors, and the raw
trajectories behind every published number.

---

## Citation

```bibtex
@software{arecesrivera2026bestofn,
  title  = {Best-of-N: inference-time compute for language models},
  author = {Areces Rivera, Alejandro},
  year   = {2026},
  doi    = {10.5281/zenodo.21936832},
  url    = {https://github.com/voidlinestudios12-jpg/Interlace-AI}
}
```

---

## License

**Apache License 2.0** — free to use, modify and redistribute, including
commercially.

Copyright 2026 Alejandro Areces Rivera — Interlace AI

Questions and collaboration: `interlaceIA@gmail.com`
Changes in this release: [CHANGELOG.md](CHANGELOG.md)
