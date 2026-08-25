<div align="center">

<img src="https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/interlace-logo.png" width="96" alt="Interlace AI">

**INTERLACE&nbsp;AI**

# Best-of-N

### Your model already knows more than it tells you

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

## The idea in one paragraph

A language model does not give you an answer. It gives you a **probability
distribution over answers**, and generating text draws one sample from it. Ask
the same question twice and you can get two different results. Most people
treat that as a defect.

We treat it as an untapped resource. Sample the same frozen model N times and
select well, and accuracy climbs sharply — **no training, no fine-tuning, no
new weights**. The knowledge was always in there. It just needed asking more
than once.

<div align="center">

| | single sample | **Best-of-128** |
|---|---:|---:|
| **GSM8K**, Qwen2.5-0.5B frozen | 41.8% | **65.0%** |

**+23.2 points. Nothing was trained.**

</div>

---

## Why voting works so well

Here is the asymmetry that makes the whole thing run, and it is more elegant
than it first looks:

> **Correct answers agree with each other. Wrong answers scatter.**

A wrong trajectory has a thousand different ways to be wrong and picks a
different one each time, so errors split into singletons. The correct answer is
the only thing several attempts can converge on together.

That is why counting votes recovers answers only a small minority reached. The
model does not need to be right most of the time — it needs to be right *more
consistently than it is wrong in any one particular way*.

---

## Quickstart

```python
from bestofn import BestOfN

engine = BestOfN("Qwen/Qwen2.5-0.5B-Instruct", n=16)
r = engine.solve("A train travels at 60 km/h for 3 hours. How far does it go?")

r.answer          # '180'
r.agreement       # 0.81   how strongly the trajectories agreed
r.effective_n     # 13     how many produced a usable answer
```

Works with any causal LM, at any N:

```python
BestOfN("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", n=32)
BestOfN("meta-llama/Llama-3.1-8B-Instruct", n=16)
BestOfN("/path/to/your/local/model", n=128)
```

Both backends are tested: `transformers` runs anywhere torch runs, and `vllm`
is dramatically faster for large N.

---

## Measured results

`Qwen2.5-0.5B-Instruct` on **GSM8K**, 200 problems, **128 trajectories each**,
weights frozen. Every figure is recomputed from the published trajectories by
`scripts/analyse.py`, which re-runs extraction over the raw reasoning text.

![GSM8K accuracy against N](https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/07_curve_n128.png)

| N | random | majority | 95% CI | coverage |
|---:|---:|---:|---:|---:|
| 1 | 41.8% | 41.8% | [39.5, 53.0] | 41.8% |
| 4 | 46.3% | 51.6% | [45.0, 59.0] | 62.2% |
| 8 | 46.7% | 56.5% | [49.0, 62.5] | 70.5% |
| 16 | 46.9% | 59.9% | [52.0, 66.0] | 77.6% |
| 32 | 46.6% | 62.1% | [57.0, 70.0] | 83.2% |
| 64 | 46.7% | 63.9% | [55.5, 68.5] | 87.8% |
| **128** | 46.9% | **65.0%** | [58.5, 71.5] | **91.0%** |

**41.8% to 65.0%** on a half-billion-parameter model, with the weights frozen
throughout. Against random selection at the same N, exact McNemar gives
**p = 5.7 × 10⁻⁸** — majority wins on 38 problems where random loses, and loses
on 4.

And **coverage reaches 91.0%**: on nine problems out of ten, this small model
does find the right answer somewhere in its 128 attempts. That is the number
that says how much is still on the table.

### How much of the gain is really selection

Most reports skip this, and it changes how the headline should be read:

| | | |
|---|---:|---:|
| N=1, a single sample | 41.8% | |
| N=128, **random** among the trajectories that answered | 46.9% | +5.1 |
| N=128, **majority vote** | 65.0% | **+18.1** |
| | | **+23.2 total** |

Random selection improves slightly with N without selecting anything, because
with more trajectories one of them usually did not abstain. Separating the two
shows that **the overwhelming majority of the gain — 18.1 of 23.2 points — is
genuine selection**, not an artefact of the comparison.

We report it this way because a single-sample baseline against an N-sample
system quietly folds the first part into the second.

### The accounting

| | |
|---|---|
| Trajectories generated | 25,600 |
| Cast a vote | 19,983 — 78.1% |
| Tokens generated | 7,773,981 |
| Re-extraction drift on replay | **0** |
| Generated on | one RTX 3090, vLLM backend, ~25 minutes |

All 25,600 trajectories are published in full.

---

## It tells you what to do next

This is the part no other library gives you. Beyond raising accuracy,
Best-of-N **measures the two halves of the problem separately** and tells you
which one you are actually facing:

```python
r = engine.solve(problem)

r.is_correct(gold)   # did the system return the right answer?
r.covered(gold)      # did any trajectory find it at all?
```

| returned | reachable | What it means | What to do |
|:---:|:---:|---|---|
| ✓ | ✓ | Working | You are done. Consider whether you need this much N |
| ✗ | ✓ | **The answer is in there** | A selection problem — a verifier recovers it |
| ✗ | ✗ | Not yet reachable | Raise N, improve the prompt, or use a stronger model |

Two numbers, one decision. Without them you are guessing whether to spend your
next hour on sampling or on selection.

The library also reports the accounting most tooling hides:

```python
r.effective_n     # trajectories that actually voted
r.n_abstained     # produced no usable answer
r.n_truncated     # ran out of tokens
r.total_tokens    # what it cost
```

---

## The habit worth building: always print the baseline

```python
r = engine.solve(problem, n=16)

r.select_with("random", seed=0)     # the honest baseline
r.select_with("majority")           # how much better is it, really?
```

Re-running a selector over an existing result is **free** — generation is the
expensive part — so there is no reason not to. Our own published tables carry
the random row, because a gain is only a gain relative to something.

| Method | Needs | What it is for |
|---|---|---|
| **`random`** | nothing | The baseline. Print it every time |
| **`majority`** | nothing | The default, and it is strong |
| `self_certainty` | `logprobs=True` | Weighs votes by the model's own confidence |
| **`verifier`** | a verifier callable | The one that can promote a minority answer |
| `verifier_argmax` | a verifier callable | Single best trajectory, no vote |
| `oracle` | the gold answer | Measures your headroom during development |

---

## Bring your own reward model

Best-of-N works with **any published reward model**, and it makes them work
correctly:

```python
from bestofn import BestOfN
from bestofn.verifiers import from_hub

verifier = from_hub("openbmb/Eurus-RM-7b")          # Apache-2.0
engine = BestOfN("your/model", n=16, verifier=verifier)
engine.solve(problem, method="verifier")
```

<img src="https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/verifier_bestofn.gif" width="760" alt="Plugging a reward model into Best-of-N">

Reward models emit **unbounded logits**, not probabilities, and a naive
implementation silently degrades to a plain majority vote when you hand it one
— leaving you convinced your verifier is running when it is not. This one
catches it, tells you exactly what to do, and the adapters apply the sigmoid
for you.

Licences differ and they govern how you may use the scores. `from_hub` flags
anything non-permissive, and `verifiers.license_of(model_id)` checks the Hub
live. The full table is in [USAGE.md](USAGE.md#using-someone-elses-reward-model).

---

## Everything here is checkable

```bash
python scripts/analyse.py     # no GPU needed
```

The published dataset contains the **complete reasoning text** of every
trajectory, with `finish_reason`, log-probabilities and token counts — not
answers extracted earlier by someone else. The analysis re-runs extraction over
that raw text, computes exact McNemar against the random baseline, and reports
bootstrap confidence intervals.

That means the numbers above are not asserted, they are **reproduced** — by
you, in about a minute, without hardware. Very little published work in this
area can say that.

To regenerate from scratch:

```bash
python scripts/run_gsm8k.py --backend vllm --n 128 --batch 25
```

---

## Where it shines

Best-of-N pays off most on:

- **Tasks with one comparable final answer** — mathematics, multiple choice,
  short factual questions, unit-testable code.
- **Models that are sometimes right.** The gain is largest when per-sample
  accuracy sits around 20–60%, which is exactly where small models live on
  hard problems.
- **Hardware you already own.** On your own GPU the extra samples cost you
  nothing but time you were not using. This is the one setting where the
  economics are simply free.

Practical guidance on choosing N, plugging in a verifier and measuring your own
task is in [USAGE.md](USAGE.md).

---

## Built on solid ground

Inference-time compute is one of the most active areas in the field, and this
sits squarely inside it:

- Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*, 2022
- Lightman et al., *Let's Verify Step by Step*, 2023
- Snell et al., *Scaling LLM Test-Time Compute Optimally*, 2024
- Brown et al., *Large Language Monkeys*, 2024

What this adds: the **selection layer the serving stacks deliberately leave
out** — vLLM removed `best_of` in 2025, SGLang discourages `n>1`, LMDeploy
supports only 1 — a single small API over six interchangeable selectors, and
the raw trajectories behind every number we publish.

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
Release notes: [CHANGELOG.md](CHANGELOG.md)
