---
license: apache-2.0
library_name: bestofn
tags:
  - inference-time-compute
  - test-time-compute
  - best-of-n
  - reasoning
  - self-consistency
  - reward-model
  - evaluation
language:
  - en
---

<div align="center">

<img src="https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/interlace-logo.png" width="96" alt="Interlace AI">

**INTERLACE&nbsp;AI**

# Best-of-N

### Your model already knows more than it tells you

[![PyPI](https://img.shields.io/pypi/v/bestofn?color=blue)](https://pypi.org/project/bestofn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936832.svg)](https://doi.org/10.5281/zenodo.21936832)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-177%20passing-brightgreen)](tests/test_selectors.py)
[![Reproducible](https://img.shields.io/badge/every%20figure-reproducible%20in%202%20min-blueviolet)](results/)

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
| **GSM8K**, Qwen2.5-0.5B frozen | 45.0% | **66.5%** |

**+21.5 points. Nothing was trained.**

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
r.effective_n     # 15     how many produced a usable answer
```

Works with any causal LM, at any N:

```python
BestOfN("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", n=32)
BestOfN("meta-llama/Llama-3.1-8B-Instruct", n=16)
BestOfN("/path/to/your/local/model", n=128)
```

Both backends are tested on real hardware: `transformers` runs anywhere torch
runs, and `vllm` is dramatically faster at large N — it is the one the
measurements below were generated with.

---

## Measured results

`Qwen2.5-0.5B-Instruct` on **GSM8K**, 200 problems, **128 trajectories each**,
weights frozen. Every figure is recomputed from the published trajectories by
`scripts/analyse.py`, which re-runs extraction over the raw reasoning text.

![GSM8K accuracy against N](https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/07_curve_n128.png)

| N | random | majority | 95% CI | coverage |
|---:|---:|---:|---:|---:|
| 1 | 45.0% | 45.0% | [39.0, 52.0] | 45.1% |
| 4 | 46.5% | 53.0% | [44.0, 57.0] | 66.7% |
| 8 | 45.9% | 58.3% | [53.0, 66.5] | 75.0% |
| 16 | 46.2% | 61.6% | [55.5, 69.0] | 81.7% |
| 32 | 46.0% | 64.1% | [57.5, 70.5] | 86.7% |
| 64 | 46.2% | 65.4% | [59.0, 72.0] | 90.6% |
| **128** | 46.3% | **66.5%** | [60.5, 73.0] | **93.5%** |

**45.0% to 66.5%** on a half-billion-parameter model, with the weights frozen
throughout. Against random selection at the same N, exact McNemar gives
**p ≤ 1.4 × 10⁻⁵**, and the median across 200 random seeds is 1.5 × 10⁻⁹.

We quote the worst seed rather than the best. `random` draws a different
trajectory every time you run it, so a p-value taken from one seed is partly a
property of that seed: ours ranged from 2 × 10⁻¹⁴ to 1.4 × 10⁻⁵ depending on
which one we picked. The conclusion holds at every one of them.

And **coverage reaches 93.5%**: on more than nine problems out of ten, this
small model does find the right answer somewhere in its 128 attempts. That is
the number that says how much is still on the table.

### How much of the gain is really selection

Most reports skip this, and it is the part that decides whether a headline
means anything:

| | | |
|---|---:|---:|
| N=1, a single sample | 45.0% | |
| N=128, **random** among the trajectories that answered | 46.3% | +1.3 |
| N=128, **majority vote** | 66.5% | **+20.2** |
| | | **+21.5 total** |

Random selection improves slightly with N without selecting anything, because
with more trajectories one of them usually did not abstain. Separating the two
shows that **20.2 of the 21.5 points — 94% of the gain — is genuine
selection**, not an artefact of comparing a one-sample baseline against an
N-sample system.

We report it this way because that comparison quietly folds the first number
into the second, and the size of the fold is not knowable in advance. Here it
is small. It is small *because* the token budget lets trajectories finish; at a
tighter budget the same experiment would have credited five times as much of
the gain to the method.

### Every selector, against the baseline

| selector at N=128 | accuracy | 95% CI |
|---|---:|---:|
| `random` (mean of 200 seeds, sd 2.5) | 46.3% | [41.0, 55.0] |
| `majority` | **66.5%** | [60.5, 73.0] |
| `self_certainty` | 66.5% | [60.5, 73.0] |
| `oracle` (diagnostic ceiling) | 93.5% | [90.0, 96.5] |

### The accounting

| | |
|---|---|
| Trajectories generated | 25,600 |
| Cast a vote | 24,797 — 96.9% |
| Truncated at the token limit | 216 — 0.8% |
| Tokens generated | 8,434,157 |
| Re-extraction drift on replay | **0** |
| Generated on | one RTX 3090, vLLM backend, ~25 minutes |

All 25,600 trajectories are published in full.

---

## It tells you what to do next

Beyond raising accuracy, Best-of-N **measures the two halves of the problem
separately** and tells you which one you are actually facing. The framing is
not ours — `pass@k` beside `maj@n` appears in the evaluation harnesses and in
Snell et al. (2024). What is ours is that it takes one call, and that the
answer arrives as a decision rather than as two numbers to interpret:

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

## The same pool always gives the same answer

Shuffle the trajectory pool and every selector returns the same answer.
`random` is the exception, and only because picking at random is what it is
for.

That sounds like it should be free. It is not, and it took three separate
things to make true:

**The equivalence classes cannot be built greedily.** Symbolic equivalence is
not transitive — a parser accepts `0.3333333333` against `1/3`, and `1/3`
against `0.33333333333333`, while rejecting the two decimals against each
other. Compare each new answer against one representative per class and the
partition you get depends on which answer the model emitted first. We take the
transitive closure over all pairs instead.

**Weights cannot be accumulated in arrival order.** Float addition is not
associative, so two permutations of the same weights can differ in the last
bit — enough to flip a near-tie in a verifier-weighted vote. Totals are summed
with `math.fsum` over a sorted list, which is exactly rounded.

**Ties cannot break on whichever came first.** Every tie, in every selector,
resolves on the canonical key.

We check it two ways. Directly: shuffling all 200 published pools eight times
each moves **no** returned answer. And visibly, in the output — at N=128 the
resampled curve draws 128 trajectories from a pool of 128, so every draw is a
permutation of one pool and the reported spread is **0.00**. That second one is
a necessary condition rather than a proof, which is why the shuffle test exists
too.

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
you, in **about two minutes** on a laptop CPU, with no hardware. Very little
published work in this area can say that.

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
