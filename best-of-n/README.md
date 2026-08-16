<div align="center">

<img src="https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/interlace-logo.png" width="96" alt="Interlace AI">

**INTERLACE&nbsp;AI**

# Best-of-N

### Inference-time compute for any language model

**Sample N reasoning trajectories from a frozen model and select the best one.**
No weights are modified. All gains come from *how* the model is used.

Works with **any causal LM** and **any N** — you configure both.

[![PyPI](https://img.shields.io/pypi/v/bestofn?color=blue)](https://pypi.org/project/bestofn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936833.svg)](https://doi.org/10.5281/zenodo.21936833)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-45%20passing-brightgreen)](tests/test_selectors.py)

```bash
pip install bestofn
```

<img src="https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/terminal_bestofn.gif" width="760" alt="Best-of-N in the terminal: 16 samples, Canberra x12 vs Sydney x4">

</div>

---

## What it does

A language model does not produce *an* answer. It produces a **distribution over
answers**, and generating text draws one sample from it. Ask the same question
twice at non-zero temperature and you can get two different results.

That variability is usually treated as a nuisance. Best-of-N treats it as a
resource.

Suppose a model solves a given problem correctly 30% of the time. Ask once and
you are right 30% of the time. Ask 32 times and the probability that **at least
one** attempt is correct is `1 − 0.7³² ≈ 99.99%`. The knowledge is there; a
single sample just fails to retrieve it reliably.

So the problem splits in two:

| | |
|---|---|
| **Coverage** | Did any of the N attempts get it right? `1 − (1−p)^N` |
| **Selection** | Did we manage to pick that one out of the N? |

This library implements both halves: sampling N trajectories, and four different
ways of choosing between them.

![A 1.5B model against the field](figures/04_vs_other_models.png)

Against published results from other laboratories. Ours is majority vote at
N=128 — the figure a system can actually return; the others are their published
pass@1. Coverage is shown separately and labelled as a ceiling.

![AIME results](figures/01_aime_base_vs_bestofn.png)

Measured on `DeepSeek-R1-Distill-Qwen-1.5B`, **AIME 2024**, weights untouched:

| N | Majority vote | Coverage (pass@N) |
|---:|---:|---:|
| 1 | 23.3% | 23.3% |
| 8 | 40.0% | 60.0% |
| 32 | 50.0% | 73.3% |
| **128** | **53.3%** | **83.3%** |

![Across benchmarks](figures/02_three_benchmarks.png)

And across domains:

| Benchmark | N | Single sample | Majority vote | Δ |
|---|---:|---:|---:|---:|
| AIME 2024 | 128 | 23.3% | **53.3%** | +30.0 |
| GPQA-Diamond | 32 | 33.8% | **43.4%** | +9.6 |
| GSM8K | 4 | 87.2% | **92.8%** | +5.6 |

All three are majority vote — the selector you would actually deploy. AIME
coverage at N=128 is higher (83.3%) but is a ceiling, not a returnable result.

### It transfers to models it was never built for

Everything above was measured on one reasoning-distilled model. So we ran it
again from scratch on a completely different one — `Qwen2.5-0.5B-Instruct`, a
general instruct model, a third of the size, on GSM8K:

![A different model entirely](figures/05_otro_modelo_gsm8k.png)

| N | Majority vote | ± sd | Coverage |
|---:|---:|---:|---:|
| 1 | 38.2% | 2.31 | 38.2% |
| 4 | 45.5% | 2.04 | 60.0% |
| 8 | 50.9% | 1.62 | 70.1% |
| **16** | **53.3%** | **1.01** | 79.5% |

**+15.1 points**, on one consumer GPU, weights untouched. Note the standard
deviation falling from 2.31 to 1.01 as N grows: Best-of-N does not only raise
accuracy, it makes the system **more predictable**.

Every trajectory behind that table is published in
[`results/generalisation/`](https://github.com/voidlinestudios12-jpg/Interlace-AI/tree/main/best-of-n/results/generalisation),
with the script that produced it.

### How a call works

1. **Generate.** N reasoning trajectories are sampled in parallel from the frozen
   model at `temperature > 0`, so each explores a different path.
2. **Extract.** The final answer is pulled out of each trajectory (by default the
   last `oxed{...}`) and normalised, so `"204"`, `"204.0"` and `" 204 "` count
   as the same answer.
3. **Select.** A selector decides which answer to return — by frequency, by the
   model's own confidence, or by an external verifier's score.
4. **Return.** You get the answer plus every sample, so you can inspect them or
   apply a different selector for free.

---

## Quickstart

```python
from bestofn import BestOfN

engine = BestOfN("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", n=32)

result = engine.solve("What is the remainder when 7^100 is divided by 13?")
print(result.answer)
print(result.agreement)     # how much the 32 samples agreed
```

Any model, any N:

```python
BestOfN("Qwen/Qwen2.5-Math-7B-Instruct", n=64)
BestOfN("meta-llama/Llama-3.1-8B-Instruct", n=16)
BestOfN("/path/to/your/local/model", n=128)
```

Install:

```bash
pip install bestofn
```

Plus a backend to run the model with:

```bash
pip install torch transformers   # works everywhere
pip install vllm                 # recommended; required for large N
```

---

## Why voting works

**Correct answers agree with each other. Wrong answers scatter.**

A wrong trajectory has a thousand different ways to be wrong and picks a
different one each time, so the errors split into singletons. The correct
answer is the only thing multiple trajectories can converge on. That is why
counting votes is enough to find an answer only a small minority reached.

From the published AIME trajectories, one problem at a time:

| Correct trajectories | Competing wrong answers | Majority vote returns |
|---:|---:|---|
| 5 of 32 | 25 different ones | **correct** |
| 7 of 32 | 11 different ones | **correct** |
| 9 of 32 | 17 different ones | **correct** |
| 13 of 32 | 14 different ones | **correct** |

Five correct out of thirty-two, against twenty-five rival answers, and the
vote still returns the right one. Every row is in
[`tests/measured_aime_n32.jsonl`](tests/measured_aime_n32.jsonl) — check them.

## Going further: the verifier

Voting is the free option: it needs nothing but the samples you already paid
for. It also has a natural limit, since it can only return what most
trajectories agree on.

![Headroom above majority voting](figures/03_selection_gap.png)

So we trained a **verifier** — a model that scores each trajectory on its own
merits instead of counting votes, and can therefore promote a correct answer
the majority missed. Here is what it buys:

| Selector (N=32, 90 AIME problems) | Accuracy |
|---|---:|
| Self-certainty | 18.9% |
| Majority vote | 35.6% |
| Verifier — argmax trajectory | 43.3% |
| **Verifier — confidence-weighted vote** | **52.2%** |

```python
def my_verifier(problem: str, trajectory: str) -> float:
    return probability_it_is_correct        # 0.0 – 1.0

engine = BestOfN(model, n=32, verifier=my_verifier)
engine.solve(problem, method="verifier")     # +16.6 points over majority
```

<img src="https://huggingface.co/InterlaceAI/best-of-n/resolve/main/figures/verificador_bestofn.gif" width="760" alt="Plugging a verifier into Best-of-N: three lines, +16.6 points">

**+16.6 points over majority voting**, on top of everything Best-of-N already
gave you — and still without touching a single weight of the base model.

---

## Selectors

| Method | Needs | Best for |
|---|---|---|
| `majority` | nothing | The default. Free, robust, no setup. |
| `self_certainty` | log-probs (automatic) | When trajectories rarely agree on anything. |
| **`verifier`** | a verifier callable | **Highest accuracy. Promotes minority-correct answers.** |
| `verifier_argmax` | a verifier callable | Single best trajectory rather than a weighted vote. |
| `oracle` | the gold answer | Measuring your headroom during development. |

Generation is the expensive part — reusing the samples with another selector is free:

```python
r = engine.solve(problem, n=32)
r.answer                          # majority
r.select_with("self_certainty")
r.covered(gold)                   # was the answer reachable at all? (pass@N)
```

---

## Configuration

```python
BestOfN(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    n=32,                  # your compute budget
    temperature=0.6,       # must be > 0
    top_p=0.95,
    max_tokens=8192,
    extractor="boxed",     # boxed | number | letter | regex | your callable
    backend="auto",        # auto | vllm | transformers
    verifier=None,
)
```

**Full documentation: [USAGE.md](USAGE.md)** — how to choose N, how to plug in a
verifier, how to measure the selection gap on your own task, and the failure
modes worth knowing about.

---

## Verification

The published numbers are not asserted, they are replayed. This repository ships
the trajectories that produced them and a script that feeds them back through
the library:

```bash
python tests/verify_against_measured.py
```

```
Samples per problem (N)          : 32
  majority selection reproduced  : 30/30  (100.0%)
  majority vote, this library    : 63.3%   [recorded run: 63.3%]
  coverage pass@32                 : 76.7%
  gain from Best-of-N            : +31.7 points
PASS - reproduces the published measurements exactly
```

> **A note on which number we quote.** We ran this benchmark twice. AIME has
> only 30 problems, so a single run carries a standard error of about 9 points,
> and our two runs came out at 63.3% and 50.0% at N=32. **The tables above quote
> the lower one.** This script replays the higher one, whose trajectories are
> shipped in `tests/` so you can check both. Reporting the conservative figure
> is deliberate: it is the number we are willing to defend.

Plus 45 unit tests covering extraction, normalisation, every selector, the
minority-rescue mechanism and error handling — no GPU required:

```bash
python tests/test_selectors.py     # 45 passed, 0 failed
```

---

## Where it works best

Best-of-N pays off most on:

- **Tasks with one comparable final answer** — mathematics, multiple choice,
  short factual questions, unit-testable code.
- **Models that are sometimes right.** The gain is largest when per-sample
  accuracy sits in the middle, around 20–60%. That is where most small models
  live on hard tasks.
- **Anywhere accuracy is worth more than latency.** Cost scales linearly with N,
  so you are trading compute for correctness — deliberately.

It is not the right tool for open-ended text such as essays or chat, since
there is no well-defined vote over free prose, and it cannot invent knowledge
the model does not have: if the model never reaches the answer, no selector
can return it.

Point it at your own task and it will tell you which case you are in:

```python
results = engine.solve_batch(problems, n=32)
selected  = sum(r.answer == g for r, g in zip(results, golds)) / len(golds)
reachable = sum(r.covered(g)  for r, g in zip(results, golds)) / len(golds)
print(f"returned {selected:.1%} · reachable {reachable:.1%}")
```

If `reachable` is well above `returned`, a verifier will pay for itself. If
`reachable` is low, you want a stronger base model rather than more samples.

---

## Built on solid ground

Inference-time compute is one of the most active lines of work in the field,
and this implementation sits squarely inside it:

- Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021 — verifier reranking of N samples
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*, 2022 — majority voting over samples
- Lightman et al., *Let's Verify Step by Step*, 2023 — process vs outcome supervision
- Snell et al., *Scaling LLM Test-Time Compute Optimally*, 2024 — compute vs parameters
- Brown et al., *Large Language Monkeys*, 2024 — coverage scaling with repeated sampling

What is published here that usually is not: a working library you can install
in one command, a trained verifier, and **every raw trajectory behind every
number above** — so the results can be reproduced rather than taken on trust.

Full discussion and references in [the technical report](https://doi.org/10.5281/zenodo.21936833).

---

## Citation

```bibtex
@techreport{arecesrivera2026interlace,
  title  = {Modern Architecture On Advanced LLM: Best-of-N Sampling,
            Learned Verification and Tree Search as a Substitute for Parameter Scale},
  author = {Areces Rivera, Alejandro},
  year   = {2026},
  number = {TR-2026-01},
  institution = {Interlace AI},
  doi    = {10.5281/zenodo.21936833}
}
```

Technical report: [doi.org/10.5281/zenodo.21936833](https://doi.org/10.5281/zenodo.21936833)
Code and raw outputs: [github.com/.../Interlace-AI/best-of-n](https://github.com/voidlinestudios12-jpg/Interlace-AI/tree/main/best-of-n)

---

## License

**Apache License 2.0** — free to use, modify and redistribute, including
commercially. Use it with any model, at any N, in any project.

Copyright 2026 Alejandro Areces Rivera — Interlace AI

Questions and collaboration: `interlaceIA@gmail.com`
