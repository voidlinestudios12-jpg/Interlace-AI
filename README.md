<div align="center">

<img src="assets/interlace-logo.svg" width="88" alt="Interlace AI">

# Interlace AI

**Getting more out of the models people can actually run.**

</div>

---

## Who we are

Interlace AI is an independent research project working on one question:

> How much more can a small language model do, if you stop asking it once?

Frontier systems now run into the trillions of parameters. Training them, and
increasingly even serving them, is out of reach for almost everybody. We work
on the other side of that problem: getting more out of models that fit on one
consumer graphics card, and publishing everything so the results can be checked
rather than believed.

| | |
|---|---|
| **Open by default** | Apache 2.0. Free for research and commercial use, no strings |
| **Checkable** | Every number we publish ships with the raw data that produced it |
| **Accessible** | Techniques that work on one GPU, not on a cluster |

---

## Best-of-N

Our first release, and the idea behind it is simple enough to state in a
sentence:

> A language model does not give you an answer. It gives you a distribution
> over answers — and one sample is a poor way to read it.

Sample the same **frozen** model N times, select well, and accuracy climbs
sharply. No training, no fine-tuning, no new weights. The knowledge was always
in the model; it just needed asking more than once.

```bash
pip install bestofn
```

```python
from bestofn import BestOfN

engine = BestOfN("your/model", n=16)
r = engine.solve(problem)

r.answer             # what the system returns
r.is_correct(gold)   # was it right?
r.covered(gold)      # was it reachable at all?
```

<div align="center">

| | single sample | **Best-of-128** |
|---|---:|---:|
| **GSM8K**, Qwen2.5-0.5B frozen | 45.0% | **66.5%** |

**+21.5 points, and the weights were never touched.**

</div>

**→ [`best-of-n/`](best-of-n/) — code, measurements, raw trajectories and full documentation**

Also on **[PyPI](https://pypi.org/project/bestofn/)** ·
**[Hugging Face](https://huggingface.co/InterlaceAI/best-of-n)** ·
**[live demo](https://huggingface.co/spaces/InterlaceAI/best-of-n-demo)**

[![PyPI](https://img.shields.io/pypi/v/bestofn?color=blue)](https://pypi.org/project/bestofn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936832.svg)](https://doi.org/10.5281/zenodo.21936832)
[![License](https://img.shields.io/badge/code-Apache%202.0-green)](LICENSE)

---

## What makes it different

Plenty of projects sample a model N times. Three things here are ours:

**It tells you which problem you have.** Coverage and accuracy are measured
separately, so you know whether to invest in more sampling or in better
selection — a decision most tooling leaves you to guess at.

**Every selector is compared against random.** A selector that cannot beat
picking a trajectory at random is not selecting anything. Our tables carry that
row, and so should everyone's.

**Selection is invariant under permutation of the pool.** Shuffle the samples,
get the same answer. Symbolic equivalence is not transitive, so this takes a
transitive closure over the answer classes and a tie-break that does not depend
on arrival order — and you can see it in our published curve, where the spread
at exhausted N is exactly zero.

**We report how much of the gain is not the method.** With more samples, one of
them usually did not abstain, and that alone lifts a random pick. Here it is
1.3 of 21.5 points; we print it beside the headline instead of banking it.

**The raw trajectories are published.** Not a summary — the complete reasoning
text of every sample, with token counts, finish reasons and log-probabilities.
The analysis script re-extracts answers from that text, so it can catch a
parsing bug rather than inherit one. Anyone can reproduce every figure we
publish in about two minutes, with no GPU.

---

## What we are working on

**Nem** — a reasoning model of our own. In development, not yet released.

**Best-of-N v2** — extending inference-time compute to agentic tasks and code
generation, where a verifier can *run* the candidate instead of estimating
whether it works.

---

## Repository layout

```
best-of-n/
  bestofn/          The library: engine, extractors, selectors, verifier adapters
  scripts/          Run the benchmark, re-derive every published number
  results/          Complete reasoning trajectories behind the measurements
  tests/            175 tests, no GPU required
  figures/          Charts and the code that produces them
  CHANGELOG.md      What changed and why

assets/             Brand assets
LICENSE             Apache 2.0
```

---

## Licensing

**Apache License 2.0** — free to use, modify and redistribute, including
commercially.

Evaluation data: GSM8K is distributed by OpenAI under the MIT licence, so it is
freely redistributable. Our published trajectories contain the model's own
generated text alongside the problem statements.

---

## Contact

**Alejandro Areces Rivera** — Interlace AI

`interlaceIA@gmail.com`

Open to collaboration, review and questions.
