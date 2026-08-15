<div align="center">

<img src="assets/interlace-logo.svg" width="88" alt="Interlace AI">

# Interlace AI

**Open reasoning models that run on hardware people actually have.**

</div>

---

## Who we are

Interlace AI is an independent research project working on one question:

> How much of the reasoning ability of a frontier model can be recovered from a
> small one — openly, reproducibly, and on modest hardware?

Frontier reasoning systems now run into the trillions of parameters. Training
them, and increasingly even serving them, is out of reach for anyone without a
large compute budget. That excludes almost everybody.

We work on the other side of that problem: getting more out of small models, and
publishing everything so the results can be checked rather than believed.

## What we care about

| | |
|---|---|
| **Open by default** | Models and code released under permissive licences, free for research and commercial use. |
| **Verifiable** | Evaluation code and raw per-problem outputs are published. Every number we report can be reproduced or refuted. |
| **Accessible** | Techniques that work on one consumer GPU, not on a cluster. |

We would rather publish a smaller result that holds up than a larger one that
does not.

---

## What we have released

### Best-of-N — inference-time compute for any language model

Our first public release. Sample N reasoning trajectories from a **frozen**
model and select the best one. No weights are modified; all gains come from how
the model is used.

On AIME 2024, a frozen 1.5B model goes from **23.3%** (single sample) to
**53.3%** with majority voting at N=128 — the deployable figure. Its coverage,
the ceiling any selector could reach, is **83.3%**. It transfers to
graduate-level science and arithmetic reasoning without retraining.

```python
from bestofn import BestOfN

engine = BestOfN("your/model", n=32)
engine.solve(problem).answer
```

**→ [`best-of-n/`](best-of-n/) — code, technical report, raw outputs and full documentation**

Also on the Hub: **[huggingface.co/InterlaceAI/best-of-n](https://huggingface.co/InterlaceAI/best-of-n)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936833.svg)](https://doi.org/10.5281/zenodo.21936833)
[![Code](https://img.shields.io/badge/code-Apache%202.0-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-CC%20BY--NC%204.0-lightgrey)](LICENSE-PAPER)

### Technical report TR-2026-01

*Modern Architecture On Advanced LLM: Best-of-N Sampling, Learned Verification
and Tree Search as a Substitute for Parameter Scale.*

Formalises the generation–selection decomposition, identifies the **selection
gap** as the binding constraint of inference-time compute systems, and derives
Best-of-N MCTS.

**→ [`best-of-n/paper/`](best-of-n/paper/Modern_Architecture_On_Advanced_LLM.md)**

---

## What we are working on

**Nem** — a reasoning model of our own, still in development. Not yet released.

**Best-of-N v2** — extending inference-time compute to agentic tasks and code
generation, where a verifier can execute the candidate rather than estimate
whether it is correct.

---

## Repository layout

```
best-of-n/          Everything for the Best-of-N release
  src/              Inference engine, verifier training, evaluation harness
  paper/            Technical report TR-2026-01
  results/          Raw per-problem outputs — baselines, best_of_n, verifier
  data/             Evaluation problem sets
  figures/          Charts and plotting scripts

assets/             Brand assets
LICENSE             Apache 2.0 — source code
LICENSE-PAPER       CC BY-NC 4.0 — technical report
```

---

## Licensing

| Component | Licence |
|---|---|
| Source code | [Apache License 2.0](LICENSE) |
| Technical report, figures and tables | [CC BY-NC 4.0](LICENSE-PAPER) |

**Commercial use of the technical report requires prior written authorisation.**
The source code is free for commercial use under Apache 2.0.

### A note on evaluation data

GPQA is a gated dataset whose authors ask that questions not be republished in
plain text. Our GPQA result files therefore contain metrics only — predictions,
reference answers and correctness — with question statements and reasoning
traces removed. See [`best-of-n/results/baselines/NOTE_GPQA.md`](best-of-n/results/baselines/NOTE_GPQA.md).
AIME and GSM8K outputs are published in full.

---

## Contact

**Alejandro Areces Rivera** — Interlace AI

interlaceIA@gmail.com

Open to collaboration, review and questions. Commercial authorisation requests
to the same address.
