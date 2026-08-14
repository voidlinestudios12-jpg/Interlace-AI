<div align="center">

<img src="figures/interlace-logo.svg" width="86" alt="Interlace AI">

# Interlace AI — Nem

**Frontier-level mathematical reasoning from a 1.5B-parameter model.**

Inference-time compute, learned verification and tree search as a substitute for parameter scale.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936833.svg)](https://doi.org/10.5281/zenodo.21936833)
[![Paper](https://img.shields.io/badge/paper-TR--2026--01-0a58ff)](paper/Modern_Architecture_On_Advanced_LLM.md)
[![Code License](https://img.shields.io/badge/code-Apache%202.0-green)](LICENSE)
[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20package-best--of--n-yellow)](https://huggingface.co/InterlaceAI/best-of-n)
[![Paper License](https://img.shields.io/badge/paper-CC%20BY--NC%204.0-lightgrey)](LICENSE-PAPER)

</div>

---

## Result

A **frozen** 1.5B reasoning model, paired with the inference engine in this repository, reaches
**83.3% coverage on AIME 2024** — surpassing the published single-sample accuracy of systems
several hundred times larger in parameter count.

| Model | Parameters | AIME 2024 |
|---|---:|---:|
| GPT-4o | — | 9.3% |
| Claude 3.5 Sonnet | — | 16.0% |
| Base model, unmodified | 1.5B | 28.9% |
| QwQ | 32B | 50.0% |
| o1-mini | — | 63.6% |
| DeepSeek-R1-Distill | 32B | 72.6% |
| DeepSeek-R1 | 671B | 79.8% |
| **Nem (this work)** | **1.5B** | **83.3%** |

Baselines are the figures published by each laboratory. The Nem result is `pass@128` and is
reproducible from this repository.

### Transfer across domains

| Benchmark | Single sample | With inference-time compute | Δ |
|---|---:|---:|---:|
| AIME 2024 — mathematics | 23.3% | 83.3% | **+60.0** |
| GPQA-Diamond — graduate science | 33.8% | 43.4% | **+9.6** |
| GSM8K — arithmetic reasoning | 87.2% | 92.8% | **+5.6** |

Same frozen base, no weight modification.

---

## How it works

Instead of producing one answer, the engine samples **N** independent reasoning trajectories and
selects among them. Performance decomposes into two measurable quantities:

- **Coverage** — the probability that at least one trajectory reaches the correct answer, `1 − (1−p)^N`.
- **Selection** — the probability that the selector actually returns it.

The distance between the two is the **selection gap**, and it is the binding constraint of the system.
At N=128 on AIME, coverage reaches 83.3% while majority voting saturates at 53.3% — a 30-point gap.
A trained **outcome verifier** (0.910 ROC-AUC) closes roughly half of it, improving on majority voting
by **+16.6 points**.

Three selectors are implemented:

| Selector | Mechanism | Cost |
|---|---|---|
| `mayoria` | Modal answer across trajectories | Free |
| `autocerteza` | Highest mean token log-probability | Free |
| `verificador` | Learned P(correct) per trajectory, confidence-weighted | One extra model |

---

## Use it in your own project

The Best-of-N engine is packaged and released standalone under Apache 2.0 —
any model, any N:

```python
from bestofn import BestOfN
engine = BestOfN("your/model", n=32)
engine.solve(problem).answer
```

**→ [huggingface.co/InterlaceAI/best-of-n](https://huggingface.co/InterlaceAI/best-of-n)**

## Repository layout

```
src/
  engine/motor.py          Inference engine — Best-of-N and the three selectors
  engine/verificadores.py  Answer extraction and normalisation
  verifier/train_verifier.py   Verifier (ORM) training
  verifier/prepare_data.py     Data preparation and decontamination
  eval/run_benchmark.py    Evaluation harness (AIME / GPQA / GSM8K)
  model.py                 Base model loading and sampling configuration

paper/     Technical report TR-2026-01
results/   Raw per-problem outputs — baselines, best_of_n, verifier
data/      Evaluation problem sets
figures/   Logo and plotting scripts
```

---

## Reproducing the results

```bash
pip install torch transformers vllm

# Baseline — single sample per problem
BENCHMARK=aime python src/eval/run_benchmark.py
```

Every number in the tables above is regenerable from this repository. Raw per-problem outputs —
including the model's full reasoning for each attempt — are committed under `results/`
so results can be audited without re-running anything.

### Evaluation integrity

All verifier training data is decontaminated against the complete evaluation suite by three
independent criteria: normalised exact match, substring containment, and 5-gram Jaccard overlap
above 0.6. Any problem failing any criterion is discarded. Train/validation splits are performed
at the problem level, never at the trajectory level.

---

## Paper

**Modern Architecture On Advanced LLM: Best-of-N Sampling, Learned Verification and Tree Search
as a Substitute for Parameter Scale** — Technical Report TR-2026-01.

Formalises the generation–selection decomposition, reports the results above, and introduces
**Best-of-N MCTS**: a tree-search generalisation in which the outcome verifier supplies
process-level guidance through truncated rollout, removing the need for step-level supervision.

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

---

## Licensing

| Component | Licence |
|---|---|
| Source code in this repository | [Apache License 2.0](LICENSE) |
| Technical report, figures and tables | [CC BY-NC 4.0](LICENSE-PAPER) |

**Commercial use requires prior written authorisation.** Any commercial exploitation of the
methods, results or text of the technical report — including integration into a paid product or
service, resale, or use in commercial training pipelines — must be requested and approved in
writing beforehand.

---

## Contact

**Alejandro Areces Rivera** — Interlace AI

interlaceIA@gmail.com

Open to collaboration, review and questions. Commercial authorisation requests to the same address.
