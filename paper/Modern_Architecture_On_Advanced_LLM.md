<div align="center">

<img src="../figures/interlace-logo.svg" width="64" alt="Interlace AI">

**INTERLACE AI** · Technical Report TR-2026-01

# Modern Architecture On Advanced LLM

### Best-of-N Sampling, Learned Verification and Tree Search as a Substitute for Parameter Scale

**Alejandro Areces Rivera**

*12 years · Interlace AI · 2026*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936833.svg)](https://doi.org/10.5281/zenodo.21936833)

</div>

---

## Abstract

The prevailing approach to capability in large language models couples performance to parameter
count. We present an alternative architecture in which a *frozen* 1.5B-parameter reasoning model is
paired with a structured inference-time compute layer, and show that the combination recovers a
substantial fraction of the capability normally attributed to scale. On AIME 2024, the frozen base
improves from 23.3% single-sample accuracy to 83.3% coverage at N=128; a trained outcome verifier
converts that coverage into 52.2% realised accuracy at N=32, a 16.6-point gain over majority voting.
We formalise the generation–selection decomposition, identify the **selection gap** as the governing
bottleneck of such systems, report transfer to graduate-level science and arithmetic reasoning, and
derive **Best-of-N MCTS** — a tree-search generalisation in which the outcome verifier supplies
process-level guidance through truncated rollout, removing the need for step-level supervision. All
results are reproducible from a public repository.

---

## Contents

1. [Introduction](#1-introduction)
2. [The Generation–Selection Decomposition](#2-the-generationselection-decomposition)
3. [System Architecture](#3-system-architecture)
4. [The Learned Verifier](#4-the-learned-verifier)
5. [Empirical Results](#5-empirical-results)
6. [Best-of-N MCTS](#6-best-of-n-mcts)
7. [Discussion](#7-discussion)
8. [Limitations](#8-limitations)
9. [Conclusion](#9-conclusion)

---

## 1. Introduction

Between 2023 and 2026 the parameter count of frontier reasoning systems grew by roughly three orders
of magnitude, from tens of billions into the trillion range. This trajectory is inaccessible to
independent researchers: training, and increasingly even serving, such models requires capital
expenditure available to a small number of laboratories. The question motivating this report is
therefore practical rather than theoretical: *how much frontier-level reasoning can be recovered from
a small model without modifying its weights?*

We argue the answer is considerably more than is generally assumed, provided the system is
restructured. Rather than treating a language model as a function mapping a problem to an answer, we
treat it as a *stochastic proposal distribution* over reasoning trajectories, and shift the burden of
correctness onto a separate selection mechanism. This decomposition is the core contribution of the
report, and everything that follows — the architecture of §3, the verifier of §4, and the tree search
of §6 — is a consequence of it.

Our contributions are:

1. A formal decomposition of inference-time compute into coverage and selection, with the selection
   gap identified as the binding constraint.
2. A complete architecture realising it on a frozen 1.5B base, with measured results on three
   benchmarks.
3. A trained outcome verifier that recovers roughly half the available selection gap.
4. Best-of-N MCTS, a tree-search generalisation that uses outcome supervision for process-level
   guidance.

---

## 2. The Generation–Selection Decomposition

Let *M* be a frozen autoregressive reasoning model and *x* a problem instance. Sampling *M* at
temperature *T* > 0 induces a distribution over complete reasoning trajectories *y*. We draw *N*
independent trajectories and write *a(y)* for the answer extracted from trajectory *y*. The system
output is determined by a selector *S* over the sampled set:

> **(1)**  ŷ = S( { y₁, …, y_N } ),  where  yᵢ ~ M( · | x, T )

This separates two quantities conflated in single-sample evaluation. The first is **coverage**: the
probability that at least one sampled trajectory reaches the correct answer *a\**. Writing *p* for the
per-sample success probability,

> **(2)**  Cov(N) = 1 − ( 1 − p )^N,  where  p = P_{y~M}[ a(y) = a\* ]

The second is **selection accuracy**: the probability that *S* returns the correct answer given that
it is present among the candidates. Realised accuracy is their product, and coverage upper-bounds any
selector:

> **(3)**  Acc(N) = Cov(N) · P[ S = a\* | a\* ∈ candidates ]  ≤  Cov(N)

### 2.1 The selection gap

We define the *selection gap* **G(N) = Cov(N) − Acc(N)**: the accuracy a perfect selector would
recover but the deployed selector does not. Equation (2) shows coverage saturating logarithmically in
N, so beyond a moderate budget additional samples contribute little. The gap, by contrast, does not
close on its own. It is therefore the correct optimisation target for any inference-time compute
architecture, and §5 confirms that it dominates system performance in practice.

---

## 3. System Architecture

The system comprises four stages around an unmodified base. The base remains frozen throughout: no
gradient reaches its weights. This guarantees the architecture cannot degrade the underlying model —
a failure mode we observed repeatedly when attempting supervised adaptation of the same base, where
an ill-chosen learning rate or a duplicated control token collapsed reasoning performance.

### 3.1 Diverse proposal generation

N trajectories are sampled in parallel with nucleus sampling (T = 0.6, top-p = 0.95), served through
a paged-attention engine so the N sequences share prefix key–value cache. Temperature is the control
variable of the coverage–consensus trade-off: low temperature concentrates probability mass on the
modal answer, raising consensus but lowering coverage; high temperature does the reverse.

### 3.2 Answer normalisation

Trajectories terminate in a delimited answer field. Extraction must be robust to nested LaTeX
environments, unit annotations and formatting variation, since a normalisation failure is
indistinguishable from a reasoning failure at the metric level. We use bracket-balanced parsing of the
terminal `\boxed{}` expression followed by numeric canonicalisation.

### 3.3 Selection

Three selectors of increasing sophistication are supported:

| Selector | Mechanism | Can recover a minority-correct answer? |
|---|---|:---:|
| Majority vote | Modal normalised answer | No |
| Self-certainty | Highest mean token log-probability | Rarely |
| Learned verifier | Estimates P(correct \| x, y) per trajectory | **Yes** |

Majority voting is structurally incapable of recovering a correct answer held by a minority of
trajectories. Self-certainty measures fluency rather than correctness. Only the learned verifier
attacks the selection gap directly.

---

## 4. The Learned Verifier

The verifier is an outcome reward model (ORM): a classification head over a frozen copy of the base,
adapted with low-rank matrices. Training pairs (*x*, *y*) are labelled by agreement with reference
answers, drawn exclusively from training-split problems.

### 4.1 Decontamination

Every candidate training problem is checked against the full evaluation suite by three independent
criteria: normalised exact match, substring containment, and Jaccard overlap of 5-gram shingles above
0.6. A problem failing any criterion is discarded. Given how easily contamination inflates apparent
performance, we regard this step as a precondition for reporting results at all, not an optimisation.

### 4.2 Training

Self-generated data is positively skewed, so the loss is class-weighted. The train/validation split is
performed at the *problem* level rather than the trajectory level, preventing trajectories of the same
problem from appearing on both sides. The resulting verifier reaches **0.910 ROC-AUC** and **84.1%
accuracy** on held-out problems.

### 4.3 Confidence-weighted aggregation

Rather than returning the single highest-scored trajectory, we aggregate votes weighted by verifier
confidence:

> **(4)**  â = argmax_a Σ_{i : a(yᵢ) = a} V( x, yᵢ )

```python
# Confidence-weighted vote — the strongest selector measured
scores = defaultdict(float)
for y in trajectories:
    scores[normalise(answer(y))] += verifier.p_correct(x, y)
return max(scores, key=scores.get)
```

This outperforms taking the argmax trajectory: it retains the variance reduction of aggregation while
allowing a confident minority to overturn an unconfident majority. Empirically the difference is
substantial — 52.2% versus 43.3% at N = 32 (Table 2).

---

## 5. Empirical Results

All measurements use a 1.5B-parameter frozen reasoning base. Scaling results are reported on AIME 2024
(30 problems); verifier evaluation uses the combined AIME 2023–2025 sets (90 problems) for statistical
reliability.

**Table 1 — AIME 2024: coverage scales with N while majority voting saturates.**

| N | Majority vote | Coverage (pass@N) | Selection gap |
|---:|---:|---:|---:|
| 1 | 23.3% | 23.3% | — |
| 8 | 40.0% | 60.0% | 20.0 pts |
| 16 | 46.7% | 63.3% | 16.7 pts |
| 32 | 50.0% | 73.3% | 23.3 pts |
| 64 | 50.0% | 80.0% | 30.0 pts |
| **128** | **53.3%** | **83.3%** | **30.0 pts** |

Two observations follow. First, coverage at N = 128 reaches 83.3%, exceeding the published
single-sample accuracy of reasoning systems several hundred times larger in parameter count. Second,
majority voting recovers less than two-thirds of that coverage: the ceiling of the system is set by
selection, not by generation.

**Table 2 — Selector comparison, 90 problems (AIME 2023–2025), N = 32.**

| Selector | Accuracy |
|---|---:|
| Self-certainty | 18.9% |
| Majority vote | 35.6% |
| Verifier — argmax trajectory | 43.3% |
| **Verifier — confidence-weighted vote** | **52.2%** |

The learned verifier improves on majority voting by **16.6 points**, converting roughly half the
available selection gap into realised accuracy.

**Table 3 — Transfer across domains. Same frozen base, no weight modification.**

| Benchmark | Single sample | With inference-time compute | Δ |
|---|---:|---:|---:|
| AIME 2024 — mathematics | 23.3% | 83.3% | **+60.0** |
| GPQA-Diamond — graduate science | 33.8% | 43.4% | **+9.6** |
| GSM8K — arithmetic reasoning | 87.2% | 92.8% | **+5.6** |

Gains are largest where headroom is largest.

---

## 6. Best-of-N MCTS

Flat Best-of-N treats trajectories as atomic: N complete solutions are generated independently and
scored only at termination. Two inefficiencies follow directly from the measurements above.

**Independence waste.** A trajectory that commits an arithmetic error in its second step is
nonetheless generated to completion — often several thousand tokens. Across our AIME runs, roughly 65%
of sampled trajectories terminated by token limit rather than natural completion, meaning most of the
compute budget extended paths already unrecoverable.

**Evidence fragmentation.** When k of N trajectories share a correct opening derivation and then
diverge, flat sampling treats them as k unrelated observations. The shared prefix — the part most
likely to be correct — accumulates no credit. This is precisely why majority voting saturates at 53.3%
while coverage continues climbing.

### 6.1 Formulation

We restructure sampling as search over a tree *T* whose nodes are *partial* trajectories. The root is
the problem statement; an edge is the generation of one reasoning segment, bounded at a natural
boundary such as a line break or completed derivation step; a leaf contains a terminal answer. Each
node *s* maintains a visit count *n(s)*, accumulated value *W(s)* and mean *Q(s) = W(s)/n(s)*.

Descent from the root follows a PUCT rule balancing exploitation of high-value prefixes against
exploration of under-visited ones, with the model's own sequence likelihood as prior:

> **(5)**  a\* = argmax_a [ Q(s,a) + c_puct · P(a|s) · √( Σ_b n(s,b) ) / ( 1 + n(s,a) ) ]

The prior anchors search to trajectories the base considers fluent; *Q* allows the verifier to
override fluency when it disagrees — the same mechanism responsible for the 16.6-point gain in the
flat setting.

### 6.2 Outcome supervision for process-level guidance

The verifier was trained on complete trajectories, so applying it to prefixes would violate its
training distribution. We instead evaluate a node by *truncated rollout*: from prefix *s*, one
continuation is sampled to termination, the verifier scores the completed trajectory, and the scalar
is backed up along the path to the root:

> **(6)**  W(sᵢ) ← W(sᵢ) + V( rollout(s) ),  ∀ sᵢ ∈ path(root → s)

This preserves the verifier's training distribution — it only ever scores complete trajectories —
while propagating credit to the prefixes responsible for them. It is the key design choice of the
method: it enables an *outcome*-supervised verifier to provide process-level guidance without
step-level labels, whose annotation cost is the principal barrier to process reward modelling.

```python
# Best-of-N MCTS — one search iteration
def iterate(root, model, verifier, c_puct):
    node = root
    while node.expanded:                                  # 1. selection
        node = max(node.children, key=lambda ch: puct(ch, c_puct))

    segs = model.sample_segments(node.prefix, k=BRANCH)    # 2. expansion
    node.expand(segs, priors=model.seq_logprobs(segs))

    child   = node.children[0]
    rollout = model.complete(child.prefix, temperature=0.6)  # 3. rollout
    value   = verifier.p_correct(root.problem, rollout)

    for s in path_to_root(child):                          # 4. backup
        s.n += 1; s.W += value
```

At exhaustion of the budget, answers are aggregated by visit-weighted verifier value:

> **(7)**  â = argmax_a Σ_{ℓ : answer(ℓ) = a} n(ℓ) · V(ℓ)

Equation (7) generalises both baselines: setting *V* ≡ 1 recovers visit-weighted majority voting, and
restricting the tree to depth one recovers equation (4), the strongest selector measured.

### 6.3 Cost model and prediction

For comparison at parity, both methods must be held to an identical token budget. Flat Best-of-N with
budget N consumes N · L tokens for mean trajectory length L. The tree method with *I* iterations
consumes approximately *I* · (L_seg · B + L_roll), where B is the branching factor. Because rollouts
begin from a prefix rather than the problem statement, L_roll < L, and the number of distinct
trajectories explored per token exceeds that of flat sampling.

> ⚠️ **Status of results.** Tables 1–3 report measured outcomes of flat Best-of-N and verifier
> selection. The tree method of §6 is derived from those measurements; its evaluation requires
> sustained accelerator access beyond currently available resources, and **no results for it are
> claimed here.**

**Prediction.** At a token budget matched to N = 128, Best-of-N MCTS should exceed flat
confidence-weighted selection on problems whose flat coverage lies strictly between 0 and 1, and be
indistinguishable from it where coverage is 0 or 1 — in those regimes there is no gap to recover. A
negative result on the intermediate regime falsifies the method.

---

## 7. Discussion

### 7.1 Where the capability comes from

A frozen 1.5B model attaining 83.3% coverage on competition mathematics implies the knowledge required
is already present in the weights; what is unreliable is the *retrieval* of a correct trajectory on any
single attempt. Parameter scaling improves per-sample reliability directly. Inference-time compute
attacks the same quantity from the opposite direction, amortising unreliability across samples. Within
the range measured here, the two are substitutable.

### 7.2 Economic asymmetry

The substitution is favourable for small models. Drawing N = 128 trajectories from a 1.5B base remains
cheaper in FLOPs than a single forward pass through a trillion-parameter system, and the memory
footprint permits local deployment on commodity hardware. Inference-time compute trades latency for
accuracy at fixed memory — precisely the trade an independent researcher can afford to make.

---

## 8. Limitations

- **Coverage bound.** Where *p* ≈ 0 for a problem class, equation (2) returns zero for every N. No
  selector and no search recovers a correct answer that was never generated. Extending beyond this
  bound requires improving the proposal distribution itself, which is a training problem and therefore
  a compute problem.
- **Verifier ceiling.** Selection quality is bounded by the verifier's 0.910 ROC-AUC. Under tree search
  this bound tightens rather than loosens: search actively concentrates budget on regions the verifier
  rates highly, so systematic verifier error is amplified rather than averaged out.
- **Segment boundaries.** Decomposition into reasoning steps is heuristic; poor boundaries produce a
  tree whose structure does not match the logical structure of the derivation.
- **Evaluation variance.** AIME comprises 30 problems per year. We report the combined 2023–2025 set
  (90 problems) wherever a comparison is load-bearing, and single-year results are noted as
  higher-variance.

---

## 9. Conclusion

Capability in language models is not a monolithic property of parameter count. Decomposed into proposal
quality and selection quality, a substantial portion becomes addressable at inference time, on frozen
weights, at hardware cost within reach of an individual. The binding constraint of such systems is the
selection gap, and a learned verifier is the correct instrument against it — recovering 16.6 points over
majority voting in our measurements. Best-of-N MCTS extends the same principle from independent sampling
to structured search, using outcome supervision to guide a process it was never explicitly trained on.
The remaining constraint — improving the proposal distribution itself — is where compute becomes
irreducible, and marks the boundary of what this architecture achieves alone.

---

## Reproducibility

Inference engine, evaluation harness, verifier training code, decontamination scripts and raw
per-problem outputs are published at
[github.com/voidlinestudios12-jpg/Interlace-AI-Nem](https://github.com/voidlinestudios12-jpg/Interlace-AI-Nem).
Every figure and table in this report is regenerable from that repository.

## Citation

```bibtex
@techreport{arecesrivera2026interlace,
  title  = {Modern Architecture On Advanced LLM: Best-of-N Sampling,
            Learned Verification and Tree Search as a Substitute for Parameter Scale},
  author = {Areces Rivera, Alejandro},
  year   = {2026},
  number = {TR-2026-01},
  institution = {Interlace AI},
  doi    = {10.5281/zenodo.21936833},
  url    = {https://doi.org/10.5281/zenodo.21936833}
}
```

---

## Licence and terms of use

| Component | Licence |
|---|---|
| This document (text, figures, tables) | **CC BY-NC 4.0** — free to read, share, quote and cite with attribution, for non-commercial purposes |
| Accompanying source code | **Apache License 2.0** |

> **Commercial use of this work requires prior written authorisation.** Any commercial exploitation of
> the methods, results or text of this report — including integration into a paid product or service,
> resale, or use in commercial training pipelines — must be requested and approved in writing
> beforehand.
>
> Academic use, teaching, citation, review, personal study and non-commercial research are expressly
> permitted and require no separate authorisation.

## Contact

**Alejandro Areces Rivera** — Interlace AI

📧 [interlaceIA@gmail.com](mailto:interlaceIA@gmail.com)

Open to collaboration, review and questions. Commercial authorisation requests to the same address.

---

<div align="center">
<sub>© 2026 Alejandro Areces Rivera — Interlace AI · Technical Report TR-2026-01</sub>
</div>
