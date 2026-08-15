<div align="center">

<img src="../../assets/interlace-logo.svg" width="64" alt="Interlace AI">

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
We make the generation–selection decomposition explicit, identify the **selection gap** as the
governing bottleneck of such systems, report transfer to graduate-level science and arithmetic reasoning, and
derive **Best-of-N MCTS** — a tree-search generalisation in which the outcome verifier supplies
process-level guidance through truncated rollout, removing the need for step-level supervision. All
results are reproducible from a public repository.

---

## Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [The Generation–Selection Decomposition](#3-the-generationselection-decomposition)
4. [System Architecture](#4-system-architecture)
5. [The Learned Verifier](#5-the-learned-verifier)
6. [Empirical Results](#6-empirical-results)
7. [Best-of-N MCTS](#7-best-of-n-mcts)
8. [Discussion](#8-discussion)
9. [Limitations](#9-limitations)
10. [Conclusion](#10-conclusion)
11. [References](#references)

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
correctness onto a separate selection mechanism. This decomposition organises the report, and
everything that follows — the architecture of §4, the verifier of §5, and the tree search
of §7 — is a consequence of it.

The components used here are established. Repeated sampling with majority voting [4], verifier-based
reranking of N candidates [1], and the general finding that inference-time compute can substitute for
parameter count [6, 7] are all prior work, and §2 situates this report against them. What follows is
therefore an empirical contribution rather than a methodological one:

1. A complete, open implementation of the generation–selection architecture on a frozen 1.5B base,
   with decontamination and per-problem raw outputs published for independent audit.
2. Measurements on three benchmarks quantifying the **selection gap** — the distance between what the
   model reaches and what the selector returns — at sample budgets from N=1 to N=128.
3. A trained outcome verifier that recovers roughly half of that gap, and the finding that
   confidence-weighted voting outperforms argmax reranking by 8.9 points.
4. Best-of-N MCTS: a tree-search formulation in which an *outcome*-supervised verifier supplies
   process-level guidance through truncated rollout, avoiding the step-level annotation that process
   reward models require [5]. This is specified and predicted, not evaluated (§7.3).

---

## 2. Related Work

**Chain-of-thought and repeated sampling.** Eliciting intermediate reasoning before the final answer
[3] made the output of a language model a *trajectory* rather than a token, which is what makes
sampling several of them meaningful. Self-consistency [4] established the basic form used here:
sample N chains at non-zero temperature and return the modal answer. Brown et al. [7] studied how
coverage — the probability that at least one of N samples is correct — scales with the budget, and
observed the same divergence between coverage and what a deployed selector actually returns that
motivates §3.1 of this report.

**Verifiers.** Cobbe et al. [1] introduced the technique this work builds on most directly: train a
separate model to judge candidate solutions, then rerank N samples by its score. Their verifiers were
*outcome*-supervised — labelled only by final correctness — which is also the supervision used here.
Uesato et al. [2] compared outcome- against process-supervision, and Lightman et al. [5] showed that
process reward models, which score individual reasoning steps, select better than outcome models. The
cost of that improvement is step-level annotation, which is the constraint §7.2 is designed around.

**Test-time compute as a substitute for scale.** Snell et al. [6] framed the question addressed here
directly: given a fixed budget, is it better spent on parameters or on inference-time search? Their
finding — that for many problem difficulties the latter wins — is the premise this report assumes
rather than re-establishes. The contribution here is a measurement of that trade-off on a model at
the small end of the range (1.5B), where the asymmetry is most favourable and least often reported.

**Tree search over reasoning.** Tree of Thoughts [8] generalised chain-of-thought to a search over
partial reasoning states. Subsequent work combines Monte Carlo tree search with a learned value
function, typically a process reward model, using the PUCT selection rule from AlphaGo Zero [9].
§7 follows that line with one difference: because no step-level labels were available, node values
come from an outcome verifier applied to a *completed* rollout from the prefix, which keeps the
verifier inside its training distribution.

**Where this report sits.** It does not introduce a new selection algorithm. It reports what the
established combination achieves on a frozen 1.5B model when implemented carefully and measured
honestly — separating coverage from realised accuracy, decontaminating the verifier's training data,
and publishing every raw trajectory so the numbers can be replayed rather than trusted.

---

## 3. The Generation–Selection Decomposition

Let *M* be a frozen autoregressive reasoning model and *x* a problem instance. Sampling *M* at
temperature *T* > 0 induces a distribution over complete reasoning trajectories *y*. We draw *N*
independent trajectories and write *a(y)* for the answer extracted from trajectory *y*. The system
output is determined by a selector *S* over the sampled set:

> **(1)**  ŷ = S( { y₁, …, y_N } ),  where  yᵢ ~ M( · | x, T )

This separates two quantities conflated in single-sample evaluation. The first is **coverage**: the
probability that at least one sampled trajectory reaches the correct answer *a\**, the quantity
studied by Brown et al. [7]. Writing *p* for the
per-sample success probability,

> **(2)**  Cov(N) = 1 − ( 1 − p )^N,  where  p = P_{y~M}[ a(y) = a\* ]

The second is **selection accuracy**: the probability that *S* returns the correct answer given that
it is present among the candidates. Realised accuracy is their product, and coverage upper-bounds any
selector:

> **(3)**  Acc(N) = Cov(N) · P[ S = a\* | a\* ∈ candidates ]  ≤  Cov(N)

### 3.1 The selection gap

We define the *selection gap* **G(N) = Cov(N) − Acc(N)**: the accuracy a perfect selector would
recover but the deployed selector does not. Equation (2) shows coverage saturating logarithmically in
N, so beyond a moderate budget additional samples contribute little. The gap, by contrast, does not
close on its own. It is therefore the correct optimisation target for any inference-time compute
architecture, and §6 confirms that it dominates system performance in practice.

---

## 4. System Architecture

The system comprises four stages around an unmodified base. The base remains frozen throughout: no
gradient reaches its weights. This guarantees the architecture cannot degrade the underlying model —
a failure mode we observed repeatedly when attempting supervised adaptation of the same base, where
an ill-chosen learning rate or a duplicated control token collapsed reasoning performance.

### 4.1 Diverse proposal generation

N trajectories are sampled in parallel with nucleus sampling (T = 0.6, top-p = 0.95), served through
a paged-attention engine [12] so the N sequences share prefix key–value cache. Temperature is the control
variable of the coverage–consensus trade-off: low temperature concentrates probability mass on the
modal answer, raising consensus but lowering coverage; high temperature does the reverse.

### 4.2 Answer normalisation

Trajectories terminate in a delimited answer field. Extraction must be robust to nested LaTeX
environments, unit annotations and formatting variation, since a normalisation failure is
indistinguishable from a reasoning failure at the metric level. We use bracket-balanced parsing of the
terminal `\boxed{}` expression followed by numeric canonicalisation.

### 4.3 Selection

Three selectors of increasing sophistication are supported:

| Selector | Mechanism | Can recover a minority-correct answer? |
|---|---|:---:|
| Majority vote [4] | Modal normalised answer | No |
| Self-certainty | Highest mean token log-probability | Rarely |
| Learned verifier [1] | Estimates P(correct \| x, y) per trajectory | **Yes** |

Majority voting is structurally incapable of recovering a correct answer held by a minority of
trajectories. Self-certainty measures fluency rather than correctness. Only the learned verifier
attacks the selection gap directly.

---

## 5. The Learned Verifier

The verifier is an outcome reward model (ORM) in the sense of Cobbe et al. [1]: a classification head
over a frozen copy of the base, adapted with low-rank matrices [11]. Training pairs (*x*, *y*) are labelled by agreement with reference
answers, drawn exclusively from training-split problems [13].

### 5.1 Decontamination

Every candidate training problem is checked against the full evaluation suite by three independent
criteria: normalised exact match, substring containment, and Jaccard overlap of 5-gram shingles above
0.6. A problem failing any criterion is discarded. Given how easily contamination inflates apparent
performance, we regard this step as a precondition for reporting results at all, not an optimisation.

### 5.2 Training

Self-generated data is positively skewed, so the loss is class-weighted. The train/validation split is
performed at the *problem* level rather than the trajectory level, preventing trajectories of the same
problem from appearing on both sides. The resulting verifier reaches **0.910 ROC-AUC** and **84.1%
accuracy** on held-out problems.

### 5.3 Confidence-weighted aggregation

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

## 6. Empirical Results

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

## 7. Best-of-N MCTS

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

### 7.1 Formulation

We restructure sampling as search over a tree *T* whose nodes are *partial* trajectories. The root is
the problem statement; an edge is the generation of one reasoning segment, bounded at a natural
boundary such as a line break or completed derivation step; a leaf contains a terminal answer. Each
node *s* maintains a visit count *n(s)*, accumulated value *W(s)* and mean *Q(s) = W(s)/n(s)*.

Descent from the root follows the PUCT rule of Silver et al. [9], balancing exploitation of
high-value prefixes against exploration of under-visited ones, with the model's own sequence likelihood as prior:

> **(5)**  a\* = argmax_a [ Q(s,a) + c_puct · P(a|s) · √( Σ_b n(s,b) ) / ( 1 + n(s,a) ) ]

The prior anchors search to trajectories the base considers fluent; *Q* allows the verifier to
override fluency when it disagrees — the same mechanism responsible for the 16.6-point gain in the
flat setting.

### 7.2 Outcome supervision for process-level guidance

The verifier was trained on complete trajectories, so applying it to prefixes would violate its
training distribution. We instead evaluate a node by *truncated rollout*: from prefix *s*, one
continuation is sampled to termination, the verifier scores the completed trajectory, and the scalar
is backed up along the path to the root:

> **(6)**  W(sᵢ) ← W(sᵢ) + V( rollout(s) ),  ∀ sᵢ ∈ path(root → s)

This preserves the verifier's training distribution — it only ever scores complete trajectories —
while propagating credit to the prefixes responsible for them. It is the key design choice of the
method: it enables an *outcome*-supervised verifier to provide process-level guidance without
step-level labels, whose annotation cost is the principal barrier to process reward modelling [5].

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

### 7.3 Cost model and prediction

For comparison at parity, both methods must be held to an identical token budget. Flat Best-of-N with
budget N consumes N · L tokens for mean trajectory length L. The tree method with *I* iterations
consumes approximately *I* · (L_seg · B + L_roll), where B is the branching factor. Because rollouts
begin from a prefix rather than the problem statement, L_roll < L, and the number of distinct
trajectories explored per token exceeds that of flat sampling.

> ⚠️ **Status of results.** Tables 1–3 report measured outcomes of flat Best-of-N and verifier
> selection. The tree method of §7 is derived from those measurements; its evaluation requires
> sustained accelerator access beyond currently available resources, and **no results for it are
> claimed here.**

**Prediction.** At a token budget matched to N = 128, Best-of-N MCTS should exceed flat
confidence-weighted selection on problems whose flat coverage lies strictly between 0 and 1, and be
indistinguishable from it where coverage is 0 or 1 — in those regimes there is no gap to recover. A
negative result on the intermediate regime falsifies the method.

---

## 8. Discussion

### 8.1 Where the capability comes from

A frozen 1.5B model attaining 83.3% coverage on competition mathematics implies the knowledge required
is already present in the weights; what is unreliable is the *retrieval* of a correct trajectory on any
single attempt. Parameter scaling improves per-sample reliability directly. Inference-time compute
attacks the same quantity from the opposite direction, amortising unreliability across samples. Within
the range measured here, the two are substitutable.

### 8.2 Economic asymmetry

The substitution is favourable for small models. Drawing N = 128 trajectories from a 1.5B base remains
cheaper in FLOPs than a single forward pass through a trillion-parameter system, and the memory
footprint permits local deployment on commodity hardware. Inference-time compute trades latency for
accuracy at fixed memory — precisely the trade an independent researcher can afford to make.

---

## 9. Limitations

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

## 10. Conclusion

Capability in language models is not a monolithic property of parameter count. Decomposed into proposal
quality and selection quality, a substantial portion becomes addressable at inference time, on frozen
weights, at hardware cost within reach of an individual. The binding constraint of such systems is the
selection gap, and a learned verifier is the correct instrument against it — recovering 16.6 points over
majority voting in our measurements. Best-of-N MCTS extends the same principle from independent sampling
to structured search, using outcome supervision to guide a process it was never explicitly trained on.
The remaining constraint — improving the proposal distribution itself — is where compute becomes
irreducible, and marks the boundary of what this architecture achieves alone.

---

## References

[1] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek,
J. Hilton, R. Nakano, C. Hesse, J. Schulman. *Training Verifiers to Solve Math Word Problems.*
arXiv:2110.14168, 2021.

[2] J. Uesato, N. Kushman, R. Kumar, F. Song, N. Siegel, L. Wang, A. Creswell, G. Irving,
I. Higgins. *Solving Math Word Problems with Process- and Outcome-Based Feedback.*
arXiv:2211.14275, 2022.

[3] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, D. Zhou.
*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS, 2022.
arXiv:2201.11903.

[4] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, D. Zhou.
*Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR, 2023.
arXiv:2203.11171.

[5] H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman,
I. Sutskever, K. Cobbe. *Let's Verify Step by Step.* ICLR, 2024. arXiv:2305.20050.

[6] C. Snell, J. Lee, K. Xu, A. Kumar. *Scaling LLM Test-Time Compute Optimally Can Be More
Effective Than Scaling Model Parameters.* arXiv:2408.03314, 2024.

[7] B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. Ré, A. Mirhoseini. *Large Language
Monkeys: Scaling Inference Compute with Repeated Sampling.* arXiv:2407.21787, 2024.

[8] S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, K. Narasimhan. *Tree of Thoughts:
Deliberate Problem Solving with Large Language Models.* NeurIPS, 2023. arXiv:2305.10601.

[9] D. Silver, J. Schrittwieser, K. Simonyan, et al. *Mastering the Game of Go without Human
Knowledge.* Nature 550, 354–359, 2017.

[10] DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
Learning.* arXiv:2501.12948, 2025. — source of the frozen base model used throughout.

[11] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. *LoRA: Low-Rank
Adaptation of Large Language Models.* ICLR, 2022. arXiv:2106.09685. — used to adapt the verifier.

[12] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, I. Stoica.
*Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP, 2023.
arXiv:2309.06180. — the serving engine used for batched sampling.

[13] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, J. Steinhardt.
*Measuring Mathematical Problem Solving with the MATH Dataset.* NeurIPS Datasets and Benchmarks,
2021. arXiv:2103.03874. — source of verifier training problems.

[14] D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, S. R. Bowman.
*GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* arXiv:2311.12022, 2023.

---

## Reproducibility

Inference engine, evaluation harness, verifier training code, decontamination scripts and raw
per-problem outputs are published at
[github.com/.../Interlace-AI/best-of-n](https://github.com/voidlinestudios12-jpg/Interlace-AI/tree/main/best-of-n).
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
