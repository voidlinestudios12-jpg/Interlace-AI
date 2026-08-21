# Changelog

All notable changes to `bestofn`.

## 1.1.0 — 2026-08-17

A correctness release. Every defect below was found by an adversarial audit of
the published 1.0.0 package and reproduced before being fixed; each has a
regression test named after it in `tests/test_selectors.py`.

**Numbers reported by 1.0.0 should not be relied on.** Answer extraction was
corrupting the inputs to the vote, so the accuracies it produced — and the
selection gaps derived from them — are not reproducible with this version. The
measurements published alongside this release were regenerated from scratch.

### Fixed — answer extraction

- **Fractions, radicals and powers no longer collapse to their first digit.**
  `extract_boxed` counted braces correctly and then discarded the result,
  applying a `-?\d+\.?\d*` regex to the content. `\boxed{\frac{1}{2}}`,
  `\boxed{\frac{1}{3}}`, `\boxed{-\frac{1}{2}}` and `\boxed{1}` all returned
  `"1"`, so four different answers voted as one. They now return `1/2`, `1/3`,
  `-1/2` and `1`.
- **Commas are no longer deleted before parsing.** `","` was in the LaTeX noise
  list, so `\boxed{(3,4)}` returned `"34"` — not a lost answer but a
  manufactured one, and `34` is a plausible AIME answer that could match the
  gold by accident. Digit grouping is now handled with context: `1,000` and
  `1{,}000` become `1000`, while `(3,4)` stays a pair.
- **Truncated trajectories abstain instead of guessing.** With no `\boxed{}`,
  extraction fell back to the last number anywhere in the text, so an
  unfinished chain of thought voted with whatever digit it had reached last.
  `extract_boxed` now returns `""` unless `allow_fallback=True`.
- **Units are stripped, prose answers are not.** `\boxed{204\text{ km}}` gives
  `204`; `\boxed{\text{Canberra}}` gives `Canberra`.

### Fixed — selection

- **Verifier scores outside `[0, 1]` now raise instead of silently becoming a
  majority vote.** Most published reward models emit unbounded logits. Passing
  them in produced a plain majority vote with no warning, so users believed
  they were running their verifier when they were not. The error message says
  to apply a sigmoid; `bestofn.verifiers` does it for you.
- **`select()` returns the canonical form.** It previously returned the raw
  string while `coverage()` compared normalised ones, so `204.0` counted as a
  miss against a gold of `204` in the numerator but as a hit in the
  denominator. That inflated every measured selection gap.
- **`agreement()` no longer counts abstentions in the denominator only.**
  Three identical votes plus two unparseable trajectories reported `0.6`; it
  now reports `1.0` and exposes the abstentions separately.
- **`verifier_argmax` filters non-finite scores.** A single `NaN` made the
  result depend on list order.
- **`coverage()` rejects a gold that does not normalise.** `coverage([""],
  "inf")` returned `True`, marking answerless trajectories as covered.
- **Equivalent answers are pooled.** With `math-verify` installed, `1/2`,
  `0.5` and `2/4` count as one vote bloc instead of three separate ones.

### Fixed — engine

- **The `transformers` backend no longer runs out of memory with the library's
  own defaults.** It requested `output_scores=True` and stacked the result,
  allocating roughly `n × max_tokens × vocab` floats — about 119 GB at n=8,
  8192 tokens and a 152k vocabulary. Since `transformers` is the automatic
  fallback when vLLM is absent, the default configuration could not run.
  Log-probabilities are now off by default, generated in memory-bounded
  sub-batches, and reduced blockwise. Measured peak on an RTX 3060: 1.1 GB
  without them, 3.9 GB with.
- **Both backends now report the same quantity.** The `transformers` path used
  `output_scores`, which is the distribution *after* temperature and top-p, so
  `self_certainty` was not comparable across backends. It now uses raw logits.
- **`solve_batch` validates temperature.** `BestOfN(m, n=1, temperature=0)
  .solve_batch(problems, n=32)` previously generated 32 identical samples at 32
  times the cost.
- **A failing chat template warns instead of passing silently.** Losing it on
  an instruction-tuned model degrades output badly, and the user would blame
  the model.

### Added

- **`random` selector.** Uniform choice among trajectories that produced an
  answer, seedable for reproducibility. It is the baseline every other
  selector has to beat: one below it is anti-correlated with correctness. Every
  table this project publishes now carries this row.
- **`bestofn.verifiers`** — adapters for published reward models, applying the
  sigmoid the selector requires and reporting the model's licence, which
  governs how its scores may be used.
- **Truncation and abstention accounting** on `Result`: `effective_n`,
  `n_abstained`, `n_truncated`, `total_tokens`. The N you paid for is not the N
  that voted.
- **`math` extra** (`pip install "bestofn[math]"`) for symbolic answer
  comparison via `math-verify`.
- **`scripts/run_gsm8k.py` and `scripts/analyse.py`.** The evaluation now
  publishes complete reasoning text, `finish_reason`, log-probabilities and
  token counts, and the analysis re-extracts from that text rather than from
  answers extracted earlier. A replay that cannot re-run extraction cannot
  detect an extraction bug, and the 1.0.0 one could not.

### Removed

- **Claims about a first-party verifier.** 1.0.0 reported a trained outcome
  reward model with a ROC-AUC of 0.910 and a 16.6-point gain over majority
  voting. Those figures are withdrawn: the model is not published, the gain
  was measured on a problem set that was not published either, and in the one
  evidence file that *was* published the verifier selects an identical answer
  to majority voting on 30 of 30 problems, for a gain of 0.0 points. The
  `verifier` selectors remain, and now work properly with third-party reward
  models.
- The `verify_against_measured.py` script, replaced by `scripts/analyse.py`.

### Changed

- Citations point at the Zenodo **concept** DOI `10.5281/zenodo.21936832`,
  which resolves to the current version, rather than the v1 DOI.
- `requires-python` unchanged at `>=3.8`.

---

## 1.0.0 — 2026-08-15

First release.
