# Changelog

All notable changes to `bestofn`.

## 1.1.4 — 2026-08-25

A third adversarial audit, run against 1.1.3. It reproduced 25 of 26 published
numbers independently and the summary regenerates bit-identically — but it
found two genuinely fatal defects in the extractor, **both of them introduced
or left by the round-2 fixes**, and it caught three items the 1.1.3 changelog
listed as closed that were not.

### Fixed — extraction

- **`\boxed{\left(5\right)}` voted as `<=ft(5)`.** The round-2 fix that added
  plain forms for `\ge`, `\le` and friends used a bare `str.replace` with no
  word boundary, so every command whose name merely *starts* with one of them
  lost its backslash: `\left` → `<=ft`, `\gets` → `>=ts`, `\newline` →
  `!=wline`. The mangled text no longer looked like a command, so the
  unresolved-command check never fired and a fabricated answer went into the
  tally — the exact failure that check exists to prevent, reintroduced by the
  fix for it. It is now one anchored regex, longest name first, with
  `(?![A-Za-z])`. Verified against a catalogue of 89 real LaTeX commands: none
  fabricates a vote, and all 12 legitimate forms still vote.
  Effect on the published GSM8K run: **none** — its answers are bare integers,
  and no trajectory contained a colliding command. On a MATH-style corpus it
  would have been serious.
- **`normalise` raised instead of abstaining.** The round-2 fix for very large
  integers called `int()`, and CPython refuses to convert a string of more than
  4,300 digits. `normalise` sits behind `Sample.key`, so a single pathological
  trajectory took down `select`, `agreement`, `effective_n` and `abstentions`
  for the whole pool of 128 with an uncaught `ValueError`. Strictly worse than
  the silent abstention it replaced. Canonicalised by string now; no `int()`.

### Fixed — the invariance claim

- **`oracle` was still order-dependent,** and the property was asserted for
  every selector. When two distinct canonical keys are each equivalent to the
  gold answer — a pool holding both `1/2` and `0.5` against a gold of
  `\frac{1}{2}` — it returned whichever arrived first. Now the minimum by key.
- **The claim had no test.** 1.1.3's changelog said "verified across 24,000
  shuffles"; that was run by hand and never committed, and the only coverage in
  the repository was 60 shuffles of `majority`. There is now a fuzz over all
  five deterministic selectors, plus the specific pools that falsified `oracle`
  and the two argmax tie cases, and the merge-cap fallback path. Confirmed to
  fail when each bug is reintroduced.

### Fixed — measurement

- **One criterion for correctness.** `scripts/analyse.py` scored accuracy with
  exact key equality while `coverage()` in the same table — and
  `Result.is_correct`, which is what a reader will actually run — used symbolic
  equivalence. The coverage column was therefore scored more generously than
  the accuracy columns beside it. Everything now goes through one `hit()`
  helper using the library's own rule. This moves the published figures by a
  tenth of a point: single sample 45.0 → **45.1**, total gain 21.5 → **21.4**,
  selection 20.2 → **20.1**. Majority (66.5%) and coverage (93.5%) are
  unchanged.
- **The abstention split was inferred, not counted.** Subtracting truncations
  from abstentions assumes every truncated trajectory abstains. One did not —
  it boxed its answer and then ran out of room — so the real split is **215
  truncated-and-abstained plus 588 others**, not 216 + 587. Counted directly
  now, and both halves are in the summary.
- **The two lead figures still printed the cherry-picked p-value.** 1.1.3 moved
  the text to the worst of 200 seeds but `figures/marca.py` was still reading
  the single-seed field, so `07_curve_n128.png` and `08_bars_n128.png` showed
  `p = 1.2e-07` beside a README saying `p ≤ 1.4e-05`. They read
  `p_value_worst` now and say how many seeds it is the worst of.
- **The skip count was reported before it finished counting.** Without
  math-verify, pytest said "1 checks skipped" where six are. The notice now
  runs after every gate.

### Fixed — claims

- `verifier_argmax` returned an unmerged key, so the same pool could answer
  `1/2` under argmax and `0.5` under `verifier`.
- The licence check blocked for up to 15 seconds at construction on an
  unreachable host, in a constructor documented as cheap. Three seconds now,
  and `BESTOFN_NO_LICENCE_CHECK=1` skips the network entirely.
- The technical note claimed the published dataset records the sampling
  parameters, seed and library version. It records the prompt suffix. That is
  now stated plainly, along with the fact that generation is not seeded — the
  run reproduces as a distribution, not token for token. `run_gsm8k.py` writes
  the full configuration into every record from this release on.
- The 1.1.3 changelog listed a fix for transient symbolic-backend failures
  being memoised as permanent. Only an unused exception class was added;
  nothing raises it. The entry has been withdrawn rather than left standing.
- `matplotlib` was an undeclared import of `figures/make_figures.py`. It is now
  the `[figures]` extra, and the README says which commands need which extra.
- The technical note's abstract still claimed the coverage/selection
  decomposition had not been reported elsewhere, while §1 of the same document
  credited Snell et al. and the evaluation harnesses.
- `USAGE.md` said per-trajectory accuracy is "exactly what random selection
  gets you". It is not — random picks among trajectories that answered, so it
  skips the abstentions and comes out about a point higher. That gap is the
  whole "not the method" term the report is built on.
- The README's selector table is headed for what it contains: `verifier` and
  `verifier_argmax` cannot appear, because this release ships no reward model
  and the published trajectories carry no scores.

### Note on the numbers

45.1% → 66.5% at N=128, coverage 93.5%, McNemar p ≤ 1.4 × 10⁻⁵ over 200 seeds.
The only movement from 1.1.3 is the tenth of a point from unifying the
correctness criterion. 187 tests.

## 1.1.3 — 2026-08-25

A second adversarial audit, run against 1.1.2 immediately after its release.
It re-derived every published headline independently and all of them held to
the decimal — but it found seventeen problems in the layer of claims wrapped
around them, including one property this project had made a selling point and
which turned out to be true for only two of the six selectors. All seventeen
are closed here.

### Fixed

- **Permutation invariance held for `majority` and `oracle` only.** 1.1.2
  claimed, in the README and in the technical note, that shuffling the pool
  never changes the answer. Two leaks made that false elsewhere:
  `verifier_argmax` used `max(..., key=score)`, and `max` returns the *first*
  maximal element, so an exact tie resolved by arrival order; and `_tally`
  accumulated weights with `+=` in list order, which for `self_certainty` and
  `verifier` makes the total depend on that order because floating-point
  addition is not associative — two permutations can differ in the last bit,
  and that is enough to flip a near-tie. Argmax now breaks ties on the
  canonical key, and totals are summed with `math.fsum` over a sorted list,
  which is exactly rounded. Verified across 24,000 shuffles: no selector moves.
- **The merge-cap counter under-reported by 73×.** `symbolic_merge_capped_calls`
  existed to justify raising `_MERGE_LIMIT`, and it published 28 where the
  true figure is 2,040. It counted *warnings*, and Python shows a given warning
  once per location, so repeats never reached the handler. It is now counted in
  `_merge_map` itself and exposed as `bestofn.select.merge_cap_hits()`.
- **`pytest` failed on a default `pip install bestofn`.** Four checks silently
  required the optional `[math]` extra, so the suite went red for anyone who
  installed the package as the documentation describes. They are gated now, the
  skip count is reported through `warnings` — which pytest displays, unlike the
  `print` that was there before — and two of the mixed-number checks were
  rewritten to test what the extractor emits, which needs no symbolic layer.
- **Unresolved LaTeX was voting instead of abstaining.** Rule 1 of the
  extraction module is "never invent an answer", and it was only being enforced
  for `\frac` and `\sqrt`. Everything else had its backslash stripped:
  `\binom{5}{2}`, which is 10, became the vote `binom52`; a `pmatrix` became a
  run of digits; `\geq` and `\ge` produced two different votes for one answer.
  Any command still present after canonicalisation now abstains. Degrees and
  the comparison operators gained plain forms rather than being lost —
  `45^{\circ}` is 45, not `45**()`.
- **A verifier returning the wrong number of scores was still tolerated.**
  1.1.2 added the length check but put it inside the `try`, so its own
  `except Exception` caught it, reported "batched scoring failed" — which had
  not happened — and silently rescored the pool through a different code path.
  The check is outside the `try` now and raises.
- **The reproduction time, the self_certainty score and the merge-cap
  measurement were each stale in one place** while correct in another: nine
  minutes in the changelog against two in the README, 65.0% in `select.py`
  against 66.5% in `USAGE.md`, 66.0% in a code comment against the 66.5% it was
  meant to support. All reconciled.
- **`normalise` destroyed integers above ~1e308.** `float()` overflows to
  infinity and non-finite values are rejected, so a 400-digit integer — exact
  as written, needing no float at all — normalised to the empty string and
  abstained.
- **Only `n` and `temperature` were validated.** `top_p=0`, `top_p=5`,
  `max_tokens=0`, a negative `max_tokens`, a float `n` and `max_parallel=0`
  were all accepted here and failed later inside a backend, where the user
  reasonably blames the backend.
- **Two chart scripts shipped with hand-typed numbers** from the previous
  release and wrote the same filenames as the summary-driven ones, so the last
  script run decided which figures shipped. Both now read
  `results/gsm8k_summary.json`, and each figure has exactly one script that
  produces it. The brand logo in them was also drawn by hand as two 0.17 × 0.30
  rectangles; the real mark is two rounded squares, so it rendered as two tall
  boxes with sharp corners. It is loaded from the PNG now and cannot be
  distorted.
- **The abstention figure attributed all 803 abstentions to truncation.** Only
  215 are; 588 are trajectories that finished normally without leaving a usable
  answer. Exact counts are published in the summary so a figure never has to
  multiply a rounded percentage back out.
- **The GitHub repository description** carried the 1.1.1 numbers under a
  changelog entry claiming it had been corrected.
- Smaller: `verifier_argmax` said nothing when every score was zero while
  `verifier` warned; the vLLM terminal-token correction was skipped in silence
  when the log-probability count did not match the token count; the test
  suite's "no network" claim was untrue; `README_PYPI.md`'s relative links
  resolved to the PyPI project page rather than to the files.

### Changed

- **The headline p-value is now the worst of 200 random seeds, not one seed.**
  `random` draws differently every time it runs, so a p-value taken from a
  single seed is partly a property of that seed: across 200 of them ours ranged
  from 2.4 × 10⁻¹⁴ to 1.4 × 10⁻⁵. We publish **p ≤ 1.4 × 10⁻⁵** with a median
  of 1.5 × 10⁻⁹. The conclusion holds at every seed; the single figure we had
  been quoting did not describe the data as firmly as it appeared to.
- **The random baseline is reported once, consistently.** Two different values
  — 46.3% from the resampled curve and 47.5% from one seeded draw — were being
  published side by side as though they were the same quantity, in the README,
  in the technical note and in two different figures. Both now report the mean
  over 200 seeds with its spread: **46.3%, sd 2.5**.
- **The benchmark records the configuration it ran with.** Model, N,
  temperature, top-p, max tokens, backend and library version now go into every
  record. The docstring had claimed this for two releases; only the prompt
  suffix was actually stored, which is not enough to regenerate a single
  trajectory.
- `README.md` no longer claims the coverage/selection split is something "no
  other library gives you". `pass@k` beside `maj@n` exists in the evaluation
  harnesses, the technical note said so, and the two documents disagreed.
- 175 → 177 tests.

### Note on the numbers

The published GSM8K results are unchanged by all of this: 45.1% → 66.5% at
N=128, coverage 93.5%. Two trajectories that had been voting with a mangled
repeating decimal now abstain, which moves the abstention count from 801 to
803 and nothing else.

## 1.1.2 — 2026-08-25

A second adversarial audit went after the parts of 1.1.1 that had not been
attacked yet: the equivalence partition, the case-folding step, the token
budget the published run used, and the claims the documentation makes about
all three. The numbers it re-derived from the published trajectories matched
what we had published, but it found real defects around them. This release
closes every one, and the GSM8K measurement has been regenerated with a token
budget that lets the model finish.

### Fixed

- **The equivalence partition depended on the order of the pool.** `equivalent`
  is not transitive — `math-verify` accepts `0.3333333333` against `1/3` and
  `1/3` against `0.33333333333333`, but rejects the two decimals against each
  other. Grouping greedily against one representative per class therefore
  produced a different partition depending on which answer the model happened
  to emit first, so the same trajectories in a different order could return a
  different winner. Grouping is now the transitive closure over all pairs,
  computed with a union-find, which is unique by construction. The old test
  compared only the *sizes* of the classes and so passed over the violation;
  it now checks the partition itself, across every permutation.
- **Case folding destroyed LaTeX.** `normalise` ended in `.upper()`, turning
  `\frac{1}{2}` into `\FRAC{1}{2}` — a token no symbolic parser accepts. The
  effect was that `is_correct()` returned False on answers `covered()` called
  True, inflating the very selection gap this library exists to measure.
  Folding now steps over control words and still applies everywhere else, so
  `x=5` and `X=5` remain one vote.
- **Mixed numbers were read as products.** Juxtaposition means multiplication
  everywhere else in the LaTeX converter, so `2\frac{1}{2}` — two and a half —
  became `2*(1/2)` = 1. Not a lost answer but a fabricated one. Integers
  written directly against a `\frac` are now resolved as a sum, sign included;
  `x\frac{1}{2}` really is a product and is untouched.
- **A verifier returning the wrong number of scores was silently tolerated.**
  The scores are zipped against the trajectories, so a short list dropped the
  tail of the pool: the vote still ran, still returned a plausible answer, and
  never mentioned that it had ignored trajectories. The length is now checked,
  and a mismatch falls back to per-trajectory scoring with a warning.
- **The two backends still disagreed about the mean log-probability.** 1.1
  excluded the terminal EOS token from the vLLM token *count* but left its
  contribution in `cumulative_logprob`, so the numerator and the denominator
  counted different things — about a 12.5% error at this model's answer
  lengths. The EOS term is now subtracted from both.
- **`warn_if_no_math_verify` was never called.** Without `math-verify`,
  equivalent answers written differently vote as separate blocs and the merge
  silently does nothing. The warning now fires once, at engine construction.
- **The published summary reported `p_value: 0.0`.** Rounding 5.653e-08 to six
  decimal places produces a zero, which reads as an exact result rather than
  the number we computed. It is now written in scientific notation.
- **`pytest` could not collect the suites.** Both test files ended in a bare
  `sys.exit`, which aborts collection with an INTERNALERROR — and `pytest` is
  the first thing anyone who clones the repository types. They run under
  `pytest` and as scripts now.

### Changed

- **The GSM8K run was regenerated with `max_tokens=1024`.** At 400 the model
  ran out of room on 20.8% of trajectories, and a truncated trajectory
  abstains: it costs full generation time and casts no vote. Truncation is now
  0.8% and 96.9% of trajectories answer. Every published figure has been
  re-derived from the new trajectories, which are published in full as before.
  The default in `scripts/run_gsm8k.py` has been raised to match.
- **Every selector is now reported against the random baseline**, not just
  majority. `self_certainty` had been described in three places as frequently
  losing to random; nobody had measured it. It does not lose to random — the
  claim is gone and the measurement is in its place.
- **The symbolic-merge cap is counted instead of silenced.** The comment
  justifying the old behaviour claimed GSM8K answers are plain integers so
  merging has nothing to do. That is not true, and the audit demonstrated it.
  `scripts/analyse.py` now reports how many `select()` calls hit the cap so
  the effect on the published numbers is measured rather than assumed.

### Documentation

- The GitHub repository description and topics still advertised the withdrawn
  1.0.0 result — a trained verifier and an AIME score — on the repository
  homepage, where they were the first thing a visitor read. Replaced with what
  the project actually does.
- Reproducing the published figures takes about two minutes on a laptop CPU,
  not "about a minute". Corrected everywhere it appeared. (It was nine minutes
  before `equivalent()` was memoised, which is where that figure came from.)
- `USAGE.md` said the vLLM backend had not been re-verified on hardware while
  `README.md` said both backends were tested. Both are tested; the run behind
  the published numbers uses vLLM.
- `examples/quickstart.py` and `USAGE.md` both compared `r.answer == gold`,
  which is exactly the comparison the library's own docstring tells callers not
  to make. Both use `is_correct()` now.

## 1.1.1 — 2026-08-17

An adversarial audit of 1.1.0 found that some of its fixes had introduced
defects of the same kind they were meant to remove. This release closes those,
and adds the test coverage whose absence let them through.

### Fixed

- **`\boxed{(2)(3)}` returned `23`.** Redundant-parenthesis stripping did not
  recognise multiplication by juxtaposition, so `(2)(3)` — six — became
  twenty-three, and `(10)(10)` became `1010`. This is the same class of defect
  as 1.0.0's comma handling: not a lost answer but a fabricated, plausible one.
  A parenthesis touching another parenthesis is now preserved.
- **Set braces were being deleted.** `\boxed{\{1,2,3\}}` compared equal to the
  tuple `(1,2,3)`, and `\{1\}` compared equal to the scalar `1`. Set delimiters
  now survive brace removal.
- **Deeply nested LaTeX produced plausible-looking garbage.** Resolution
  stopped after four levels, leaving fragments that collapsed into strings like
  `fracfrac...22`. The bound is now 24, and anything still unresolved abstains
  rather than voting with a fragment.
- **Unit stripping was inconsistent.** `204\text{ km}` gave `204` but
  `204\text{ m/s}` gave `204m/s`, splitting the vote between them. Units may
  now contain digits, slashes and exponents.
- **The equivalence-class representative is now the most-voted member.** It was
  the alphabetically first, so `["2*3", "6", "6"]` returned the unevaluated
  `2*3` even though two of three trajectories said `6`.
- **`select(..., "oracle")` uses the same equivalence rule as `coverage()`.**
  The two disagreed: `covered(gold)` reported an answer as reachable while the
  oracle selector reported it as missing.
- **Symbolic merging is bounded and computed once per call.** It ran twice per
  `select()` and had no size limit; at 128 distinct answers a single call took
  twenty seconds. Two distinct finite numbers now short-circuit without
  invoking the symbolic parser at all, and pools above 24 distinct answers fall
  back to exact matching with a warning.
- **A positive log-probability no longer raises `OverflowError`.**
- **`verifier_argmax` validates every score**, not only those attached to a
  usable answer. The same pool raised under `verifier` and passed here.
- **The sigmoid stays inside `(0, 1)`.** It saturated to exactly `0.0`, and a
  pool of zeros degenerates into a majority vote — the failure the score-range
  check exists to prevent.
- **The verifier is called once per batch**, not once per trajectory. The
  documented `batch_size` had no effect when used through `BestOfN`.
- **Both backends exclude the terminal EOS** from the token count and the mean
  log-probability. 1.1.0 claimed they measured the same quantity; they did not.

### Added

- **`Result.is_correct(gold)`** — equivalence-aware scoring. Use it instead of
  `result.answer == gold`: a pool that agreed on `0.5` is a correct answer to a
  gold of `1/2`, and a string comparison would score it wrong while `covered()`
  scored it right, inflating the very gap this library exists to measure.
- **`tests/test_engine_and_verifiers.py`** — 67 tests covering the engine, the
  verifier adapters and the equivalence merging. None of those had any coverage
  in 1.1.0, which is where all three of the audit's worst findings were.
- Per-N confidence intervals in `scripts/analyse.py`.

### Measurements

- **Re-run at N=128** on the same 200 GSM8K problems, on an RTX 3090 through
  the **vLLM backend** (which this release therefore exercises end to end):
  41.8% at a single sample to **65.0%** with majority voting, coverage **91.0%**,
  exact McNemar against random `p = 5.7e-08`. All 25,600 trajectories are
  published in full.

### Changed

- **The published accuracy curve now goes through `select()`** rather than
  reimplementing the vote with a `Counter`. A curve that bypasses the library
  does not exercise its tie-breaking or its answer merging, and can drift from
  what users actually get. Majority at N=16 moves from 54.7% to 54.8% as a
  result.
- Total tests: 155.

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
