# Using Best-of-N 1.1

A practical guide: how to install it, how to read what it tells you, how to
plug in a reward model, and how to measure the gain on your own task so the
number you report is one you can stand behind.

- [Install](#install)
- [The thirty-second version](#the-thirty-second-version)
- [Reading the diagnostic](#reading-the-diagnostic)
- [Choosing N](#choosing-n)
- [Selectors](#selectors)
- [Using someone else's reward model](#using-someone-elses-reward-model)
- [Answer extraction](#answer-extraction)
- [Measuring honestly](#measuring-honestly)
- [Troubleshooting](#troubleshooting)

---

## Install

```bash
pip install bestofn
```

The package itself has no dependencies, so the selectors run anywhere. Add what
you need:

```bash
pip install "bestofn[math]"           # symbolic answer comparison, recommended
pip install torch transformers        # to run a model locally
pip install vllm                      # much faster; required for large N
```

`[math]` pulls in `math-verify`. Without it, `1/2` and `0.5` are counted as two
different votes rather than one. For integer answers it makes no difference;
for anything else it matters.

---

## The thirty-second version

```python
from bestofn import BestOfN

engine = BestOfN("Qwen/Qwen2.5-0.5B-Instruct", n=16)
r = engine.solve("A train travels at 60 km/h for 3 hours. How far?")

r.answer          # '180'
r.agreement       # 0.81  -- how much the voters agreed
r.effective_n     # 13    -- how many of the 16 actually voted
```

---

## Reading the diagnostic

This is the part worth learning, because it is what turns the library from a
wrapper into a tool.

```python
r = engine.solve(problem)

r.n              # trajectories you paid for
r.effective_n    # trajectories that produced a usable answer
r.n_abstained    # produced nothing
r.n_truncated    # ran out of tokens
r.total_tokens   # what it cost
r.agreement      # agreement among the ones that voted
```

**When `effective_n` is well below `n`, fix that before anything else.** You are
paying for trajectories that contribute nothing to the answer, and every
per-sample statistic you compute is wrong. The usual cause is `max_tokens`
being too low for the model's reasoning style — check `n_truncated` — or the
model not using `\boxed{}` — check the prompt.

Then compare what was returned against what was reachable:

```python
returned  = r.is_correct(gold)   # not `r.answer == gold` -- see below
reachable = r.covered(gold)
```

| `returned` | `reachable` | What it means | What to do |
|---|---|---|---|
| ✗ | ✗ | The model never found it | More samples will not help. Better model, or better prompt |
| ✗ | ✓ | Found it and threw it away | A **selection** problem. A verifier can fix this |
| ✓ | ✓ | Working | Consider whether you need this much N |

That table is the whole method. Coverage tells you what is inside the model;
the selector tells you how much of it you can get out.

> **On `covered()`:** it is a diagnostic, not a target. The problems that
> sustain its tail are single correct answers among N — by definition never the
> mode, and indistinguishable from noise without the label. Treating the gap
> between coverage and accuracy as "headroom available" is too optimistic:
> some of it is structurally out of reach.

---

## Choosing N

Cost is linear in N; accuracy is not. Coverage grows as `1 − (1−p)^N`, which
saturates.

| Per-sample accuracy `p` | Useful range of N |
|---|---|
| under 10% | Best-of-N will not save you |
| 20–60% | **the sweet spot** — this is where it pays |
| over 80% | you are mostly paying N times for the same answer |

Measure your own `p` first — the mean correctness across trajectories. It is
what a single sample gets you, and it is the number the whole curve is built
on.

It is *close to*, but not the same as, what `random` scores at large N. Random
picks among the trajectories that produced an answer, so it skips the
abstentions and comes out slightly higher: on our published run p is 45.3% and
`random` at N=128 is 46.3%. That roughly one-point wedge is the "not the
method" term in the decomposition the [README](README.md#how-much-of-the-gain-is-really-selection)
reports, and conflating the two is what makes it disappear from a write-up.

```python
from bestofn.extract import equivalent

results = engine.solve_batch(problems, n=16)
p = sum(bool(s.answer) and equivalent(s.answer, g)
        for r, g in zip(results, golds) for s in r.samples) \
    / sum(r.n for r in results)
```

Use `equivalent`, not `==`, and not `normalise(a) == normalise(b)` either.
On our own published data the three give 0.4534, 0.4425 and 0.4527: plain `==`
scores a trajectory wrong for writing `1,000` where the gold says `1000`, and
comparing canonical keys still misses `0.5` against `1/2`. `equivalent` is what
`Result.is_correct` and `coverage` both use, so it is the one that makes your
number comparable to the ones in the README.

**There is little point in `n=2`.** With two trajectories there is no majority
to speak of, so it costs twice as much as one sample and gains
nothing. Start at 8.

---

## Selectors

Every selector runs over an existing result at no cost, so compare them:

```python
r = engine.solve(problem, n=16)

r.select_with("random", seed=0)     # the baseline
r.select_with("majority")
r.select_with("verifier")           # needs verifier=... at construction
```

| Method | Needs | Notes |
|---|---|---|
| `random` | nothing | **The baseline.** Anything that does not beat it is doing harm |
| `majority` | nothing | The sensible default |
| `self_certainty` | `logprobs=True` | Weights votes by the model's own confidence. **66.5%** on our GSM8K run, level with `majority` |
| `verifier` | a verifier callable | The only one that can promote a minority answer |
| `verifier_argmax` | a verifier callable | Single best trajectory, no vote |
| `oracle` | the gold answer | Diagnostic only, never deployable |

**Always print the `random` row.** It costs nothing and it is the difference
between "my selector gets 53%" and "my selector gets 53% where guessing gets
51%".

---

## Using someone else's reward model

This library does not ship a reward model. It works with published ones.

```python
from bestofn import BestOfN
from bestofn.verifiers import from_hub

verifier = from_hub("openbmb/Eurus-RM-7b")          # Apache-2.0
engine = BestOfN("your/model", n=16, verifier=verifier)
engine.solve(problem, method="verifier")
```

### The trap this used to fall into

Reward models emit **unbounded logits**, not probabilities. Weighting a vote by
`-3.7` is meaningless. Passing raw logits now raises an error rather than
quietly falling back to a majority vote:

```
ValueError: verifier scores must be probabilities in [0, 1], got
[-0.7, -0.01]. Reward models usually return unbounded logits: apply a
sigmoid first...
```

The adapters in `bestofn.verifiers` do this for you. For your own scorer:

```python
from bestofn.verifiers import from_callable

verifier = from_callable(my_reward_model)                      # logits
verifier = from_callable(my_scorer, already_probability=True)  # already [0,1]
```

### Licences

A reward model's licence governs how you may use its scores. They are not
uniform, and several popular ones are more restrictive than they look. Checked
on the Hub in August 2026:

| Model | Licence | Notes |
|---|---|---|
| `openbmb/Eurus-RM-7b` | **apache-2.0** | Cleanest of the group |
| `OpenAssistant/reward-model-deberta-v3-large-v2` | **mit** | Small; general preference, not maths |
| `internlm/internlm2-1_8b-reward` | other | 1.8B, the smallest usable one. Read the repo terms |
| `Skywork/Skywork-Reward-V2-Llama-3.1-8B` | llama3.1 | Meta acceptable-use policy applies |
| `Qwen/Qwen2.5-Math-PRM-7B` | other | Step-level PRM, not a drop-in ORM |

`from_hub` warns when a licence is anything other than permissive, and
`bestofn.verifiers.license_of(model_id)` queries the Hub live so you are not
relying on this table staying current.

> **Pick your verifier by measuring it, not by its size.** Reward-model
> quality varies enormously and does not track parameter count: the 7B models
> in the table above are the ones with a track record, while some sub-2B ones
> score at or below chance on PRMBench. You do not have to guess — running a
> verifier against `random` on a hundred of your own problems takes minutes and
> settles it. That comparison is one line, and this library prints it for you.

---

## Answer extraction

```python
BestOfN(model, extractor="boxed")     # default: last \boxed{...}
BestOfN(model, extractor="number")    # last number
BestOfN(model, extractor="letter")    # multiple choice
BestOfN(model, extractor=my_function) # anything str -> str
```

Two behaviours worth knowing:

**A trajectory with no answer abstains.** If there is no `\boxed{}`, extraction
returns `""` and that trajectory does not vote. It does *not* guess at the last
number in the text — an unfinished chain of thought would otherwise cast a vote
indistinguishable from a real one. If your model reliably answers without
`\boxed{}`, opt in:

```python
from bestofn.extract import extract_boxed
BestOfN(model, extractor=lambda t: extract_boxed(t, allow_fallback=True))
```

**Structure is preserved.** `\boxed{\frac{1}{2}}` gives `1/2`, not `1`;
`\boxed{(3,4)}` gives `(3,4)`, not `34`. With `math-verify` installed,
equivalent answers written differently are pooled into one vote.

---

## Measuring honestly

If you are going to publish a number, these four things are what a reviewer
will ask for, and all four are cheap.

**1. The random baseline, in the same table.** Non-negotiable.

**2. Effective N, not N.** Report how many trajectories actually voted.

**3. A paired test, not a difference of percentages.** Two selectors on the
same problems are paired data. What matters is how many problems A got right
and B got wrong, and vice versa. `scripts/analyse.py` computes exact McNemar
from those counts.

**4. Confidence intervals.** A 30-problem benchmark at p≈0.5 has a standard
error of about 9 points. Differences smaller than that are not differences.

`scripts/analyse.py` produces all four from a trajectory file, and it
re-extracts answers from the raw text rather than trusting the ones stored
alongside them — so it can catch an extraction bug rather than inheriting one.

---

## Troubleshooting

**`method='self_certainty' requires log-probabilities`**
Construct with `logprobs=True`. They are off by default because collecting them
is the most memory-hungry part of generation on the `transformers` backend.

**Out of memory with `logprobs=True`**
Lower `max_parallel`. The engine sizes it from free VRAM, but a shared GPU can
change underneath it:

```python
BestOfN(model, n=32, logprobs=True, max_parallel=4)
```

**Everything abstains**
The model is not boxing its answers. Three things fix this, in order of how
often they work: check `prompt_suffix` actually asks for `\boxed{}`, raise
`max_tokens` so the reasoning has room to reach the box, or pass a different
extractor. If you know the model's answers are reliable without the box,
`allow_fallback=True` takes the last number instead.

**`temperature must be > 0 for Best-of-N`**
At temperature 0 all N samples are identical. Use 0.6–1.0.

**The two backends give different trajectories**
Expected, and harmless. Different kernels and different batching draw different
samples from the same distribution, so accuracy matches but the individual
texts do not. Both report the same *quantity* for `logprob` as of 1.1, so
`self_certainty` behaves identically across them.

> **Both backends are tested on real hardware.** `transformers` runs anywhere
> torch runs; `vllm` is much faster at large N and is the one the published
> GSM8K run was generated with. As of 1.1 they report the same quantity for
> `logprob`, including identical treatment of the terminal token.

---

## Reproducing the published numbers

```bash
python scripts/run_gsm8k.py       # regenerate trajectories (needs a GPU)
python scripts/analyse.py         # re-derive every number (no GPU)
```

The second command works on the published trajectory file without regenerating
anything, so you can check the results without hardware.
