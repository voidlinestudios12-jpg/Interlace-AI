# Usage and configuration

Complete guide to `bestofn`. If you only read one section, read
[Choosing N](#choosing-n) and [Choosing a selector](#choosing-a-selector) —
those two decisions determine most of your accuracy.

---

## Install

```bash
pip install torch transformers
pip install vllm          # strongly recommended; required for large N
```

Then download this repository:

```python
from huggingface_hub import snapshot_download
snapshot_download("InterlaceAI/best-of-n", local_dir="best-of-n")
```

```bash
cd best-of-n && python tests/test_selectors.py    # 45 tests, no GPU needed
```

---

## Quickstart

```python
from bestofn import BestOfN

engine = BestOfN("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", n=32)

result = engine.solve("Find the number of ordered pairs (a,b) with a+b=100 and a,b>0.")

print(result.answer)       # selected answer
print(result.n)            # 32
print(result.agreement)    # 0.0-1.0, how much the samples agreed
```

Works with **any causal language model** on the Hub — swap the first argument.

---

## Choosing N

`N` is your compute budget: cost and latency scale linearly with it, accuracy
does not. Coverage follows `1 − (1−p)^N`, so it saturates.

Measured on DeepSeek-R1-Distill-Qwen-1.5B, AIME 2024:

| N | Majority vote | Coverage (ceiling) |
|---:|---:|---:|
| 1 | 23.3% | 23.3% |
| 8 | 40.0% | 60.0% |
| 16 | 46.7% | 63.3% |
| 32 | 50.0% | 73.3% |
| 64 | 50.0% | 80.0% |
| 128 | 53.3% | 83.3% |

Practical guidance:

| N | When to use |
|---|---|
| 1 | Baseline. No benefit from this library. |
| 4–8 | Easy tasks, tight latency. Most of the cheap gain is already here. |
| **16–32** | **Best default.** Strong gains, manageable cost. |
| 64–128 | Hard tasks (competition maths) where accuracy dominates cost. |
| >128 | Rarely worth it — coverage has flattened. Improve the *selector* instead. |

Above N≈32 majority voting stops improving while coverage keeps rising. That
divergence is the **selection gap**, and closing it needs a better selector,
not more samples.

---

## Choosing a selector

```python
engine.solve(problem, method="majority")     # default
```

| Method | Needs | Recovers a minority-correct answer? |
|---|---|:---:|
| `majority` | nothing | No |
| `self_certainty` | log-probs (automatic) | Rarely |
| `verifier` | a verifier callable | **Yes** |
| `verifier_argmax` | a verifier callable | Yes, but noisier |
| `oracle` | the gold answer | Diagnostic only |

**`majority`** — the modal answer. Free and robust. Structurally cannot pick an
answer that most trajectories disagree with, which is exactly what happens on
hard problems.

**`self_certainty`** — weights each vote by `exp(mean token log-prob)`. Helps at
moderate N. Measures fluency, which correlates with correctness but is not it.

**`verifier`** — weights each vote by an external `P(correct)`. The only
selector that attacks the selection gap. Measured on AIME (90 problems, N=32):
majority 35.6% → verifier **52.2%**, a gain of 16.6 points.

**`verifier_argmax`** — takes the single highest-scored trajectory. Simpler but
measured 8.9 points *worse* than the weighted vote (43.3% vs 52.2%): one
overconfident score can outvote a solid consensus.

**`oracle`** — needs the gold answer, so it can never be deployed. Use it to
measure your ceiling:

```python
r = engine.solve(problem, n=64)
print("selected:", r.answer)
print("was it reachable?", r.covered(gold))   # pass@N
```

### Reusing samples for free

Generation is the expensive part. Once you have a `Result`, trying other
selectors costs nothing:

```python
r = engine.solve(problem, n=32)
r.answer                        # majority
r.select_with("self_certainty")
r.select_with("oracle", gold="42")
```

---

## Using a verifier

A verifier is any callable `(problem, trajectory_text) -> float in [0,1]`.

```python
def my_verifier(problem: str, text: str) -> float:
    return score_between_0_and_1

engine = BestOfN(model, n=32, verifier=my_verifier)
engine.solve(problem, method="verifier")
```

With a trained reward model:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("your/verifier")
vm = AutoModelForSequenceClassification.from_pretrained(
    "your/verifier", num_labels=2, device_map={"": 0})
vm.eval()

def verifier(problem, text):
    prompt = f"Problem:\n{problem}\n\nProposed solution:\n{text}\n\nIs it correct?"
    ids = tok(prompt, return_tensors="pt", truncation=True,
              max_length=2048).to(vm.device)
    with torch.no_grad():
        return torch.softmax(vm(**ids).logits.float(), -1)[0, 1].item()
```

Two things matter more than the architecture:

1. **Train on problems disjoint from your evaluation set.** Contamination
   inflates results silently. Check by normalised exact match, substring
   containment, and n-gram overlap.
2. **Split train/validation by problem, not by trajectory.** Otherwise
   trajectories from the same problem land on both sides and the score is
   meaningless.

---

## Answer extraction

The default expects a final `\boxed{...}`. An extraction failure looks exactly
like a reasoning failure in your metrics, so match the extractor to your task.

```python
BestOfN(model, extractor="boxed")    # \boxed{...}  (default)
BestOfN(model, extractor="number")   # last number
BestOfN(model, extractor="letter")   # A/B/C/D multiple choice
BestOfN(model, extractor="regex", pattern=r"Answer:\s*(\w+)")
BestOfN(model, extractor=lambda text: text.strip().split()[-1])
```

Check it before a long run — silently broken extraction is the most common way
to waste a night of compute:

```python
from bestofn import get_extractor
ex = get_extractor("boxed")
print(ex(r"...therefore \boxed{42}."))   # '42'
```

---

## Full configuration

```python
BestOfN(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    n=32,                    # trajectories per problem
    temperature=0.6,         # MUST be > 0
    top_p=0.95,
    max_tokens=8192,         # cap per trajectory
    extractor="boxed",
    prompt_suffix="\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
    backend="auto",          # "auto" | "vllm" | "transformers"
    verifier=None,
    # extra kwargs go to the backend:
    gpu_memory_utilization=0.90,
    max_model_len=16384,
)
```

| Parameter | Notes |
|---|---|
| `temperature` | **Must be > 0.** At 0 all N samples are identical and Best-of-N gains nothing. The constructor raises if you try. 0.6–0.8 works well; higher raises coverage but lowers consensus. |
| `max_tokens` | Reasoning trajectories are long. Too low a cap truncates them and quietly costs accuracy — in our AIME runs ~65% of trajectories hit the cap. |
| `prompt_suffix` | Set `""` if your prompts already specify the output format. |
| `backend` | `vllm` shares the prompt KV-cache across the N samples, so N=32 costs far less than 32 separate calls. Use it whenever you can. |

---

## Batching

Always batch when solving many problems — much faster than looping:

```python
problems = [p1, p2, p3]
results = engine.solve_batch(problems, n=32)
accuracy = sum(r.answer == g for r, g in zip(results, golds)) / len(golds)
```

---

## Early stopping

`agreement` is the fraction of samples backing the modal answer. High agreement
early means more samples are unlikely to change the outcome:

```python
r = engine.solve(problem, n=8)
if r.agreement < 0.5:                 # samples disagree: spend more
    r = engine.solve(problem, n=64)
```

---

## Measuring the selection gap on your own task

This tells you whether to buy more samples or a better selector:

```python
results = engine.solve_batch(problems, n=32)

selected = sum(r.answer == g for r, g in zip(results, golds)) / len(golds)
reachable = sum(r.covered(g) for r, g in zip(results, golds)) / len(golds)

print(f"selected  {selected:.1%}")
print(f"coverage  {reachable:.1%}")
print(f"gap       {reachable - selected:.1%}")
```

- **Large gap** → the answer is being generated but not chosen. Invest in a
  verifier.
- **Small gap, low coverage** → the model rarely finds the answer at all. More
  samples or a stronger base model.

---

## Limitations

- **Coverage is a hard ceiling.** If the model never produces the correct
  answer, no selector recovers it. `1 − (1−p)^N` is 0 for every N when p = 0.
- **Cost is linear in N.** N=128 costs 128 generations.
- **Needs an extractable answer.** Designed for tasks with a comparable final
  answer (maths, multiple choice, short factual). Open-ended generation has no
  well-defined vote.
- **Verifier errors amplify.** A selector is bounded by its verifier; systematic
  verifier bias is not averaged out.

---

## Reference

Areces Rivera, A. (2026). *Modern Architecture On Advanced LLM: Best-of-N
Sampling, Learned Verification and Tree Search as a Substitute for Parameter
Scale.* Technical Report TR-2026-01, Interlace AI.
[doi.org/10.5281/zenodo.21936833](https://doi.org/10.5281/zenodo.21936833)

Questions: `interlaceIA@gmail.com`
