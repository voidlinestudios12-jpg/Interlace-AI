# Does it work on a model it was not developed against?

The measurements in the technical report were all taken on
`DeepSeek-R1-Distill-Qwen-1.5B`, a reasoning-distilled model, on competition
mathematics. A fair question is whether the method transfers, or whether it
happens to suit that one model.

This directory answers it with a fresh, independent run.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` — a general instruct model, not a reasoning distill, and a third of the size |
| Benchmark | GSM8K test set, 200 problems sampled with seed 20260815 |
| Sampling | 16 trajectories per problem, temperature 0.7, top-p 0.95, 400 max tokens |
| Hardware | one consumer GPU (RTX 3060, 12 GB) |
| Weights changed | none |

The 16 trajectories were generated **once**. The curve for N = 1, 2, 4, 8, 16 is
obtained by subsampling those same trajectories 400 times at each N, which is
why standard deviations are reported.

## Result

| N | Majority vote | ± sd | Coverage |
|---:|---:|---:|---:|
| 1 | 38.2% | 2.31 | 38.2% |
| 2 | 38.4% | 2.20 | 49.6% |
| 4 | 45.5% | 2.04 | 60.0% |
| 8 | 50.9% | 1.62 | 70.1% |
| **16** | **53.3%** | **1.01** | **79.5%** |

**+15.1 points from a single sample to N=16**, on a model the method was not
developed against, on a task it was not tuned for.

Two things worth noting beyond the headline:

- **The standard deviation shrinks as N grows** (2.31 → 1.01). Best-of-N does
  not only raise accuracy, it makes the system *more predictable* — a
  single-sample system is a coin flip in a way an N=16 system is not.
- **N=2 is not better than N=1.** With two trajectories there is no majority to
  speak of; ties are broken arbitrarily. The mechanism needs a few voters
  before it has anything to work with.

## Files

| File | What it is |
|---|---|
| `gsm8k_qwen05b_n16.jsonl` | One line per problem: the gold answer, the answer extracted from each of the 16 trajectories, and how many were correct |
| `summary.json` | The curve above, machine-readable |
| `run_benchmark.py` | The script that produced it, resumable |

## Reproducing it

```bash
pip install bestofn torch transformers datasets
python run_benchmark.py
```

Roughly 70 minutes on an RTX 3060. Or recompute the curve from the published
trajectories without a GPU at all:

```python
import json, random
from collections import Counter
from bestofn import normalise

rows = [json.loads(l) for l in open("gsm8k_qwen05b_n16.jsonl", encoding="utf-8")]
rng = random.Random(1)
for n in (1, 2, 4, 8, 16):
    hits = 0
    for row in rows:
        picked = [normalise(a) for a in rng.sample(row["respuestas"], n) if a]
        if picked and Counter(picked).most_common(1)[0][0] == normalise(row["correcta"]):
            hits += 1
    print(f"N={n:>2}  {100*hits/len(rows):.1f}%")
```

GSM8K is distributed by OpenAI under the MIT licence, so the problems
themselves are freely redistributable — but only the extracted answers are kept
here, since that is all the curve needs.
