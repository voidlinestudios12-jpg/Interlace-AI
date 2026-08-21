# -*- coding: utf-8 -*-
"""Best-of-N on GSM8K, publishing everything needed to re-derive the numbers.

What makes this different from the 1.0.0 evaluation, and the reason it exists:
the file it writes contains the **full reasoning text** of every trajectory,
not the answer somebody already extracted from it. That is the difference
between a replay script that can catch an extraction bug and one that cannot.

Each record carries, per trajectory:

    text           the complete generated reasoning
    answer         what this version of the extractor made of it
    finish_reason  "stop" or "length"
    logprob        mean token log-probability under the model
    n_tokens       generated tokens

plus the prompt, the sampling parameters, the seed and the library version, so
the run is reproducible rather than merely reported.

Resumable: interrupt it and run it again.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
import argparse
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bestofn                                              # noqa: E402
from bestofn import BestOfN                                 # noqa: E402

DEFAULTS = dict(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    n=16,
    problems=200,
    max_tokens=400,
    temperature=0.7,
    top_p=0.95,
    seed=20260815,
)


def gold_answer(solution: str):
    """GSM8K stores the reference as a line ending in '#### 42'."""
    m = re.search(r"####\s*(-?[\d.,]+)", solution)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def load_problems(n_problems: int, seed: int):
    import random
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)

    out = []
    for i in order:
        if len(out) >= n_problems:
            break
        row = ds[i]
        gold = gold_answer(row["answer"])
        if gold is not None:
            out.append({"id": int(i), "question": row["question"], "gold": gold})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, val in DEFAULTS.items():
        ap.add_argument(f"--{key}", type=type(val), default=val)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-logprobs", action="store_true")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "gsm8k_trajectories.jsonl",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 70)
    print("BEST-OF-N ON GSM8K  ·  full trajectories published")
    print("=" * 70)
    print(f"  bestofn      : {bestofn.__version__}")
    print(f"  model        : {args.model}")
    print(f"  problems     : {args.problems}")
    print(f"  n            : {args.n}")
    print(f"  max_tokens   : {args.max_tokens}")
    print(f"  temperature  : {args.temperature}   top_p: {args.top_p}")
    print(f"  seed         : {args.seed}")
    print(f"  logprobs     : {not args.no_logprobs}")
    print("=" * 70, flush=True)

    problems = load_problems(args.problems, args.seed)
    print(f"\n  loaded {len(problems)} problems\n", flush=True)

    done = 0
    if os.path.exists(out_path):
        with io.open(out_path, encoding="utf-8") as fh:
            done = sum(1 for line in fh if line.strip())
        if done:
            print(f"  resuming: {done} already generated, "
                  f"{len(problems) - done} to go\n", flush=True)

    engine = BestOfN(
        args.model, n=args.n, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, extractor="boxed",
        logprobs=not args.no_logprobs, backend="transformers",
    )

    started = time.time()
    with io.open(out_path, "a", encoding="utf-8") as fh:
        for k, item in enumerate(problems, 1):
            if k <= done:
                continue
            res = engine.solve(item["question"])
            record = {
                "i": k,
                "gsm8k_id": item["id"],
                "question": item["question"],
                "gold": item["gold"],
                "prompt_suffix": engine.prompt_suffix,
                "trajectories": [
                    {
                        "text": s.text,
                        "answer": s.answer,
                        "finish_reason": s.finish_reason,
                        "logprob": s.logprob,
                        "n_tokens": s.n_tokens,
                    }
                    for s in res.samples
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

            if k % 10 == 0 or k == len(problems):
                elapsed = time.time() - started
                rate = elapsed / max(1, k - done)
                print(f"  {k:>3}/{len(problems)}  "
                      f"{rate:.0f}s/problem  "
                      f"~{rate * (len(problems) - k) / 60:.0f} min left",
                      flush=True)

    print(f"\n  finished in {(time.time() - started) / 60:.1f} min")
    print(f"  written to {out_path}")
    print("\n  Now run:  python scripts/analyse.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
