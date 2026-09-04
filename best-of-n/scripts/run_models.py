# -*- coding: utf-8 -*-
"""Run the same GSM8K protocol across several models.

One model on one benchmark says something about that model. The question this
answers is the obvious next one: does the shape hold -- does coverage stay well
above what majority voting returns, and does the gain survive as the base model
gets better?

Deliberately identical to ``run_gsm8k.py`` in every respect except the model:
same 200 problems, same prompt, same temperature, same token budget, same N.
A comparison where the protocol drifts between rows is not a comparison.

Each model writes its own trajectory file, so ``scripts/analyse.py`` can be
pointed at any one of them and re-derive that row from scratch.

Run:  python scripts/run_models.py --problems-file problems.json --n 64

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
import argparse
import gc
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bestofn import BestOfN, coverage, normalise, select        # noqa: E402
from bestofn import __version__ as _VERSION                     # noqa: E402
from bestofn.extract import equivalent, extract_boxed           # noqa: E402

#: The comparison set. Chosen to vary two things independently -- size, and
#: whether the model was trained for mathematics -- because those are the two
#: axes a reader will ask about.
MODELS = [
    # (id, label, N) -- N drops for the large models because the cost is linear
    # in it and the point of including them is the shape of the curve, not a
    # like-for-like N against the 0.5B.
    ("Qwen/Qwen2.5-0.5B-Instruct",         "0.5B general",            64),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "1.7B, different family",  64),
    ("Qwen/Qwen2.5-1.5B-Instruct",         "1.5B general",            64),
    ("Qwen/Qwen2.5-Math-1.5B-Instruct",    "1.5B maths-tuned",        64),
    ("Qwen/Qwen2.5-3B-Instruct",           "3B general",              64),
    ("microsoft/Phi-3-mini-4k-instruct",   "3.8B, different family",  64),
    ("Qwen/Qwen2.5-7B-Instruct",           "7B general",              64),
    # A vision-language model, run text-only. That is a legitimate way to use
    # it and GSM8K has no images, but it is not the same kind of object as the
    # rows above and the label has to say so -- calling it "27B
    # frontier-class" beside six text models would imply a comparison the
    # architecture does not support.
    # FP8, not bf16, and the label says so. Qwen publishes both; FP8 is 30.9 GB
    # of weights against 55.6, which fits a 48 GB card instead of needing an
    # 80 GB one at three times the price. Substituting a quantised build and
    # labelling it as the base model would be exactly the sort of quiet swap
    # this library was written to detect.
    ("Qwen/Qwen3.8-27B-FP8",  "27B vision-language, text-only, FP8", 64),
]

#: meta-llama/Llama-3.2-1B-Instruct was here and is gated: it returns 403
#: without an accepted licence, and the failure took the whole sweep down
#: rather than skipping the row. Everything above is ungated, so the table can
#: be reproduced by anyone.

SUFFIX = ("\n\nPlease reason step by step, and put your final answer "
          "within \\boxed{}.")


def hit(answer, gold):
    """One correctness rule, the same one scripts/analyse.py uses."""
    return bool(answer) and equivalent(answer, gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems-file", required=True)
    ap.add_argument("--n", type=int, default=0,
                    help="override N for every model; 0 uses the per-model default")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out-dir", default="results/models")
    ap.add_argument("--only", default=None,
                    help="substring; run just the models matching it")
    args = ap.parse_args()

    problems = json.load(io.open(args.problems_file, encoding="utf-8"))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(problems)} problems, N={args.n or 'per-model'}, "
          f"max_tokens={args.max_tokens}, bestofn {_VERSION}")

    summary, skipped = [], []
    for model_id, label, n_model in MODELS:
        if args.only and args.only.lower() not in model_id.lower():
            continue
        n_here = args.n if args.n else n_model
        slug = model_id.split("/")[-1]
        path = os.path.join(args.out_dir, f"{slug}.jsonl")
        if os.path.exists(path) and _matches(path, problems, n_here, args):
            print(f"\n== {model_id}  (already done with the same settings, "
                  f"skipping)")
        else:
            print(f"\n== {model_id}  [{label}]")
            t0 = time.time()
            try:
                engine = BestOfN(model_id, n=n_here, temperature=args.temperature,
                                 top_p=args.top_p, max_tokens=args.max_tokens,
                                 extractor="boxed", prompt_suffix=SUFFIX,
                                 backend="vllm", logprobs=True,
                                 gpu_memory_utilization=0.90,
                                 # Prompt AND generation share this budget, so
                                 # it has to exceed max_tokens or the engine
                                 # truncates at max_model_len and the token
                                 # budget you asked for is fiction. A 3072-token
                                 # run against a 2048 window truncated 4.4% of
                                 # trajectories and paid for generation it could
                                 # never produce.
                                 max_model_len=args.max_tokens + 2048)
            except BaseException as exc:                        # noqa: BLE001
                # BaseException, not Exception: a gated repo surfaced as an
                # OSError that escaped and killed the sweep on its second row,
                # and a vLLM worker can exit in ways Exception does not cover.
                # One model failing to load is a missing row, not a lost run.
                print(f"   COULD NOT LOAD: {type(exc).__name__}: "
                      f"{str(exc)[:200]}", flush=True)
                skipped.append((model_id, label, type(exc).__name__))
                continue

            with io.open(path, "w", encoding="utf-8") as fh:
                for start in range(0, len(problems), args.batch):
                    chunk = problems[start:start + args.batch]
                    results = engine.solve_batch([p["question"] for p in chunk])
                    for item, res in zip(chunk, results):
                        fh.write(json.dumps({
                            "gsm8k_id": item.get("id"),
                            "question": item["question"],
                            "gold": item["gold"],
                            "prompt_suffix": SUFFIX,
                            "config": {
                                "model": model_id, "n": n_here,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "max_tokens": args.max_tokens,
                                "backend": "vllm",
                                "bestofn_version": _VERSION,
                            },
                            "trajectories": [
                                {"text": s.text, "answer": s.answer,
                                 "finish_reason": s.finish_reason,
                                 "logprob": s.logprob, "n_tokens": s.n_tokens}
                                for s in res.samples],
                        }, ensure_ascii=False) + "\n")
                        fh.flush()
                    done = min(start + args.batch, len(problems))
                    el = time.time() - t0
                    print(f"   {done}/{len(problems)}  {el/done:.1f}s/problem  "
                          f"~{el/done*(len(problems)-done)/60:.0f} min left",
                          flush=True)
            del engine
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                                   # noqa: BLE001
                pass

        summary.append(score(path, model_id, label, n_here))

    out = os.path.join(args.out_dir, "summary.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + "=" * 78)
    print(f"{'model':<34} {'N=1':>7} {'maj':>7} {'rand':>7} {'cover':>7} "
          f"{'gain':>7} {'drift':>6}")
    print("-" * 78)
    for r in summary:
        print(f"{r['model'].split('/')[-1]:<34} {r['single']:>6.1f}% "
              f"{r['majority']:>6.1f}% {r['random']:>6.1f}% "
              f"{r['coverage']:>6.1f}% {r['gain']:>+6.1f} "
              f"{r['reextraction_drift']:>6}")
    if skipped:
        print("\nNOT MEASURED (%d):" % len(skipped))
        for mid, lbl, exc in skipped:
            print(f"  {mid:<38} {lbl:<28} {exc}")
        print("  These are absent from the table above and from the chart.")
    print("=" * 78)
    print(f"written to {out}")
    return 0


def _matches(path, problems, n, args) -> bool:
    """Whether an existing file was produced by the settings asked for now.

    The gate used to be a bare line count, so re-running with a different
    ``--n``, ``--max-tokens``, ``--temperature`` or a different problem file of
    the same length silently reused the old trajectories -- and the chart then
    labelled them with the N that had been requested rather than the N that
    was generated. Compare the stored config; regenerate on any mismatch.
    """
    try:
        rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
    except Exception:                                           # noqa: BLE001
        return False
    if len(rows) < len(problems):
        return False
    cfg = rows[0].get("config") or {}
    if len(rows[0].get("trajectories", [])) != n:
        return False
    for key, want in (("n", n), ("temperature", args.temperature),
                      ("top_p", args.top_p), ("max_tokens", args.max_tokens)):
        if cfg.get(key) != want:
            return False
    return True


def score(path, model_id, label, n):
    """Re-derive this model's row from its own trajectory file.

    **Answers are re-extracted from the raw text, not read from the stored
    ``answer`` field.** Reading the stored one would make this a replay that
    re-counts somebody else's extraction, which by construction cannot detect
    an extraction bug -- it inherits it. Every audit round so far has found at
    least one defect in the extractor, two of which fabricated votes rather
    than losing them, so a table built on stored answers would quietly carry
    whichever version was live when the GPU ran.

    The drift between stored and re-derived is reported. On a fresh run it is
    zero; run this again after changing the extractor and it is the count of
    trajectories whose answer moved.
    """
    rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
    # N comes from the data, never from the command line. Taking it from argv
    # let a chart claim N=128 over trajectories generated at N=64.
    n = len(rows[0]["trajectories"]) if rows else n
    from bestofn import Sample
    pools, golds, drift = [], [], 0
    for r in rows:
        pool = []
        for t in r["trajectories"]:
            fresh = extract_boxed(t["text"])
            if fresh != t.get("answer"):
                drift += 1
            pool.append(Sample(answer=fresh, text=t["text"],
                               logprob=t.get("logprob"),
                               finish_reason=t.get("finish_reason"),
                               n_tokens=t.get("n_tokens")))
        pools.append(pool)
        golds.append(r["gold"])

    total = sum(len(p) for p in pools)
    voting = sum(1 for p in pools for x in p if x.key)
    trunc = sum(1 for p in pools for x in p if x.truncated)
    tokens = sum(x.n_tokens or 0 for p in pools for x in p)

    # Single-sample accuracy is exactly the per-trajectory accuracy: no need to
    # estimate it by resampling, and estimating it adds noise to the headline.
    single = 100.0 * sum(1 for p, g in zip(pools, golds)
                         for x in p if hit(x.key, g)) / total

    # Random-among-voters likewise has a closed form.
    rnd = 0.0
    for p, g in zip(pools, golds):
        voters = [x for x in p if x.key]
        if voters:
            rnd += sum(1 for x in voters if hit(x.key, g)) / len(voters)
    rnd = 100.0 * rnd / len(golds)

    maj = 100.0 * sum(1 for p, g in zip(pools, golds)
                      if hit(select(p, "majority"), g)) / len(golds)
    cov = 100.0 * sum(1 for p, g in zip(pools, golds)
                      if coverage(p, g)) / len(golds)
    return {
        "model": model_id, "label": label, "n": n,
        "problems": len(rows), "trajectories": total,
        "single": round(single, 2), "random": round(rnd, 2),
        "majority": round(maj, 2), "coverage": round(cov, 2),
        "gain": round(maj - single, 2),
        "selection_gain": round(maj - rnd, 2),
        "abstention_gain": round(rnd - single, 2),
        "headroom": round(cov - maj, 2),
        "reextraction_drift": drift,
        "voting_rate": round(100.0 * voting / total, 2),
        "truncation_rate": round(100.0 * trunc / total, 2),
        "tokens": tokens,
    }


def rescore(out_dir="results/models"):
    """Re-derive every model's row from the raw text already on disk.

    The point of publishing complete trajectories is that a change to the
    extractor or the selectors costs a couple of CPU minutes, not another GPU
    run. Generation is the expensive and irreproducible part; scoring is
    neither.
    """
    summary = []
    for model_id, label, n_model in MODELS:
        path = os.path.join(out_dir, model_id.split("/")[-1] + ".jsonl")
        if os.path.exists(path):
            summary.append(score(path, model_id, label, n_model))
    out = os.path.join(out_dir, "summary.json")
    io.open(out, "w", encoding="utf-8").write(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"{'model':<34} {'N=1':>7} {'maj':>7} {'rand':>7} {'cover':>7} "
          f"{'gain':>7} {'drift':>6}")
    print("-" * 78)
    for r in summary:
        print(f"{r['model'].split('/')[-1]:<34} {r['single']:>6.1f}% "
              f"{r['majority']:>6.1f}% {r['random']:>6.1f}% "
              f"{r['coverage']:>6.1f}% {r['gain']:>+6.1f} "
              f"{r['reextraction_drift']:>6}")
    print(f"\nwritten to {out}")
    return summary


if __name__ == "__main__":
    if "--rescore" in sys.argv:
        i = sys.argv.index("--rescore")
        d = (sys.argv[i + 1] if len(sys.argv) > i + 1
             and not sys.argv[i + 1].startswith("-") else "results/models")
        rescore(d)
        sys.exit(0)
    sys.exit(main())
