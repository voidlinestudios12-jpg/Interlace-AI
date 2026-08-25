# -*- coding: utf-8 -*-
"""Best-of-N in five minutes.

    pip install "bestofn[math]" torch transformers
    python examples/quickstart.py

Runs a 0.5B model on one consumer GPU. The point of the example is not the
answer -- it is the three numbers printed alongside it, which are what tell you
whether Best-of-N is helping on *your* task.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
from bestofn import BestOfN, have_math_verify

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROBLEMS = [
    ("A shop sells pens at 3 euros each. Ana buys 7 pens and pays with a "
     "50 euro note. How much change does she get?", "29"),
    ("A train travels at 60 km/h for 3 hours. How far does it go?", "180"),
    ("Luis has 24 biscuits and gives half to his sister, then eats 3. "
     "How many are left?", "9"),
]


def main():
    if not have_math_verify():
        print('note: install "bestofn[math]" for symbolic answer comparison\n')

    engine = BestOfN(MODEL, n=8, temperature=0.7, max_tokens=400)
    print(f"model: {MODEL}   n=8\n")

    selected = reachable = 0
    for problem, gold in PROBLEMS:
        r = engine.solve(problem)

        # The habit worth forming: never read a selector's accuracy without
        # the random baseline next to it. A selector that cannot beat picking
        # a trajectory at random is not selecting, it is adding noise.
        rnd = r.select_with("random", seed=0)

        # Not `r.answer == gold`: is_correct() applies the same
        # canonicalisation the vote used, so 0.5 scores against 1/2.
        hit = r.is_correct(gold)
        selected += hit
        reachable += r.covered(gold)

        print(f"  {problem[:58]}...")
        print(f"    gold {gold:<6} majority {r.answer or '-':<8} "
              f"random {rnd or '-':<8} {'correct' if hit else 'wrong'}")
        print(f"    voted {r.effective_n}/{r.n}   "
              f"abstained {r.n_abstained}   truncated {r.n_truncated}   "
              f"agreement {r.agreement:.0%}\n")

    n = len(PROBLEMS)
    print(f"  returned  {selected}/{n}   what the system actually answers")
    print(f"  reachable {reachable}/{n}   whether any trajectory found it\n")

    if reachable > selected:
        print("  Some answers were generated and then not selected. That is a")
        print("  selection problem: a verifier can help. See bestofn.verifiers.")
    elif reachable < n:
        print("  Some answers were never generated at all. More samples will")
        print("  not fix that -- you need a stronger model or a better prompt.")
    else:
        print("  Everything reachable was returned. Nothing to fix here.")


if __name__ == "__main__":
    main()
