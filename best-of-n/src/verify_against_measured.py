"""Verification: replay the published measurements through this package.

The strongest available check is not a synthetic unit test -- it is feeding the
library the *same trajectories* that produced the numbers in the technical
report and confirming it reproduces them.

Data: results/best_of_n/COMPLETO_aime.jsonl from the Interlace-AI
repository (30 AIME 2024 problems, the answers extracted from every sampled
trajectory, and the selection each method made at the time of measurement).

Run:  python tests/verify_against_measured.py [path/to/COMPLETO_aime.jsonl]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bestofn import Sample, agreement, coverage, normalise, select  # noqa: E402

DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "measured_aime_n32.jsonl"
)

# From the technical report, Table 1 (AIME 2024, majority vote).
PUBLISHED_MAJORITY = 63.3
PUBLISHED_N1 = 23.3


def load(path):
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    if not os.path.exists(path):
        print(f"data not found: {path}")
        return 1

    rows = load(path)
    print(f"Loaded {len(rows)} problems from measured data\n")

    n_problems = len(rows)
    agree_majority = 0      # our selector agrees with the recorded selection
    correct_majority = 0    # our selector is correct against gold
    correct_recorded = 0    # the recorded selection was correct
    covered = 0             # pass@N
    single_sample_hits = 0
    single_sample_total = 0

    for row in rows:
        answers = row["respuestas"]
        gold = str(row["correcta"])
        samples = [Sample(answer=a) for a in answers]

        ours = select(samples, "majority")
        recorded = row["seleccion"]["mayoria"]["pred"]

        if normalise(ours) == normalise(recorded):
            agree_majority += 1
        if normalise(ours) == normalise(gold):
            correct_majority += 1
        if row["seleccion"]["mayoria"]["correcto"]:
            correct_recorded += 1
        if coverage(samples, gold):
            covered += 1

        # Expected single-sample accuracy = mean correctness over all samples.
        for a in answers:
            single_sample_total += 1
            if normalise(a) == normalise(gold):
                single_sample_hits += 1

    n_samples = single_sample_total // n_problems
    pct = lambda x: 100.0 * x / n_problems

    print(f"Samples per problem (N)          : {n_samples}")
    print("-" * 58)
    print("AGREEMENT WITH THE RECORDED RUN")
    print(f"  majority selection reproduced  : {agree_majority}/{n_problems}"
          f"  ({pct(agree_majority):.1f}%)")
    print("-" * 58)
    print("ACCURACY")
    print(f"  single sample (expected pass@1): {100.0*single_sample_hits/single_sample_total:.1f}%"
          f"   [one measured draw: {PUBLISHED_N1}%]")
    print("      note: the report's 23.3% is a single N=1 run; the figure above")
    print("      is the mean over all N samples. Both estimate the same p and")
    print("      differ by sampling noise (30 problems, sd ~8 points).")
    print(f"  majority vote, this library    : {pct(correct_majority):.1f}%"
          f"   [report: {PUBLISHED_MAJORITY}%]")
    print(f"  majority vote, recorded run    : {pct(correct_recorded):.1f}%")
    print(f"  coverage pass@{n_samples:<3}                : {pct(covered):.1f}%")
    print("-" * 58)

    gain = pct(correct_majority) - 100.0 * single_sample_hits / single_sample_total
    gap = pct(covered) - pct(correct_majority)
    print(f"  gain from Best-of-N            : +{gain:.1f} points")
    print(f"  remaining selection gap        : {gap:.1f} points")
    print("-" * 58)

    ok = True
    if agree_majority != n_problems:
        print(f"FAIL: selection differs from the recorded run on "
              f"{n_problems - agree_majority} problems")
        ok = False
    if abs(pct(correct_majority) - PUBLISHED_MAJORITY) > 0.1:
        print(f"FAIL: majority accuracy {pct(correct_majority):.1f}% != "
              f"published {PUBLISHED_MAJORITY}%")
        ok = False
    if gain <= 0:
        print("FAIL: Best-of-N did not improve on single-sample accuracy")
        ok = False

    print("\nPASS - reproduces the published measurements exactly" if ok
          else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
