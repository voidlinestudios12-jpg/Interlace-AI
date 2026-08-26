# -*- coding: utf-8 -*-
"""Rewrite every published number in the documentation from the summary.

Four releases in a row, an audit found a table somewhere carrying the previous
release's figures: the PyPI description, the landing README, a chart caption, a
post. Each time the cause was the same -- a human copying a number from one
file into another.

So the numbers are generated. Every block this script owns is delimited by

    <!-- auto:NAME -->  ...  <!-- /auto:NAME -->

and its contents are replaced wholesale from ``results/gsm8k_summary.json``.
Prose stays hand-written; only the figures inside the markers move.

Run after ``scripts/analyse.py``:

    python scripts/analyse.py && python scripts/sync_docs.py

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESKTOP = os.path.dirname(HERE)
SUMMARY = os.path.join(HERE, "results", "gsm8k_summary.json")


def load():
    d = json.load(io.open(SUMMARY, encoding="utf-8"))
    c = d["curve"]
    sel = {r["selector"]: r for r in d["selectors_at_n_max"]}
    mc = d["mcnemar_majority_vs_random"]
    last = c[-1]
    d.update(
        curve_rows=c, sel=sel, mc=mc,
        n_top=last["n"], base=c[0]["majority"], top=last["majority"],
        cov=last["coverage"], rnd=last["random"],
        gain=round(last["majority"] - c[0]["majority"], 1),
        from_sel=round(last["majority"] - last["random"], 1),
        from_abs=round(last["random"] - c[0]["majority"], 1),
        p_worst=mc["p_value_worst"], p_med=mc["p_value_median"],
        seeds=mc["seeds"],
    )
    d["share"] = round(100 * d["from_sel"] / d["gain"])
    return d


SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def sci(x):
    """1.401e-05 -> '1.5 × 10⁻⁵', for prose rather than code.

    Rounds the mantissa **up**, because every caller prints this after "p ≤"
    and rounding to nearest makes the inequality false half the time. The sign
    of the exponent is read from the number rather than assumed negative.
    """
    import math
    if x <= 0 or not math.isfinite(x):
        return "%g" % x
    e = math.floor(math.log10(x)) - 1
    m = math.ceil(x / 10 ** e) / 10.0          # two sig figs, rounded up
    if m >= 10:                                # 9.96 -> 10.0 -> 1.0e+1
        m, e = m / 10.0, e + 1
    exp = e + 1
    sign = "⁻" if exp < 0 else ""
    return "%.1f × 10%s%s" % (m, sign, "".join(SUP[int(c)]
                                               for c in str(abs(exp))))


def blocks(d):
    """The generated fragments, keyed by marker name."""
    out = {}

    rows = "\n".join(
        "| %s | %.1f%% | %s%.1f%%%s | [%.1f, %.1f] | %s%.1f%%%s |"
        % ("**%d**" % r["n"] if r["n"] == d["n_top"] else r["n"],
           r["random"],
           "**" if r["n"] == d["n_top"] else "", r["majority"],
           "**" if r["n"] == d["n_top"] else "",
           r["majority_ci95"][0], r["majority_ci95"][1],
           "**" if r["n"] == d["n_top"] else "", r["coverage"],
           "**" if r["n"] == d["n_top"] else "")
        for r in d["curve_rows"])
    out["curve"] = ("| N | random | majority | 95% CI | coverage |\n"
                    "|---:|---:|---:|---:|---:|\n" + rows)

    out["headline"] = (
        "| | single sample | **Best-of-%d** |\n|---|---:|---:|\n"
        "| **GSM8K**, Qwen2.5-0.5B frozen | %.1f%% | **%.1f%%** |\n\n"
        "**+%.1f points. Nothing was trained.**"
        % (d["n_top"], d["base"], d["top"], d["gain"]))

    out["decomposition"] = (
        "| | | |\n|---|---:|---:|\n"
        "| N=1, a single sample | %.1f%% | |\n"
        "| N=%d, **random** among the trajectories that answered | %.1f%% | +%.1f |\n"
        "| N=%d, **majority vote** | %.1f%% | **+%.1f** |\n"
        "| | | **+%.1f total** |"
        % (d["base"], d["n_top"], d["rnd"], d["from_abs"],
           d["n_top"], d["top"], d["from_sel"], d["gain"]))

    out["selectors"] = (
        "| selector at N=%d | accuracy | 95%% CI |\n|---|---:|---:|\n"
        "| `random` (exact expectation) | %.1f%% | [%.1f, %.1f] |\n"
        "| `majority` | **%.1f%%** | [%.1f, %.1f] |\n"
        "| `self_certainty` | %.1f%% | [%.1f, %.1f] |\n"
        "| `oracle` (diagnostic ceiling) | %.1f%% | [%.1f, %.1f] |"
        % (d["n_top"],
           d["sel"]["random"]["accuracy"], *d["sel"]["random"]["ci95"],
           d["sel"]["majority"]["accuracy"], *d["sel"]["majority"]["ci95"],
           d["sel"]["self_certainty"]["accuracy"],
           *d["sel"]["self_certainty"]["ci95"],
           d["sel"]["oracle"]["accuracy"], *d["sel"]["oracle"]["ci95"]))

    out["accounting"] = (
        "| | |\n|---|---|\n"
        "| Trajectories generated | %s |\n"
        "| Cast a vote | %s — %.1f%% |\n"
        "| Abstained | %s — %.1f%% |\n"
        "| Truncated at the token limit | %s — %.1f%% (%s of them abstained) |\n"
        "| Tokens generated | %s |\n"
        "| Re-extraction drift on replay | **%d** |"
        % (f"{d['n_trajectories']:,}", f"{d['n_voting']:,}",
           100 - d["abstention_rate"], f"{d['n_abstained']:,}",
           d["abstention_rate"], f"{d['n_truncated']:,}",
           d["truncation_rate"], f"{d['n_truncated_and_abstained']:,}",
           f"{d['total_tokens']:,}", d["reextraction_drift"]))

    out["significance"] = (
        "**%.1f%% to %.1f%%** on a half-billion-parameter model, with the "
        "weights frozen throughout. Against random selection at the same N, "
        "exact McNemar gives **p ≤ %s** at every one of %d random seeds, "
        "with a median of %s.\n\n"
        "We quote the worst seed rather than the best, and we say what it is: "
        "the worst of the %d enumerated, not a bound. `random` draws "
        "differently every run, so its p-value has a distribution — a "
        "wider sweep will find a worse seed. What does not move is the "
        "conclusion."
        % (d["base"], d["top"], sci(d["p_worst"]), d["seeds"],
           sci(d["p_med"]), d["seeds"]))

    out["gain_prose"] = (
        "Random selection improves slightly with N without selecting anything, "
        "because with more trajectories one of them usually did not abstain. "
        "Separating the two shows that **%.1f of the %.1f points — %d%% of "
        "the gain — is genuine selection**, not an artefact of comparing a "
        "one-sample baseline against an N-sample system."
        % (d["from_sel"], d["gain"], d["share"]))

    # ------------------------------------------------ the multi-model table
    models = os.path.join(HERE, "results", "models", "summary.json")
    if os.path.exists(models):
        rows = json.load(io.open(models, encoding="utf-8"))
        rows.sort(key=lambda r: r["single"])
        out["models"] = (
            "| model | one sample | **Best-of-N** | gain | coverage |\n"
            "|---|---:|---:|---:|---:|\n" + "\n".join(
                "| `%s`<br><sub>%s · N=%d</sub> | %.1f%% | **%.1f%%** | "
                "**+%.1f** | %.1f%% |"
                % (r["model"], r["label"], r["n"], r["single"],
                   r["majority"], r["gain"], r["coverage"])
                for r in rows))
        gains = [r["gain"] for r in rows]
        heads = [r["headroom"] for r in rows]
        improved = sum(1 for g in gains if g > 0)
        out["models_prose"] = (
            "%s of the %d models we measured improved, by between **+%.1f** "
            "and **+%.1f** points, and none of them was trained. The gain "
            "shrinks as the base model gets better — which is what should "
            "happen, and is worth saying plainly rather than hiding behind "
            "the largest number in the table.\n\n"
            "The row that matters more is coverage. It stays above what the "
            "vote returns on **every** model, by a median of %.1f points: "
            "even the strongest one here is still failing to return answers "
            "it already found. That gap is the whole reason to work on "
            "selection rather than on sampling harder."
            % ("Every one" if improved == len(rows) else "%d" % improved,
               len(rows), min(gains), max(gains),
               sorted(heads)[len(heads) // 2]))

    return out


MARK = re.compile(r"<!-- auto:(\w+) -->.*?<!-- /auto:\1 -->", re.S)


def apply_to(path, out):
    if not os.path.exists(path):
        return 0
    s = io.open(path, encoding="utf-8").read()
    n = [0]

    def rep(m):
        name = m.group(1)
        if name not in out:
            return m.group(0)
        n[0] += 1
        return "<!-- auto:%s -->\n%s\n<!-- /auto:%s -->" % (
            name, out[name], name)

    new = MARK.sub(rep, s)
    if new != s:
        io.open(path, "w", encoding="utf-8", newline="\n").write(new)
    return n[0]


def render_pypi():
    """Regenerate README_PYPI.md from README.md.

    PyPI does not read the Hugging Face YAML header and renders it as raw
    text, and relative links there resolve to the project page rather than to
    the files, so both are rewritten.

    This used to be a manual step. It was skipped on one release, and the live
    PyPI page served the previous version's numbers until the next audit found
    them. It is part of the sync now: there is no way to run the docs update
    and leave it behind.
    """
    src = os.path.join(HERE, "README.md")
    dst = os.path.join(HERE, "README_PYPI.md")
    gh = ("https://github.com/voidlinestudios12-jpg/Interlace-AI"
          "/blob/main/best-of-n/")
    body = re.sub(r"\A---\n.*?\n---\n", "",
                  io.open(src, encoding="utf-8").read(), flags=re.S)
    body = re.sub(r"\]\((?!http|#)([^)]+)\)",
                  lambda m: "](" + gh + m.group(1) + ")", body)
    head = ("<!-- GENERATED from README.md by scripts/sync_docs.py.\n"
            "     Do not edit: changes are overwritten on the next sync. -->\n\n")
    io.open(dst, "w", encoding="utf-8", newline="\n").write(head + body)
    left = re.findall(r"\]\((?!http|#)([^)]+)\)", body)
    print("  README_PYPI.md regenerated from README.md"
          + ("  RELATIVE LINKS LEFT: %s" % left if left else ""))



def main():
    if not os.path.exists(SUMMARY):
        print("run scripts/analyse.py first")
        return 1
    d = load()
    out = blocks(d)
    targets = [
        os.path.join(HERE, "README.md"),
        os.path.join(HERE, "USAGE.md"),
        os.path.join(HERE, "PAPER.md"),
        os.path.join(HERE, "TR-2026-02.md"),
        os.path.join(DESKTOP, "TR-2026-02.md"),
        os.path.join(DESKTOP, "POSTS_Y_ZENODO.md"),
        os.path.join(DESKTOP, "_bench_work", "repo", "README.md"),
    ]
    total = 0
    for t in targets:
        k = apply_to(t, out)
        total += k
        if k:
            print("  %-52s %d block%s"
                  % (os.path.relpath(t, DESKTOP), k, "" if k == 1 else "s"))
    print("\n%d generated blocks updated from results/gsm8k_summary.json"
          % total)
    if not total:
        print("  (no <!-- auto:NAME --> markers found yet)")
    render_pypi()
    return 0


if __name__ == "__main__":
    sys.exit(main())
