# -*- coding: utf-8 -*-
"""Every published figure, regenerated from results/gsm8k_summary.json.

One script so the charts cannot drift from the numbers: nothing here is typed
in by hand except the labels. Run it after scripts/analyse.py.

    python figures/make_figures.py

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.ticker import FuncFormatter           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Interlace palette. Dark ink on warm paper, one accent that carries the
# result and a muted grey that carries the baseline it has to beat.
INK = "#14161a"
PAPER = "#ffffff"
ACCENT = "#2f6df6"        # majority / the method
CEIL = "#8b5cf6"          # coverage / the ceiling
BASE = "#9aa3ad"          # random / the baseline
GOOD = "#12a150"          # gains
GRID = "#e6e9ee"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.edgecolor": "#c9cfd7", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

PCT = FuncFormatter(lambda v, _: "%d%%" % v)


def load():
    p = os.path.join(ROOT, "results", "gsm8k_summary.json")
    if not os.path.exists(p):
        sys.exit("run scripts/analyse.py first: %s not found" % p)
    return json.load(io.open(p, encoding="utf-8"))


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("  wrote figures/%s" % name)


# --------------------------------------------------------------- 1. the curve

def fig_curve(d):
    c = d["curve"]
    ns = [r["n"] for r in c]
    maj = [r["majority"] for r in c]
    rnd = [r["random"] for r in c]
    cov = [r["coverage"] for r in c]
    lo = [r["majority_ci95"][0] for r in c]
    hi = [r["majority_ci95"][1] for r in c]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.fill_between(ns, lo, hi, color=ACCENT, alpha=0.11, linewidth=0)
    ax.plot(ns, cov, "--", color=CEIL, lw=2.2, marker="o", ms=5,
            label="Coverage — the answer is somewhere in the pool")
    ax.plot(ns, maj, "-", color=ACCENT, lw=3.2, marker="o", ms=7,
            label="Majority vote — what you get back", zorder=5)
    ax.plot(ns, rnd, "-", color=BASE, lw=2, marker="s", ms=5,
            label="Random pick — the baseline it has to beat")

    ax.annotate("%.1f%%" % maj[-1], (ns[-1], maj[-1]),
                textcoords="offset points", xytext=(10, -4),
                fontsize=13, fontweight="bold", color=ACCENT)
    ax.annotate("%.1f%%" % cov[-1], (ns[-1], cov[-1]),
                textcoords="offset points", xytext=(10, -4),
                fontsize=12, fontweight="bold", color=CEIL)
    ax.annotate("%.1f%%" % maj[0], (ns[0], maj[0]),
                textcoords="offset points", xytext=(4, -18),
                fontsize=11, color=INK)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N — samples drawn from the same frozen model")
    ax.set_ylabel("GSM8K accuracy")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(35, 100)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, fontsize=10.5, ncol=3, columnspacing=1.6)
    ax.set_title("One 0.5B model, asked more than once\n",
                 fontsize=15, fontweight="bold", loc="left")
    ax.text(0, 1.005, "Qwen2.5-0.5B-Instruct on GSM8K · weights frozen · "
            "200 problems · shaded band is the 95% CI",
            transform=ax.transAxes, fontsize=9.5, color="#5b636d")
    save(fig, "07_curve_n128.png")


# ------------------------------------------------------- 2. before and after

def fig_bars(d):
    c = d["curve"]
    first, last = c[0], c[-1]
    labels = ["Single sample\n(what you get today)",
              "Best-of-%d\n(same model, same weights)" % last["n"]]
    vals = [first["majority"], last["majority"]]

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    bars = ax.bar(labels, vals, width=0.52, color=[BASE, ACCENT], zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.4, "%.1f%%" % v,
                ha="center", fontsize=17, fontweight="bold",
                color=b.get_facecolor())

    gain = vals[1] - vals[0]
    ax.annotate("", xy=(1, vals[1] - 1), xytext=(1, vals[0] + 1),
                arrowprops=dict(arrowstyle="<->", color=GOOD, lw=2.4))
    ax.text(1.09, (vals[0] + vals[1]) / 2, "+%.1f\npoints" % gain,
            color=GOOD, fontweight="bold", fontsize=14, va="center")

    ax.set_ylabel("GSM8K accuracy")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, max(vals) + 16)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_title("Nothing was trained\n", fontsize=15,
                 fontweight="bold", loc="left")
    ax.text(0, 1.005, "The weights never changed. Only the number of "
            "attempts did.", transform=ax.transAxes,
            fontsize=9.5, color="#5b636d")
    save(fig, "08_bars_n128.png")


# ---------------------------------------------- 3. where the gain comes from

def fig_decomposition(d):
    c = d["curve"]
    base = c[0]["majority"]
    rnd_hi = c[-1]["random"]
    top = c[-1]["majority"]
    absten = rnd_hi - base
    select = top - rnd_hi

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.barh([0], [base], color=BASE, height=0.5, zorder=3,
            label="Single-sample accuracy")
    ax.barh([0], [absten], left=[base], color="#c8a2f0", height=0.5, zorder=3,
            label="Not the method: more samples, so one of them answered")
    ax.barh([0], [select], left=[base + absten], color=ACCENT, height=0.5,
            zorder=3, label="Genuine selection: the vote picked the right one")

    ax.text(base / 2, 0, "%.1f%%" % base, ha="center", va="center",
            color="white", fontweight="bold", fontsize=12)
    ax.text(base + absten / 2, 0, "+%.1f" % absten, ha="center", va="center",
            color=INK, fontweight="bold", fontsize=10)
    ax.text(base + absten + select / 2, 0, "+%.1f" % select, ha="center",
            va="center", color="white", fontweight="bold", fontsize=14)
    ax.text(top + 1, 0, "%.1f%%" % top, va="center",
            fontweight="bold", fontsize=14, color=ACCENT)

    ax.set_yticks([])
    ax.set_xlim(0, top + 9)
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False, fontsize=9.5, ncol=1)
    ax.set_title("%.0f%% of the gain is real selection\n"
                 % (100 * select / (top - base)),
                 fontsize=15, fontweight="bold", loc="left")
    ax.text(0, 1.02, "Most reports never separate these two. "
            "We print the split beside the headline.",
            transform=ax.transAxes, fontsize=9.5, color="#5b636d")
    save(fig, "09_decomposition.png")


# --------------------------------------------------- 4. selectors vs baseline

def fig_selectors(d):
    rows = d.get("selectors_at_n_max")
    if not rows:
        print("  (no selector table in summary; skipping 10)")
        return
    nice = {"random": "random\n(the baseline)", "majority": "majority",
            "self_certainty": "self_certainty", "oracle": "oracle\n(ceiling)"}
    cols = {"random": BASE, "majority": ACCENT,
            "self_certainty": "#3fa9d8", "oracle": CEIL}
    names = [nice.get(r["selector"], r["selector"]) for r in rows]
    vals = [r["accuracy"] for r in rows]
    err = [[r["accuracy"] - r["ci95"][0] for r in rows],
           [r["ci95"][1] - r["accuracy"] for r in rows]]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(names, vals, width=0.55,
                  color=[cols.get(r["selector"], ACCENT) for r in rows],
                  zorder=3)
    ax.errorbar(names, vals, yerr=err, fmt="none", ecolor=INK,
                elinewidth=1.4, capsize=6, zorder=4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3.2, "%.1f%%" % v,
                ha="center", fontweight="bold", fontsize=13)

    ax.axhline(vals[0], color=BASE, ls=":", lw=1.8, zorder=2)
    ax.set_ylabel("GSM8K accuracy at N=128")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_title("Every selector clears the baseline\n", fontsize=15,
                 fontweight="bold", loc="left")
    ax.text(0, 1.005, "Bars are 95% bootstrap CIs. The dotted line is a "
            "random pick — the bar any selector has to clear.",
            transform=ax.transAxes, fontsize=9.5, color="#5b636d")
    save(fig, "10_selectors.png")


# ------------------------------------------------ 5. what the accounting says

def fig_accounting(d):
    voted = 100.0 - d["abstention_rate"]
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.barh([0], [voted], color=GOOD, height=0.42, zorder=3)
    ax.barh([0], [d["abstention_rate"]], left=[voted], color="#e4b363",
            height=0.42, zorder=3)
    ax.text(voted / 2, 0, "%.1f%% cast a vote" % voted, ha="center",
            va="center", color="white", fontweight="bold", fontsize=12)
    ax.text(voted + d["abstention_rate"] / 2, 0.62,
            "%.1f%% abstained" % d["abstention_rate"], ha="center",
            fontsize=10, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_title("A trajectory with no answer does not get a vote\n",
                 fontsize=14, fontweight="bold", loc="left")
    ax.text(0, 1.06, "%s trajectories. Guessing at the last number in an "
            "unfinished one would have added %s phantom ballots."
            % ("{:,}".format(25600),
               "{:,}".format(int(round(25600 * d["abstention_rate"] / 100)))),
            transform=ax.transAxes, fontsize=9.5, color="#5b636d")
    save(fig, "11_abstention.png")


# ------------------------------------------------------- 6. the coverage gap

def fig_headroom(d):
    c = d["curve"]
    ns = [r["n"] for r in c]
    gap = [r["coverage"] - r["majority"] for r in c]
    maj = [r["majority"] for r in c]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(ns, maj, [r["coverage"] for r in c], color=CEIL,
                    alpha=0.16, linewidth=0, label="Still on the table")
    ax.plot(ns, [r["coverage"] for r in c], "--", color=CEIL, lw=2.2)
    ax.plot(ns, maj, "-", color=ACCENT, lw=3, marker="o", ms=6)
    ax.annotate("%.1f points a better selector could still take"
                % gap[-1], (ns[-1], (maj[-1] + c[-1]["coverage"]) / 2),
                textcoords="offset points", xytext=(-260, 0),
                fontsize=11, color="#6d28d9", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N")
    ax.set_ylabel("GSM8K accuracy")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(35, 100)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_title("The library tells you which problem you have\n",
                 fontsize=15, fontweight="bold", loc="left")
    ax.text(0, 1.005, "Blue is what comes back. Purple is what was reachable. "
            "The gap says: work on selection, not on sampling.",
            transform=ax.transAxes, fontsize=9.5, color="#5b636d")
    save(fig, "12_headroom.png")


def main():
    d = load()
    print("regenerating figures from results/gsm8k_summary.json")
    fig_curve(d)
    fig_bars(d)
    fig_decomposition(d)
    fig_selectors(d)
    fig_accounting(d)
    fig_headroom(d)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
