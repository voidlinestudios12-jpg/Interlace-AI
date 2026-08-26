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
import subprocess
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
    total = d.get("n_trajectories", d["problems"] * d["n_generated"])
    voted = 100.0 - d["abstention_rate"]
    n_abs = d.get("n_abstained")
    # The truncated-AND-abstained count, not the truncated count: one
    # truncated trajectory boxed its answer before running out of room,
    # so using the latter made the chart sum to 804 out of 803.
    n_tr = d.get("n_truncated_and_abstained", d.get("n_truncated"))
    n_un = d.get("n_abstained_not_truncated")
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.barh([0], [voted], color=GOOD, height=0.42, zorder=3)
    ax.barh([0], [d["abstention_rate"]], left=[voted], color="#e4b363",
            height=0.42, zorder=3)
    ax.text(voted / 2, 0, "%.1f%% cast a vote" % voted, ha="center",
            va="center", color="white", fontweight="bold", fontsize=12)
    ax.annotate("%.1f%% abstained" % d["abstention_rate"],
                xy=(voted + d["abstention_rate"] / 2, 0.21),
                xytext=(voted + d["abstention_rate"] / 2 + 1.5, 0.95),
                ha="left", fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color="#b9c0c8", lw=1))
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 1.3)
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_title("A trajectory with no answer does not get a vote\n",
                 fontsize=14, fontweight="bold", loc="left")
    # Not all abstentions are truncations. The chart used to imply they were,
    # and attributed every one of them to an unfinished trajectory.
    ax.text(0, 1.10,
            "%s trajectories. %s abstained: %s ran out of tokens and %s "
            "finished without boxing an answer."
            % ("{:,}".format(total), "{:,}".format(n_abs),
               "{:,}".format(n_tr), "{:,}".format(n_un)),
            transform=ax.transAxes, fontsize=9.5, color="#5b636d")
    ax.text(0, 1.02,
            "Every one of them would have cast a phantom ballot under a "
            "\u201clast number in the text\u201d fallback.",
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
    ax.annotate("%.1f points between what came back and what was reachable"
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
    """Regenerate every published figure.

    07 and 08 are the two the model card leads with and they carry the brand
    header, so they are produced by plot_curva.py and plot_bars.py. Those two
    scripts and the functions in this module used to write the same filenames,
    which meant the last one run decided what shipped -- and for one release
    the pair that won were carrying hand-typed numbers from a previous version.
    One entry point, one owner per file.
    """
    d = load()
    print("regenerating figures from results/gsm8k_summary.json")

    for script in ("plot_curva.py", "plot_bars.py"):
        path = os.path.join(HERE, script)
        r = subprocess.run([sys.executable, path], cwd=HERE,
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout + r.stderr)
            return r.returncode
        print("  " + r.stdout.strip())

    fig_decomposition(d)
    fig_selectors(d)
    fig_accounting(d)
    fig_headroom(d)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
