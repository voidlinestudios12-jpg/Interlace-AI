# -*- coding: utf-8 -*-
"""Selector comparison bars, Interlace brand styling.

Every number is read from results/gsm8k_summary.json. Nothing is typed here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import FancyBboxPatch                  # noqa: E402

import marca as M                                              # noqa: E402

d = M.data()
fam = M.font()
n = d["n_top"]

ROWS = [
    ("Single sample", "N=1", d["base"], M.GRIS),
    ("Random", "N=%d, among those that answered" % n, d["rnd_top"], M.GRIS),
    ("Majority vote", "N=%d" % n, d["top"], M.AZUL),
    ("Coverage", "N=%d, what the model can reach" % n, d["cov_top"], M.VERDE),
]

fig = plt.figure(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(M.FONDO)

M.header(
    fig,
    "One frozen 0.5B model, asked %d times" % n,
    "Qwen2.5-0.5B-Instruct on GSM8K. 200 problems, %d trajectories each, "
    "weights never modified." % n)

ax = fig.add_axes([0.034, 0.13, 0.932, 0.60])
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

ax.text(0, 96, "S E L E C T O R", color=M.NOTA, fontsize=10.5,
        fontweight="bold", family=fam)
ax.text(30, 96, "A C C U R A C Y", color=M.NOTA, fontsize=10.5,
        fontweight="bold", family=fam)

X0, WIDTH, TALL = 30.0, 50.0, 7.4
y = 78.0
for label, sub, value, color in ROWS:
    ax.text(0, y + TALL / 2 + 1.4, label, color=M.BLANCO, fontsize=15.5,
            fontweight="bold", family=fam, va="center")
    ax.text(0, y + TALL / 2 - 3.0, sub, color=M.NOTA, fontsize=10.5,
            family=fam, va="center")

    ax.add_patch(FancyBboxPatch(
        (X0, y), WIDTH, TALL, boxstyle="round,pad=0,rounding_size=1.5",
        facecolor="#10141c", edgecolor="none", zorder=2))
    ax.add_patch(FancyBboxPatch(
        (X0, y), WIDTH * value / 100.0, TALL,
        boxstyle="round,pad=0,rounding_size=1.5",
        facecolor=color, edgecolor="none", zorder=3))

    ax.text(X0 + WIDTH + 3.2, y + TALL / 2, "%.1f%%" % value,
            color=M.BLANCO, fontsize=20, fontweight="bold", family=fam,
            va="center")
    y -= 18.5

gain = d["top"] - d["base"]
selection = d["top"] - d["rnd_top"]
fig.text(0.034, 0.108,
         "Of the +%.1f points over a single sample, +%.1f come from selection "
         "itself. Exact McNemar vs random: p \u2264 %.0e at every one of %d "
         "random seeds." % (gain, selection, d["p"], d["p_seeds"]),
         color=M.NOTA, fontsize=11.5, family=fam)
fig.text(0.034, 0.075,
         "Coverage is what the model reaches somewhere in its %d attempts "
         "— the headroom a verifier can still claim." % n,
         color=M.NOTA, fontsize=11.5, family=fam)

M.footer(fig, None, "every number recomputable: scripts/analyse.py")

fig.savefig(M.out("08_bars_n128.png"), facecolor=M.FONDO, dpi=100)
print("08_bars_n128.png  (%.1f%% -> %.1f%%, +%.1f points)"
      % (d["base"], d["top"], gain))
