# -*- coding: utf-8 -*-
"""Accuracy against N, Interlace brand styling.

Every number is read from results/gsm8k_summary.json. Nothing is typed here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

import marca as M                                              # noqa: E402

d = M.data()
fam = M.font()

fig = plt.figure(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(M.FONDO)

M.header(
    fig,
    "%.1f%%  \u2192  %.1f%%   by asking the same frozen model more than once"
    % (d["base"], d["top"]),
    "Qwen2.5-0.5B-Instruct on GSM8K \u00b7 200 problems \u00b7 "
    "%d trajectories each \u00b7 weights never modified" % d["n_top"])

ax = fig.add_axes([0.075, 0.135, 0.60, 0.595])
ax.set_facecolor(M.FONDO)
for s in ax.spines.values():
    s.set_color(M.BORDE)
ax.grid(True, color=M.REJILLA, lw=0.9)
ax.set_axisbelow(True)

ax.fill_between(d["N"], d["MAJ"], d["COV"], color=M.VERDE, alpha=0.09, lw=0)
ax.plot(d["N"], d["COV"], "-o", color=M.VERDE, lw=2.6, ms=6,
        label="coverage  (reachable)")
ax.plot(d["N"], d["MAJ"], "-o", color=M.AZUL, lw=3.4, ms=7,
        label="majority vote  (returned)", zorder=5)
ax.plot(d["N"], d["RND"], "--s", color=M.GRIS, lw=1.9, ms=5,
        label="random  (baseline)")

ax.set_xscale("log", base=2)
ax.set_xticks(d["N"])
ax.set_xticklabels([str(x) for x in d["N"]], color=M.TENUE, fontsize=11)
ax.set_yticks(range(40, 101, 10))
ax.set_yticklabels(["%d%%" % y for y in range(40, 101, 10)],
                   color=M.TENUE, fontsize=11)
ax.set_ylim(38, 99)
ax.set_xlim(0.85, 190)
ax.set_xlabel("trajectories sampled  (N)", color=M.TENUE, fontsize=12,
              family=fam, labelpad=10)
ax.tick_params(colors=M.TENUE, length=0)

for serie, color, size in ((d["COV"], M.VERDE, 15), (d["MAJ"], M.AZUL, 15),
                           (d["RND"], M.GRIS, 12)):
    ax.annotate("%.1f%%" % serie[-1], (d["N"][-1], serie[-1]),
                textcoords="offset points", xytext=(12, -5),
                color=color, fontsize=size, fontweight="bold", family=fam)
ax.annotate("%.1f%%" % d["base"], (d["N"][0], d["base"]),
            textcoords="offset points", xytext=(2, -22),
            color=M.BLANCO, fontsize=12, fontweight="bold", family=fam)

leg = ax.legend(loc="upper left", frameon=False, fontsize=11.5,
                labelcolor=M.TENUE)
for t in leg.get_texts():
    t.set_family(fam)

# ------------------------------------------------------------- side panel
pan = fig.add_axes([0.735, 0.135, 0.235, 0.595])
pan.axis("off")
pan.set_xlim(0, 1)
pan.set_ylim(0, 1)

pan.text(0, 0.985, "W H E R E   T H E   G A I N   I S", color=M.NOTA,
         fontsize=10.5, fontweight="bold", family=fam)

rows = [
    ("single sample", "%.1f%%" % d["base"], M.BLANCO),
    ("random at N=%d" % d["n_top"], "%.1f%%" % d["rnd_top"], M.GRIS),
    ("majority at N=%d" % d["n_top"], "%.1f%%" % d["top"], M.AZUL),
]
y = 0.87
for label, value, color in rows:
    pan.text(0, y, label, color=M.TENUE, fontsize=12, family=fam, va="center")
    pan.text(1, y, value, color=color, fontsize=15, fontweight="bold",
             family=fam, ha="right", va="center")
    y -= 0.105

pan.plot([0, 1], [y + 0.035, y + 0.035], color=M.BORDE, lw=1)
y -= 0.045

pan.text(0, y, "from voting", color=M.TENUE, fontsize=12, family=fam,
         va="center")
pan.text(1, y, "+%.1f" % (d["top"] - d["rnd_top"]), color=M.AZUL,
         fontsize=17, fontweight="bold", family=fam, ha="right", va="center")
y -= 0.105

pan.text(0, y, "exact McNemar vs random", color=M.TENUE, fontsize=11,
         family=fam, va="center")
pan.text(1, y, "p = %.1e" % d["p"], color=M.VERDE, fontsize=13,
         fontweight="bold", family=fam, ha="right", va="center")

pan.text(0, 0.20,
         "Coverage %.1f%% at N=%d:\nthe right answer is in the\npool on more "
         "than nine\nproblems out of ten." % (d["cov_top"], d["n_top"]),
         color=M.NOTA, fontsize=11, family=fam, va="top", linespacing=1.65)

M.footer(fig,
         "%s trajectories published in full. Every number recomputable with "
         "scripts/analyse.py, no GPU." % format(d["trajectories"], ","),
         "pip install bestofn")

fig.savefig(M.out("07_curve_n128.png"), facecolor=M.FONDO, dpi=100)
print("07_curve_n128.png  (%.1f%% -> %.1f%%, coverage %.1f%%)"
      % (d["base"], d["top"], d["cov_top"]))
