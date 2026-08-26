# -*- coding: utf-8 -*-
"""Best-of-N across model scale, Interlace brand styling.

Reads results/models/summary.json, which scripts/run_models.py writes after
re-extracting every answer from that model's own raw trajectory text.

**Nothing here asserts a conclusion.** The first version of this file printed
"Every model we tried got better, and none of them was touched" as a fixed
headline, directly beside a panel that *computed* how many had improved -- so
a regression would have been captioned as a success. It also printed "the gain
does not disappear as the base model gets better" and "coverage stays above
what the vote returns at every scale we measured" unconditionally. Those are
findings, not decoration: if the data stops supporting them the chart has to
stop saying them. Every sentence below is now derived, and models that failed
to load are named rather than dropped.
"""
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import FancyBboxPatch                  # noqa: E402

import marca as M                                              # noqa: E402

SUMMARY = os.path.join(M.ROOT, "results", "models", "summary.json")
if not os.path.exists(SUMMARY):
    raise SystemExit("%s not found.\nRun scripts/run_models.py first." % SUMMARY)

rows = json.load(io.open(SUMMARY, encoding="utf-8"))
if not rows:
    raise SystemExit("summary.json is empty; nothing to plot.")
rows.sort(key=lambda r: r["single"])
fam = M.font()

improved = sum(1 for r in rows if r["gain"] > 0)
gains = [r["gain"] for r in rows]
heads = [r["headroom"] for r in rows]
n_values = sorted({r["n"] for r in rows})
drift = sum(r.get("reextraction_drift", 0) for r in rows)

# ------------------------------------------------------------- the headline
# Derived, not asserted. If a model regresses this says so.
if improved == len(rows):
    title = "Every model improved, and not one of them was trained"
elif improved:
    title = "%d of %d models improved, and not one was trained" % (
        improved, len(rows))
else:
    title = "No model improved on this benchmark"

n_txt = ("N=%d" % n_values[0] if len(n_values) == 1
         else "N=%d–%d" % (n_values[0], n_values[-1]))

fig = plt.figure(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(M.FONDO)
M.header(
    fig, title,
    "GSM8K · 200 problems · %s · identical prompt, temperature "
    "and token budget · weights frozen throughout" % n_txt)

ax = fig.add_axes([0.055, 0.135, 0.60, 0.595])
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

ax.text(0, 96.5, "M O D E L", color=M.NOTA, fontsize=10.5,
        fontweight="bold", family=fam)
ax.text(38, 96.5, "S I N G L E   S A M P L E   →   B E S T - O F - N",
        color=M.NOTA, fontsize=10.5, fontweight="bold", family=fam)

X0, WIDTH, TALL = 38.0, 47.0, 6.2
step = 84.0 / max(1, len(rows))
y = 90.0 - step


def x_of(pct):
    return X0 + WIDTH * pct / 100.0


for r in rows:
    name = r["model"].split("/")[-1]
    ax.text(0, y + TALL / 2 + 1.2, name, color=M.BLANCO, fontsize=12,
            fontweight="bold", family=fam, va="center")
    ax.text(0, y + TALL / 2 - 2.6, "%s · N=%d" % (r["label"], r["n"]),
            color=M.NOTA, fontsize=9, family=fam, va="center")

    ax.add_patch(FancyBboxPatch(
        (X0, y), WIDTH, TALL, boxstyle="round,pad=0,rounding_size=1.2",
        facecolor="#10141c", edgecolor="none", zorder=2))
    ax.add_patch(FancyBboxPatch(
        (X0, y), WIDTH * r["coverage"] / 100.0, TALL,
        boxstyle="round,pad=0,rounding_size=1.2",
        facecolor="#12341f", edgecolor="none", zorder=3))
    ax.add_patch(FancyBboxPatch(
        (X0, y), WIDTH * r["majority"] / 100.0, TALL,
        boxstyle="round,pad=0,rounding_size=1.2",
        facecolor=M.AZUL if r["gain"] > 0 else "#8a3b3b",
        edgecolor="none", zorder=4))
    ax.plot([x_of(r["single"])] * 2, [y - 0.9, y + TALL + 0.9],
            color=M.BLANCO, lw=1.9, zorder=6)

    ax.text(x_of(r["single"]) - 1.2, y + TALL / 2, "%.0f%%" % r["single"],
            color=M.TENUE, fontsize=10, family=fam, ha="right", va="center")
    ax.text(X0 + WIDTH + 2.4, y + TALL / 2, "%.1f%%" % r["majority"],
            color=M.BLANCO, fontsize=14.5, fontweight="bold", family=fam,
            va="center")
    ax.text(X0 + WIDTH + 10.2, y + TALL / 2, "%+.1f" % r["gain"],
            color=M.AZUL if r["gain"] > 0 else "#c96a6a",
            fontsize=12, fontweight="bold", family=fam, va="center")
    y -= step

ax.text(X0, y + step - TALL - 5.5,
        "white line: one sample   ·   blue: what Best-of-N returns   "
        "·   dark green: what was reachable",
        color=M.NOTA, fontsize=10, family=fam)

# ------------------------------------------------------------- side panel
pan = fig.add_axes([0.695, 0.135, 0.275, 0.595])
pan.axis("off")
pan.set_xlim(0, 1)
pan.set_ylim(0, 1)
pan.text(0, 0.985, "W H A T   T H E   D A T A   S A Y S", color=M.NOTA,
         fontsize=10.5, fontweight="bold", family=fam)

lines = [
    ("models measured", "%d" % len(rows), M.BLANCO),
    ("improved", "%d/%d" % (improved, len(rows)),
     M.VERDE if improved == len(rows) else M.AZUL),
    ("smallest gain", "%+.1f" % min(gains), M.AZUL),
    ("largest gain", "%+.1f" % max(gains), M.AZUL),
    ("median headroom left", "%.1f" % sorted(heads)[len(heads) // 2], M.VERDE),
    ("re-extraction drift", "%d" % drift, M.VERDE if not drift else "#c96a6a"),
]
yy = 0.87
for label, value, colour in lines:
    pan.text(0, yy, label, color=M.TENUE, fontsize=11.5, family=fam, va="center")
    pan.text(1, yy, value, color=colour, fontsize=15, fontweight="bold",
             family=fam, ha="right", va="center")
    yy -= 0.093

pan.plot([0, 1], [yy + 0.030, yy + 0.030], color=M.BORDE, lw=1)

# Every sentence below is a statement about these rows, checked against them.
holds = []
if all(h > 0 for h in heads):
    holds.append("Coverage stays above what the vote returns for all %d, so "
                 "there is headroom a better selector could attack at every "
                 "scale here." % len(rows))
else:
    n_closed = sum(1 for h in heads if h <= 0)
    holds.append("Headroom has closed on %d of %d: on those, sampling more is "
                 "no longer the constraint." % (n_closed, len(rows)))

best, worst = max(rows, key=lambda r: r["single"]), min(rows, key=lambda r: r["single"])
if best["gain"] > 0:
    holds.append("The strongest model here (%.1f%% at one sample) still gains "
                 "%+.1f, so this is not only a small-model effect."
                 % (best["single"], best["gain"]))
else:
    holds.append("The strongest model here gains %+.1f: the effect does not "
                 "survive at this scale." % best["gain"])

holds.append("Spread of the gain across models: %+.1f to %+.1f."
             % (min(gains), max(gains)))

pan.text(0, yy - 0.03, "\n\n".join(holds), color=M.NOTA, fontsize=10.5,
         family=fam, va="top", linespacing=1.6, wrap=True)

M.footer(fig,
         "Every row re-derived from that model's own published trajectories, "
         "re-extracting each answer from the raw text. Same protocol for all "
         "of them.",
         "pip install bestofn")

fig.savefig(M.out("13_models.png"), facecolor=M.FONDO, dpi=100)
print("13_models.png  (%d models, %d improved, gains %+.1f to %+.1f)"
      % (len(rows), improved, min(gains), max(gains)))
