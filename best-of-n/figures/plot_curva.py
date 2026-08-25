# -*- coding: utf-8 -*-
"""Curva de precision frente a N, estetica de marca Interlace."""
import matplotlib; matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt, numpy as np

FONDO, REJILLA = "#06070d", "#161b22"
AZUL, VERDE, GRIS = "#1cbcf0", "#28d58b", "#565e72"
BLANCO, TENUE, NOTA = "#ffffff", "#9aa4b0", "#5c6470"

N    = [1, 2, 4, 8, 16, 32, 64, 128]
MAY  = [41.8, 45.0, 51.6, 56.5, 59.9, 62.1, 63.9, 65.0]
COB  = [41.8, 52.7, 62.2, 70.5, 77.6, 83.2, 87.8, 91.0]
RND  = [41.8, 45.1, 46.3, 46.7, 46.9, 46.6, 46.7, 46.9]

fig = plt.figure(figsize=(16, 9), dpi=100); fig.patch.set_facecolor(FONDO)
fam = "DejaVu Sans"
for c in ("Segoe UI", "Inter", "Arial"):
    if any(f.name == c for f in font_manager.fontManager.ttflist): fam = c; break

cab = fig.add_axes([0, 0.80, 1, 0.20]); cab.axis("off")
cab.set_xlim(0,16); cab.set_ylim(0,1.8)
cab.add_patch(Rectangle((0.55,1.30),0.17,0.30,facecolor="none",edgecolor=AZUL,lw=2.1))
cab.add_patch(Rectangle((0.63,1.14),0.17,0.30,facecolor=FONDO,edgecolor="#0b5bff",lw=2.1))
cab.text(0.97,1.42,"I N T E R L A C E",color=BLANCO,fontsize=12.5,fontweight="bold",family=fam,va="center")
cab.text(0.97,1.06,"A I",color=AZUL,fontsize=8,fontweight="bold",family=fam,va="center")
cab.text(0.55,0.55,"41.8%  \u2192  65.0%   by asking the same frozen model more than once",
         color=BLANCO,fontsize=27,fontweight="bold",family=fam,va="center")
cab.text(0.55,0.12,"Qwen2.5-0.5B-Instruct on GSM8K \u00b7 200 problems \u00b7 128 trajectories each \u00b7 weights never modified",
         color=TENUE,fontsize=13,family=fam,va="center")

ax = fig.add_axes([0.075, 0.13, 0.62, 0.60]); ax.set_facecolor(FONDO)
for s in ax.spines.values(): s.set_color("#21262d")
ax.grid(True, color=REJILLA, lw=0.9); ax.set_axisbelow(True)
ax.set_xscale("log", base=2); ax.set_xticks(N)
ax.set_xticklabels([str(x) for x in N], color=TENUE, fontsize=11)
ax.set_yticks(range(30, 101, 10))
ax.set_yticklabels([f"{y}%" for y in range(30,101,10)], color=TENUE, fontsize=11)
ax.set_ylim(33, 97); ax.set_xlim(0.85, 165)
ax.tick_params(colors="#30363d", length=3)
ax.set_xlabel("trajectories sampled  (N)", color=TENUE, fontsize=12.5, labelpad=9)

ax.fill_between(N, MAY, COB, color=VERDE, alpha=0.055)
ax.plot(N, COB, color=VERDE, lw=2.6, marker="o", ms=6, label="coverage  (reachable)")
ax.plot(N, MAY, color=AZUL,  lw=3.4, marker="o", ms=7, label="majority vote  (returned)")
ax.plot(N, RND, color=GRIS,  lw=2.0, marker="o", ms=5, ls="--", label="random  (baseline)")

for x, y, c, dy in [(128, 65.0, AZUL, 2.6), (128, 91.0, VERDE, 2.4), (128, 46.9, GRIS, -4.4)]:
    ax.annotate(f"{y}%", (x, y), textcoords="offset points", xytext=(-16, dy*4),
                color=c, fontsize=15, fontweight="bold", family=fam, ha="right")
ax.annotate("41.8%", (1, 41.8), textcoords="offset points", xytext=(6, -19),
            color=BLANCO, fontsize=13, fontweight="bold", family=fam)

lg = ax.legend(loc="lower right", frameon=True, fontsize=11.5,
               facecolor="#0f1119", edgecolor="#21262d", labelcolor=TENUE)

pan = fig.add_axes([0.735, 0.13, 0.225, 0.60]); pan.axis("off")
pan.set_xlim(0,1); pan.set_ylim(0,1)
pan.text(0, 0.96, "W H E R E   T H E   G A I N   I S", color=NOTA, fontsize=9.5,
         fontweight="bold", family=fam, va="top")
filas = [("single sample", "41.8%", TENUE), ("random at N=128", "46.9%", GRIS),
         ("majority at N=128", "65.0%", AZUL)]
y = 0.85
for etiqueta, val, col in filas:
    pan.text(0, y, etiqueta, color=TENUE, fontsize=11.5, family=fam, va="center")
    pan.text(1, y, val, color=col, fontsize=14, fontweight="bold", family=fam,
             va="center", ha="right")
    y -= 0.105
pan.plot([0,1],[y+0.045,y+0.045], color="#21262d", lw=1)
pan.text(0, y-0.03, "from voting", color=BLANCO, fontsize=11.5, family=fam, va="center")
pan.text(1, y-0.03, "+18.1", color=AZUL, fontsize=16, fontweight="bold",
         family=fam, va="center", ha="right")
pan.text(0, y-0.15, "exact McNemar vs random", color=TENUE, fontsize=10.5, family=fam, va="center")
pan.text(1, y-0.15, "p = 5.7e-08", color=VERDE, fontsize=12, fontweight="bold",
         family=fam, va="center", ha="right")
pan.text(0, y-0.34,
         "Coverage 91.0% at N=128:\nnine problems in ten, this\n0.5B model does find the\nanswer somewhere.",
         color=NOTA, fontsize=10.5, family=fam, va="top", linespacing=1.6)

pie = fig.add_axes([0,0,1,0.09]); pie.axis("off"); pie.set_xlim(0,16); pie.set_ylim(0,1)
pie.text(0.55,0.62,"25,600 trajectories published in full. Every number recomputable with scripts/analyse.py, no GPU.",
         color=NOTA,fontsize=11,family=fam,va="center")
pie.text(15.45,0.62,"pip install bestofn",color="#3b82f6",fontsize=11.5,
         fontweight="bold",family=fam,va="center",ha="right")

fig.savefig("07_curve_n128.png", facecolor=FONDO, dpi=100)
print("guardado 07_curve_n128.png")
