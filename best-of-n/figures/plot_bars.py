# -*- coding: utf-8 -*-
"""Grafico 1.1: de donde viene la ganancia, con el baseline aleatorio."""
import json, os
import matplotlib; matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt, numpy as np

FONDO, TRACK = "#06070d", "#0f1119"
GRIS  = ("#363c4a", "#565e72")
AZUL  = ("#0b5bff", "#1cbcf0")
VERDE = ("#0e795b", "#28d58b")
BLANCO, TENUE, NOTA, ENLACE = "#ffffff", "#9aa4b0", "#5c6470", "#3b82f6"

FILAS = [
    ("Single sample",  "N=1",                             41.8, GRIS),
    ("Random",         "N=128, among those that answered", 46.9, GRIS),
    ("Majority vote",  "N=128",                           65.0, AZUL),
    ("Coverage",       "N=128, what the model can reach", 91.0, VERDE),
]
TITULO = "One frozen 0.5B model, asked 128 times"
SUB = ("Qwen2.5-0.5B-Instruct on GSM8K. 200 problems, 128 trajectories each, "
       "weights never modified.")
NOTAS = [
    "Of the +23.2 points over a single sample, +18.1 come from selection itself. Exact McNemar vs random: p = 5.7e-08.",
    "Coverage is what the model reaches somewhere in its 128 attempts — the headroom a verifier can still claim.",
]

def deg(c0,c1): return LinearSegmentedColormap.from_list("g",[c0,c1],N=256)

def barra(ax,x,y,w,h,cols,r=0.11):
    g=np.linspace(0,1,256).reshape(1,-1)
    im=ax.imshow(g,extent=[x,x+w,y,y+h],aspect="auto",cmap=deg(*cols),zorder=3)
    im.set_clip_path(FancyBboxPatch((x+r,y+r),max(w-2*r,.01),h-2*r,
        boxstyle=f"round,pad={r}",linewidth=0,transform=ax.transData))

fig=plt.figure(figsize=(16,9),dpi=100); fig.patch.set_facecolor(FONDO)
ax=fig.add_axes([0,0,1,1]); ax.set_facecolor(FONDO)
ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")
fam="DejaVu Sans"
for c in ("Segoe UI","Inter","Arial"):
    if any(f.name==c for f in font_manager.fontManager.ttflist): fam=c; break

ax.add_patch(Rectangle((0.55,8.28),0.17,0.17,facecolor="none",edgecolor="#1cbcf0",linewidth=2.1,zorder=5))
ax.add_patch(Rectangle((0.63,8.19),0.17,0.17,facecolor=FONDO,edgecolor="#0b5bff",linewidth=2.1,zorder=6))
ax.text(0.97,8.34,"I N T E R L A C E",color=BLANCO,fontsize=12.5,fontweight="bold",family=fam,va="center")
ax.text(0.97,8.13,"A I",color="#1cbcf0",fontsize=8,fontweight="bold",family=fam,va="center")
ax.text(0.55,7.42,TITULO,color=BLANCO,fontsize=30,fontweight="bold",family=fam,va="center")
ax.text(0.55,6.98,SUB,color=TENUE,fontsize=13,family=fam,va="center")
ax.text(0.55,6.48,"S E L E C T O R",color=NOTA,fontsize=9.5,fontweight="bold",family=fam,va="center")
ax.text(5.10,6.48,"A C C U R A C Y",color=NOTA,fontsize=9.5,fontweight="bold",family=fam,va="center")

x0,x1=5.10,13.10; util=x1-x0; alto=0.66; paso=1.16; tope=5.85
for i,(n,s,v,c) in enumerate(FILAS):
    y=tope-i*paso-alto
    ax.add_patch(FancyBboxPatch((x0+0.11,y+0.11),util-0.22,alto-0.22,
        boxstyle="round,pad=0.11",facecolor=TRACK,edgecolor="none",zorder=2))
    barra(ax,x0,y,util*v/100.0,alto,c)
    ax.text(0.55,y+alto*0.64,n,color=BLANCO,fontsize=15.5,fontweight="bold",family=fam,va="center")
    ax.text(0.55,y+alto*0.20,s,color=NOTA,fontsize=10.5,family=fam,va="center")
    ax.text(13.45,y+alto/2,f"{v:.1f}%",color=BLANCO,fontsize=21,fontweight="bold",family=fam,va="center")

for j,l in enumerate(NOTAS):
    ax.text(0.55,0.95-j*0.32,l,color=NOTA,fontsize=10.5,family=fam,va="center")
ax.text(15.45,0.63,"every number recomputable: scripts/analyse.py",color=ENLACE,
        fontsize=10.5,family=fam,va="center",ha="right")

fig.savefig("08_bars_n128.png",facecolor=FONDO,dpi=100)
print("guardado 08_bars_n128.png")
