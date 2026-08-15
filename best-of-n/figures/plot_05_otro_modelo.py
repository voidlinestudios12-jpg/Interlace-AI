# -*- coding: utf-8 -*-
"""Grafico 05 - Best-of-N sobre un modelo distinto al del informe.

Mismo estilo exacto que los cuatro graficos anteriores: paleta muestreada de
01_aime_base_vs_bestofn.png para que la serie sea coherente.

Datos: gsm8k_qwen05b_resumen.json, 200 problemas del test de GSM8K con
Qwen2.5-0.5B-Instruct, 16 trayectorias por problema, medido en una RTX 3060.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------ paleta
FONDO = "#06070d"
TRACK = "#0f1119"
GRIS = ("#363c4a", "#565e72")
AZUL = ("#0b5bff", "#1cbcf0")
VERDE = ("#0e795b", "#28d58b")
BLANCO = "#ffffff"
TENUE = "#9aa4b0"
NOTA = "#5c6470"
ENLACE = "#3b82f6"

# ------------------------------------------------------------------- datos
TITULO = "GSM8K — a model this was never built for"
SUBTITULO = ("Qwen2.5-0.5B-Instruct, weights frozen. A different architecture, "
             "a different task, no retuning.")

FILAS = [
    ("Base model",   "single sample",  38.2, GRIS),
    ("Best-of-4",    "majority vote",  45.5, AZUL),
    ("Best-of-8",    "majority vote",  50.9, AZUL),
    ("Best-of-16",   "majority vote",  53.3, AZUL),
    ("Best-of-16",   "coverage (pass@N)", 79.5, VERDE),
]
MAXIMO = 100.0

NOTAS = [
    "200 problems from the GSM8K test set, 16 trajectories per problem, "
    "measured on one consumer GPU.",
    "+15.1 points over a single sample, on a model the method was not "
    "developed against. Raw trajectories published.",
]

SALIDA = "05_otro_modelo_gsm8k.png"


def degradado(c0, c1):
    return LinearSegmentedColormap.from_list("g", [c0, c1], N=256)


def barra_redonda(ax, x, y, ancho, alto, colores, radio=0.11):
    """Barra con degradado horizontal y esquinas redondeadas."""
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    im = ax.imshow(grad, extent=[x, x + ancho, y, y + alto], aspect="auto",
                   cmap=degradado(*colores), zorder=3)
    recorte = FancyBboxPatch(
        (x + radio, y + radio), max(ancho - 2 * radio, 0.01), alto - 2 * radio,
        boxstyle=f"round,pad={radio}", linewidth=0, transform=ax.transData)
    im.set_clip_path(recorte)
    return im


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(FONDO)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(FONDO)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    fam = "DejaVu Sans"
    for cand in ("Segoe UI", "Inter", "Helvetica Neue", "Arial"):
        if any(f.name == cand for f in font_manager.fontManager.ttflist):
            fam = cand
            break

    # ------------------------------------------------------------ marca
    ax.add_patch(Rectangle((0.55, 8.28), 0.17, 0.17, facecolor="none",
                           edgecolor="#1cbcf0", linewidth=2.1, zorder=5))
    ax.add_patch(Rectangle((0.63, 8.19), 0.17, 0.17, facecolor=FONDO,
                           edgecolor="#0b5bff", linewidth=2.1, zorder=6))
    ax.text(0.97, 8.34, "I N T E R L A C E", color=BLANCO, fontsize=12.5,
            fontweight="bold", family=fam, va="center")
    ax.text(0.97, 8.13, "A I", color="#1cbcf0", fontsize=8,
            fontweight="bold", family=fam, va="center")

    # ---------------------------------------------------------- titulares
    ax.text(0.55, 7.42, TITULO, color=BLANCO, fontsize=33, fontweight="bold",
            family=fam, va="center")
    ax.text(0.55, 6.98, SUBTITULO, color=TENUE, fontsize=13.5, family=fam,
            va="center")

    # -------------------------------------------------------- encabezados
    ax.text(0.55, 6.52, "S E L E C T O R", color=NOTA, fontsize=9.5,
            fontweight="bold", family=fam, va="center")
    ax.text(4.30, 6.52, "A C C U R A C Y", color=NOTA, fontsize=9.5,
            fontweight="bold", family=fam, va="center")

    # -------------------------------------------------------------- barras
    x0, x1 = 4.30, 13.28
    util = x1 - x0
    alto = 0.60
    paso = 0.96
    tope = 6.00

    for i, (nombre, sub, valor, colores) in enumerate(FILAS):
        y = tope - i * paso - alto

        ax.add_patch(FancyBboxPatch(
            (x0 + 0.11, y + 0.11), util - 0.22, alto - 0.22,
            boxstyle="round,pad=0.11", facecolor=TRACK, edgecolor="none",
            zorder=2))

        barra_redonda(ax, x0, y, util * valor / MAXIMO, alto, colores)

        ax.text(0.55, y + alto * 0.62, nombre, color=BLANCO, fontsize=15.5,
                fontweight="bold", family=fam, va="center")
        ax.text(0.55, y + alto * 0.18, sub, color=NOTA, fontsize=11,
                family=fam, va="center")

        ax.text(13.62, y + alto / 2, f"{valor:.1f}%", color=BLANCO,
                fontsize=21, fontweight="bold", family=fam, va="center",
                path_effects=[pe.Normal()])

    # -------------------------------------------------------------- notas
    for j, linea in enumerate(NOTAS):
        ax.text(0.55, 0.92 - j * 0.30, linea, color=NOTA, fontsize=11,
                family=fam, va="center")

    ax.text(15.45, 0.62, "doi.org/10.5281/zenodo.21936833", color=ENLACE,
            fontsize=11, family=fam, va="center", ha="right")

    destino = os.path.join(aqui, SALIDA)
    fig.savefig(destino, facecolor=FONDO, dpi=100)
    print("guardado:", destino)


if __name__ == "__main__":
    main()
