# -*- coding: utf-8 -*-
"""Shared brand header and data loading for the Interlace charts.

Two reasons this is a module rather than code repeated in each script:

1. **The logo is an image, not a drawing.** It used to be hand-drawn as two
   ``Rectangle`` patches sized 0.17 x 0.30. The real mark is two *squares* with
   rounded corners that interlace, so what shipped was two tall thin boxes with
   sharp corners. Loading the PNG cannot be distorted.
2. **The numbers are read, never typed.** They come from
   ``results/gsm8k_summary.json``, which is what ``scripts/analyse.py``
   produces. Two scripts here used to carry hand-copied values and silently
   went stale across a release, overwriting the figures the model card leads
   with. A chart that can disagree with the data will eventually do so.

Copyright 2026 Alejandro Areces Rivera - Interlace AI. Apache License 2.0.
"""
import io
import json
import os

import math

import matplotlib.image as mpimg
from matplotlib import font_manager
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# ------------------------------------------------------------------- palette
FONDO = "#06070d"
REJILLA = "#161b22"
AZUL = "#1cbcf0"
VERDE = "#28d58b"
GRIS = "#565e72"
BLANCO = "#ffffff"
TENUE = "#9aa4b0"
NOTA = "#5c6470"
BORDE = "#21262d"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # the package root
LOGO = os.path.join(HERE, "interlace-logo.png")
SUMMARY = os.path.join(ROOT, "results", "gsm8k_summary.json")


def ceil_sig(x, digits=2):
    """Round ``x`` UP to ``digits`` significant figures.

    Needed wherever a number is printed after "p <=". ``"%.1e" % 1.44e-05``
    gives ``1.4e-05``, which is smaller than the value it claims to bound, so
    the printed inequality is false. This shipped once as ``%.0e`` on a worst
    case of 1.401e-05; the fix then was to add a digit, which is the same bug
    one digit over. Rounding up is the fix that does not depend on the
    mantissa being lucky.
    """
    if not x or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x))) - (digits - 1)
    return math.ceil(x / 10 ** e) * 10 ** e


def font():
    """The first font in the list that is actually installed."""
    for c in ("Segoe UI", "Inter", "Arial"):
        if any(f.name == c for f in font_manager.fontManager.ttflist):
            return c
    return "DejaVu Sans"


def data():
    """The published numbers, read from the summary analyse.py writes."""
    if not os.path.exists(SUMMARY):
        raise SystemExit(
            "%s not found.\nRun scripts/analyse.py first." % SUMMARY)
    d = json.load(io.open(SUMMARY, encoding="utf-8"))
    c = d["curve"]
    d["N"] = [r["n"] for r in c]
    d["MAJ"] = [r["majority"] for r in c]
    d["COV"] = [r["coverage"] for r in c]
    d["RND"] = [r["random"] for r in c]
    d["base"] = c[0]["majority"]
    d["top"] = c[-1]["majority"]
    d["cov_top"] = c[-1]["coverage"]
    d["rnd_top"] = c[-1]["random"]
    d["n_top"] = c[-1]["n"]
    # The worst of the seeds, not the one arbitrary draw. "p_value" is the
    # single-seed figure kept for continuity; printing it on the two lead
    # charts put a cherry-picked 1.2e-07 beside the README's honest 1.4e-05.
    mc = d["mcnemar_majority_vs_random"]
    d["p"] = ceil_sig(mc.get("p_value_worst", mc["p_value"]))
    d["p_exact"] = mc.get("p_value_worst", mc["p_value"])
    d["p_median"] = mc.get("p_value_median")
    d["p_seeds"] = mc.get("seeds")
    d["trajectories"] = d["problems"] * d["n_generated"]
    return d


def header(fig, title, subtitle, height=0.20):
    """Stamp the real logo, the wordmark, a headline and a subtitle.

    The logo goes in as an image at a fixed ``zoom``, so it keeps its square
    proportions whatever happens to the figure size.
    """
    fam = font()
    cab = fig.add_axes([0, 1.0 - height, 1, height])
    cab.axis("off")
    cab.set_xlim(0, 16)
    cab.set_ylim(0, 1.8)

    if os.path.exists(LOGO):
        img = OffsetImage(mpimg.imread(LOGO), zoom=0.115)
        cab.add_artist(AnnotationBbox(img, (0.72, 1.30), frameon=False,
                                      box_alignment=(0.5, 0.5)))
    cab.text(1.12, 1.40, "I N T E R L A C E", color=BLANCO, fontsize=12.5,
             fontweight="bold", family=fam, va="center")
    cab.text(1.12, 1.10, "A I", color=AZUL, fontsize=8, fontweight="bold",
             family=fam, va="center")

    cab.text(0.55, 0.58, title, color=BLANCO, fontsize=28,
             fontweight="bold", family=fam, va="center")
    cab.text(0.55, 0.13, subtitle, color=TENUE, fontsize=13, family=fam,
             va="center")
    return cab


def footer(fig, left, right=None):
    """Bottom line: provenance on the left, call to action on the right."""
    fam = font()
    if left:
        fig.text(0.034, 0.035, left, color=NOTA, fontsize=11, family=fam)
    if right:
        fig.text(0.966, 0.035, right, color=AZUL, fontsize=12,
                 fontweight="bold", family=fam, ha="right")


def out(name):
    """Absolute path beside this module, so cwd does not decide where it lands."""
    return os.path.join(HERE, name)
