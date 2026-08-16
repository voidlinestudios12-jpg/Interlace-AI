# -*- coding: utf-8 -*-
"""GIF: como enchufar un verificador a Best-of-N.

Terminal dibujado por codigo: no aparece nada del ordenador real.
"""
import os, subprocess, shutil
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

W, H, FPS = 1280, 720, 25
FONDO, BARRA = "#0d1117", "#161b22"
VERDE, AZUL, BLANCO, TENUE = "#3fb950", "#4a9eff", "#e6edf3", "#8b949e"
AMARILLO, ROSA, NARANJA = "#d29922", "#ff7b72", "#ffa657"

GUION = [
    ("comentario", "# 1. Your verifier: any function that scores a solution", 1.4),
    ("kw",         ">>> def my_verifier(problem, trajectory):", 0.9),
    ("py",         "...     return probability_it_is_correct   # 0.0 - 1.0", 1.6),
    ("blanco", "", 0.5),
    ("comentario", "# 2. Hand it to the engine", 1.2),
    ("py",         ">>> engine = BestOfN(model, n=32, verifier=my_verifier)", 1.6),
    ("blanco", "", 0.5),
    ("comentario", "# 3. Pick the selector", 1.2),
    ("py",         ">>> engine.solve(problem, method='verifier')", 1.4),
    ("blanco", "", 0.7),
    ("sep",        "AIME 2024  ·  N=32  ·  90 problems", 0.9),
    ("res_gris",   "  majority vote                35.6%", 0.9),
    ("res_azul",   "  verifier, weighted vote      52.2%", 1.2),
    ("res_verde",  "                              +16.6 points", 2.4),
    ("blanco", "", 0.4),
    ("comentario", "# Same model. Same weights. Same 32 samples.", 2.2),
]

def fuente(t):
    for r in [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf"]:
        if os.path.exists(r): return ImageFont.truetype(r, t)
    return ImageFont.load_default()

F, FT = fuente(21), fuente(15)
COL = {"comentario": TENUE, "kw": ROSA, "py": BLANCO, "sep": NARANJA,
       "res_gris": TENUE, "res_azul": AZUL, "res_verde": VERDE}

def dibujar(lineas, cursor=None, visible=True):
    im = Image.new("RGB", (W, H), FONDO)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 42], fill=BARRA)
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        d.ellipse([22 + i*22, 15, 34 + i*22, 27], fill=c)
    d.text((W//2 - 70, 13), "bestofn - verifier", font=FT, fill=TENUE)
    y = 74
    for j, (tipo, txt) in enumerate(lineas):
        if tipo == "sep":
            d.line([32, y + 12, W - 60, y + 12], fill="#21262d", width=1)
            d.text((32, y + 20), txt, font=FT, fill=NARANJA)
            y += 52
            continue
        neg = tipo.startswith("res_") and tipo != "res_gris"
        d.text((32, y), txt, font=F, fill=COL.get(tipo, BLANCO))
        if neg:
            d.text((32.7, y), txt, font=F, fill=COL.get(tipo, BLANCO))
        if cursor == j and visible:
            a = d.textlength(txt, font=F)
            d.rectangle([32 + a + 2, y + 2, 32 + a + 11, y + 25], fill=BLANCO)
        y += 32
    return im

frames, hechas = [], []
for tipo, texto, pausa in GUION:
    if tipo == "blanco":
        hechas.append((tipo, ""))
        for _ in range(int(pausa*FPS)):
            frames.append(dibujar(hechas))
        continue
    hechas.append((tipo, ""))
    i = len(hechas) - 1
    escribe = tipo in ("py", "kw")
    if escribe:
        for k in range(len(texto) + 1):
            hechas[i] = (tipo, texto[:k])
            frames.append(dibujar(hechas, i))
            frames.append(dibujar(hechas, i))
    hechas[i] = (tipo, texto)
    for _ in range(int(pausa*FPS)):
        frames.append(dibujar(hechas, i if escribe else None,
                              visible=(len(frames)//8) % 2 == 0))
    if len(hechas) > 15:
        hechas = hechas[-15:]

print(f"fotogramas: {len(frames)} ({len(frames)/FPS:.1f}s)")
tmp = "_fv"; os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames): f.save(f"{tmp}/{i:05d}.png")
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff,"-y","-framerate",str(FPS),"-i",f"{tmp}/%05d.png","-c:v","libx264",
                "-pix_fmt","yuv420p","-crf","20","verificador_bestofn.mp4"],capture_output=True)
subprocess.run([ff,"-y","-i","verificador_bestofn.mp4","-vf",
                "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                "-loop","0","verificador_bestofn.gif"],capture_output=True)
shutil.rmtree(tmp)
for f in ["verificador_bestofn.mp4","verificador_bestofn.gif"]:
    print(f"  {f}: {os.path.getsize(f)/1e6:.2f} MB")
