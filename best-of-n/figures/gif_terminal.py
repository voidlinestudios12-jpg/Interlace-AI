# -*- coding: utf-8 -*-
"""Genera un GIF/MP4 de un terminal escribiendose solo.

No graba la pantalla: dibuja un terminal falso, asi que no aparece nada del
ordenador (ni escritorio, ni usuario, ni rutas). Salida en 1280x720.
"""
import os, subprocess, shutil
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

W, H, FPS = 1280, 720, 25
FONDO, BARRA = "#0d1117", "#161b22"
VERDE, AZUL, BLANCO, TENUE = "#3fb950", "#4a9eff", "#e6edf3", "#8b949e"
AMARILLO = "#d29922"

GUION = [
    ("prompt", "pip install bestofn", 2.0),
    ("salida", "Successfully installed bestofn-1.0.0", 0.8),
    ("blanco", "", 0.4),
    ("prompt", "python", 0.6),
    ("py", ">>> from bestofn import BestOfN", 1.2),
    ("py", ">>> engine = BestOfN('Qwen/Qwen2.5-0.5B-Instruct', n=16)", 2.2),
    ("py", ">>> r = engine.solve('What is the capital of Australia?')", 2.2),
    ("blanco", "", 0.5),
    ("comentario", "# 16 samples: Canberra x12, Sydney x4", 1.2),
    ("blanco", "", 0.3),
    ("py", ">>> r.answer", 0.8),
    ("resultado", "'Canberra'", 0.5),
    ("py", ">>> r.agreement", 0.8),
    ("resultado", "0.75", 0.5),
    ("blanco", "", 1.6),
]

def fuente(tam):
    for ruta in [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf"]:
        if os.path.exists(ruta):
            return ImageFont.truetype(ruta, tam)
    return ImageFont.load_default()

F = fuente(22)
FT = fuente(15)

def color_de(tipo):
    return {"prompt": BLANCO, "salida": VERDE, "py": BLANCO,
            "comentario": TENUE, "resultado": AMARILLO}.get(tipo, BLANCO)

def dibujar(lineas, cursor_en=None, cursor_col=0, visible=True):
    im = Image.new("RGB", (W, H), FONDO)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 42], fill=BARRA)
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        d.ellipse([22 + i*22, 15, 34 + i*22, 27], fill=c)
    d.text((W//2 - 40, 13), "bestofn", font=FT, fill=TENUE)
    y = 70
    for j, (tipo, txt) in enumerate(lineas):
        x = 32
        if tipo == "prompt":
            d.text((x, y), "$", font=F, fill=VERDE); x += 22
        d.text((x, y), txt, font=F, fill=color_de(tipo))
        if cursor_en == j and visible:
            ancho = d.textlength(txt, font=F)
            d.rectangle([x + ancho + 2, y + 2, x + ancho + 12, y + 26], fill=BLANCO)
        y += 34
    return im

frames = []
hechas = []
for tipo, texto, pausa in GUION:
    if tipo == "blanco":
        hechas.append((tipo, ""))
        for _ in range(int(pausa * FPS)):
            frames.append(dibujar(hechas, len(hechas)-1, visible=(len(frames)//8) % 2 == 0))
        continue
    hechas.append((tipo, ""))
    idx = len(hechas) - 1
    escribible = tipo in ("prompt", "py")
    if escribible:
        for k in range(len(texto) + 1):
            hechas[idx] = (tipo, texto[:k])
            frames.append(dibujar(hechas, idx))
            if k < len(texto):
                frames.append(dibujar(hechas, idx))
    else:
        hechas[idx] = (tipo, texto)
    hechas[idx] = (tipo, texto)
    for _ in range(int(pausa * FPS)):
        frames.append(dibujar(hechas, idx if escribible else None,
                              visible=(len(frames)//8) % 2 == 0))
    if len(hechas) > 14:
        hechas = hechas[-14:]

print(f"fotogramas: {len(frames)}  ({len(frames)/FPS:.1f}s)")
tmp = "_frames"; os.makedirs(tmp, exist_ok=True)
for i, f in enumerate(frames):
    f.save(os.path.join(tmp, f"{i:05d}.png"))

ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-framerate", str(FPS), "-i", f"{tmp}/%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                "terminal_bestofn.mp4"], capture_output=True)
subprocess.run([ff, "-y", "-i", "terminal_bestofn.mp4", "-vf",
                "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                "-loop", "0", "terminal_bestofn.gif"], capture_output=True)
shutil.rmtree(tmp)
for f in ["terminal_bestofn.mp4", "terminal_bestofn.gif"]:
    print(f"  {f}: {os.path.getsize(f)/1e6:.2f} MB")
