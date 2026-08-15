# -*- coding: utf-8 -*-
"""Best-of-N sobre GSM8K con un modelo distinto al del informe.

Modelo: Qwen/Qwen2.5-0.5B-Instruct  (no es el DeepSeek-R1-Distill-Qwen-1.5B
con el que se midio TR-2026-01, ni un modelo de razonamiento).

Se generan N_MAX trayectorias por problema UNA sola vez. La curva para
N = 1, 2, 4, 8, 16 se obtiene submuestreando esas mismas trayectorias muchas
veces, que es lo estandar: evita volver a generar y da barras de error.

Guarda las respuestas extraidas de cada trayectoria en un JSONL para que
cualquiera pueda recalcular los numeros sin GPU.
"""
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELO = "Qwen/Qwen2.5-0.5B-Instruct"
N_MAX = 16
N_PROBLEMAS = 200
MAX_TOKENS = 400
TEMPERATURA = 0.7
TOP_P = 0.95
SEMILLA = 20260815
REPETICIONES = 400          # submuestreos por cada N
CURVA = [1, 2, 4, 8, 16]

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "gsm8k_qwen05b_n16.jsonl")
RESUMEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "gsm8k_qwen05b_resumen.json")


# --------------------------------------------------------------- utilidades
def respuesta_oro(texto):
    """GSM8K guarda la solucion terminada en '#### 42'."""
    m = re.search(r"####\s*(-?[\d.,]+)", texto)
    if not m:
        return None
    return m.group(1).replace(",", "").rstrip(".")


def media(xs):
    return sum(xs) / len(xs) if xs else 0.0


def desviacion(xs):
    if len(xs) < 2:
        return 0.0
    m = media(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    from bestofn import BestOfN, normalise

    print("=" * 70)
    print("BEST-OF-N SOBRE GSM8K  ·  MODELO DISTINTO AL DEL INFORME")
    print("=" * 70)
    print(f"  modelo      : {MODELO}")
    print(f"  problemas   : {N_PROBLEMAS} del test de GSM8K")
    print(f"  N generado  : {N_MAX} trayectorias por problema")
    print(f"  temperatura : {TEMPERATURA}   top_p: {TOP_P}   "
          f"max_tokens: {MAX_TOKENS}")
    print("=" * 70, flush=True)

    # ---------------------------------------------------------- datos
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(SEMILLA)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:N_PROBLEMAS]

    problemas, oros = [], []
    for i in indices:
        fila = ds[i]
        oro = respuesta_oro(fila["answer"])
        if oro is None:
            continue
        problemas.append(fila["question"])
        oros.append(oro)
    print(f"\n  cargados {len(problemas)} problemas con respuesta valida\n",
          flush=True)

    # ---------------------------------------------------------- generacion
    import torch
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n", flush=True)

    motor = BestOfN(MODELO, n=N_MAX, temperature=TEMPERATURA, top_p=TOP_P,
                    max_tokens=MAX_TOKENS, extractor="boxed",
                    backend="transformers")

    # Reanudar: si ya hay trabajo hecho, no se vuelve a generar. El orden de
    # los problemas es determinista (misma semilla), asi que el indice i
    # identifica siempre al mismo problema.
    filas = []
    if os.path.exists(SALIDA):
        with io.open(SALIDA, encoding="utf-8") as f:
            for linea in f:
                if linea.strip():
                    filas.append(json.loads(linea))
    hechos = len(filas)
    if hechos:
        print(f"  reanudando: {hechos} problemas ya generados, "
              f"faltan {len(problemas)-hechos}\n", flush=True)

    t_inicio = time.time()
    with io.open(SALIDA, "a", encoding="utf-8") as f:
        for k, (problema, oro) in enumerate(zip(problemas, oros), 1):
            if k <= hechos:
                continue
            r = motor.solve(problema)
            respuestas = [s.answer for s in r.samples]
            fila = {
                "i": k,
                "correcta": oro,
                "respuestas": respuestas,
                "aciertos": sum(1 for a in respuestas
                                if normalise(a) == normalise(oro)),
            }
            filas.append(fila)
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()

            if k % 10 == 0 or k == len(problemas):
                transcurrido = time.time() - t_inicio
                por_problema = transcurrido / max(1, k - hechos)
                quedan = por_problema * (len(problemas) - k)
                p1 = media([x["aciertos"] / N_MAX for x in filas])
                print(f"  {k:>3}/{len(problemas)}  "
                      f"pass@1 parcial {p1:.1%}  "
                      f"{por_problema:.1f}s/problema  "
                      f"quedan ~{quedan/60:.0f} min", flush=True)

    print(f"\n  generacion terminada en {(time.time()-t_inicio)/60:.1f} min")
    print(f"  trayectorias guardadas en {os.path.basename(SALIDA)}\n",
          flush=True)

    # ---------------------------------------------------------- analisis
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"\n  {'N':>3}  {'voto mayoria':>16}  {'cobertura':>12}  "
          f"{'hueco':>8}")
    print("  " + "-" * 46)

    rng2 = random.Random(SEMILLA + 1)
    resumen = {"modelo": MODELO, "benchmark": "GSM8K test",
               "problemas": len(filas), "n_generado": N_MAX,
               "temperatura": TEMPERATURA, "max_tokens": MAX_TOKENS,
               "curva": []}

    for n in CURVA:
        precisiones, coberturas = [], []
        for _ in range(REPETICIONES):
            aciertos_v = 0
            aciertos_c = 0
            for fila in filas:
                muestra = rng2.sample(fila["respuestas"], n)
                validas = [normalise(a) for a in muestra if a]
                oro_n = normalise(fila["correcta"])
                if validas:
                    voto = Counter(validas).most_common(1)[0][0]
                    if voto == oro_n:
                        aciertos_v += 1
                if oro_n in validas:
                    aciertos_c += 1
            precisiones.append(100.0 * aciertos_v / len(filas))
            coberturas.append(100.0 * aciertos_c / len(filas))

        pv, pc = media(precisiones), media(coberturas)
        sv = desviacion(precisiones)
        print(f"  {n:>3}  {pv:>13.1f}%    {pc:>10.1f}%  {pc-pv:>7.1f}")
        resumen["curva"].append({
            "n": n, "mayoria": round(pv, 2), "mayoria_sd": round(sv, 2),
            "cobertura": round(pc, 2), "hueco": round(pc - pv, 2),
        })

    base = resumen["curva"][0]["mayoria"]
    tope = resumen["curva"][-1]["mayoria"]
    resumen["ganancia"] = round(tope - base, 2)

    print("  " + "-" * 46)
    print(f"\n  de {base:.1f}% con una sola muestra "
          f"a {tope:.1f}% con {N_MAX}")
    print(f"  ganancia: +{tope-base:.1f} puntos, sin tocar los pesos")

    with io.open(RESUMEN, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"\n  resumen guardado en {os.path.basename(RESUMEN)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
