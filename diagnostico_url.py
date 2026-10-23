# -*- coding: utf-8 -*-
"""
Diagnostico: por que Wikipedia y Mercado Libre (sitios legitimos y muy conocidos)
salieron clasificados como phishing con probabilidad casi 1.0?

Este script imprime las 111 caracteristicas calculadas para cada URL, marcando en
rojo (con "<-- FALLO/SENTINEL") las que no se pudieron calcular (valor -1, que en
teoria deberia ser raro para un sitio real y bien establecido). Tambien mide cuanto
tarda cada consulta de red por separado (WHOIS, DNS, TLS, HTTP) para detectar si
alguna se esta cayendo por timeout en vez de por un problema real del sitio.
"""
import warnings
warnings.filterwarnings("ignore")
import time

from url_features_full import extraer_features_full_dict, SENTINEL_ERROR
from prototipo_url_final import cargar_modelo

modelo, feature_names = cargar_modelo()

urls = [
    "https://es.wikipedia.org/wiki/Phishing",
    "https://www.mercadolibre.com.co",
]

for url in urls:
    print("\n" + "=" * 70)
    print(f"URL: {url}")
    print("=" * 70)

    t0 = time.time()
    valores = extraer_features_full_dict(url, timeout=5.0)
    print(f"(extraccion completa tardo {time.time() - t0:.2f}s)\n")

    fallos = []
    for nombre, valor in valores.items():
        marca = ""
        if valor == SENTINEL_ERROR:
            marca = "  <-- SENTINEL_ERROR (fallo la consulta)"
            fallos.append(nombre)
        print(f"  {nombre:28s}: {valor}{marca}")

    print(f"\nTotal features en SENTINEL_ERROR: {len(fallos)}  -> {fallos}")

    # Vector en el orden exacto que espera el modelo, y prediccion
    features_vec = [valores.get(n, SENTINEL_ERROR) for n in feature_names]
    faltantes = [n for n in feature_names if n not in valores]
    print(f"Features que el modelo espera pero NO estan en el dict calculado: {faltantes}")

    prob = modelo.predict_proba([features_vec])[0][1]
    print(f"\nProbabilidad de phishing segun el modelo: {prob:.4f}")

print("\n" + "=" * 70)
print(f"Total de features que el modelo espera (feature_names): {len(feature_names)}")
