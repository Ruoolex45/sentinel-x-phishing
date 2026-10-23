"""
Prediccion de phishing para URLs -- VERSION FINAL (XGBoost, 111 caracteristicas).

Esta es la version recomendada para produccion/front-end: usa el modelo XGBoost
entrenado en modelo_3.py sobre las 111 caracteristicas completas del dataset
(97.2% de accuracy en el conjunto de prueba), en vez del Random Forest liviano de
15 caracteristicas de prototipo1.py (93.4% de accuracy). El costo es que esta
version necesita hacer consultas de red en tiempo real (WHOIS, DNS, SSL, HTTP) por
cada URL -- ver url_features_full.py -- por lo que una clasificacion tarda
tipicamente 1 a 5 segundos, en vez de ser instantanea.

prototipo1.py (Random Forest, 15 features) se conserva como alternativa liviana:
por ejemplo, para un modo "rapido" en el front-end, o como respaldo si en algun
momento las consultas de red no estan disponibles.

Requiere haber corrido modelo_3.py al menos una vez para generar modelo_pro_xgb.pkl.
"""

import re
import joblib
import numpy as np
from urllib.parse import urlparse

from url_features_full import extraer_features_full_dict, SENTINEL_ERROR

RUTA_MODELO = "modelo_pro_xgb.pkl"

_WHITELIST = {
    "www.youtube.com", "www.google.com", "www.facebook.com",
    "umb.instructure.com",
}

_IP_REGEX = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def cargar_modelo(ruta=RUTA_MODELO):
    """Carga el modelo XGBoost y la lista de features guardada junto con el."""
    artefacto = joblib.load(ruta)
    return artefacto["model"], artefacto["feature_names"]


def clasificar_url(url: str, modelo=None, feature_names=None, timeout: float = 3.0) -> dict:
    """
    Clasifica una URL y devuelve un resultado estructurado, con las mismas
    categorias que la tabla `resultados` del backend (ver Manual Tecnico:
    clasificacion, probabilidad, nivel_riesgo, puntaje_heuristico, explicacion).

    A diferencia de prototipo1.py, esta funcion hace consultas de red reales
    (WHOIS/DNS/SSL/HTTP) para calcular las 111 features del modelo XGBoost, asi
    que puede tardar varios segundos por URL.
    """
    if modelo is None or feature_names is None:
        modelo, feature_names = cargar_modelo()

    dominio = urlparse(url if "://" in url else "http://" + url).hostname or ""
    dominio = dominio.lower()

    if dominio in _WHITELIST:
        return {"clasificacion": "seguro", "probabilidad": 0.0, "nivel_riesgo": "bajo",
                "puntaje_heuristico": 0.0, "explicacion": "Dominio en lista blanca de sitios conocidos."}

    if "@" in url or _IP_REGEX.match(dominio):
        return {"clasificacion": "phishing", "probabilidad": 1.0, "nivel_riesgo": "alto",
                "puntaje_heuristico": 1.0,
                "explicacion": "La URL usa '@' o una direccion IP en el dominio, "
                               "un patron tipico de suplantacion."}

    valores_dict = extraer_features_full_dict(url, timeout=timeout)
    features = np.array([valores_dict.get(nombre, SENTINEL_ERROR) for nombre in feature_names]).reshape(1, -1)
    probabilidad = float(modelo.predict_proba(features)[0][1])

    if probabilidad >= 0.7:
        clasificacion, nivel_riesgo = "phishing", "alto"
    elif probabilidad >= 0.4:
        clasificacion, nivel_riesgo = "sospechoso", "medio"
    else:
        clasificacion, nivel_riesgo = "seguro", "bajo"

    return {
        "clasificacion": clasificacion, "probabilidad": round(probabilidad, 4),
        "nivel_riesgo": nivel_riesgo, "puntaje_heuristico": None,
        "explicacion": "Clasificado por el modelo XGBoost a partir de las 111 "
                       "caracteristicas del dataset (lexicas, DNS, WHOIS, SSL y HTTP).",
    }


if __name__ == "__main__":
    modelo, feature_names = cargar_modelo()
    url = input("Ingrese la URL: ")
    print("\nAnalizando (esto puede tardar unos segundos por las consultas de red)...")
    resultado = clasificar_url(url, modelo, feature_names)
    print(f"\nClasificacion  : {resultado['clasificacion']}")
    print(f"Probabilidad   : {resultado['probabilidad']}")
    print(f"Nivel de riesgo: {resultado['nivel_riesgo']}")
    print(f"Explicacion    : {resultado['explicacion']}")
