"""
Prediccion de phishing para URLs.

Este modulo expone `clasificar_url(url)` para que pueda importarse
directamente desde el backend/frontend mas adelante (por ejemplo desde
una vista de FastAPI/Flask), y tambien se puede ejecutar directamente
para probarlo por consola, igual que antes.
"""

import re
import joblib
import numpy as np
from urllib.parse import urlparse

from url_features import extraer_features, FEATURE_NAMES

RUTA_MODELO = "modelo_pro.pkl"

_WHITELIST = {
    "www.youtube.com",
    "www.google.com",
    "www.facebook.com",
    "umb.instructure.com",
}

_IP_REGEX = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def cargar_modelo(ruta=RUTA_MODELO):
    """Carga el modelo y la lista de features guardada junto con el."""
    artefacto = joblib.load(ruta)

    # Compatibilidad con modelos guardados en el formato antiguo (solo el
    # modelo, sin diccionario de metadatos).
    if isinstance(artefacto, dict):
        return artefacto["model"], artefacto["feature_names"]

    return artefacto, FEATURE_NAMES


def clasificar_url(url: str, modelo=None, feature_names=None) -> dict:
    """
    Clasifica una URL y devuelve un resultado estructurado, con las mismas
    categorias que la tabla `resultados` del backend (ver Manual Tecnico:
    clasificacion, probabilidad, nivel_riesgo, puntaje_heuristico, explicacion).
    """
    if modelo is None or feature_names is None:
        modelo, feature_names = cargar_modelo()

    dominio = urlparse(url).netloc.lower()

    # Reglas heuristicas (se evaluan antes que el modelo)
    if dominio in _WHITELIST:
        return {
            "clasificacion": "seguro",
            "probabilidad": 0.0,
            "nivel_riesgo": "bajo",
            "puntaje_heuristico": 0.0,
            "explicacion": "Dominio en lista blanca de sitios conocidos.",
        }

    if "@" in url or _IP_REGEX.match(dominio):
        return {
            "clasificacion": "phishing",
            "probabilidad": 1.0,
            "nivel_riesgo": "alto",
            "puntaje_heuristico": 1.0,
            "explicacion": "La URL usa '@' o una direccion IP en el dominio, "
                           "un patron tipico de suplantacion.",
        }

    # Modelo de machine learning
    features = np.array(extraer_features(url, feature_names)).reshape(1, -1)
    probabilidad = float(modelo.predict_proba(features)[0][1])  # prob. de phishing

    if probabilidad >= 0.7:
        clasificacion, nivel_riesgo = "phishing", "alto"
    elif probabilidad >= 0.4:
        clasificacion, nivel_riesgo = "sospechoso", "medio"
    else:
        clasificacion, nivel_riesgo = "seguro", "bajo"

    return {
        "clasificacion": clasificacion,
        "probabilidad": round(probabilidad, 4),
        "nivel_riesgo": nivel_riesgo,
        "puntaje_heuristico": None,
        "explicacion": "Clasificado por el modelo Random Forest a partir de "
                       "las caracteristicas lexicas de la URL.",
    }


if __name__ == "__main__":
    modelo, feature_names = cargar_modelo()
    url = input("Ingrese la URL: ")
    resultado = clasificar_url(url, modelo, feature_names)

    print(f"\nClasificacion  : {resultado['clasificacion']}")
    print(f"Probabilidad   : {resultado['probabilidad']}")
    print(f"Nivel de riesgo: {resultado['nivel_riesgo']}")
    print(f"Explicacion    : {resultado['explicacion']}")
