"""
Extraccion de caracteristicas lexicas de una URL.

Estas son las unicas caracteristicas que se pueden calcular al instante,
solo con el texto de la URL, sin llamadas externas (WHOIS, DNS, SSL, etc.).
Es exactamente el mismo conjunto de columnas usado para entrenar el modelo
de Random Forest en modelo_2.py, asi que cualquier cambio en este archivo
debe reflejarse tambien alla (y viceversa). Al centralizar la logica aqui
evitamos que el orden de las features se desincronice entre entrenamiento
e inferencia, que era un riesgo real en la version anterior del prototipo.
"""

import re
from urllib.parse import urlparse

# Debe coincidir con las columnas usadas al entrenar el modelo en modelo_2.py
FEATURE_NAMES = [
    "length_url",
    "qty_dot_url",
    "qty_hyphen_url",
    "qty_underline_url",
    "qty_slash_url",
    "qty_questionmark_url",
    "qty_equal_url",
    "qty_at_url",
    "qty_and_url",
    "qty_exclamation_url",
    "qty_percent_url",
    "qty_tld_url",
    "domain_length",
    "domain_in_ip",
    "qty_dot_domain",
]

_IP_REGEX = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def extraer_features_dict(url: str) -> dict:
    """Calcula las caracteristicas lexicas de una URL y las devuelve como dict."""
    dominio = urlparse(url).netloc

    return {
        "length_url": len(url),
        "qty_dot_url": url.count("."),
        "qty_hyphen_url": url.count("-"),
        "qty_underline_url": url.count("_"),
        "qty_slash_url": url.count("/"),
        "qty_questionmark_url": url.count("?"),
        "qty_equal_url": url.count("="),
        "qty_at_url": url.count("@"),
        "qty_and_url": url.count("&"),
        "qty_exclamation_url": url.count("!"),
        "qty_percent_url": url.count("%"),
        # Nota: se aproxima con el conteo de puntos de la URL (igual que el
        # prototipo original). El dataset real cuenta TLDs validos con una
        # lista de sufijos publicos, lo cual seria una mejora futura.
        "qty_tld_url": url.count("."),
        "domain_length": len(dominio),
        "domain_in_ip": 1 if _IP_REGEX.match(dominio) else 0,
        "qty_dot_domain": dominio.count("."),
    }


def extraer_features(url: str, feature_names=None):
    """Devuelve las caracteristicas en el orden que espera el modelo."""
    feature_names = feature_names or FEATURE_NAMES
    valores = extraer_features_dict(url)
    return [valores[nombre] for nombre in feature_names]
