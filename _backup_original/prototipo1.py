import joblib
import numpy as np
from urllib.parse import urlparse
import re


modelo = joblib.load("modelo_pro.pkl")


def extraer_features(url):

    dominio = urlparse(url).netloc

    return [
        len(url),
        url.count("."),
        url.count("-"),
        url.count("_"),
        url.count("/"),
        url.count("?"),
        url.count("="),
        url.count("@"),
        url.count("&"),
        url.count("!"),
        url.count("%"),
        url.count("."),
        len(dominio),
        1 if re.match(r"\d+\.\d+\.\d+\.\d+", dominio) else 0,
        dominio.count(".")
    ]


def clasificar_url(url, modelo):

    features = np.array(extraer_features(url)).reshape(1, -1)
    pred = modelo.predict(features)[0]

    dominio = urlparse(url).netloc


    if dominio in ["www.youtube.com", "www.google.com", "www.facebook.com","https://umb.instructure.com/"]:
        return "  (whitelist)"

    if "@" in url or re.match(r"\d+\.\d+\.\d+\.\d+", dominio):
        return " Phishing probable"

    if pred == 1:
        return "Phishing detectado"

    return " Probablemente seguro"


url = input("Ingrese la URL: ")

resultado = clasificar_url(url, modelo)

print(resultado)