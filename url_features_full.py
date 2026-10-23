"""
Extraccion completa de las 111 caracteristicas del dataset GregaVrbancic/Phishing-Dataset,
usadas para entrenar el modelo XGBoost final (modelo_3.py) para clasificacion de URLs.

A diferencia de url_features.py (15 caracteristicas lexicas, instantaneas), este modulo
calcula las 111 caracteristicas originales del dataset, lo que implica consultas de red
en tiempo real (WHOIS, DNS, SSL/TLS, HTTP) para 14 de ellas. Esto hace la clasificacion
mas lenta (aprox. 1-5 segundos por URL, segun la latencia de red) pero mucho mas precisa
(97.2% vs 93.4% de accuracy en el conjunto de prueba, ver modelo_3.py).

IMPORTANTE - limitaciones honestas de esta implementacion (documentar en la tesis):
- El repositorio original del dataset no publica el codigo fuente exacto de extraccion
  (solo el dataset ya procesado), asi que esta es una reimplementacion propia basada en
  las definiciones publicadas en su tabla de caracteristicas (README de
  GregaVrbancic/Phishing-Dataset, cita abajo). Puede haber pequenas diferencias de
  metodologia en las features mas ambiguas; cada una tiene su supuesto documentado.
- url_google_index / domain_google_index: no existe una forma gratuita y confiable de
  consultar el indice real de Google sin usar su API de pago (Custom Search) o hacer
  scraping (que Google bloquea activamente y que ademas viola sus terminos de
  servicio). Por eso estas dos features devuelven un valor fijo neutro
  (NEUTRO_GOOGLE_INDEX) en vez de calcularse de verdad. Es una limitacion conocida que
  vale la pena mencionar en el capitulo de discusion de la tesis.
- asn_ip, ttl_hostname, time_response: la metodologia exacta original (que servidor DNS
  se uso, en que unidad de tiempo, etc.) no esta documentada publicamente; se
  implementaron con supuestos razonables, explicados en cada funcion.

Cita del dataset:
G. Vrbancic, I. Jr. Fister, V. Podgorelec. "Datasets for Phishing Websites Detection."
Data in Brief, Vol. 33, 2020. DOI: 10.1016/j.dib.2020.106438
"""

import re
import socket
import ssl
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qsl

import dns.resolver
import requests
import tldextract
import whois


# Valor neutro para las dos features de indexacion en Google (ver limitaciones arriba).
# Se usa 1 ("si esta indexado") porque es el supuesto menos alarmista: no queremos que
# un sitio legitimo aparezca como mas sospechoso solo por esta limitacion tecnica.
NEUTRO_GOOGLE_INDEX = 1

# Valor centinela para features de red que no se pudieron calcular (timeout, dominio
# caido, WHOIS privado/sin respuesta, etc.), en vez de tumbar toda la extraccion.
SENTINEL_ERROR = -1

_CARACTERES = [
    (".", "dot"), ("-", "hyphen"), ("_", "underline"), ("/", "slash"),
    ("?", "questionmark"), ("=", "equal"), ("@", "at"), ("&", "and"),
    ("!", "exclamation"), (" ", "space"), ("~", "tilde"), (",", "comma"),
    ("+", "plus"), ("*", "asterisk"), ("#", "hashtag"), ("$", "dollar"),
    ("%", "percent"),
]

_VOCALES = set("aeiouAEIOU")

_ACORTADORES = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "mcaf.ee", "su.pr", "twurl.nl", "snipurl.com",
    "short.to", "budurl.com", "tr.im", "cli.gs", "shorte.st", "cutt.ly",
    "rebrand.ly", "shorturl.at", "s.id", "v.gd", "tiny.cc", "lnkd.in",
    "soo.gd", "clck.ru",
}

_IP_REGEX = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
_EMAIL_REGEX = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")

# suffix_list_urls=() desactiva la descarga en vivo de la lista de sufijos publicos
# (publicsuffix.org) y usa solo la version incluida con la libreria. Sin esto,
# tldextract intenta salir a internet la primera vez que se usa, lo cual es lento,
# ruidoso (imprime un traceback si no hay red) e innecesario para este caso de uso.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


def _descomponer_url(url: str):
    """Separa la URL en: url completa, domain, directory, file, params (query string)."""
    if "://" not in url:
        url = "http://" + url  # urlparse necesita un esquema para separar netloc de path

    partes = urlparse(url)
    dominio = (partes.hostname or "").lower()
    path = partes.path or ""

    if path in ("", "/") or path.endswith("/"):
        directorio, archivo = path, ""
    else:
        directorio, _, archivo = path.rpartition("/")
        directorio += "/"

    parametros = partes.query or ""
    return url, dominio, directorio, archivo, parametros


def _contar_caracteres(texto: str, sufijo: str) -> dict:
    return {f"qty_{nombre}_{sufijo}": texto.count(caracter) for caracter, nombre in _CARACTERES}


def extraer_features_instantaneas(url: str) -> dict:
    """
    Calcula las 97 caracteristicas que dependen solo del texto de la URL (sin red).
    Se pueden usar solas para pruebas rapidas offline antes de pagar el costo de las
    consultas de red de extraer_features_red().
    """
    url_completa, dominio, directorio, archivo, parametros = _descomponer_url(url)
    resultado = {}

    # ---- URL completa (19 features) ----
    resultado.update(_contar_caracteres(url_completa, "url"))
    resultado["qty_tld_url"] = len(_TLD_EXTRACTOR(url_completa).suffix)
    resultado["length_url"] = len(url_completa)

    # ---- Dominio (21 features) ----
    resultado.update(_contar_caracteres(dominio, "domain"))
    resultado["qty_vowels_domain"] = sum(1 for c in dominio if c in _VOCALES)
    resultado["domain_length"] = len(dominio)
    resultado["domain_in_ip"] = 1 if _IP_REGEX.match(dominio) else 0
    resultado["server_client_domain"] = 1 if ("server" in dominio or "client" in dominio) else 0

    # ---- Directorio (18 features) ----
    resultado.update(_contar_caracteres(directorio, "directory"))
    resultado["directory_length"] = len(directorio)

    # ---- Archivo (18 features) ----
    resultado.update(_contar_caracteres(archivo, "file"))
    resultado["file_length"] = len(archivo)

    # ---- Parametros / query string (21 features) ----
    resultado.update(_contar_caracteres(parametros, "params"))
    resultado["params_length"] = len(parametros)
    tld_dominio = _TLD_EXTRACTOR(url_completa).suffix
    resultado["tld_present_params"] = 1 if (tld_dominio and tld_dominio in parametros.lower()) else 0
    resultado["qty_params"] = len(parse_qsl(parametros)) if parametros else 0
    resultado["email_in_url"] = 1 if _EMAIL_REGEX.search(url_completa) else 0

    return resultado


def _consultar_rdap_fechas(dominio_registrable: str, timeout: float = 3.0):
    """
    Consulta RDAP (el reemplazo moderno de WHOIS, via HTTPS en el puerto 443) para
    obtener la fecha de registro y de vencimiento del dominio. A diferencia de WHOIS
    (puerto 43, texto libre sin formato estandar entre registradores), RDAP devuelve
    JSON estandarizado y usa el mismo puerto que cualquier pagina web (443), por lo
    que no lo bloquean los firewalls/routers que si bloquean el puerto 43 -- una causa
    comun de falsos positivos: un sitio 100% legitimo (ej. wikipedia.org) puede
    quedar con WHOIS en SENTINEL_ERROR simplemente porque la red del usuario bloquea
    ese puerto, no porque el dominio sea sospechoso.

    Devuelve (fecha_creacion, fecha_vencimiento), cada una como datetime o None si no
    se pudo obtener por esta via (en cuyo caso se intenta WHOIS clasico como respaldo).
    """
    try:
        resp = requests.get(
            f"https://rdap.org/domain/{dominio_registrable}",
            timeout=timeout,
            headers={"Accept": "application/rdap+json"},
        )
        if resp.status_code != 200:
            return None, None
        data_rdap = resp.json()
        creacion = vencimiento = None
        for evento in data_rdap.get("events", []):
            fecha = evento.get("eventDate")
            if not fecha:
                continue
            try:
                fecha_dt = datetime.fromisoformat(fecha.replace("Z", "+00:00"))
            except ValueError:
                continue
            accion = evento.get("eventAction", "")
            if accion == "registration":
                creacion = fecha_dt
            elif accion == "expiration":
                vencimiento = fecha_dt
        return creacion, vencimiento
    except Exception:
        return None, None


def extraer_features_red(url: str, timeout: float = 3.0) -> dict:
    """
    Calcula las 14 caracteristicas que requieren consultas de red en tiempo real:
    WHOIS, DNS, SSL/TLS y HTTP. Cada consulta tiene su propio manejo de errores: si una
    falla o se demora mas de `timeout` segundos, esa feature queda en SENTINEL_ERROR
    (o en 0 cuando "no se pudo confirmar" equivale razonablemente a "no tiene"), sin
    tumbar el resto de la extraccion.
    """
    _, dominio, _, _, _ = _descomponer_url(url)
    resultado = {}

    # NS, MX, SPF y WHOIS casi nunca existen a nivel de subdominio (por ejemplo
    # "www.wikipedia.org" no tiene sus propios NS/MX/SPF/WHOIS -- esos registros
    # viven en el dominio registrable "wikipedia.org"). Si consultaramos el
    # hostname literal de la URL, estas features saldrian en 0 para casi
    # cualquier sitio real, perdiendo toda su capacidad de discriminar entre
    # phishing y sitios legitimos. Por eso, solo para esas 4 consultas, se usa
    # el dominio registrable (eTLD+1); el resto de features de red (IP, TTL,
    # TLS, tiempo de respuesta, redirecciones) si usan el hostname literal,
    # porque son propiedades del host exacto al que el usuario se conecta.
    registrable = _TLD_EXTRACTOR(dominio).registered_domain or dominio

    # --- time_response: tiempo de resolucion DNS del dominio (en segundos) ---
    try:
        t0 = time.time()
        socket.gethostbyname(dominio)
        resultado["time_response"] = round(time.time() - t0, 4)
    except Exception:
        resultado["time_response"] = SENTINEL_ERROR

    resolver = dns.resolver.Resolver()
    resolver.timeout = timeout
    resolver.lifetime = timeout

    # --- Registro A: cantidad de IPs resueltas + TTL del hostname ---
    ip_resueltas = []
    try:
        respuesta = resolver.resolve(dominio, "A")
        ip_resueltas = [r.to_text() for r in respuesta]
        resultado["qty_ip_resolved"] = len(ip_resueltas)
        resultado["ttl_hostname"] = respuesta.rrset.ttl
    except Exception:
        resultado["qty_ip_resolved"] = 0
        resultado["ttl_hostname"] = SENTINEL_ERROR

    # --- NS: cantidad de nameservers (consultado sobre el dominio registrable) ---
    try:
        ns = resolver.resolve(registrable, "NS")
        resultado["qty_nameservers"] = len(ns)
    except Exception:
        resultado["qty_nameservers"] = 0

    # --- MX: cantidad de servidores de correo (dominio registrable) ---
    try:
        mx = resolver.resolve(registrable, "MX")
        resultado["qty_mx_servers"] = len(mx)
    except Exception:
        resultado["qty_mx_servers"] = 0

    # --- domain_spf: registro TXT que empieza con "v=spf1" (dominio registrable) ---
    try:
        txt = resolver.resolve(registrable, "TXT")
        resultado["domain_spf"] = 1 if any("v=spf1" in r.to_text().lower() for r in txt) else 0
    except Exception:
        resultado["domain_spf"] = 0

    # --- asn_ip: AS Number via Team Cymru (consulta DNS publica, sin API key) ---
    resultado["asn_ip"] = SENTINEL_ERROR
    if ip_resueltas:
        try:
            octetos = ip_resueltas[0].split(".")
            consulta = f"{'.'.join(reversed(octetos))}.origin.asn.cymru.com"
            txt_asn = resolver.resolve(consulta, "TXT")
            primer_campo = txt_asn[0].to_text().strip('"').split("|")[0].strip()
            resultado["asn_ip"] = int(primer_campo)
        except Exception:
            pass

    # --- Edad del dominio (dias desde creacion) y dias hasta vencimiento ---
    # Se intenta primero RDAP (HTTPS/443); si no da resultado, WHOIS clasico (puerto
    # 43) como respaldo. Ver docstring de _consultar_rdap_fechas para el porque.
    resultado["time_domain_activation"] = SENTINEL_ERROR
    resultado["time_domain_expiration"] = SENTINEL_ERROR

    creacion, vencimiento = _consultar_rdap_fechas(registrable, timeout)

    if creacion is None and vencimiento is None:
        try:
            info = whois.whois(registrable, timeout=timeout)
            creacion = info.creation_date
            vencimiento = info.expiration_date
            if isinstance(creacion, list):
                creacion = creacion[0]
            if isinstance(vencimiento, list):
                vencimiento = vencimiento[0]
        except Exception:
            pass

    try:
        ahora = datetime.now()
        if creacion:
            if creacion.tzinfo is not None:
                creacion = creacion.replace(tzinfo=None)
            resultado["time_domain_activation"] = max((ahora - creacion).days, 0)
        if vencimiento:
            if vencimiento.tzinfo is not None:
                vencimiento = vencimiento.replace(tzinfo=None)
            resultado["time_domain_expiration"] = (vencimiento - ahora).days
    except Exception:
        pass

    # --- TLS/SSL: intenta un handshake valido en el puerto 443 ---
    try:
        contexto = ssl.create_default_context()
        with socket.create_connection((dominio, 443), timeout=timeout) as sock:
            with contexto.wrap_socket(sock, server_hostname=dominio) as ssock:
                ssock.getpeercert()
        resultado["tls_ssl_certificate"] = 1
    except Exception:
        resultado["tls_ssl_certificate"] = 0

    # --- HTTP: cantidad de redirecciones siguiendo la URL ---
    try:
        r = requests.get(url if "://" in url else "http://" + url,
                          timeout=timeout, allow_redirects=True)
        resultado["qty_redirects"] = len(r.history)
    except Exception:
        resultado["qty_redirects"] = SENTINEL_ERROR

    # --- Indexacion en Google: sin metodo gratuito confiable, ver docstring del modulo ---
    resultado["url_google_index"] = NEUTRO_GOOGLE_INDEX
    resultado["domain_google_index"] = NEUTRO_GOOGLE_INDEX

    # --- Acortadores de URL conocidos ---
    resultado["url_shortened"] = 1 if dominio in _ACORTADORES else 0

    return resultado


def extraer_features_full_dict(url: str, timeout: float = 3.0, consultar_red: bool = True) -> dict:
    """
    Devuelve las 111 caracteristicas del dataset completo, como dict {nombre: valor}.
    Si consultar_red=False, las 14 features de red se omiten (solo para pruebas rapidas
    offline; el modelo entrenado SI las necesita todas para predecir correctamente).
    """
    resultado = extraer_features_instantaneas(url)
    if consultar_red:
        resultado.update(extraer_features_red(url, timeout=timeout))
    return resultado


def extraer_features_full(url: str, feature_names, timeout: float = 3.0):
    """Devuelve las caracteristicas en el orden exacto que espera el modelo (feature_names)."""
    valores = extraer_features_full_dict(url, timeout=timeout)
    return [valores.get(nombre, SENTINEL_ERROR) for nombre in feature_names]


if __name__ == "__main__":
    url_prueba = input("Ingrese la URL a analizar: ")
    t0 = time.time()
    features = extraer_features_full_dict(url_prueba)
    print(f"\n{len(features)} caracteristicas calculadas en {time.time() - t0:.2f}s:\n")
    for nombre, valor in features.items():
        print(f"  {nombre:28s}: {valor}")
