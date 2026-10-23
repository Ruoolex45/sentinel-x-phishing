# -*- coding: utf-8 -*-
"""
Sentinel-X - backend Flask
Sirve el front-end (basado en el diseno de Google Stitch) conectado de verdad a
los dos modelos finales del proyecto: XGBoost (URL) y Random Forest + embeddings
multilingues (correo).

Para correrlo:
    cd ProyectoPhishing/webapp
    python app.py
Luego abrir http://127.0.0.1:5000 en el navegador.

Requiere las mismas dependencias que ya usan los scripts del proyecto (joblib,
xgboost, scikit-learn, sentence-transformers, dnspython, requests, tldextract,
python-whois) mas Flask (`pip install flask` si no lo tienes en el venv).
"""
import os
import sys
import json
import sqlite3
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, redirect, url_for, g, flash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)  # carpeta ProyectoPhishing (un nivel arriba)
sys.path.insert(0, PROJECT_DIR)

DB_PATH = os.path.join(BASE_DIR, "historial.db")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

app = Flask(__name__)
app.secret_key = "sentinel-x-thesis-demo"  # solo para flash messages, no hay login real

# =====================================================================
# 1. CARGA DE LOS MODELOS FINALES (una sola vez, al arrancar el server)
# =====================================================================
print("=" * 60)
print("Cargando modelo de URL (XGBoost, 111 caracteristicas)...")
from prototipo_url_final import cargar_modelo as _cargar_modelo_url
from url_features_full import extraer_features_full_dict, SENTINEL_ERROR

_modelo_url, _feature_names_url = _cargar_modelo_url()
print("Modelo de URL listo.")

print("Cargando modelo de correo (Random Forest + embeddings multilingues)...")
from CorreoPrototipo1 import entrenar_o_cargar as _cargar_modelo_correo

_modelo_correo, _embedder_correo = _cargar_modelo_correo()
print("Modelo de correo listo.")
print("=" * 60)

# =====================================================================
# 2. CONFIGURACION (sensibilidad del escaner, ajustable desde Settings)
# =====================================================================
SENSIBILIDADES = {
    "relaxed": {"phishing": 0.85, "sospechoso": 0.60, "label": "Relajada"},
    "balanced": {"phishing": 0.70, "sospechoso": 0.45, "label": "Balanceada"},
    "strict": {"phishing": 0.50, "sospechoso": 0.25, "label": "Estricta"},
}

CONFIG_DEFAULT = {
    "sensibilidad": "balanced",
    "alertas_tiempo_real": True,
    "resumen_semanal": True,
    "prompts_educativos": False,
    "analytics_anonimos": True,
    "retencion_dias": "30",
}


def cargar_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = CONFIG_DEFAULT.copy()
                cfg.update(json.load(f))
                return cfg
        except Exception:
            pass
    return CONFIG_DEFAULT.copy()


def guardar_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def reclasificar(probabilidad, sensibilidad):
    """Re-aplica los umbrales seguro/sospechoso/phishing segun la sensibilidad
    elegida en Settings, a partir de la probabilidad cruda que ya devolvio el
    modelo (el modelo en si no se reentrena ni cambia)."""
    u = SENSIBILIDADES.get(sensibilidad, SENSIBILIDADES["balanced"])
    if probabilidad >= u["phishing"]:
        return "phishing", "alto"
    elif probabilidad >= u["sospechoso"]:
        return "sospechoso", "medio"
    return "seguro", "bajo"


# =====================================================================
# 3. BASE DE DATOS (historial real de escaneos, SQLite)
# =====================================================================
def get_db():
    db = getattr(g, "_db", None)
    if db is None:
        db = g._db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo TEXT NOT NULL,
            objetivo TEXT NOT NULL,
            clasificacion TEXT NOT NULL,
            nivel_riesgo TEXT NOT NULL,
            probabilidad REAL NOT NULL,
            explicacion TEXT,
            detalle_json TEXT,
            creado_en TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# =====================================================================
# 4. SEÑALES LEGIBLES A PARTIR DE LAS CARACTERISTICAS REALES DE LA URL
#    (nada de categorias inventadas -- solo lo que de verdad se calculo)
# =====================================================================
def calcular_senales_url(valores):
    senales = []

    if valores.get("domain_in_ip") == 1:
        senales.append({
            "icono": "alert-triangle",
            "titulo": "Dirección IP en vez de dominio",
            "detalle": "La URL usa una dirección IP directamente, algo muy inusual en sitios legítimos.",
        })

    if valores.get("url_shortened") == 1:
        senales.append({
            "icono": "link-2",
            "titulo": "Acortador de URL",
            "detalle": "El enlace usa un servicio acortador conocido, lo que oculta el destino real.",
        })

    if valores.get("qty_at_url", 0) and valores.get("qty_at_url", 0) > 0:
        senales.append({
            "icono": "at-sign",
            "titulo": "Símbolo \"@\" en la URL",
            "detalle": "El uso de \"@\" en una URL puede ocultar el verdadero dominio de destino.",
        })

    edad = valores.get("time_domain_activation", SENTINEL_ERROR)
    if edad != SENTINEL_ERROR and edad < 30:
        senales.append({
            "icono": "calendar",
            "titulo": "Dominio muy reciente",
            "detalle": f"Este dominio fue registrado hace solo {edad} día(s).",
        })

    if valores.get("tls_ssl_certificate") == 0:
        senales.append({
            "icono": "unlock",
            "titulo": "Sin conexión segura (SSL/TLS)",
            "detalle": "No se pudo establecer una conexión HTTPS válida con el sitio.",
        })

    redirs = valores.get("qty_redirects", 0)
    if redirs and redirs != SENTINEL_ERROR and redirs >= 3:
        senales.append({
            "icono": "shuffle",
            "titulo": "Múltiples redirecciones",
            "detalle": f"La URL redirige {redirs} veces antes de llegar al destino final.",
        })

    if valores.get("qty_hyphen_domain", 0) and valores.get("qty_hyphen_domain", 0) >= 2:
        senales.append({
            "icono": "type",
            "titulo": "Dominio con guiones múltiples",
            "detalle": "Un patrón común usado para imitar nombres de marcas conocidas.",
        })

    if not senales:
        senales.append({
            "icono": "check-circle",
            "titulo": "Sin señales de riesgo evidentes",
            "detalle": "Las características analizadas no muestran patrones típicos de phishing.",
        })

    return senales


# =====================================================================
# 5. RUTAS
# =====================================================================
@app.route("/")
def dashboard():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM scans").fetchone()["c"]
    amenazas = db.execute(
        "SELECT COUNT(*) c FROM scans WHERE clasificacion='phishing'"
    ).fetchone()["c"]
    recientes = db.execute(
        "SELECT * FROM scans ORDER BY id DESC LIMIT 8"
    ).fetchall()
    return render_template(
        "dashboard.html", total=total, amenazas=amenazas, recientes=recientes
    )


@app.route("/scan", methods=["POST"])
def scan():
    tipo = request.form.get("tipo", "url")
    valor = request.form.get("valor", "").strip()

    if not valor:
        flash("Ingresa una URL o el texto del correo antes de analizar.")
        return redirect(url_for("dashboard"))

    if tipo == "hash":
        flash("El análisis de File Hash todavía no está disponible en esta versión.")
        return redirect(url_for("dashboard"))

    cfg = cargar_config()
    detalle = {}

    if tipo == "url":
        valores = extraer_features_full_dict(valor, timeout=5.0)
        features = [valores.get(n, SENTINEL_ERROR) for n in _feature_names_url]
        probabilidad = float(_modelo_url.predict_proba([features])[0][1])
        clasificacion, nivel_riesgo = reclasificar(probabilidad, cfg["sensibilidad"])
        explicacion = (
            "Clasificado por el modelo XGBoost a partir de 111 características "
            "(léxicas, DNS, WHOIS/RDAP, SSL y HTTP)."
        )
        detalle = {
            "ssl_valido": bool(valores.get("tls_ssl_certificate")),
            "antiguedad_dias": valores.get("time_domain_activation", SENTINEL_ERROR),
            "vencimiento_dias": valores.get("time_domain_expiration", SENTINEL_ERROR),
            "redirecciones": valores.get("qty_redirects", SENTINEL_ERROR),
            "servidores_nombre": valores.get("qty_nameservers", 0),
            "servidores_correo": valores.get("qty_mx_servers", 0),
            "asn": valores.get("asn_ip", SENTINEL_ERROR),
            "senales": calcular_senales_url(valores),
        }
    else:  # tipo == "email"
        X_input = _embedder_correo.encode([valor])
        probabilidad = float(_modelo_correo.predict_proba(X_input)[0][1])
        clasificacion, nivel_riesgo = reclasificar(probabilidad, cfg["sensibilidad"])
        explicacion = (
            "Clasificado por Random Forest sobre embeddings multilingües "
            "(sentence-transformers), con soporte nativo para español."
        )
        detalle = {"texto_analizado": valor}

    db = get_db()
    cur = db.execute(
        """INSERT INTO scans (tipo, objetivo, clasificacion, nivel_riesgo,
                               probabilidad, explicacion, detalle_json, creado_en)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            tipo,
            valor if tipo == "url" else (valor[:80] + "…" if len(valor) > 80 else valor),
            clasificacion,
            nivel_riesgo,
            probabilidad,
            explicacion,
            json.dumps(detalle, ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    db.commit()
    return redirect(url_for("risk_analysis", scan_id=cur.lastrowid))


@app.route("/risk-analysis")
@app.route("/risk-analysis/<int:scan_id>")
def risk_analysis(scan_id=None):
    db = get_db()
    if scan_id is None:
        registro = db.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT 1"
        ).fetchone()
    else:
        registro = db.execute(
            "SELECT * FROM scans WHERE id = ?", (scan_id,)
        ).fetchone()

    if registro is None:
        flash("Todavía no has analizado nada. Empieza por el Dashboard.")
        return redirect(url_for("dashboard"))

    detalle = json.loads(registro["detalle_json"] or "{}")
    return render_template("risk_analysis.html", scan=registro, detalle=detalle)


@app.route("/statistics")
def statistics():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM scans").fetchone()["c"]
    amenazas = db.execute(
        "SELECT COUNT(*) c FROM scans WHERE clasificacion='phishing'"
    ).fetchone()["c"]
    seguros = db.execute(
        "SELECT COUNT(*) c FROM scans WHERE clasificacion='seguro'"
    ).fetchone()["c"]
    sospechosos = db.execute(
        "SELECT COUNT(*) c FROM scans WHERE clasificacion='sospechoso'"
    ).fetchone()["c"]
    por_dia = db.execute(
        """SELECT substr(creado_en, 1, 10) AS dia,
                  COUNT(*) AS total,
                  SUM(CASE WHEN clasificacion='phishing' THEN 1 ELSE 0 END) AS amenazas
           FROM scans GROUP BY dia ORDER BY dia DESC LIMIT 7"""
    ).fetchall()
    por_dia = list(reversed(por_dia))
    max_total = max([row["total"] for row in por_dia], default=0) or 1
    return render_template(
        "statistics.html",
        total=total,
        amenazas=amenazas,
        seguros=seguros,
        sospechosos=sospechosos,
        por_dia=por_dia,
        max_total=max_total,
    )


@app.route("/education")
def education():
    return render_template("education.html")


@app.route("/settings", methods=["GET", "POST"])
def settings():
    cfg = cargar_config()
    if request.method == "POST":
        cfg["sensibilidad"] = request.form.get("sensibilidad", "balanced")
        cfg["alertas_tiempo_real"] = "alertas_tiempo_real" in request.form
        cfg["resumen_semanal"] = "resumen_semanal" in request.form
        cfg["prompts_educativos"] = "prompts_educativos" in request.form
        cfg["analytics_anonimos"] = "analytics_anonimos" in request.form
        cfg["retencion_dias"] = request.form.get("retencion_dias", "30")
        guardar_config(cfg)
        flash("Cambios guardados.")
        return redirect(url_for("settings"))
    return render_template("settings.html", cfg=cfg, sensibilidades=SENSIBILIDADES)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
