# Sentinel-X — Detección de phishing con Machine Learning

Trabajo de grado — Ingeniería de Software, Universidad Manuela Beltrán.
Autor: Juan Pablo Camacho Casas.

Herramienta que clasifica **URLs** y **correos electrónicos** como seguros, sospechosos o phishing, usando dos modelos de machine learning entrenados específicamente para cada caso (no reglas heurísticas inventadas ni datos simulados).

## ¿Qué hace?

- **URLs**: extrae 111 características reales (léxicas, DNS, WHOIS/RDAP, SSL, HTTP) y las clasifica con un modelo **XGBoost** (97.2% de accuracy en el conjunto de prueba).
- **Correos**: convierte el texto en un embedding semántico multilingüe (`sentence-transformers`, modelo `paraphrase-multilingual-MiniLM-L12-v2`) y lo clasifica con un modelo **Random Forest**, pensado específicamente para funcionar bien en español.
- **Front-end web** (Flask): Dashboard para escanear, reporte de riesgo, historial real de escaneos (SQLite), estadísticas de uso y una sensibilidad de escáner ajustable (relajada / balanceada / estricta).

Más contexto técnico y el razonamiento detrás de cada decisión está en `ManualTecnico.docx` y en `Proyecto de investigacion.docx` (el documento de tesis).

## Estructura del proyecto

```
url_features_full.py       Extracción de las 111 características de una URL (DNS, WHOIS/RDAP, SSL, HTTP)
prototipo_url_final.py     Carga el modelo XGBoost y clasifica una URL
modelo_3.py                 Entrena el modelo XGBoost (GridSearchCV, 5-fold CV)

CorreoPrototipo1.py         Entrena/carga el modelo Random Forest + embeddings, y clasifica un correo

webapp/
  app.py                     Backend Flask — carga los dos modelos y sirve el front-end
  templates/                 Páginas (Dashboard, Risk Analysis, Statistics, Education, Settings)
  static/                    CSS y JS

url_features.py             Versión liviana (15 características léxicas, instantánea)
prototipo1.py                Modelo alternativo sobre esas 15 características (93.4% accuracy)

test_modelos_finales.py     Batería de prueba de los dos modelos finales
diagnostico_url.py          Script de diagnóstico de features para una URL puntual
```

## Requisitos

- Python 3.10+
- Un entorno virtual (`.venv`) con las dependencias del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Archivos que **no están** en este repositorio (por tamaño)

GitHub rechaza archivos de más de 100 MB, y de todas formas no tiene sentido versionar binarios entrenados ni datasets completos en git. Estos archivos existen localmente pero están en `.gitignore`:

| Archivo | Qué es | Se regenera con |
|---|---|---|
| `modelo_pro_xgb.pkl` | Modelo XGBoost entrenado (URL) | `python modelo_3.py` |
| `model_guardado.pkl` | Modelo Random Forest entrenado (correo) | `python CorreoPrototipo1.py` |
| `embeddings_guardados.pkl` | Embeddings cacheados del dataset de correos | se genera junto con el anterior |
| `phishing_dataset.csv` | Dataset de URLs (111 features) — GregaVrbancic/Phishing-Dataset | descargar aparte |
| `phishing_email.csv` | Dataset de correos (texto + etiqueta) | descargar aparte |

Si clonas este repo en una máquina nueva, necesitas conseguir esos dos `.csv` por tu cuenta y correr los dos scripts de entrenamiento antes de poder levantar el front-end.

## Cómo levantar el front-end

```powershell
# con el venv activo, parado en la carpeta del proyecto
cd webapp
python app.py
```

Abre `http://127.0.0.1:5000` en el navegador. La primera vez tarda unos segundos en arrancar porque carga los dos modelos en memoria.

## Validar los modelos

```powershell
python test_modelos_finales.py
```

Corre una batería de casos conocidos (URLs y correos legítimos y de phishing) contra los dos modelos finales y reporta aciertos.

## Nota académica

Este es un prototipo desarrollado como trabajo de grado, no un producto en producción. El Manual Técnico (`ManualTecnico.docx`) documenta explícitamente qué partes del diseño inicial se implementaron y cuáles quedaron fuera de alcance (autenticación, base de datos en la nube, despliegue en contenedores, entre otras).
