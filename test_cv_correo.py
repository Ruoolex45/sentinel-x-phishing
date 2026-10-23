"""
Prueba rapida de validacion cruzada para el modelo de correos, SIN
reentrenar ni sobrescribir model_guardado.pkl / embeddings_guardados.pkl.

Reutiliza los embeddings ya calculados en embeddings_guardados.pkl para
evitar volver a vectorizar los ~82,500 correos con sentence-transformers
(eso es lo que tarda varios minutos). Aqui solo se corre la validacion
cruzada sobre esos embeddings ya existentes, en segundos.

Requisito: haber ejecutado CorreoPrototipo1.py al menos una vez antes,
para que exista embeddings_guardados.pkl (ya lo tienes generado).
"""

import os
import pickle
import time
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

from CorreoPrototipo1 import _cargar_datos, RUTA_EMBEDDINGS

if not os.path.exists(RUTA_EMBEDDINGS):
    raise SystemExit(
        f"No se encontro '{RUTA_EMBEDDINGS}'. Corre CorreoPrototipo1.py al menos "
        "una vez para generar los embeddings antes de usar este script."
    )

print("[1/3] Cargando embeddings ya calculados (embeddings_guardados.pkl, ~127 MB)...")
t0 = time.time()
with open(RUTA_EMBEDDINGS, "rb") as f:
    X = pickle.load(f)
print(f"      Listo en {time.time() - t0:.1f}s")

print("[2/3] Cargando dataset de correos (phishing_email.csv, ~106 MB)...")
t0 = time.time()
df = _cargar_datos()
y = df['label'].values
print(f"      Listo en {time.time() - t0:.1f}s -> {len(y)} correos, embeddings shape {X.shape}")

if len(y) != len(X):
    raise SystemExit(
        f"El dataset actual tiene {len(y)} filas pero los embeddings guardados "
        f"tienen {len(X)}. Es probable que el dataset haya cambiado desde que se "
        "generaron los embeddings; borra embeddings_guardados.pkl y corre "
        "CorreoPrototipo1.py de nuevo para regenerarlos."
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# n_jobs=-1: usa todos los nucleos disponibles. Sin esto, cada uno de los 5
# entrenamientos de la validacion cruzada corre en un solo nucleo y tarda
# varias veces mas.
modelo_base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

print("[3/3] Corriendo validacion cruzada (5 folds, 200 arboles por fold)...")
print("      Esto puede tardar uno o varios minutos segun tu CPU.")
t0 = time.time()
cv_resultados = cross_validate(modelo_base, X, y, cv=cv, scoring=scoring, n_jobs=1)
print(f"      Listo en {time.time() - t0:.1f}s")

print("\n===== Validacion cruzada (5 folds) - RF + embeddings multilingues =====")
print("(usando los embeddings ya guardados, sin reentrenar el modelo final)\n")
for metrica in scoring:
    valores = cv_resultados[f"test_{metrica}"]
    print(f"{metrica:10s}: {valores.mean():.4f} (+/- {valores.std():.4f})")
