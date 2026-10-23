import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from url_features import FEATURE_NAMES

print("Cargando dataset (phishing_dataset.csv)...")
t0 = time.time()
data = pd.read_csv("phishing_dataset.csv")
print(f"  Listo en {time.time() - t0:.1f}s -> {data.shape[0]} filas")

X = data[FEATURE_NAMES]
y = data["phishing"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Validacion cruzada (5 folds) + busqueda de hiperparametros
# ==============================
# Se usa GridSearchCV (busqueda EXHAUSTIVA) en vez de RandomizedSearchCV:
# la grilla de abajo solo tiene 3x4x3x2 = 72 combinaciones, un numero
# perfectamente manejable para probarlas TODAS. Con RandomizedSearchCV y
# n_iter=10 corriamos el riesgo real de nunca probar la combinacion con
# max_depth=None (arboles sin limite de profundidad) que es justo la que
# ya funcionaba bien en la version original -- y esa fue la causa de que
# un primer intento con busqueda aleatoria diera un resultado *peor* que
# el modelo original. GridSearchCV evita ese riesgo por completo.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

param_grid = {
    "n_estimators": [150, 250, 350],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", None],
}
n_combinaciones = 3 * 4 * 3 * 2  # 72

busqueda = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring=scoring,
    refit="f1",
    n_jobs=-1,
    verbose=2,  # imprime cada entrenamiento a medida que termina
)
print(f"\nBuscando hiperparametros ({n_combinaciones} combinaciones x 5 folds = "
      f"{n_combinaciones * 5} entrenamientos)...")
print("Iras viendo una linea por cada entrenamiento que termine.\n")
t0 = time.time()
busqueda.fit(X_train, y_train)
print(f"\nBusqueda completa en {time.time() - t0:.1f}s")

mejor_idx = busqueda.best_index_
print("===== Validacion cruzada (5 folds) - mejor combinacion Random Forest URL =====")
for metrica in scoring:
    media = busqueda.cv_results_[f"mean_test_{metrica}"][mejor_idx]
    std = busqueda.cv_results_[f"std_test_{metrica}"][mejor_idx]
    print(f"{metrica:10s}: {media:.4f} (+/- {std:.4f})")

print("\nMejores hiperparametros encontrados:", busqueda.best_params_)

modelo_final = busqueda.best_estimator_

# ==============================
# Evaluacion final en el conjunto de prueba (holdout)
# ==============================
y_pred = modelo_final.predict(X_test)

print("\n===== Evaluacion en conjunto de prueba =====")
print("Matriz de confusion:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificacion:")
print(classification_report(y_test, y_pred))

# ==============================
# Guardar modelo + metadatos (listo para que el front-end lo consuma)
# ==============================
joblib.dump({
    "model": modelo_final,
    "feature_names": FEATURE_NAMES,
    "umbral_sospechoso": 0.4,
    "umbral_phishing": 0.7,
}, "modelo_pro.pkl")

print("\nModelo guardado en modelo_pro.pkl (incluye nombres de features y umbrales)")
