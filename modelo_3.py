import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# Cargar dataset (ruta relativa: coloca este script en la misma carpeta que el CSV)
print("Cargando dataset (phishing_dataset.csv)...")
t0 = time.time()
data = pd.read_csv("phishing_dataset.csv")
print(f"  Listo en {time.time() - t0:.1f}s -> {data.shape[0]} filas")

# Limpiar nombres de columnas (importante por si hay espacios)
data.columns = data.columns.str.strip()

# Separar variables
X = data.drop("phishing", axis=1)
y = data["phishing"]

# Si tienes -1 y 1, conviertelo a 0 y 1
y = y.replace(-1, 0)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Relacion entre clases negativa/positiva, para probar scale_pos_weight
ratio_desbalance = (y_train == 0).sum() / (y_train == 1).sum()

# ==============================
# Validacion cruzada (5 folds) + busqueda de hiperparametros
# ==============================
# GridSearchCV (busqueda EXHAUSTIVA): la grilla tiene 3x3x3x2 = 54
# combinaciones, manejable para probarlas todas en vez de arriesgarnos a
# que un muestreo aleatorio se salte la mejor combinacion (eso fue lo que
# paso con el modelo de URL con RandomForest: RandomizedSearchCV con
# n_iter=10 dio un resultado peor que el modelo original sin tuning).
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "scale_pos_weight": [1, ratio_desbalance],
}
n_combinaciones = 3 * 3 * 3 * 2  # 54

busqueda = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
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
print("===== Validacion cruzada (5 folds) - mejor combinacion XGBoost =====")
for metrica in scoring:
    media = busqueda.cv_results_[f"mean_test_{metrica}"][mejor_idx]
    std = busqueda.cv_results_[f"std_test_{metrica}"][mejor_idx]
    print(f"{metrica:10s}: {media:.4f} (+/- {std:.4f})")

print("\nMejores hiperparametros encontrados:", busqueda.best_params_)

xgb = busqueda.best_estimator_

# Predecir
y_pred = xgb.predict(X_test)

# Evaluar
print("\n===== Evaluacion en conjunto de prueba =====")
print("Matriz de confusion:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificacion:")
print(classification_report(y_test, y_pred))

# ==============================
# Guardar modelo + metadatos (listo para que el front-end lo consuma)
# ==============================
# IMPORTANTE: se guarda list(X.columns), es decir, el orden EXACTO de columnas tal
# como quedo en el CSV despues de X = data.drop("phishing", axis=1). Esto es a
# proposito: en vez de mantener una lista de 111 nombres a mano (facil de
# desincronizar), el propio entrenamiento le dice a la inferencia en que orden
# exacto espera las features. url_features_full.py calcula un dict {nombre: valor}
# y prototipo_url_final.py arma el vector final buscando cada nombre en ese dict,
# asi que el orden real solo importa aqui, una sola vez.
joblib.dump({
    "model": xgb,
    "feature_names": list(X.columns),
    "umbral_sospechoso": 0.4,
    "umbral_phishing": 0.7,
}, "modelo_pro_xgb.pkl")

print("\nModelo guardado en modelo_pro_xgb.pkl (incluye nombres de features y umbrales)")
print("Nota: se guardo con un nombre distinto a modelo_pro.pkl (el de modelo_2.py / RF) "
      "para que ambos artefactos puedan coexistir sin pisarse.")
