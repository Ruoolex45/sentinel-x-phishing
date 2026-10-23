# ==============================
# 1. IMPORTAR LIBRERIAS
# ==============================
import pandas as pd

# ==============================
# 2. CARGAR DATASET
# ==============================
df = pd.read_csv("phishing_email.csv", encoding='latin-1')

# ==============================
# 3. PREPARAR DATOS
# ==============================
X = df['text_combined']
y = df['label']

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# ==============================
# 5. TRAIN / TEST
# ==============================
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 6. VALIDACION CRUZADA + AJUSTE DE HIPERPARAMETROS (XGBOOST)
# ==============================
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
}
# GridSearchCV (exhaustiva, 27 combinaciones): con RandomizedSearchCV y
# n_iter=8 solo se probaba un tercio de las combinaciones al azar, con
# riesgo real de saltarse la mejor (eso paso con el modelo de URL).
busqueda = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring=scoring,
    refit="f1",
    n_jobs=-1,
)
# Nota: se quito `use_label_encoder=False`; ese parametro fue eliminado en
# versiones recientes de xgboost y ya no es necesario (ni valido).
busqueda.fit(X_train, y_train)

mejor_idx = busqueda.best_index_
print("===== Validacion cruzada (5 folds) - mejor combinacion XGBoost (correos) =====")
for metrica in scoring:
    media = busqueda.cv_results_[f"mean_test_{metrica}"][mejor_idx]
    std = busqueda.cv_results_[f"std_test_{metrica}"][mejor_idx]
    print(f"{metrica:10s}: {media:.4f} (+/- {std:.4f})")

print("\nMejores hiperparametros encontrados:", busqueda.best_params_)

model = busqueda.best_estimator_

# ==============================
# 7. EVALUACION
# ==============================
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)

print("\n===== RESULTADOS XGBOOST =====")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
