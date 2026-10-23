import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Primer paso: Cargar dataset
data = pd.read_csv("phishing_dataset.csv")

# Segundo paso: Separar variables
X = data.drop("phishing", axis=1)
y = data["phishing"]

# Tercer paso: Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Validacion cruzada (5 folds) sobre el set de entrenamiento
# ==============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelo_base = LogisticRegression(max_iter=2000)

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_resultados = cross_validate(modelo_base, X_train, y_train, cv=cv, scoring=scoring)

print("===== Validacion cruzada (5 folds) - Regresion Logistica =====")
for metrica in scoring:
    valores = cv_resultados[f"test_{metrica}"]
    print(f"{metrica:10s}: {valores.mean():.4f} (+/- {valores.std():.4f})")

# Cuarto paso: Crear y entrenar el modelo final con todo el set de entrenamiento
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Quinto paso: Predecir
y_pred = model.predict(X_test)

# Sexto paso: Evaluar en el conjunto de prueba (holdout)
print("\n===== Evaluacion en conjunto de prueba =====")
print("Matriz de confusion:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificacion:")
print(classification_report(y_test, y_pred))
