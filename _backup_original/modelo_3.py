import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# Cargar dataset
data = pd.read_csv(r"C:\Users\Juan pablo camacho\Documents\ProyectoPhishing\phishing_dataset.csv")

# Limpiar nombres de columnas (importante por si hay espacios)
data.columns = data.columns.str.strip()

# Separar variables
X = data.drop("phishing", axis=1)
y = data["phishing"]

# Si tienes -1 y 1, conviértelo a 0 y 1
y = y.replace(-1, 0)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear modelo XGBoost
xgb = XGBClassifier(
    eval_metric='logloss',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

# Entrenar
xgb.fit(X_train, y_train)

# Predecir
y_pred = xgb.predict(X_test)

# Evaluar
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
