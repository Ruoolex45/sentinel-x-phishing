import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Primer paso: Cargar dataset
data = pd.read_csv("phishing_dataset.csv")

# Segundo paso: Separar variables
X = data.drop("phishing", axis=1)
y = data["phishing"]

# Tercer paso: Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cuarto paso: Crear modelo
model = LogisticRegression(max_iter=2000)

# Quinto paso: Entrenar
model.fit(X_train, y_train)

# Sexto paso: Predecir
y_pred = model.predict(X_test)

# Septimo paso: Evaluar
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))