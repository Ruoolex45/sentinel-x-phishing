import pandas as pd

# 1. Cargar dataset
df = pd.read_csv("phishing_email.csv", encoding='latin-1')

X = df['text_combined']
y = df['label']

# 3. TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# 4. Train/Test
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Validacion cruzada (5 folds)
# ==============================
from sklearn.ensemble import RandomForestClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelo_base = RandomForestClassifier(random_state=42)

scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_resultados = cross_validate(modelo_base, X_train, y_train, cv=cv, scoring=scoring)

print("===== Validacion cruzada (5 folds) - Random Forest (correos, TF-IDF) =====")
for metrica in scoring:
    valores = cv_resultados[f"test_{metrica}"]
    print(f"{metrica:10s}: {valores.mean():.4f} (+/- {valores.std():.4f})")

# 5. Modelo final
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluacion
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print("\n===== Evaluacion en conjunto de prueba =====")
print(classification_report(y_test, y_pred))
