# ==============================
# 1. IMPORTAR LIBRERÍAS
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
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# ==============================
# 6. MODELO XGBOOST
# ==============================
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ==============================
# 7. EVALUACIÓN
# ==============================
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)

print("\n===== RESULTADOS XGBOOST =====")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))