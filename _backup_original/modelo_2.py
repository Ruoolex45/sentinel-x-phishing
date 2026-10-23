import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("phishing_dataset.csv")

X = data[[
    "length_url",
    "qty_dot_url",
    "qty_hyphen_url",
    "qty_underline_url",
    "qty_slash_url",
    "qty_questionmark_url",
    "qty_equal_url",
    "qty_at_url",
    "qty_and_url",
    "qty_exclamation_url",
    "qty_percent_url",
    "qty_tld_url",
    "domain_length",
    "domain_in_ip",
    "qty_dot_domain",

]]

y = data["phishing"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)

joblib.dump(rf, "modelo_pro.pkl")

print("Carga exitosa")