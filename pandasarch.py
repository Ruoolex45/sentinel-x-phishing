import pandas as pd

df = pd.read_csv("phishing_email.csv", encoding='latin-1')
print(df.head())
print(df.columns)