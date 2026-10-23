"""
Compara los tres modelos de correo (RF+TF-IDF, XGBoost+TF-IDF, RF+embeddings
multilingues) sobre los 18 ejemplos en espanol/contexto colombiano que ya
tenias escritos a mano en CorreoPrototipo1.py (el caso "Nequi"), para ver cual
generaliza de verdad a ese caso de uso -- en vez de solo mirar el accuracy
general de phishing_email.csv, que es mayoritariamente en ingles y no refleja
ese problema.

IMPORTANTE - fuga de datos en el modelo de embeddings original:
model_guardado.pkl ya fue entrenado INCLUYENDO estos 18 ejemplos (son los
mismos que _cargar_datos() le agrega al dataset en CorreoPrototipo1.py). Si le
va bien aqui, una parte de eso puede ser que los memorizo, no que generalice
mejor en espanol. Por eso este script tambien entrena una version "limpia" del
RF+embeddings -- sin esos 18 en el entrenamiento -- reutilizando los
embeddings ya calculados en embeddings_guardados.pkl (rapido: no hay que
volver a vectorizar los ~87000 correos con el sentence-transformer). La
version original tambien se muestra, pero marcada como no comparable.

No se modifica ni sobreescribe ningun .pkl existente. No hace falta que
sentence-transformers este instalado para correr este script: se reusan
embeddings ya calculados en disco.
"""

import pickle
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

RUTA_MODELO_EMBED = "model_guardado.pkl"
RUTA_EMBEDDINGS = "embeddings_guardados.pkl"

# Los mismos 18 ejemplos hechos a mano en CorreoPrototipo1.py::_cargar_datos().
# Si alguna vez editas esa lista alla, actualiza tambien esta copia.
TEXTOS_ES = [
    "Realizaste un envio por Bre-B y todo salio bien. Hola JUAN PABLO CAMACHO CASAS. Enviaste de manera exitosa 28.200 a la llave 0089807047 de ARCA ROCK el 5 de mayo de 2026.",
    "Tu transferencia Nequi fue exitosa. Enviaste plata de manera rapida y facil 24/7.",
    "Hola, tu pago PSE fue procesado correctamente. Tu compra en exito.com por valor de 150000 pesos fue aprobada.",
    "Bancolombia te informa que tu transferencia de 500000 pesos fue realizada exitosamente.",
    "Tu pedido de Rappi ha sido confirmado. El domiciliario llegara en 30 minutos.",
    "Hola, tu recarga de Claro por 30000 pesos fue exitosa. Tu saldo actual es de 35000 pesos.",
    "Davivienda te notifica que tu tarjeta de credito fue usada por 75000 pesos en Exito Unicentro.",
    "Tu cita medica en Compensar fue confirmada para el 10 de mayo a las 9am.",
    "EPM te informa que tu factura de energia de abril por valor de 180000 pesos esta disponible.",
    "Hola Juan Pablo, tu solicitud de soporte fue recibida. Ticket #45821.",
    "Hi Ruoolex45, What if you could accurately map the subsurface without ever seeing it? In partnership with ROGII Inc.",
    "URGENTE: Tu cuenta Bancolombia ha sido suspendida. Verifica tus datos: http://bancolombia-seguro.xyz/verificar",
    "Felicitaciones! Ganaste 5000000 pesos en el sorteo Exito. Envia tu cedula a premios@exito-sorteo.net",
    "Tu cuenta Nequi sera bloqueada en 24 horas. Haz clic aqui: http://nequi-verificacion.cc/login",
    "AVISO DIAN: Tienes una deuda de 2500000 pesos. Paga ahora: http://dian-pagos.xyz",
    "Tu paquete no pudo ser entregado. Paga 5000 pesos aqui: http://coordinadora-envios.net/pago",
    "URGENTE: Alguien intento acceder a tu cuenta Davivienda. Confirma tu identidad: http://davivienda-seguro.xyz",
    "Ganaste un bono de 500000 pesos en Rappi. Ingresa tus datos: http://rappi-bonos.net/reclamar",
]
Y_ES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
N_EJEMPLOS = len(TEXTOS_ES)

print(f"Probando con los {N_EJEMPLOS} ejemplos en espanol/contexto colombiano.\n")

# ---------------------------------------------------------------
# 1) RF + TF-IDF  y  2) XGBoost + TF-IDF, entrenados SOLO con
#    phishing_email.csv -- nunca vieron estos 18 ejemplos.
# ---------------------------------------------------------------
print("[1/3] Cargando phishing_email.csv y entrenando RF + TF-IDF y XGBoost + TF-IDF...")
t0 = time.time()
df_ingles = pd.read_csv("phishing_email.csv", encoding="latin-1")
vectorizer = TfidfVectorizer(max_features=5000)
X_ingles = vectorizer.fit_transform(df_ingles["text_combined"])
y_ingles = df_ingles["label"]
print(f"      Dataset + TF-IDF listo en {time.time() - t0:.1f}s ({X_ingles.shape[0]} correos)")

t0 = time.time()
rf_tfidf = RandomForestClassifier(random_state=42)
rf_tfidf.fit(X_ingles, y_ingles)
print(f"      RF + TF-IDF entrenado en {time.time() - t0:.1f}s")

t0 = time.time()
# Mejores hiperparametros ya encontrados por GridSearchCV en correoModel2.py
xgb_tfidf = XGBClassifier(
    eval_metric="logloss", random_state=42,
    n_estimators=300, max_depth=8, learning_rate=0.2,
)
xgb_tfidf.fit(X_ingles, y_ingles)
print(f"      XGBoost + TF-IDF entrenado en {time.time() - t0:.1f}s")

X_es_tfidf = vectorizer.transform(TEXTOS_ES)
pred_rf_tfidf = rf_tfidf.predict(X_es_tfidf)
pred_xgb_tfidf = xgb_tfidf.predict(X_es_tfidf)

# ---------------------------------------------------------------
# 3) RF + embeddings multilingues, version LIMPIA (sin fuga): se
#    reentrena sin los 18 ejemplos en espanol, reusando los embeddings
#    ya calculados (no hay que volver a correr el sentence-transformer).
# ---------------------------------------------------------------
print("\n[2/3] Cargando embeddings ya calculados y reentrenando RF sin los 18 ejemplos en espanol...")
t0 = time.time()
with open(RUTA_EMBEDDINGS, "rb") as f:
    X_embeddings = pickle.load(f)
print(f"      Embeddings cargados en {time.time() - t0:.1f}s (shape {X_embeddings.shape})")

# embeddings_guardados.pkl se genero sobre _cargar_datos() = phishing_email.csv
# + estos mismos 18 ejemplos AL FINAL, en ese orden -- por eso los ultimos
# N_EJEMPLOS renglones son justamente los que queremos excluir del entrenamiento.
if X_embeddings.shape[0] < N_EJEMPLOS + 100:
    raise SystemExit(
        f"embeddings_guardados.pkl tiene {X_embeddings.shape[0]} filas, muy pocas para "
        "asumir que los ultimos 18 corresponden a los ejemplos en espanol. Corre "
        "CorreoPrototipo1.py de nuevo para regenerar el archivo antes de usar este script."
    )

y_completo = pd.concat([df_ingles["label"], pd.Series(Y_ES)], ignore_index=True).values
X_limpio, y_limpio = X_embeddings[:-N_EJEMPLOS], y_completo[:-N_EJEMPLOS]
X_es_embed = X_embeddings[-N_EJEMPLOS:]

t0 = time.time()
rf_embed_limpio = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_embed_limpio.fit(X_limpio, y_limpio)
print(f"      RF + embeddings (limpio, sin los 18) entrenado en {time.time() - t0:.1f}s")
pred_embed_limpio = rf_embed_limpio.predict(X_es_embed)

print("\n[3/3] Cargando el modelo de produccion original (este SI vio los 18 en su entrenamiento)...")
with open(RUTA_MODELO_EMBED, "rb") as f:
    rf_embed_original = pickle.load(f)
pred_embed_original = rf_embed_original.predict(X_es_embed)

# ---------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------
etiquetas = {0: "legitimo", 1: "phishing"}
columnas = ["RF+TFIDF", "XGB+TFIDF", "RF+Embed(limpio)", "RF+Embed(original)*"]
predicciones = [pred_rf_tfidf, pred_xgb_tfidf, pred_embed_limpio, pred_embed_original]

print("\n" + "=" * 115)
print(f"{'#':<3}{'Real':<10}{'RF+TFIDF':<11}{'XGB+TFIDF':<12}{'RF+Embed(limpio)':<19}{'RF+Embed(orig)*':<17}Texto")
print("=" * 115)

aciertos = [0, 0, 0, 0]
for i in range(N_EJEMPLOS):
    real = Y_ES[i]
    fila = [pred[i] for pred in predicciones]
    for j, pred in enumerate(fila):
        if pred == real:
            aciertos[j] += 1
    texto_corto = (TEXTOS_ES[i][:40] + "...") if len(TEXTOS_ES[i]) > 40 else TEXTOS_ES[i]
    print(f"{i + 1:<3}{etiquetas[real]:<10}{etiquetas[fila[0]]:<11}{etiquetas[fila[1]]:<12}"
          f"{etiquetas[fila[2]]:<19}{etiquetas[fila[3]]:<17}{texto_corto}")

print("=" * 115)
print("ACIERTOS SOBRE LOS 18 EJEMPLOS EN ESPANOL:")
for nombre, correctos in zip(columnas, aciertos):
    print(f"  {nombre:22s}: {correctos}/{N_EJEMPLOS}")
print("=" * 115)
print("\n* RF+Embed(original) SI vio estos 18 ejemplos durante su entrenamiento (fuga de datos),")
print("  asi que no es comparable de igual a igual -- se muestra solo como referencia.")
print("  La comparacion justa es RF+TFIDF vs XGB+TFIDF vs RF+Embed(limpio): los tres se probaron")
print("  en ejemplos que nunca vieron durante el entrenamiento.")
