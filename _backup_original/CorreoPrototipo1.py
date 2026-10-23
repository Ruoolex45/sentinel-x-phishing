# ==============================
# DETECTOR DE PHISHING - CONSOLA
# ==============================

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==============================
# 1. CARGAR Y PREPARAR DATOS
# ==============================
df = pd.read_csv("phishing_email.csv", encoding='latin-1')

nuevos = pd.DataFrame({
    'text_combined': [
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
    ],
    'label': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1,
    ]
})

df = pd.concat([df, nuevos], ignore_index=True)

# ==============================
# 2. CARGAR EMBEDDER SIEMPRE
# ==============================
print("\nCargando modelo de lenguaje...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ==============================
# 3. VERIFICAR SI YA EXISTE MODELO GUARDADO
# ==============================
if os.path.exists("model_guardado.pkl") and os.path.exists("embeddings_guardados.pkl"):
    print("Modelo encontrado, cargando...")
    with open("model_guardado.pkl", "rb") as f:
        model = pickle.load(f)
    print("Listo! Modelo cargado en segundos ")

else:
    print("Primera ejecucion, entrenando modelo...")
    print("Vectorizando correos... puede tardar varios minutos...")

    X = embedder.encode(
        df['text_combined'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nEntrenando Random Forest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    print("\n===== RESULTADOS DEL MODELO =====")
    print(classification_report(y_test, model.predict(X_test)))
    print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)) * 100, 2), "%")

    # Guardar modelo
    with open("model_guardado.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("embeddings_guardados.pkl", "wb") as f:
        pickle.dump(X, f)

    print("\nModelo guardado! La proxima vez cargara instantaneamente ")

# ==============================
# 4. FUNCIÓN DE PREDICCIÓN
# ==============================
def predecir_correo(texto):
    X_input = embedder.encode([texto])
    probabilidad = model.predict_proba(X_input)[0]

    p_phishing = probabilidad[1] * 100
    p_legitimo = probabilidad[0] * 100

    if p_phishing >= 70:
        label = "PHISHING ⚠"
    elif p_phishing >= 45:
        label = "SOSPECHOSO "
    else:
        label = "LEGITIMO "

    print(f"\n{'='*50}")
    print(f"  Resultado  : {label}")
    print(f"  P(legitimo): {p_legitimo:.1f}%")
    print(f"  P(phishing): {p_phishing:.1f}%")
    print(f"{'='*50}")

# ==============================
# 5. LOOP PRINCIPAL
# ==============================
print("\n" + "="*50)
print("   DETECTOR DE PHISHING")
print("   Sentence-Transformers + Random Forest")
print("="*50)

while True:
    print("\nPega el texto del correo (escribe FIN para terminar):")
    lineas = []
    while True:
        linea = input()
        if linea.strip().upper() == "FIN":
            break
        lineas.append(linea)

    texto = "\n".join(lineas)
    if texto.strip():
        predecir_correo(texto)

    print("\n¿Que deseas hacer?")
    print("  [1] Analizar otro correo")
    print("  [q] Salir")

    opcion = input("\nOpcion: ").strip().lower()
    if opcion == "q":
        print("\nSaliendo... hasta luego!")
        break