# ==============================
# DETECTOR DE PHISHING PARA CORREOS
# ==============================
"""
Entrena (o carga, si ya existe) un modelo Random Forest sobre embeddings
multilingues de correos, y expone `predecir_correo(texto)` para que el
backend/frontend pueda reutilizarlo directamente mas adelante, en lugar
de quedar amarrado al loop de consola.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

RUTA_MODELO = "model_guardado.pkl"
RUTA_EMBEDDINGS = "embeddings_guardados.pkl"
NOMBRE_EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"


# ==============================
# 1. CARGAR Y PREPARAR DATOS
# ==============================
def _cargar_datos():
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

    return pd.concat([df, nuevos], ignore_index=True)


# ==============================
# 2. ENTRENAR O CARGAR MODELO
# ==============================
def entrenar_o_cargar(forzar_reentrenamiento=False):
    """
    Si ya existe un modelo guardado lo carga (rapido). Si no existe, o si
    `forzar_reentrenamiento=True`, vuelve a entrenar desde cero: vectoriza
    los correos, corre validacion cruzada (5 folds) para reportar metricas
    honestas antes de fijar el modelo final, y lo guarda para la proxima vez.
    """
    print("\nCargando modelo de lenguaje...")
    embedder = SentenceTransformer(NOMBRE_EMBEDDER)

    if not forzar_reentrenamiento and os.path.exists(RUTA_MODELO):
        print("Modelo encontrado, cargando...")
        with open(RUTA_MODELO, "rb") as f:
            modelo = pickle.load(f)
        print("Listo! Modelo cargado en segundos")
        return modelo, embedder

    print("Primera ejecucion (o reentrenamiento forzado)...")
    print("Vectorizando correos... puede tardar varios minutos...")

    df = _cargar_datos()
    X = embedder.encode(
        df['text_combined'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Validacion cruzada (5 folds) sobre el set de entrenamiento ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    modelo_base = RandomForestClassifier(n_estimators=200, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_resultados = cross_validate(modelo_base, X_train, y_train, cv=cv, scoring=scoring)

    print("\n===== Validacion cruzada (5 folds) - RF + embeddings multilingues =====")
    for metrica in scoring:
        valores = cv_resultados[f"test_{metrica}"]
        print(f"{metrica:10s}: {valores.mean():.4f} (+/- {valores.std():.4f})")

    # ---- Entrenamiento del modelo final ----
    print("\nEntrenando Random Forest final...")
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    print("\n===== Evaluacion en conjunto de prueba =====")
    print(classification_report(y_test, modelo.predict(X_test)))
    print("Accuracy:", round(accuracy_score(y_test, modelo.predict(X_test)) * 100, 2), "%")

    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)
    with open(RUTA_EMBEDDINGS, "wb") as f:
        pickle.dump(X, f)

    print("\nModelo guardado! La proxima vez cargara instantaneamente")
    return modelo, embedder


# ==============================
# 3. PREDICCION (lista para el front-end)
# ==============================
def predecir_correo(texto: str, modelo=None, embedder=None) -> dict:
    """
    Clasifica un correo y devuelve un resultado estructurado, con las mismas
    categorias que la tabla `resultados` del backend (ver Manual Tecnico:
    clasificacion, probabilidad, nivel_riesgo, puntaje_heuristico, explicacion).
    """
    if modelo is None or embedder is None:
        modelo, embedder = entrenar_o_cargar()

    X_input = embedder.encode([texto])
    probabilidad = modelo.predict_proba(X_input)[0]
    p_phishing = float(probabilidad[1])

    if p_phishing >= 0.7:
        clasificacion, nivel_riesgo = "phishing", "alto"
    elif p_phishing >= 0.45:
        clasificacion, nivel_riesgo = "sospechoso", "medio"
    else:
        clasificacion, nivel_riesgo = "seguro", "bajo"

    return {
        "clasificacion": clasificacion,
        "probabilidad": round(p_phishing, 4),
        "nivel_riesgo": nivel_riesgo,
        "puntaje_heuristico": None,
        "explicacion": "Clasificado por Random Forest sobre embeddings "
                       "multilingues (sentence-transformers).",
    }


# ==============================
# 4. CONSOLA (para pruebas manuales, igual que antes)
# ==============================
if __name__ == "__main__":
    modelo, embedder = entrenar_o_cargar()

    print("\n" + "=" * 50)
    print("   DETECTOR DE PHISHING - CORREOS")
    print("   Sentence-Transformers + Random Forest")
    print("=" * 50)

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
            resultado = predecir_correo(texto, modelo, embedder)
            print(f"\n{'=' * 50}")
            print(f"  Clasificacion  : {resultado['clasificacion'].upper()}")
            print(f"  Probabilidad   : {resultado['probabilidad'] * 100:.1f}%")
            print(f"  Nivel de riesgo: {resultado['nivel_riesgo']}")
            print(f"{'=' * 50}")

        print("\nQue deseas hacer?")
        print("  [1] Analizar otro correo")
        print("  [q] Salir")

        opcion = input("\nOpcion: ").strip().lower()
        if opcion == "q":
            print("\nSaliendo... hasta luego!")
            break
