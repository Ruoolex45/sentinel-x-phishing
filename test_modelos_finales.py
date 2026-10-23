# -*- coding: utf-8 -*-
"""
Prueba rapida de los dos modelos finales (URL y correo) con casos conocidos,
para confirmar que responden bien antes de conectarlos al front-end.

Ejecutar con el venv del proyecto activo (necesita internet para el modelo
de correo la primera vez que descarga el embedder, y para las consultas
DNS/WHOIS/HTTP del modelo de URL).
"""
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("PRUEBA 1: MODELO DE CORREO (Random Forest + embeddings)")
print("=" * 60)

from CorreoPrototipo1 import entrenar_o_cargar, predecir_correo

modelo_correo, embedder = entrenar_o_cargar()

casos_correo = [
    ("Legitimo ES (Nequi)", "Realizaste una transferencia por Nequi de 50000 pesos a Juan Perez. Tu saldo actual es de 120000 pesos.", 0),
    ("Legitimo ES (Bancolombia)", "Bancolombia te informa que tu factura de la tarjeta de credito con corte al 20 de agosto ya esta disponible en la app.", 0),
    ("Legitimo ES (Rappi)", "Tu pedido de Rappi numero 4521 fue entregado exitosamente. Gracias por tu compra.", 0),
    ("Legitimo ES (EPM)", "EPM te informa que tu factura de energia de julio por valor de 165000 pesos ya esta disponible para pago.", 0),
    ("Phishing ES (Nequi falso)", "URGENTE: Tu cuenta Nequi sera bloqueada en 24 horas por actividad sospechosa. Verifica ya en http://nequi-verifica.cc", 1),
    ("Phishing ES (DIAN falso)", "AVISO DIAN: Tienes una deuda pendiente de 3200000 pesos. Evita el embargo pagando aqui: http://dian-pagos.xyz/pago", 1),
    ("Phishing ES (Bancolombia falso)", "Su cuenta Bancolombia presenta actividad inusual. Confirme sus datos de inmediato en http://bancolombia-verificacion.net o sera bloqueada.", 1),
    ("Legitimo EN", "Your Amazon order has shipped and will arrive tomorrow. Track your package here: amazon.com/orders", 0),
    ("Phishing EN", "Your account has been suspended. Click here immediately to verify your identity or lose access permanently: http://paypal-secure-verify.tk", 1),
]

print(f"\n{'Caso':34s} {'Esperado':10s} {'Predicho':12s} {'Prob.phish':10s} OK?")
aciertos_correo = 0
for nombre, texto, esperado in casos_correo:
    r = predecir_correo(texto, modelo_correo, embedder)
    pred = 0 if r["clasificacion"] == "seguro" else 1
    ok = "SI" if pred == esperado else "NO"
    aciertos_correo += (pred == esperado)
    print(f"{nombre:34s} {esperado:<10d} {r['clasificacion']:12s} {r['probabilidad']:<10.3f} {ok}")
print(f"\nAciertos correo: {aciertos_correo}/{len(casos_correo)}")


print("\n" + "=" * 60)
print("PRUEBA 2: MODELO DE URL (XGBoost, 111 caracteristicas)")
print("=" * 60)

from prototipo_url_final import cargar_modelo, clasificar_url

modelo_url, feature_names = cargar_modelo()

casos_url = [
    ("Legitimo (Google)", "https://www.google.com", 0),
    ("Legitimo (Wikipedia)", "https://es.wikipedia.org/wiki/Phishing", 0),
    ("Legitimo (Mercado Libre)", "https://www.mercadolibre.com.co", 0),
    ("Phishing (dominio sospechoso)", "http://bancolombia-verificacion-urgente.xyz/login", 1),
    ("Phishing (IP en URL)", "http://192.168.1.1/nequi/login", 1),
]

print(f"\n{'Caso':40s} {'Esperado':10s} {'Predicho':14s} {'Prob.phish':10s} OK?")
aciertos_url = 0
for nombre, url, esperado in casos_url:
    try:
        r = clasificar_url(url, modelo_url, feature_names)
        pred = 0 if r["clasificacion"] == "seguro" else 1
        ok = "SI" if pred == esperado else "NO"
        aciertos_url += (pred == esperado)
        prob = r["probabilidad"] if r["probabilidad"] is not None else 0.0
        print(f"{nombre:40s} {esperado:<10d} {r['clasificacion']:14s} {prob:<10.3f} {ok}")
    except Exception as e:
        print(f"{nombre:40s} ERROR: {e}")
print(f"\nAciertos URL: {aciertos_url}/{len(casos_url)}")

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"Correo: {aciertos_correo}/{len(casos_correo)}   URL: {aciertos_url}/{len(casos_url)}")
