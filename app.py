from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Inicializar app
app = Flask(__name__)
CORS(app)

# Cargar modelo de predicción
modelo = joblib.load("modelo_sin_costo_extra.pkl")

# Servicios disponibles (extraídos del modelo original)
servicios_estaticos = [
    "Limpieza del cuerpo de aceleración",
    "Reparación de sistema de frenos (discos",
    "ABS",
    "Ajuste de Válvulas",
    "Ajuste de freno de mano",
    "Alineación de Dirección",
    "Baja",
    "Balanceo de llantas",
    "Cambio de Aceite de Motor",
    "Cambio de Banda o Cadena de Distribución",
    "Cambio de Bomba de Combustible",
    "Cambio de Empaques y Retenes",
    "Cambio de Filtro de Aceite",
    "Cambio de Filtro de Aire",
    "Cambio de Filtro de Cabina",
    "Cambio de Filtro de Gasolina",
    "Cambio de Junta de Culata",
    "Cambio de Juntas Homocinéticas",
    "Cambio de Sensores (Oxígeno",
    "Cambio de amortiguadores",
    "Cambio de bomba de agua",
    "Cambio de clutch",
    "Cambio de empaques y retenes",
    "Cambio de muelles o resortes",
    "Cambio de radiador",
    "Cambio de rótulas",
    "Cambio de termostato",
    "Diagnóstico con Escáner Automotriz",
    "Engrase de componentes",
    "Instalación de Luces LED o Xenón",
    "Intermitentes)",
    "Limpieza de Inyectores",
    "Reparación de Alternador",
    "Reparación de Dirección Eléctrica",
    "Reparación de Elevadores Eléctricos",
    "Reparación de Luces (Alta",
    "Reparación de Motor",
    "Reparación de Motor de Arranque",
    "Reparación de Módulos Electrónicos",
    "Reparación de Sistema de Cierre Centralizado",
    "Reparación de Tablero de Instrumentos",
    "Reparación de Transmisión Automática",
    "Reparación de Turbo",
    "Reparación de cabeza de motor",
    "Reparación de diferencial",
    "Reparación de dirección eléctrica",
    "Reparación de dirección hidráulica",
    "Reparación de suspensión delantera y trasera",
    "Reparación de transmisión automática",
    "Reparación de transmisión manual",
    "Reparación del Sistema de Encendido",
    "Reparación del sistema de escape",
    "Revisión de correas y mangueras",
    "Revisión de correas y mangueras ",
    "Revisión de sistema de enfriamiento",
    "Revisión de suspensión",
    "Revisión del sistema de carga de batería",
    "Revisión del sistema de escape",
    "Revisión y Ajuste de Frenos",
    "Rotación de Llantas",
    "Servicio: Revisión y Rellenado de Líquidos (frenos",
    "Stop",
    "dirección",
    "etc.)",
    "pastillas)",
    "refrigerante",
    "tambores"
]

@app.route('/')
def index():
    return jsonify({
        "mensaje": "Servicios disponibles para predicción",
        "servicios": servicios_estaticos
    })

@app.route('/servicios', methods=['GET'])
def get_servicios():
    try:
        return jsonify({
            "mensaje": "Servicios disponibles para predicción",
            "servicios": servicios_estaticos
        })
    except Exception as e:
        print(f"❌ Error al obtener servicios: {e}")
        return jsonify({"error": "No se pudieron obtener los servicios."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        marca = data['marca']
        modelo_vehiculo = data['modelo']
        servicios = data['servicios']

        marca_modelo = f"{marca}_{modelo_vehiculo}"
        num_servicios = len(servicios)

        # Convertir los servicios a columnas binarias manualmente
        servicios_binarios = {nombre: 1 if nombre in servicios else 0 for nombre in servicios_estaticos}
        servicios_df = pd.DataFrame([servicios_binarios])

        input_df = pd.DataFrame([{
            "marca": marca,
            "modelo": modelo_vehiculo,
            "marca_modelo": marca_modelo,
            "num_servicios": num_servicios
        }])

        input_completo = pd.concat([input_df, servicios_df], axis=1)

        total_estimado = modelo.predict(input_completo)[0]

        return jsonify({"total_estimado": round(float(total_estimado), 2)})

    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return jsonify({"error": "No se pudo procesar la predicción."}), 500

if __name__ == '__main__':
    app.run(debug=True)
