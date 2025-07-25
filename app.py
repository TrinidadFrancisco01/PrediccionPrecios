from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Inicializar app
app = Flask(__name__)
CORS(app)

# Cargar modelo y codificador
modelo = joblib.load("modelo_sin_costo_extra.pkl")
mlb = joblib.load("servicios_encoder.pkl")

# Servicios a excluir de la respuesta
SERVICIOS_EXCLUIDOS = {"Cambio de Amortiguadores"}

@app.route('/')
def index():
    try:
        servicios_disponibles = [
            s for s in mlb.classes_ if s not in SERVICIOS_EXCLUIDOS
        ]
        return jsonify({
            "mensaje": "Servicios disponibles para predicción",
            "servicios": servicios_disponibles
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

        # Validar si se intenta enviar un servicio excluido
        if any(servicio in SERVICIOS_EXCLUIDOS for servicio in servicios):
            return jsonify({
                "error": "Uno o más servicios no están disponibles para cotización."
            }), 400

        marca_modelo = f"{marca}_{modelo_vehiculo}"
        num_servicios = len(servicios)

        servicios_array = mlb.transform([servicios])
        servicios_df = pd.DataFrame(servicios_array, columns=mlb.classes_)

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

@app.route('/servicios', methods=['GET'])
def get_servicios():
    try:
        servicios_disponibles = [
            s for s in mlb.classes_ if s not in SERVICIOS_EXCLUIDOS
        ]
        return jsonify({"servicios": servicios_disponibles})
    except Exception as e:
        print(f"❌ Error al obtener servicios: {e}")
        return jsonify({"error": "No se pudieron obtener los servicios."}), 500

if __name__ == '__main__':
    app.run(debug=True)
