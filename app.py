from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Inicializar app
app = Flask(__name__)
CORS(app)  # Habilita CORS para que pueda ser llamado desde React

# Cargar modelo y codificador
modelo = joblib.load("modelo_sin_costo_extra.pkl")
mlb = joblib.load("servicios_encoder.pkl")

@app.route('/')
def index():
    return jsonify({"message": "API de predicción activa."})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        marca = data['marca']
        modelo_vehiculo = data['modelo']
        servicios = data['servicios']

        # Preprocesar entrada
        marca_modelo = f"{marca}_{modelo_vehiculo}"
        num_servicios = len(servicios)

        # Codificar servicios como OneHot
        servicios_array = mlb.transform([servicios])
        servicios_df = pd.DataFrame(servicios_array, columns=mlb.classes_)

        # Construir input del modelo
        input_df = pd.DataFrame([{
            "marca": marca,
            "modelo": modelo_vehiculo,
            "marca_modelo": marca_modelo,
            "num_servicios": num_servicios
        }])

        # Combinar con codificación de servicios
        input_completo = pd.concat([input_df, servicios_df], axis=1)

        # Predecir
        total_estimado = modelo.predict(input_completo)[0]

        return jsonify({"total_estimado": round(float(total_estimado), 2)})

    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return jsonify({"error": "No se pudo procesar la predicción."}), 500

if __name__ == '__main__':
    app.run(debug=True)
