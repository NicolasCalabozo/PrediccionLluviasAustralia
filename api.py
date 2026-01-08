# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Optional

from imputación import AgregadorEspacial, ImputadorNumerico, ImputadorCategorico, CodificadorCiclico, procesar_fechas

app = FastAPI()

#CARGA DE TODOS LOS MODELOS
try:
    rf_model = joblib.load("modelo_random_forest.pkl")
    hgb_model = joblib.load("modelo_hist_gradient_boosting_classifier.pkl")
    nn_model = joblib.load("modelo_red_neuronal.pkl")
    print("✅ Todos los modelos cargados correctamente.")
except Exception as e:
    print(f"Error cargando modelos: {e}")

class InputClima(BaseModel):
    
    Date: str
    Location: str
    RainToday: Optional[str] = None
    MinTemp: Optional[str] = None
    MaxTemp: Optional[str] = None
    Rainfall: Optional[str] = None
    Evaporation: Optional[str] = None
    Sunshine: Optional[str] = None
    WindGustDir: Optional[str] = None
    WindGustSpeed: Optional[str] = None
    WindDir9am: Optional[str] = None
    WindDir3pm: Optional[str] = None
    WindSpeed9am: Optional[str] = None
    WindSpeed3pm: Optional[str] = None
    Humidity9am: Optional[str] = None
    Humidity3pm: Optional[str] = None
    Pressure9am: Optional[str] = None
    Pressure3pm: Optional[str] = None
    Cloud9am: Optional[str] = None
    Cloud3pm: Optional[str] = None
    Temp9am: Optional[str] = None
    Temp3pm: Optional[str] = None

@app.post("/predecir_todo") # Cambiamos el nombre para reflejar que hace todo
def predecir_todo(datos: InputClima):
    input_dict = datos.dict()
    df_input = pd.DataFrame([input_dict])

    #LIMPIEZA
    df_input = df_input.replace(r'^\s*$', np.nan, regex=True)
    df_input = df_input.apply(pd.to_numeric, errors='ignore')

    try:
        # Hacemos las 3 predicciones
        prob_rf = rf_model.predict_proba(df_input)[0][1]
        prob_hgb = hgb_model.predict_proba(df_input)[0][1]
        prob_nn = nn_model.predict_proba(df_input)[0][1]

        # Calculamos un "Consenso" (Promedio simple)
        promedio = (prob_rf + prob_hgb + prob_nn) / 3

        return {
            "status": "ok",
            "consenso": {
                "probabilidad": float(promedio),
                "prediccion": 1 if promedio > 0.5 else 0
            },
            "detalle": {
                "Random Forest": float(prob_rf),
                "Gradient Boosting": float(prob_hgb),
                "Red Neuronal": float(prob_nn)
            }
        }
    except Exception as e:
        return {"error": str(e)}