import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# --- IMPORTANTE: Traer tus clases personalizadas ---
from imputación import (
    AgregadorEspacial, 
    ImputadorNumerico, 
    ImputadorCategorico, 
    CodificadorCiclico, 
    transformer_fechas,
    procesar_fechas
)
from clases_modelos import (
    RedLluviaPipeline, 
    HGBClassifier, 
    RFClassifier
)

# --- Configuración de Página ---
st.set_page_config(page_title="Predicción Lluvia Australia", layout="wide")

st.title("🌧️ Sistema de Predicción Meteorológica")

# --- Carga de Modelos ---
# Usamos cache tradicional para versiones viejas de Streamlit
@st.cache(allow_output_mutation=True)
def cargar_pipeline(nombre_archivo):
    try:
        return joblib.load(nombre_archivo)
    except FileNotFoundError:
        st.error(f"⚠️ No se encontró el archivo: {nombre_archivo}")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# --- 1. LISTAS DE OPCIONES ---
lista_locaciones = [
    'Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle',
    'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport',
    'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong',
    'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne',
    'Mildura', 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
    'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera',
    'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums',
    'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru'
]

lista_direcciones = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
]

# --- LÓGICA PARA VERSIÓN ANTIGUA DE STREAMLIT ---
# Como no tienes index=None, agregamos una opción "Vacío" al principio de las listas
opcion_vacia = ["Seleccionar..."]
lista_locaciones_form = opcion_vacia + lista_locaciones
lista_direcciones_form = opcion_vacia + lista_direcciones
opciones_si_no = opcion_vacia + ["No", "Yes"]

# --- 2. EL FORMULARIO (Versión Compatible Streamlit 1.12.2) ---
with st.form("form_datos_completos"):
    st.markdown("### 📝 Ingreso de Datos Meteorológicos")

    # A) DATOS GENERALES
    with st.expander("📍 Datos Básicos (Obligatorios)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            fecha_input = st.date_input("Date (Fecha)", value=datetime.date.today())
        with col2:
            location_input = st.selectbox("Location (Ubicación)", lista_locaciones_form)
        with col3:
            rain_today_input = st.selectbox("RainToday (¿Llovió hoy?)", opciones_si_no)

    # B) TEMPERATURA 
    # Usamos st.text_input y value="" (cadena vacía)
    # IMPORTANTE: No usar 'format' aquí.
    with st.expander("🌡️ Temperaturas"):
        t1, t2, t3, t4 = st.columns(4)
        with t1: min_temp = st.text_input("MinTemp", value="")
        with t2: max_temp = st.text_input("MaxTemp", value="")
        with t3: temp9am = st.text_input("Temp9am", value="")
        with t4: temp3pm = st.text_input("Temp3pm", value="")

    # C) VIENTO
    with st.expander("💨 Viento (Dirección y Velocidad)"):
        st.markdown("Si no tienes un dato, déjalo vacío.")
        w1, w2 = st.columns(2)
        with w1:
            wind_gust_dir = st.selectbox("WindGustDir (Dir. Ráfaga)", lista_direcciones_form)
        with w2:
            wind_gust_speed = st.text_input("WindGustSpeed (Vel. Ráfaga)", value="")
        
        st.markdown("---")
        wa, wb, wc, wd = st.columns(4)
        with wa: wind_dir9 = st.selectbox("WindDir9am", lista_direcciones_form)
        with wb: wind_speed9 = st.text_input("WindSpeed9am", value="")
        with wc: wind_dir3 = st.selectbox("WindDir3pm", lista_direcciones_form)
        with wd: wind_speed3 = st.text_input("WindSpeed3pm", value="")

    # D) HUMEDAD, PRESIÓN Y NUBES
    with st.expander("💧 Humedad, Presión y Otros"):
        h1, h2, p1, p2 = st.columns(4)
        with h1: humidity9 = st.text_input("Humidity9am (%)", value="")
        with h2: humidity3 = st.text_input("Humidity3pm (%)", value="")
        with p1: pressure9 = st.text_input("Pressure9am (hPa)", value="")
        with p2: pressure3 = st.text_input("Pressure3pm (hPa)", value="")
        
        c1, c2, sun, evap = st.columns(4)
        with c1: cloud9 = st.text_input("Cloud9am (octas 0-9)", value="")
        with c2: cloud3 = st.text_input("Cloud3pm (octas 0-9)", value="")
        with sun: sunshine = st.text_input("Sunshine (horas)", value="")
        with evap: evaporation = st.text_input("Evaporation (mm)", value="")
        
        rainfall_input = st.text_input("Rainfall (Lluvia acumulada hoy mm)", value="")

    # --- BOTÓN DE ENVÍO (DENTRO DEL FORM) ---
    # Esto soluciona el error "Missing Submit Button"
    submit_btn = st.form_submit_button("Realizar Predicción")

# --- Lógica de Predicción ---
if submit_btn:
    # 1. Construir DataFrame inicial (Esto se hace UNA sola vez para los 3 modelos)
    input_data = pd.DataFrame({
        'Date': [pd.to_datetime(fecha_input)],
        'Location': [location_input],
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall_input],
        'Evaporation': [evaporation],
        'Sunshine': [sunshine],
        'WindGustDir': [wind_gust_dir],
        'WindGustSpeed': [wind_gust_speed],
        'WindDir9am': [wind_dir9],
        'WindDir3pm': [wind_dir3],
        'WindSpeed9am': [wind_speed9],
        'WindSpeed3pm': [wind_speed3],
        'Humidity9am': [humidity9],
        'Humidity3pm': [humidity3],
        'Pressure9am': [pressure9],
        'Pressure3pm': [pressure3],
        'Cloud9am': [cloud9],
        'Cloud3pm': [cloud3],
        'Temp9am': [temp9am],
        'Temp3pm': [temp3pm],
        'RainToday': [rain_today_input]
    })

    # --- LIMPIEZA DE DATOS (Común para todos) ---
    # 1. Reemplazar "Seleccionar..." por np.nan
    input_data = input_data.replace("Seleccionar...", np.nan)
    
    # 2. Reemplazar strings vacíos y espacios por np.nan
    input_data = input_data.replace(r'^\s*$', np.nan, regex=True)
    
    # 3. Conversión de Tipos Numéricos
    lista_columnas_numericas = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
    ]
    
    for col in lista_columnas_numericas:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

    # --- PREDICCIÓN MÚLTIPLE ---
    try:
        with st.spinner('Consultando a los 3 modelos...'):
            # Cargamos los 3 pipelines explícitamente
            rf_pipeline = cargar_pipeline("modelo_random_forest.pkl")
            hgb_pipeline = cargar_pipeline("modelo_hist_gradient_boosting_classifier.pkl")
            nn_pipeline = cargar_pipeline("modelo_red_neuronal.pkl")

            # Verificamos que cargaron bien
            if rf_pipeline is None or hgb_pipeline is None or nn_pipeline is None:
                st.error("Error: No se pudieron cargar los archivos .pkl. Verifica que estén en la carpeta.")
            else:
                # Cada modelo hace su predicción sobre los mismos datos
                prob_rf = rf_pipeline.predict_proba(input_data)[0][1]
                prob_hgb = hgb_pipeline.predict_proba(input_data)[0][1]
                prob_nn = nn_pipeline.predict_proba(input_data)[0][1]

                # --- MOSTRAR RESULTADOS ---
                st.markdown("### 📊 Resultados del Consenso de Modelos")
                
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.info("🌲 Random Forest")
                    st.metric("Probabilidad", f"{prob_rf*100:.1f}%")
                    if prob_rf > 0.5: st.warning("Lluvia")
                    else: st.success("Sin Lluvia")

                with col_b:
                    st.info("🚀 Gradient Boosting")
                    st.metric("Probabilidad", f"{prob_hgb*100:.1f}%")
                    if prob_hgb > 0.5: st.warning("Lluvia")
                    else: st.success("Sin Lluvia")

                with col_c:
                    st.info("🧠 Red Neuronal")
                    st.metric("Probabilidad", f"{prob_nn*100:.1f}%")
                    if prob_nn > 0.5: st.warning("Lluvia")
                    else: st.success("Sin Lluvia")

                # Conclusión final (Promedio)
                promedio = (prob_rf + prob_hgb + prob_nn) / 3
                st.markdown("---")
                if promedio > 0.5:
                    st.error(f"🚨 **Conclusión Final:** El consenso de modelos indica una alta probabilidad ({promedio*100:.1f}%) de lluvia.")
                else:
                    st.balloons()
                    st.success(f"☀️ **Conclusión Final:** El consenso de modelos indica que es poco probable ({promedio*100:.1f}%) que llueva.")

    except Exception as e:
        st.error(f"Ocurrió un error inesperado durante la predicción: {e}")