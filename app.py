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

# --- Sidebar ---
with st.sidebar:
    st.header("Configuración del Modelo")
    modelo_seleccionado = st.selectbox(
        "Algoritmo de Predicción",
        [
            "Random Forest", 
            "Hist Gradient Boosting", 
            "Red Neuronal"
        ]
    )
    
    archivos_modelos = {
        "Random Forest": "modelo_random_forest.pkl",
        "Hist Gradient Boosting": "modelo_hist_gradient_boosting_classifier.pkl",
        "Red Neuronal": "modelo_red_neuronal.pkl"
    }
    
    archivo_a_cargar = archivos_modelos[modelo_seleccionado]

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

# --- 2. EL FORMULARIO ---
with st.form("form_datos_completos"):
    st.markdown("### 📝 Ingreso de Datos Meteorológicos")

    # A) DATOS GENERALES
    with st.expander("📍 Datos Básicos (Obligatorios)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            fecha_input = st.date_input("Date (Fecha)", value=datetime.date.today())
        with col2:
            # Quitamos index=None y placeholder
            location_input = st.selectbox("Location (Ubicación)", lista_locaciones_form)
        with col3:
            rain_today_input = st.selectbox("RainToday (¿Llovió hoy?)", opciones_si_no)

    # B) TEMPERATURA
    with st.expander("🌡️ Temperaturas"):
        t1, t2, t3, t4 = st.columns(4)
        with t1: min_temp = st.number_input("MinTemp", value=None, format="%.1f")
        with t2: max_temp = st.number_input("MaxTemp", value=None, format="%.1f")
        with t3: temp9am = st.number_input("Temp9am", value=None, format="%.1f")
        with t4: temp3pm = st.number_input("Temp3pm", value=None, format="%.1f")

    # C) VIENTO
    with st.expander("💨 Viento (Dirección y Velocidad)"):
        st.markdown("Si no tienes un dato, déjalo en 'Seleccionar...' o vacío.")
        w1, w2 = st.columns(2)
        with w1:
            wind_gust_dir = st.selectbox("WindGustDir (Dir. Ráfaga)", lista_direcciones_form)
        with w2:
            wind_gust_speed = st.number_input("WindGustSpeed (Vel. Ráfaga)", value=None, format="%.1f")
        
        st.markdown("---")
        wa, wb, wc, wd = st.columns(4)
        with wa: wind_dir9 = st.selectbox("WindDir9am", lista_direcciones_form)
        with wb: wind_speed9 = st.number_input("WindSpeed9am", value=None, format="%.1f")
        with wc: wind_dir3 = st.selectbox("WindDir3pm", lista_direcciones_form)
        with wd: wind_speed3 = st.number_input("WindSpeed3pm", value=None, format="%.1f")

    # D) HUMEDAD, PRESIÓN Y NUBES
    with st.expander("💧 Humedad, Presión y Otros"):
        h1, h2, p1, p2 = st.columns(4)
        # Usamos number_input en vez de slider para permitir vacíos
        with h1: humidity9 = st.number_input("Humidity9am (%)", min_value=0, max_value=100, step=1, value=None)
        with h2: humidity3 = st.number_input("Humidity3pm (%)", min_value=0, max_value=100, step=1, value=None)
        with p1: pressure9 = st.number_input("Pressure9am (hPa)", value=None, format="%.1f")
        with p2: pressure3 = st.number_input("Pressure3pm (hPa)", value=None, format="%.1f")
        
        c1, c2, sun, evap = st.columns(4)
        with c1: cloud9 = st.number_input("Cloud9am (octas)", min_value=0, max_value=9, step=1, value=None)
        with c2: cloud3 = st.number_input("Cloud3pm (octas)", min_value=0, max_value=9, step=1, value=None)
        with sun: sunshine = st.number_input("Sunshine (horas)", value=None, format="%.1f")
        with evap: evaporation = st.number_input("Evaporation (mm)", value=None, format="%.1f")
        
        rainfall_input = st.number_input("Rainfall (Lluvia acumulada hoy mm)", value=None, format="%.2f")

    submit_btn = st.form_submit_button("Realizar Predicción")

# --- Lógica de Predicción ---
if submit_btn:
    pipeline = cargar_pipeline(archivo_a_cargar)
    
    if pipeline:
        # Construir DataFrame inicial
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

        # --- CORRECCIONES CLAVE PARA EL BACKEND ---
        
        # 1. Reemplazar "Seleccionar..." por np.nan (Para tus selectbox antiguos)
        input_data = input_data.replace("Seleccionar...", np.nan)
        
        # 2. Reemplazar strings vacíos y espacios por np.nan
        input_data = input_data.replace(r'^\s*$', np.nan, regex=True)
        
        # 3. Conversión de Tipos Numéricos (Explícito)
        # Definimos explícitamente qué columnas son números para forzar la conversión
        lista_columnas_numericas = [
            'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
            'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
            'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
        ]
        
        for col in lista_columnas_numericas:
            # errors='coerce' transformará None y strings raros en NaN
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        try:
            with st.spinner('Procesando datos y generando predicción...'):
                
                # Predicción
                prediccion = pipeline.predict(input_data)[0]
                probs = pipeline.predict_proba(input_data)[0]
                prob_lluvia = probs[1] # Probabilidad de "Yes"

            # Mostrar Resultados
            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric("Probabilidad de Lluvia", f"{prob_lluvia*100:.1f}%")
            
            with col_res2:
                if prob_lluvia > 0.5:
                    st.error(f"🌧️ **Alta probabilidad de lluvia** detectada con {modelo_seleccionado}.")
                else:
                    st.success(f"☀️ **Baja probabilidad de lluvia** detectada con {modelo_seleccionado}.")
                
                st.progress(int(prob_lluvia*100))

        except Exception as e:
            st.error("Ocurrió un error en el pipeline:")
            st.code(e)
            st.info("Verifica que las columnas coincidan y que los archivos .pkl y shapefiles estén accesibles.")