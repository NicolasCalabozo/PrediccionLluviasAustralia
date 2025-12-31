import streamlit as st
import datetime
import requests  # <--- Esta es la clave para hablar con la API

# --- Configuración de Página ---
st.set_page_config(page_title="Predicción Lluvia Australia (Cliente API)", layout="wide")

st.title("🌧️ Sistema de Predicción (Modo Cliente API)")
st.markdown("Esta interfaz es ligera. Los cálculos pesados los hace el servidor (FastAPI).")

# --- LISTAS DE OPCIONES ---
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

opcion_vacia = ["Seleccionar..."]
lista_locaciones_form = opcion_vacia + lista_locaciones
lista_direcciones_form = opcion_vacia + lista_direcciones
opciones_si_no = opcion_vacia + ["No", "Yes"]

# --- FORMULARIO ---
with st.form("form_datos_api"):
    st.markdown("### 📝 Ingreso de Datos")

    # A) DATOS GENERALES
    with st.expander("📍 Datos Básicos (Obligatorios)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1: fecha_input = st.date_input("Fecha", value=datetime.date.today())
        with col2: location_input = st.selectbox("Ubicación", lista_locaciones_form)
        with col3: rain_today_input = st.selectbox("¿Llovió hoy?", opciones_si_no)

    # B) TEMPERATURA 
    with st.expander("🌡️ Temperaturas"):
        t1, t2, t3, t4 = st.columns(4)
        with t1: min_temp = st.text_input("MinTemp", value="")
        with t2: max_temp = st.text_input("MaxTemp", value="")
        with t3: temp9am = st.text_input("Temp9am", value="")
        with t4: temp3pm = st.text_input("Temp3pm", value="")

    # C) VIENTO
    with st.expander("💨 Viento"):
        w1, w2 = st.columns(2)
        with w1: wind_gust_dir = st.selectbox("Dir. Ráfaga", lista_direcciones_form)
        with w2: wind_gust_speed = st.text_input("Vel. Ráfaga", value="")
        
        wa, wb, wc, wd = st.columns(4)
        with wa: wind_dir9 = st.selectbox("WindDir9am", lista_direcciones_form)
        with wb: wind_speed9 = st.text_input("WindSpeed9am", value="")
        with wc: wind_dir3 = st.selectbox("WindDir3pm", lista_direcciones_form)
        with wd: wind_speed3 = st.text_input("WindSpeed3pm", value="")

    # D) OTROS
    with st.expander("💧 Otros"):
        h1, h2, p1, p2 = st.columns(4)
        with h1: humidity9 = st.text_input("Humidity9am", value="")
        with h2: humidity3 = st.text_input("Humidity3pm", value="")
        with p1: pressure9 = st.text_input("Pressure9am", value="")
        with p2: pressure3 = st.text_input("Pressure3pm", value="")
        
        c1, c2, sun, evap = st.columns(4)
        with c1: cloud9 = st.text_input("Cloud9am", value="")
        with c2: cloud3 = st.text_input("Cloud3pm", value="")
        with sun: sunshine = st.text_input("Sunshine", value="")
        with evap: evaporation = st.text_input("Evaporation", value="")
        
        rainfall_input = st.text_input("Rainfall (Lluvia acumulada)", value="")

    submit_btn = st.form_submit_button("Consultar API")

# --- LÓGICA DE ENVÍO A LA API ---
if submit_btn:
    datos_a_enviar = {
        "Date": str(fecha_input),
        "Location": location_input,
        "RainToday": rain_today_input,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall_input,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir9,
        "WindDir3pm": wind_dir3,
        "WindSpeed9am": wind_speed9,
        "WindSpeed3pm": wind_speed3,
        "Humidity9am": humidity9,
        "Humidity3pm": humidity3,
        "Pressure9am": pressure9,
        "Pressure3pm": pressure3,
        "Cloud9am": cloud9,
        "Cloud3pm": cloud3,
        "Temp9am": temp9am,
        "Temp3pm": temp3pm
    }

    try:
        with st.spinner('Consultando al Comité de IA (RF, HGB, NN)...'):
            # Nota: cambiamos la URL a /predict_all
            respuesta = requests.post("http://127.0.0.1:8000/predict_all", json=datos_a_enviar)
            
        if respuesta.status_code == 200:
            data = respuesta.json()
            
            if "error" in data:
                st.error(f"Error en API: {data['error']}")
            else:
                # 1. RESULTADO PRINCIPAL (CONSENSO)
                prob_cons = data["consenso"]["probabilidad"]
                st.markdown("## 🔮 Resultado del Consenso")
                
                col_res, col_msg = st.columns([1, 2])
                with col_res:
                    st.metric("Probabilidad Promedio", f"{prob_cons*100:.1f}%")
                with col_msg:
                    if prob_cons > 0.5:
                        st.error(f"⚠️ **ALERTA DE LLUVIA:** Los modelos coinciden en que es probable que llueva.")
                    else:
                        st.success(f"☀️ **BUEN TIEMPO:** El consenso indica que no lloverá.")

                # 2. DETALLE TÉCNICO (Lo que querías: ver cada uno)
                st.markdown("---")
                with st.expander("🔍 Ver opinión detallada de cada modelo"):
                    st.write("Aquí puedes ver qué 'pensó' cada algoritmo por separado:")
                    
                    detalles = data["detalle"]
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.info("🌲 Random Forest")
                        st.progress(detalles["Random Forest"])
                        st.write(f"**{detalles['Random Forest']*100:.1f}%**")
                        
                    with c2:
                        st.info("🚀 Gradient Boosting")
                        st.progress(detalles["Gradient Boosting"])
                        st.write(f"**{detalles['Gradient Boosting']*100:.1f}%**")
                        
                    #with c3:
                        #st.info("🧠 Red Neuronal")
                        #st.progress(detalles["Red Neuronal"])
                        #st.write(f"**{detalles['Red Neuronal']*100:.1f}%**")

        else:
            st.error("Error de conexión con el servidor.")
            
    except Exception as e:
        st.error(f"Error crítico: {e}")