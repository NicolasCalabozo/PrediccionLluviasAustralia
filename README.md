# Predicción de Lluvias en Australia 🌧️🦘

## Descripción
Este proyecto implementa un sistema predictivo basado en Machine Learning para determinar si lloverá al día siguiente en diversas ubicaciones de Australia. A partir de variables climáticas como la temperatura, velocidad del viento, presión, y humedad, el sistema utiliza un **consenso de modelos** para proporcionar predicciones precisas y robustas.

## Arquitectura
El sistema sigue una arquitectura cliente-servidor, separando la lógica pesada de inferencia de la interfaz de usuario:

1. **Backend (API Predictiva)**: Desarrollado con **FastAPI**. Se encarga de procesar los datos de entrada y realizar predicciones en tiempo real utilizando tres algoritmos distintos:
   - Random Forest 🌲
   - HistGradientBoosting 🚀
   - Red Neuronal 🧠
   El backend calcula una probabilidad promedio (consenso) de los tres modelos para generar el resultado final.

2. **Frontend (Interfaz de Usuario)**: Desarrollado con **Streamlit**. Proporciona un formulario intuitivo donde los usuarios pueden ingresar manualmente las condiciones climáticas del día. Muestra el resultado final (Alerta de Lluvia / Buen Tiempo) y el detalle de "opinión" de cada modelo de manera gráfica.

## Estructura del Proyecto
- `app.py`: Código principal del Frontend (Streamlit).
- `api.py`: Código principal del Backend (FastAPI).
- `imputación.py` y `clases_modelos.py`: Módulos personalizados con la lógica de limpieza de datos, imputación (espacial, numérica, categórica) y definición de clases base.
- `Analisis_Datos.ipynb`: Notebook de Jupyter con el Análisis Exploratorio de Datos (EDA).
- `Pipeline_Produccion.ipynb`: Notebook que contiene el pipeline de entrenamiento de los modelos y exportación.
- `*.pkl`: Modelos de Machine Learning previamente entrenados y serializados listos para su uso.
- `docker-compose.yml` y `Dockerfile`: Configuración para el despliegue del proyecto en contenedores Docker.
- `requirements.txt`: Lista de dependencias del proyecto.

## Tecnologías Utilizadas
- **Lenguaje**: Python
- **Ciencia de Datos / ML**: Scikit-Learn, TensorFlow (CPU), Pandas, NumPy, Joblib, Geopandas
- **Frameworks Web**: FastAPI, Streamlit
- **Infraestructura**: Docker, Docker Compose

## Instrucciones de Ejecución

### Opción 1: Despliegue con Docker (Recomendado)
Es la manera más rápida y segura de probar la aplicación sin afectar tu entorno local.
1. Asegúrate de tener [Docker](https://docs.docker.com/get-docker/) y Docker Compose instalados.
2. Abre una terminal en la raíz del proyecto.
3. Ejecuta el siguiente comando para construir las imágenes y levantar los contenedores:
   ```bash
   docker-compose up --build
   ```
4. Accede a las aplicaciones desde tu navegador:
   - **Frontend (Interfaz de Usuario)**: [http://localhost:8501](http://localhost:8501)
   - **Documentación de la API (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Opción 2: Ejecución Local
Ideal para desarrollar y realizar modificaciones en el código.
1. Crea un entorno virtual para el proyecto:
   ```bash
   python -m venv venv
   ```
2. Activa el entorno virtual:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Instala todas las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
4. **Levantar la API**: En una terminal, ejecuta:
   ```bash
   uvicorn api:app --reload --port 8000
   ```
5. **Levantar el Frontend**: En *otra* terminal (con el entorno virtual activado), ejecuta:
   ```bash
   streamlit run app.py
   ```
6. Automáticamente se debería abrir el frontend en tu navegador o puedes acceder manualmente a `http://localhost:8501`.
