import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Clasificación - Salud Colombia",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏥 Sistema de Clasificación - Modelo de Salud Colombia")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("Navegación")
app_mode = st.sidebar.selectbox(
    "Seleccione una opción:",
    ["🏠 Inicio", "📊 Análisis Exploratorio", "🔮 Predicción Individual", "📁 Predicción por Lotes", "📈 Resultados", "ℹ️ Acerca del Modelo"]
)

# URL base de la API - CON MODO SIMULACIÓN
API_BASE_URL = st.sidebar.text_input("URL de la API:", "http://localhost:5000")
MODO_SIMULACION = st.sidebar.checkbox("🔧 Usar modo simulación", value=True)

# Función para verificar estado de la API
def check_api_health():
    if MODO_SIMULACION:
        return True, {
            "message": "API de Modelo de Machine Learning - Sistema de Salud",
            "model_loaded": True,
            "status": "operacional",
            "endpoints": {
                "/batch_predict": "POST - Predicciones por lotes (SIMULADO)",
                "/health": "GET - Estado del servicio (SIMULADO)",
                "/predict": "POST - Realizar predicciones individuales (SIMULADO)"
            },
            "modo": "simulación"
        }
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Error {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

# Función de predicción simulada
def prediccion_simulada(input_data):
    """Simula predicciones basadas en reglas simples"""
    # Reglas de simulación (basadas en datos reales)
    riesgo_alto_indicadores = 0
    
    # Regla 1: Adultos mayores tienen mayor riesgo
    if input_data.get("Grupo_etario") in ["> 75", "70 a 75", "65 a 70"]:
        riesgo_alto_indicadores += 2
    
    # Regla 2: Régimen subsidiado puede indicar mayor vulnerabilidad
    if input_data.get("Régimen") == "Subsidiado":
        riesgo_alto_indicadores += 1
    
    # Regla 3: Zonas rurales pueden tener menor acceso
    if "Rural" in input_data.get("Zona", ""):
        riesgo_alto_indicadores += 1
    
    # Regla 4: Nivel Sisbén bajo indica vulnerabilidad
    if input_data.get("Nivel_Sisben") in ["1", "2"]:
        riesgo_alto_indicadores += 1
    
    # Determinar predicción basada en reglas
    if riesgo_alto_indicadores >= 2:
        prediction = 1  # Alto riesgo
        confidence = min(0.3 + (riesgo_alto_indicadores * 0.15), 0.95)
    else:
        prediction = 0  # Bajo riesgo
        confidence = min(0.7 - (riesgo_alto_indicadores * 0.1), 0.95)
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "modo": "simulación",
        "indicadores_riesgo": riesgo_alto_indicadores
    }

# Función de predicción por lotes simulada
def prediccion_lotes_simulada(records):
    """Simula predicciones por lotes"""
    predictions = []
    for record in records:
        prediction = prediccion_simulada(record)
        predictions.append(prediction)
    
    return {
        "predictions": predictions,
        "total_records": len(records),
        "modo": "simulación"
    }

# Página de Inicio
if app_mode == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Clasificación de Salud")
    
    # Verificar estado de la API
    st.subheader("🔍 Estado del Sistema")
    api_healthy, api_status = check_api_health()
    
    if api_healthy:
        if MODO_SIMULACION:
            st.warning("🔧 **MODO SIMULACIÓN ACTIVADO** - Usando predicciones simuladas")
        else:
            st.success("✅ API conectada y operacional")
        st.json(api_status)
    else:
        st.error(f"❌ Error conectando con la API: {api_status['error']}")
        st.info("Active el modo simulación para probar la aplicación")
    
    # Resumen del sistema
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modelo", "Cargado" if api_healthy and api_status.get('model_loaded') else "No disponible")
    
    with col2:
        if MODO_SIMULACION:
            st.metric("Estado", "🔧 Simulación")
        else:
            st.metric("Estado", "Operacional" if api_healthy else "Offline")
    
    with col3:
        st.metric("Endpoints", len(api_status.get('endpoints', {})) if api_healthy else 0)

# Página de Predicción Individual
elif app_mode == "🔮 Predicción Individual":
    st.header("Predicción Individual")
    
    if not check_api_health()[0] and not MODO_SIMULACION:
        st.error("La API no está disponible. Por favor, verifique la conexión o active el modo simulación.")
        st.stop()
    
    if MODO_SIMULACION:
        st.warning("🔧 **MODO SIMULACIÓN**: Usando reglas predefinidas para predicciones")
    
    # Formulario para entrada de datos
    with st.form("prediction_form"):
        st.subheader("Ingrese los datos para la predicción:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            genero = st.selectbox("Género", ["Masculino", "Femenino"])
            grupo_etario = st.selectbox("Grupo Etario", [
                "< 1", "1 a 5", "5 a 15", "15 a 19", "19 a 45", 
                "45 a 50", "50 a 55", "55 a 60", "60 a 65", 
                "65 a 70", "70 a 75", "> 75"
            ])
            regimen = st.selectbox("Régimen", ["Contributivo", "Subsidiado"])
            tipo_afiliado = st.selectbox("Tipo de Afiliado", [
                "COTIZANTE", "BENEFICIARIO", "CABEZA DE FAMILIA", 
                "ADICIONAL", "OTRO MIEMBRO DEL NUCLEO FAMILIAR"
            ])
        
        with col2:
            departamento = st.selectbox("Departamento", [
                "BOGOTA D.C.", "ANTIOQUIA", "VALLE", "CUNDINAMARCA", 
                "ATLANTICO", "SANTANDER", "BOLIVAR", "NARINO", 
                "BOYACA", "CORDOBA", "META", "TOLIMA"
            ])
            municipio = st.text_input("Municipio", "BOGOTA")
            zona = st.selectbox("Zona de Afiliación", [
                "Urbana", "Rural", "Urbana-Cabecera Municipal",
                "Rural - Dispersal", "Rural - Resto Rural",
                "Urbana - Centro Poblado"
            ])
            nivel_sisben = st.selectbox("Nivel Sisbén", [
                "1", "2", "3", "4", "NO APLICA", "POBLACIÓN CON SISBEN",
                "VÍCTIMAS DEL CONFLICTO ARMADO INTERNO"
            ])
        
        submitted = st.form_submit_button("Realizar Predicción")
        
        if submitted:
            # Preparar datos para la API
            input_data = {
                "Genero": genero,
                "Grupo_etario": grupo_etario,
                "Régimen": regimen,
                "Tipo_afiliado": tipo_afiliado,
                "Departamento": departamento,
                "Municipio": municipio,
                "Zona": zona,
                "Nivel_Sisben": nivel_sisben
            }
            
            # Realizar predicción
            with st.spinner("Realizando predicción..."):
                try:
                    if MODO_SIMULACION:
                        result = prediccion_simulada(input_data)
                        st.success("✅ Predicción completada exitosamente! (Modo Simulación)")
                    else:
                        response = requests.post(
                            f"{API_BASE_URL}/predict",
                            json=input_data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Predicción completada exitosamente!")
                        else:
                            st.error(f"Error en la predicción: {response.status_code}")
                            st.stop()
                    
                    # Mostrar resultados
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Resultado de la Predicción")
                        prediction = result.get('prediction', 'N/A')
                        confidence = result.get('confidence', 0)
                        
                        if prediction == 1:
                            st.error(f"🔴 Clasificación: Alto Riesgo")
                        else:
                            st.success(f"🟢 Clasificación: Bajo Riesgo")
                        
                        st.metric("Confianza", f"{confidence:.2%}")
                        
                        if MODO_SIMULACION:
                            st.info(f"📊 Indicadores de riesgo: {result.get('indicadores_riesgo', 0)}")
                    
                    with col2:
                        st.subheader("Datos Ingresados")
                        st.json(input_data)
                        
                except Exception as e:
                    st.error(f"Error en la predicción: {str(e)}")

# Página de Predicción por Lotes
elif app_mode == "📁 Predicción por Lotes":
    st.header("Predicción por Lotes")
    
    if not check_api_health()[0] and not MODO_SIMULACION:
        st.error("La API no está disponible. Por favor, verifique la conexión o active el modo simulación.")
        st.stop()
    
    if MODO_SIMULACION:
        st.warning("🔧 **MODO SIMULACIÓN**: Usando reglas predefinidas para predicciones por lotes")
    
    st.info("""
    **Instrucciones:** 
    - Suba un archivo CSV con los datos para predicción
    - El archivo debe tener las columnas: Genero, Grupo_etario, Régimen, etc.
    - Descargue los resultados después de la predicción
    """)
    
    uploaded_file = st.file_uploader("Seleccione archivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Leer el archivo
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"Archivo cargado: {uploaded_file.name}")
            
            # Mostrar vista previa
            st.subheader("Vista Previa del Archivo")
            st.dataframe(batch_data.head())
            
            st.write(f"**Total de registros:** {len(batch_data)}")
            
            if st.button("Ejecutar Predicción por Lotes"):
                with st.spinner("Procesando predicciones por lotes..."):
                    try:
                        # Convertir a formato JSON para la API
                        records = batch_data.to_dict('records')
                        
                        if MODO_SIMULACION:
                            results = prediccion_lotes_simulada(records)
                            st.success(f"✅ {len(records)} predicciones simuladas completadas")
                        else:
                            response = requests.post(
                                f"{API_BASE_URL}/batch_predict",
                                json={"data": records},
                                timeout=60
                            )
                            
                            if response.status_code == 200:
                                results = response.json()
                                st.success(f"✅ {len(records)} predicciones completadas")
                            else:
                                st.error(f"Error en la predicción por lotes: {response.status_code}")
                                st.stop()
                        
                        # Agregar predicciones al DataFrame
                        predictions = results.get('predictions', [])
                        batch_data['prediccion'] = [p.get('prediction', 'N/A') for p in predictions]
                        batch_data['confianza'] = [p.get('confidence', 0) for p in predictions]
                        
                        # Mostrar resumen
                        st.subheader("Resumen de Predicciones")
                        col1, col2, col3 = st.columns(3)
                        
                        # Manejar diferentes tipos de datos en prediccion
                        if batch_data['prediccion'].dtype == 'object':
                            alto_riesgo = len(batch_data[batch_data['prediccion'] == '1'])
                            bajo_riesgo = len(batch_data[batch_data['prediccion'] == '0'])
                        else:
                            alto_riesgo = batch_data['prediccion'].sum()
                            bajo_riesgo = len(batch_data) - alto_riesgo
                        
                        with col1:
                            st.metric("Total", len(batch_data))
                        with col2:
                            st.metric("Alto Riesgo", alto_riesgo)
                        with col3:
                            st.metric("Bajo Riesgo", bajo_riesgo)
                        
                        # Mostrar resultados
                        st.subheader("Resultados Detallados")
                        st.dataframe(batch_data)
                        
                        # Descargar resultados
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resultados CSV",
                            data=csv,
                            file_name=f"resultados_prediccion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Guardar en session state para análisis
                        st.session_state.batch_results = batch_data
                        
                    except Exception as e:
                        st.error(f"Error procesando el lote: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error leyendo el archivo: {str(e)}")

# ... (el resto de tu código se mantiene igual)

# Footer
st.markdown("---")
st.markdown(
    "**Sistema de Clasificación - Modelo de Salud Colombia** | "
    "Desarrollado con Streamlit 🚀"
)
