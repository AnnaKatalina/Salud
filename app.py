import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import io
import sys
import os

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Clasificación - Salud Colombia",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
    }
    .prediction-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
    }
    .api-status-connected {
        color: #4CAF50;
        font-weight: bold;
    }
    .api-status-disconnected {
        color: #F44336;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🏥 Sistema de Clasificación - Modelo de Salud Colombia</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("Navegación")
app_mode = st.sidebar.selectbox(
    "Seleccione una opción:",
    ["🏠 Inicio", "🔮 Predicción Individual", "📁 Predicción por Lotes", "📊 Análisis de Resultados", "⚙️ Configuración"]
)

# Configuración de la API
st.sidebar.markdown("---")
st.sidebar.subheader("Configuración API")

# Selector de URL de API con opciones predefinidas
api_options = {
    "Local (puerto 5000)": "http://localhost:5000",
    "Local (puerto 8000)": "http://localhost:8000", 
    "Personalizada": "personalizada"
}

selected_api = st.sidebar.selectbox("URL de la API:", list(api_options.keys()))

if selected_api == "Personalizada":
    API_BASE_URL = st.sidebar.text_input("Ingrese URL personalizada:", "http://localhost:5000")
else:
    API_BASE_URL = api_options[selected_api]

# Función mejorada para verificar estado de la API
def check_api_health():
    try:
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            return True, data, response_time
        else:
            return False, {"error": f"Error HTTP {response.status_code}"}, response_time
    except requests.exceptions.Timeout:
        return False, {"error": "Timeout - La API no respondió en 10 segundos"}, 0
    except requests.exceptions.ConnectionError:
        return False, {"error": "Error de conexión - Verifique la URL y que la API esté ejecutándose"}, 0
    except Exception as e:
        return False, {"error": f"Error inesperado: {str(e)}"}, 0

# Verificar estado de la API
api_healthy, api_status, response_time = check_api_health()

# Mostrar estado en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Estado de Conexión")

if api_healthy:
    st.sidebar.success("✅ API Conectada")
    st.sidebar.metric("Tiempo Respuesta", f"{response_time:.0f} ms")
    
    if api_status.get('model_loaded'):
        st.sidebar.success(f"✅ Modelo: {api_status.get('model_type', 'Cargado')}")
    else:
        st.sidebar.error("❌ Modelo No Cargado")
else:
    st.sidebar.error("❌ API No Disponible")
    st.sidebar.error(f"Error: {api_status.get('error', 'Desconocido')}")

# Instrucciones de solución de problemas
with st.sidebar.expander("🔧 Solución de Problemas"):
    st.markdown("""
    **Si la API no está disponible:**
    
    1. **Ejecutar la API:**
    ```bash
    python api.py
    ```
    
    2. **Verificar puertos:**
    ```bash
    netstat -an | findstr :5000
    ```
    
    3. **Probar manualmente:**
    ```bash
    curl http://localhost:5000/health
    ```
    """)

# Página de Inicio
if app_mode == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Clasificación de Salud")
    
    if not api_healthy:
        st.error(f"""
        ⚠️ **La API no está disponible**
        
        **URL intentada:** `{API_BASE_URL}`
        
        **Para solucionar este problema:**
        
        1. **Ejecutar la API Flask:**
           ```bash
           python api.py
           ```
           
        2. **Verificar en tu navegador:**
           Visita: {API_BASE_URL}/health
           
        3. **Puertos alternativos:**
           Si el puerto 5000 está ocupado, prueba con:
           ```bash
           python api.py --port 8000
           ```
           
        4. **Verificar firewall:**
           Asegúrate que el puerto no esté bloqueado
        """)
        
        # Comandos útiles
        with st.expander("🛠️ Comandos útiles para diagnóstico"):
            st.code("""
# Ver procesos usando puerto 5000
netstat -ano | findstr :5000

# Ejecutar API en puerto diferente
python api.py --port 8000

# Probar conexión con curl
curl -X GET http://localhost:5000/health
            """, language="bash")
    
    else:
        st.success("✅ Sistema operativo correctamente")
    
    # Resumen del sistema
    st.subheader("📊 Resumen del Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        status_text = "Operacional" if api_healthy else "Offline"
        status_class = "api-status-connected" if api_healthy else "api-status-disconnected"
        st.markdown(f'<span class="{status_class}">Estado API: {status_text}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        model_status = api_status.get('model_type', 'No disponible') if api_healthy else "No disponible"
        st.metric("Modelo ML", model_status)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tiempo Respuesta", f"{response_time:.0f} ms" if api_healthy else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("URL API", API_BASE_URL.split('//')[-1])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Características del modelo si está disponible
    if api_healthy:
        try:
            features_response = requests.get(f"{API_BASE_URL}/features", timeout=5)
            if features_response.status_code == 200:
                features_data = features_response.json()
                st.info(f"🎯 El modelo espera {features_data.get('count', 0)} características: {', '.join(features_data.get('expected_features', []))}")
        except:
            pass

    # Guía rápida
    st.subheader("🚀 Guía Rápida de Uso")
    
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.markdown("""
        **🔮 Predicción Individual**
        - Complete el formulario
        - Obtenga resultados inmediatos  
        - Vea el nivel de confianza
        """)
        if api_healthy:
            st.success("Disponible")
        else:
            st.error("Requiere API")
    
    with guide_col2:
        st.markdown("""
        **📁 Predicción por Lotes**
        - Suba archivo CSV
        - Procese múltiples registros
        - Descargue resultados
        """)
        if api_healthy:
            st.success("Disponible")
        else:
            st.error("Requiere API")
    
    with guide_col3:
        st.markdown("""
        **📊 Análisis de Resultados**
        - Visualice distribuciones
        - Analice por características
        - Exporte reportes
        """)
        st.info("Usar con datos existentes")

# Página de Predicción Individual (similar estructura pero mejorada)
elif app_mode == "🔮 Predicción Individual":
    st.header("Predicción Individual")
    
    if not api_healthy:
        st.error(f"❌ No se puede realizar la predicción. La API en {API_BASE_URL} no está disponible.")
        st.info("Por favor, ejecuta la API Flask primero y verifica que esté corriendo en el puerto correcto.")
        st.stop()
    
    # Obtener características esperadas del modelo
    try:
        features_response = requests.get(f"{API_BASE_URL}/features", timeout=5)
        expected_features = []
        if features_response.status_code == 200:
            features_data = features_response.json()
            expected_features = features_data.get('expected_features', [])
    except:
        expected_features = []
    
    # Formulario para entrada de datos
    with st.form("prediction_form"):
        st.subheader("📝 Ingrese los datos para la predicción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            genero = st.selectbox("Género *", ["Masculino", "Femenino", "Otro"])
            grupo_etario = st.selectbox("Grupo Etario *", [
                "< 1", "1 a 5", "5 a 15", "15 a 19", "19 a 45",
                "45 a 50", "50 a 55", "55 a 60", "60 a 65",
                "65 a 70", "70 a 75", "> 75"
            ])
            tipo_afiliado = st.selectbox("Tipo de Afiliado *", [
                "COTIZANTE", "BENEFICIARIO", "CABEZA DE FAMILIA",
                "ADICIONAL", "OTRO MIEMBRO DEL NUCLEO FAMILIAR"
            ])
        
        with col2:
            departamento = st.selectbox("Departamento *", [
                "BOGOTA D.C.", "ANTIOQUIA", "VALLE", "CUNDINAMARCA",
                "ATLANTICO", "SANTANDER", "BOLIVAR", "NARIÑO",
                "BOYACA", "CORDOBA", "META", "TOLIMA", "OTRO"
            ])
            municipio = st.text_input("Municipio *", "BOGOTA")
            zona = st.selectbox("Zona de Afiliación *", [
                "Urbana", "Rural", "Urbana-Cabecera Municipal",
                "Rural - Dispersal", "Rural - Resto Rural",
                "Urbana - Centro Poblado"
            ])
            nivel_sisben = st.selectbox("Nivel Sisbén *", [
                "1", "2", "3", "4", "NO APLICA", "POBLACIÓN CON SISBEN",
                "VÍCTIMAS DEL CONFLICTO ARMADO INTERNO", "MIGRACION"
            ])
        
        st.markdown("**Campos obligatorios ***")
        submitted = st.form_submit_button("🎯 Realizar Predicción", type="primary")
        
        if submitted:
            # Validar campos obligatorios
            if not municipio.strip():
                st.error("❌ Por favor complete el campo Municipio")
                st.stop()
            
            # Preparar datos para la API
            input_data = {
                "Genero": genero,
                "Grupo_etario": grupo_etario,
                "Tipo_afiliado": tipo_afiliado,
                "Departamento": departamento,
                "Municipio": municipio,
                "Zona": zona,
                "Nivel_Sisben": nivel_sisben
            }
            
            # Realizar predicción
            with st.spinner("🔍 Analizando datos y realizando predicción..."):
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{API_BASE_URL}/predict",
                        json=input_data,
                        timeout=30
                    )
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Predicción completada en {response_time:.0f}ms")
                        
                        # Mostrar resultados (código existente)
                        # ... (mantener el código de visualización de resultados)
                        
                    else:
                        st.error(f"❌ Error en la API: {response.status_code}")
                        try:
                            error_detail = response.json()
                            st.write("Detalles del error:", error_detail)
                        except:
                            st.write("Respuesta:", response.text)
                
                except requests.exceptions.Timeout:
                    st.error("⏰ Timeout - La API no respondió en 30 segundos")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Error de conexión - Verifique que la API esté ejecutándose")
                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")

# Página de Configuración
elif app_mode == "⚙️ Configuración":
    st.header("Configuración del Sistema")
    
    st.subheader("🔧 Estado del Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Información de la API:**")
        st.write(f"- URL: {API_BASE_URL}")
        st.write(f"- Estado: {'🟢 Conectada' if api_healthy else '🔴 Desconectada'}")
        st.write(f"- Tiempo respuesta: {response_time:.0f} ms" if api_healthy else "- Tiempo respuesta: N/A")
        
        if api_healthy:
            st.write(f"- Modelo: {api_status.get('model_type', 'Desconocido')}")
            st.write(f"- Cargado: {'✅ Sí' if api_status.get('model_loaded') else '❌ No'}")
    
    with col2:
        st.write("**Información de Streamlit:**")
        st.write(f"- Versión Python: {sys.version.split()[0]}")
        st.write(f"- Versión Streamlit: {st.__version__}")
        st.write(f"- Directorio actual: {os.getcwd()}")
    
    st.subheader("🛠️ Herramientas de Diagnóstico")
    
    if st.button("🔄 Probar Conexión API"):
        api_healthy, api_status, response_time = check_api_health()
        if api_healthy:
            st.success("✅ Conexión exitosa con la API")
        else:
            st.error(f"❌ Error de conexión: {api_status.get('error')}")
    
    if st.button("📋 Obtener Información del Modelo"):
        if api_healthy:
            try:
                features_response = requests.get(f"{API_BASE_URL}/features", timeout=5)
                if features_response.status_code == 200:
                    features_data = features_response.json()
                    st.write("**Características del modelo:**")
                    st.json(features_data)
                else:
                    st.error("No se pudieron obtener las características del modelo")
            except Exception as e:
                st.error(f"Error obteniendo características: {e}")
        else:
            st.error("API no disponible")
    
    st.subheader("📚 Documentación")
    
    with st.expander("Ver comandos de instalación y ejecución"):
        st.markdown("""
        **Instalación de dependencias:**
        ```bash
        pip install -r requirements.txt
        ```
        
        **Ejecutar API Flask:**
        ```bash
        python api.py
        ```
        
        **Ejecutar aplicación Streamlit:**
        ```bash
        streamlit run app.py
        ```
        
        **Ejecutar en puerto específico:**
        ```bash
        streamlit run app.py --server.port 8501
        ```
        """)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown(
        "**Sistema de Clasificación - Modelo de Salud Colombia** | "
        "Desarrollado con Streamlit 🚀 y Flask ⚙️"
    )

with footer_col2:
    if api_healthy:
        st.markdown(f'<span class="api-status-connected">🟢 API: {API_BASE_URL.split("//")[-1]}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="api-status-disconnected">🔴 API: No conectada</span>', unsafe_allow_html=True)

with footer_col3:
    st.markdown(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
