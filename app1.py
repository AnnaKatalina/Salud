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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Clasificación - Salud Colombia",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados mejorados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-high {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-low {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-importance {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .api-status-connected {
        color: #4caf50;
        font-weight: bold;
    }
    .api-status-disconnected {
        color: #f44336;
        font-weight: bold;
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
    ["🏠 Inicio", "📊 Análisis Exploratorio", "🔮 Predicción Individual", "📁 Predicción por Lotes", "📈 Resultados", "🔍 Análisis del Modelo", "ℹ️ Acerca del Modelo"]
)

# URL base de la API
API_BASE_URL = st.sidebar.text_input("URL de la API:", "http://localhost:5000")

# Información de conexión en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Estado de Conexión")

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
            return False, {"error": f"Error {response.status_code}"}, response_time
    except requests.exceptions.Timeout:
        return False, {"error": "Timeout - La API no respondió en 10 segundos"}, 0
    except requests.exceptions.ConnectionError:
        return False, {"error": "Error de conexión - Verifique la URL"}, 0
    except Exception as e:
        return False, {"error": str(e)}, 0

# Verificar estado de la API
api_healthy, api_status, response_time = check_api_health()

# Mostrar estado en sidebar
status_col1, status_col2 = st.sidebar.columns([1, 2])
with status_col1:
    if api_healthy:
        st.success("✅")
    else:
        st.error("❌")
with status_col2:
    if api_healthy:
        st.markdown('<span class="api-status-connected">API Conectada</span>', unsafe_allow_html=True)
        st.metric("Tiempo Respuesta", f"{response_time:.0f} ms")
        if api_status.get('model_loaded'):
            st.success("✅ Modelo Cargado")
        else:
            st.error("❌ Modelo No Cargado")
    else:
        st.markdown('<span class="api-status-disconnected">API No Disponible</span>', unsafe_allow_html=True)
        st.error(f"Error: {api_status.get('error', 'Desconocido')}")

# Página de Inicio
if app_mode == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Clasificación de Salud")
    
    if not api_healthy:
        st.error("""
        ⚠️ **La API no está disponible**
        
        Para usar la aplicación, asegúrese de:
        1. Tener la API ejecutándose en la URL especificada
        2. Verificar que el puerto 5000 esté disponible
        3. Que el modelo esté correctamente cargado
        
        **Solución rápida:** Ejecute el siguiente comando en su terminal:
        ```bash
        python api_flask.py
        ```
        """)
    
    # Resumen del sistema
    st.subheader("📊 Resumen del Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Estado API", "Operacional" if api_healthy else "Offline")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        model_status = "Cargado" if api_healthy and api_status.get('model_loaded') else "No disponible"
        st.metric("Modelo ML", model_status)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tiempo Respuesta", f"{response_time:.0f} ms" if api_healthy else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        endpoint_count = len(api_status.get('endpoints', {})) if api_healthy else 0
        st.metric("Endpoints", endpoint_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Información sobre los datos y modelo
    st.subheader("🎯 Objetivo del Modelo")
    
    st.info("""
    **Este sistema utiliza modelos de Machine Learning para:**
    - 🔍 **Clasificar** afiliados entre régimen contributivo y subsidiado
    - 📈 **Predecir** riesgo basado en características demográficas
    - 🎯 **Optimizar** la asignación de recursos en salud
    - 📊 **Analizar** patrones en los datos del sistema de salud
    """)
    
    # Información sobre los datos
    st.subheader("📋 Bases de Datos Utilizadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🏢 Base de Datos - Régimen Contributivo", expanded=True):
            st.markdown("""
            - **Fuente**: Base de Datos Única de Afiliados (BDUA)
            - **Registros**: ~641,000
            - **Actualización**: Julio 2025
            - **Variables principales**:
              * Género y grupo etario
              * Tipo de afiliado
              * Ubicación geográfica
              * Nivel Sisbén
              * Estado del afiliado
            """)
    
    with col2:
        with st.expander("🏘️ Base de Datos - Régimen Subsidiado", expanded=True):
            st.markdown("""
            - **Fuente**: Entidades Promotoras de Salud (EPSS)
            - **Registros**: ~1,000,000+
            - **Actualización**: Julio 2025
            - **Variables principales**:
              * Género y grupo etario
              * Tipo de afiliación
              * Zona geográfica
              * Nivel Sisbén
              * Grupo poblacional
            """)
    
    # Guía rápida
    st.subheader("🚀 Guía Rápida de Uso")
    
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.markdown("""
        **🔮 Predicción Individual**
        - Complete el formulario interactivo
        - Obtenga resultados en tiempo real
        - Vea probabilidades y explicaciones
        """)
    
    with guide_col2:
        st.markdown("""
        **📁 Predicción por Lotes**
        - Suba archivo CSV con múltiples registros
        - Procesamiento eficiente en lote
        - Descargue resultados completos
        """)
    
    with guide_col3:
        st.markdown("""
        **📈 Análisis Avanzado**
        - Visualice distribuciones
        - Analice importancia de características
        - Exporte reportes ejecutivos
        """)

# Página de Análisis Exploratorio
elif app_mode == "📊 Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    
    # Opciones de análisis
    analysis_type = st.radio(
        "Tipo de análisis:",
        ["Datos de Ejemplo", "Subir Datos Propios"],
        horizontal=True
    )
    
    if analysis_type == "Datos de Ejemplo":
        if st.button("🎲 Generar Datos de Ejemplo", type="primary"):
            with st.spinner("Generando datos de ejemplo basados en la estructura real..."):
                # Simular datos realistas basados en el notebook
                np.random.seed(42)
                n_samples = 2000
                
                # Crear datos balanceados entre contributivo y subsidiado
                sample_data = pd.DataFrame({
                    'Genero': np.random.choice(['Masculino', 'Femenino'], n_samples, p=[0.48, 0.52]),
                    'Grupo_etario': np.random.choice([
                        '< 1', '1 a 5', '5 a 15', '15 a 19', '19 a 45',
                        '45 a 50', '50 a 55', '55 a 60', '60 a 65',
                        '65 a 70', '70 a 75', '> 75'
                    ], n_samples, p=[0.02, 0.05, 0.08, 0.1, 0.25, 0.15, 0.12, 0.08, 0.06, 0.05, 0.03, 0.01]),
                    'Régimen': np.random.choice(['Contributivo', 'Subsidiado'], n_samples, p=[0.4, 0.6]),
                    'Tipo_afiliado': np.random.choice([
                        'COTIZANTE', 'BENEFICIARIO', 'CABEZA DE FAMILIA'
                    ], n_samples, p=[0.4, 0.4, 0.2]),
                    'Departamento': np.random.choice([
                        'BOGOTA D.C.', 'ANTIOQUIA', 'VALLE', 'CUNDINAMARCA', 
                        'ATLANTICO', 'SANTANDER', 'BOLIVAR', 'NARIÑO', 'BOYACA'
                    ], n_samples, p=[0.2, 0.15, 0.12, 0.1, 0.08, 0.08, 0.07, 0.1, 0.1]),
                    'Municipio': np.random.choice([
                        'BOGOTA', 'MEDELLIN', 'CALI', 'BARRANQUILLA',
                        'CARTAGENA', 'BUCARAMANGA', 'CUCUTA', 'VILLAVICENCIO'
                    ], n_samples),
                    'Zona': np.random.choice([
                        'Urbana', 'Rural', 'Urbana-Cabecera Municipal'
                    ], n_samples, p=[0.7, 0.2, 0.1]),
                    'Nivel_Sisben': np.random.choice([
                        '1', '2', '3', '4', 'NO APLICA'
                    ], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
                    'Estado_afiliado': np.random.choice([
                        'Activo', 'Inactivo', 'Protección Laboral C'
                    ], n_samples, p=[0.85, 0.1, 0.05])
                })
                
                st.session_state.sample_data = sample_data
                st.success(f"✅ Se generaron {n_samples} registros de ejemplo realistas!")
    
    else:  # Subir Datos Propios
        uploaded_file = st.file_uploader("📤 Subir archivo CSV", type="csv")
        if uploaded_file is not None:
            try:
                sample_data = pd.read_csv(uploaded_file)
                st.session_state.sample_data = sample_data
                st.success(f"✅ Archivo cargado: {uploaded_file.name}")
                st.info(f"📊 Dimensiones: {sample_data.shape[0]} filas × {sample_data.shape[1]} columnas")
            except Exception as e:
                st.error(f"❌ Error al cargar el archivo: {str(e)}")
    
    # Mostrar análisis si hay datos
    if 'sample_data' in st.session_state:
        data = st.session_state.sample_data
        
        # Mostrar datos
        st.subheader("📋 Vista Previa de Datos")
        
        # Filtros interactivos
        col1, col2, col3 = st.columns(3)
        with col1:
            show_rows = st.slider("Filas a mostrar", 5, 100, 10)
        with col2:
            selected_columns = st.multiselect(
                "Columnas a mostrar",
                data.columns.tolist(),
                default=data.columns.tolist()[:min(8, len(data.columns))]
            )
        
        st.dataframe(data[selected_columns].head(show_rows), use_container_width=True)
        
        # Estadísticas básicas
        st.subheader("📊 Estadísticas Descriptivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Información General**")
            st.write(f"📈 Registros totales: {len(data):,}")
            st.write(f"📊 Variables: {len(data.columns)}")
            st.write(f"💾 Memoria usada: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.write(f"🔍 Valores nulos: {data.isnull().sum().sum()}")
            
            # Tipos de datos
            st.write("**📝 Tipos de Datos**")
            dtype_counts = data.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columnas")
        
        with col2:
            st.write("**🔢 Resumen Numérico**")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write(data[numeric_cols].describe())
            else:
                st.info("No hay columnas numéricas en los datos")
        
        # Visualizaciones interactivas
        st.subheader("📈 Visualizaciones Interactivas")
        
        # Seleccionar variables para visualizar
        available_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if available_columns:
            viz_col1, viz_col2 = st.columns([1, 2])
            
            with viz_col1:
                chart_type = st.selectbox(
                    "Tipo de gráfico:",
                    ["Barras", "Torta", "Histograma", "Boxplot", "Dispersión"]
                )
                
                x_axis = st.selectbox(
                    "Variable X:",
                    available_columns
                )
                
                # Opciones adicionales según el tipo de gráfico
                if chart_type in ["Dispersión", "Boxplot"] and len(available_columns) > 1:
                    y_axis = st.selectbox(
                        "Variable Y:",
                        [col for col in available_columns if col != x_axis]
                    )
                else:
                    y_axis = None
                
                color_by = st.selectbox(
                    "Color por (opcional):",
                    ["Ninguno"] + available_columns
                )
            
            with viz_col2:
                if chart_type == "Barras":
                    fig = px.bar(data, x=x_axis, title=f'Distribución de {x_axis}')
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Torta":
                    counts = data[x_axis].value_counts()
                    fig = px.pie(values=counts.values, names=counts.index, 
                                title=f'Distribución de {x_axis}')
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Histograma" and x_axis in data.select_dtypes(include=[np.number]).columns:
                    fig = px.histogram(data, x=x_axis, title=f'Histograma de {x_axis}')
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Boxplot" and y_axis:
                    fig = px.box(data, x=x_axis, y=y_axis, title=f'Boxplot: {x_axis} vs {y_axis}')
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Dispersión" and y_axis:
                    color_param = None if color_by == "Ninguno" else color_by
                    fig = px.scatter(data, x=x_axis, y=y_axis, color=color_param,
                                   title=f'Dispersión: {x_axis} vs {y_axis}')
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.write(f"**Distribución de {x_axis}:**")
                    counts = data[x_axis].value_counts()
                    st.dataframe(counts)
            
            # Análisis cruzado avanzado
            if len(available_columns) > 1:
                st.subheader("🔍 Análisis Cruzado Avanzado")
                
                col_x = st.selectbox("Variable para filas:", available_columns, key='x_var_cross')
                col_y = st.selectbox("Variable para columnas:", available_columns, key='y_var_cross')
                
                if col_x != col_y:
                    # Tabla de contingencia
                    crosstab = pd.crosstab(data[col_x], data[col_y], normalize='index') * 100
                    
                    # Heatmap interactivo
                    fig = px.imshow(crosstab, 
                                  labels=dict(x=col_y, y=col_x, color="Porcentaje"),
                                  x=crosstab.columns,
                                  y=crosstab.index,
                                  title=f'Relación entre {col_x} y {col_y}',
                                  aspect="auto")
                    fig.update_xaxes(side="bottom")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar tabla numérica
                    with st.expander("📋 Ver tabla numérica detallada"):
                        st.dataframe(crosstab.style.background_gradient(cmap='Blues'))

# Página de Predicción Individual
elif app_mode == "🔮 Predicción Individual":
    st.header("Predicción Individual")
    
    if not api_healthy:
        st.error("""
        ❌ **API no disponible**
        
        No se puede realizar la predicción porque la API no está conectada.
        Verifique:
        1. Que la API esté ejecutándose en: {API_BASE_URL}
        2. Que el modelo esté cargado
        3. La conexión de red
        
        **Solución:** Ejecute `python api_flask.py` en su terminal
        """)
        st.stop()
    
    # Formulario para entrada de datos
    with st.form("prediction_form"):
        st.subheader("📝 Ingrese los datos para la predicción")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Información Personal**")
            genero = st.selectbox("Género *", ["Masculino", "Femenino"])
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
            st.markdown("**📍 Información Geográfica**")
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
        
        with col3:
            st.markdown("**📊 Información de Salud**")
            nivel_sisben = st.selectbox("Nivel Sisbén *", [
                "1", "2", "3", "4", "NO APLICA", "POBLACIÓN CON SISBEN",
                "VÍCTIMAS DEL CONFLICTO ARMADO INTERNO", "MIGRACION"
            ])
            estado_afiliado = st.selectbox("Estado del Afiliado", [
                "Activo", "Inactivo", "Protección Laboral C", "NO APLICA"
            ])
            condicion_beneficiario = st.selectbox("Condición del Beneficiario", [
                "NO APLICA", "ESTUDIANTE", "PENSIONADO", "DISCAPACITADO"
            ])
        
        st.markdown("**📌 Campos obligatorios ***")
        submitted = st.form_submit_button("🎯 Realizar Predicción", type="primary")
        
        if submitted:
            # Validar campos obligatorios
            required_fields = [municipio]
            if not all(required_fields):
                st.error("Por favor complete todos los campos obligatorios (*)")
                st.stop()
            
            # Preparar datos para la API
            input_data = {
                "Genero": genero,
                "Grupo_etario": grupo_etario,
                "Tipo_afiliado": tipo_afiliado,
                "Departamento": departamento,
                "Municipio": municipio,
                "Zona": zona,
                "Nivel_Sisben": nivel_sisben,
                "Estado_afiliado": estado_afiliado,
                "Condicion_beneficiario": condicion_beneficiario
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
                        
                        # Mostrar resultados
                        st.success(f"✅ Predicción completada en {response_time:.0f}ms")
                        
                        # Layout de resultados
                        res_col1, res_col2 = st.columns([2, 1])
                        
                        with res_col1:
                            st.subheader("🎯 Resultado de la Predicción")
                            
                            prediction = result['predictions'][0] if 'predictions' in result else None
                            
                            if prediction is not None:
                                if prediction == 1:
                                    st.markdown('<div class="prediction-high">', unsafe_allow_html=True)
                                    st.error("🔴 **CLASIFICACIÓN: RÉGIMEN SUBSIDIADO**")
                                    st.write("""
                                    **Interpretación:** Este caso presenta características que lo clasifican en el régimen subsidiado.
                                    
                                    **Recomendaciones:**
                                    - Verificar elegibilidad para programas sociales
                                    - Revisar documentación de Sisbén
                                    - Evaluar necesidades específicas de salud
                                    """)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="prediction-low">', unsafe_allow_html=True)
                                    st.success("🟢 **CLASIFICACIÓN: RÉGIMEN CONTRIBUTIVO**")
                                    st.write("""
                                    **Interpretación:** Este caso presenta características que lo clasifican en el régimen contributivo.
                                    
                                    **Características típicas:**
                                    - Afiliación mediante cotizaciones
                                    - Capacidad de pago demostrada
                                    - Situación laboral formal
                                    """)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Mostrar probabilidades si están disponibles
                                if 'probabilities' in result:
                                    probs = result['probabilities'][0]
                                    prob_contributivo = probs[0] * 100
                                    prob_subsidiado = probs[1] * 100
                                    
                                    # Métricas de probabilidad
                                    prob_col1, prob_col2 = st.columns(2)
                                    with prob_col1:
                                        st.metric("Probabilidad Contributivo", f"{prob_contributivo:.1f}%")
                                    with prob_col2:
                                        st.metric("Probabilidad Subsidiado", f"{prob_subsidiado:.1f}%")
                                    
                                    # Gráfico de probabilidades interactivo
                                    fig = go.Figure(data=[
                                        go.Bar(name='Probabilidades', 
                                              x=['Contributivo', 'Subsidiado'], 
                                              y=[prob_contributivo, prob_subsidiado],
                                              marker_color=['#4CAF50', '#F44336'])
                                    ])
                                    fig.update_layout(
                                        title='Probabilidades de Clasificación',
                                        yaxis_title='Probabilidad (%)',
                                        yaxis_range=[0, 100]
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            else:
                                st.warning("⚠️ No se pudo obtener una predicción válida")
                        
                        with res_col2:
                            st.subheader("📋 Datos Ingresados")
                            
                            # Mostrar datos en formato más legible
                            for key, value in input_data.items():
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                            
                            # Opción para guardar la predicción
                            if st.button("💾 Guardar Predicción en Sesión"):
                                if 'saved_predictions' not in st.session_state:
                                    st.session_state.saved_predictions = []
                                
                                saved_pred = {
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'input': input_data,
                                    'prediction': prediction,
                                    'probabilities': result.get('probabilities', [None])[0] if 'probabilities' in result else None
                                }
                                st.session_state.saved_predictions.append(saved_pred)
                                st.success("✅ Predicción guardada en sesión actual")
                    
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

# Página de Predicción por Lotes (manteniendo la estructura pero optimizada)
elif app_mode == "📁 Predicción por Lotes":
    st.header("Predicción por Lotes")
    
    if not api_healthy:
        st.error("La API no está disponible. Por favor, verifique la conexión.")
        st.stop()
    
    st.info("""
    **📋 Instrucciones para Predicción por Lotes:**
    
    1. **Preparar datos**: Su archivo CSV debe contener las columnas requeridas por el modelo
    2. **Formato**: Asegúrese de que los datos estén en el formato correcto
    3. **Tamaño**: Archivos hasta 200MB (dependiendo de su configuración de Streamlit)
    4. **Procesamiento**: Las predicciones se realizarán en lote y podrá descargar los resultados
    """)
    
    # Plantilla de datos mejorada
    with st.expander("📥 Descargar Plantilla de Datos", expanded=True):
        template_data = pd.DataFrame({
            'Genero': ['Masculino', 'Femenino', 'Masculino'],
            'Grupo_etario': ['19 a 45', '45 a 50', '60 a 65'],
            'Tipo_afiliado': ['COTIZANTE', 'BENEFICIARIO', 'CABEZA DE FAMILIA'],
            'Departamento': ['BOGOTA D.C.', 'ANTIOQUIA', 'VALLE'],
            'Municipio': ['BOGOTA', 'MEDELLIN', 'CALI'],
            'Zona': ['Urbana', 'Urbana', 'Rural'],
            'Nivel_Sisben': ['1', '2', 'NO APLICA'],
            'Estado_afiliado': ['Activo', 'Activo', 'Inactivo'],
            'Condicion_beneficiario': ['NO APLICA', 'ESTUDIANTE', 'NO APLICA']
        })
        
        csv = template_data.to_csv(index=False)
        st.download_button(
            label="📋 Descargar Plantilla CSV Completa",
            data=csv,
            file_name="plantilla_datos_modelo_salud.csv",
            mime="text/csv",
            help="Use esta plantilla como referencia para preparar sus datos"
        )
    
    uploaded_file = st.file_uploader(
        "📤 Seleccione archivo CSV para predicción por lotes", 
        type="csv",
        help="Suba un archivo CSV con los datos para predicción"
    )
    
    if uploaded_file is not None:
        try:
            # Leer el archivo
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
            
            # Mostrar información del archivo
            st.subheader("📊 Información del Archivo")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📈 Registros", f"{len(batch_data):,}")
            
            with col2:
                st.metric("📊 Columnas", len(batch_data.columns))
            
            with col3:
                file_size = uploaded_file.size / 1024  # KB
                st.metric("💾 Tamaño", f"{file_size:.1f} KB")
            
            with col4:
                null_count = batch_data.isnull().sum().sum()
                st.metric("⚠️ Valores Nulos", null_count)
            
            # Mostrar vista previa con pestañas
            tab1, tab2, tab3 = st.tabs(["👀 Vista Previa", "🔍 Estructura", "📝 Muestra Aleatoria"])
            
            with tab1:
                st.dataframe(batch_data.head(10), use_container_width=True)
            
            with tab2:
                st.write("**Tipos de datos:**")
                dtype_info = batch_data.dtypes.reset_index()
                dtype_info.columns = ['Columna', 'Tipo de Dato']
                st.dataframe(dtype_info, use_container_width=True)
            
            with tab3:
                sample_size = min(10, len(batch_data))
                st.dataframe(batch_data.sample(sample_size), use_container_width=True)
            
            # Validar datos antes de procesar
            st.subheader("🔍 Validación de Datos")
            
            required_columns = ['Genero', 'Grupo_etario', 'Tipo_afiliado', 'Departamento', 'Municipio', 'Zona', 'Nivel_Sisben']
            missing_columns = [col for col in required_columns if col not in batch_data.columns]
            
            if missing_columns:
                st.error(f"❌ Faltan columnas requeridas: {', '.join(missing_columns)}")
                st.info("""
                **Columnas requeridas:**
                - Genero
                - Grupo_etario  
                - Tipo_afiliado
                - Departamento
                - Municipio
                - Zona
                - Nivel_Sisben
                """)
            else:
                st.success("✅ Todas las columnas requeridas están presentes")
                
                # Mostrar resumen de datos por columna
                st.write("**Resumen por columna:**")
                for col in required_columns:
                    unique_vals = batch_data[col].nunique()
                    sample_vals = batch_data[col].dropna().head(3).tolist()
                    st.write(f"- **{col}**: {unique_vals} valores únicos (ej: {', '.join(map(str, sample_vals))})")
            
            # Procesar predicción
            if st.button("🚀 Ejecutar Predicción por Lotes", type="primary", disabled=bool(missing_columns)):
                with st.spinner(f"📊 Procesando {len(batch_data):,} registros..."):
                    try:
                        # Convertir a formato JSON para la API
                        records = batch_data.to_dict('records')
                        
                        # Barra de progreso
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        start_time = time.time()
                        response = requests.post(
                            f"{API_BASE_URL}/batch_predict",
                            json={"records": records},
                            timeout=120  # Mayor timeout para lotes grandes
                        )
                        processing_time = (time.time() - start_time)
                        
                        progress_bar.progress(100
