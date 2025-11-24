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
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🏥 Sistema de Clasificación - Modelo de Salud Colombia</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("Navegación")
app_mode = st.sidebar.selectbox(
    "Seleccione una opción:",
    ["🏠 Inicio", "📊 Análisis Exploratorio", "🔮 Predicción Individual", "📁 Predicción por Lotes", "📈 Resultados", "ℹ️ Acerca del Modelo"]
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

if api_healthy:
    st.sidebar.success("✅ API Conectada")
    st.sidebar.metric("Tiempo Respuesta", f"{response_time:.0f} ms")
    if api_status.get('model_loaded'):
        st.sidebar.success("✅ Modelo Cargado")
    else:
        st.sidebar.error("❌ Modelo No Cargado")
else:
    st.sidebar.error("❌ API No Disponible")
    st.sidebar.error(f"Error: {api_status.get('error', 'Desconocido')}")

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
        - Complete el formulario
        - Obtenga resultados inmediatos
        - Vea el nivel de confianza
        """)
    
    with guide_col2:
        st.markdown("""
        **📁 Predicción por Lotes**
        - Suba archivo CSV
        - Procese múltiples registros
        - Descargue resultados
        """)
    
    with guide_col3:
        st.markdown("""
        **📈 Análisis de Resultados**
        - Visualice distribuciones
        - Analice por características
        - Exporte reportes
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
            with st.spinner("Generando datos de ejemplo..."):
                # Simular datos más realistas basados en la estructura del notebook
                np.random.seed(42)
                n_samples = 2000
                
                # Crear datos balanceados entre contributivo y subsidiado
                regimen_choices = ['Contributivo', 'Subsidiado']
                regimen_probs = [0.4, 0.6]  # Más subsidiado como en los datos reales
                
                sample_data = pd.DataFrame({
                    'Genero': np.random.choice(['Masculino', 'Femenino'], n_samples, p=[0.48, 0.52]),
                    'Grupo_etario': np.random.choice([
                        '15 a 19', '19 a 45', '45 a 50', '50 a 55', 
                        '55 a 60', '60 a 65', '65 a 70', '70 a 75', '> 75'
                    ], n_samples, p=[0.1, 0.25, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.07]),
                    'Régimen': np.random.choice(regimen_choices, n_samples, p=regimen_probs),
                    'Tipo_afiliado': np.random.choice([
                        'COTIZANTE', 'BENEFICIARIO', 'CABEZA DE FAMILIA', 'ADICIONAL'
                    ], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
                    'Departamento': np.random.choice([
                        'BOGOTA D.C.', 'ANTIOQUIA', 'VALLE', 'CUNDINAMARCA', 
                        'ATLANTICO', 'SANTANDER', 'BOLIVAR', 'NARIÑO'
                    ], n_samples),
                    'Zona': np.random.choice([
                        'Urbana', 'Rural', 'Urbana-Cabecera Municipal'
                    ], n_samples, p=[0.6, 0.25, 0.15]),
                    'Nivel_Sisben': np.random.choice([
                        '1', '2', '3', '4', 'NO APLICA'
                    ], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
                })
                
                st.session_state.sample_data = sample_data
                st.success(f"✅ Se generaron {n_samples} registros de ejemplo!")
    
    else:  # Subir Datos Propios
        uploaded_file = st.file_uploader("📤 Subir archivo CSV", type="csv")
        if uploaded_file is not None:
            try:
                sample_data = pd.read_csv(uploaded_file)
                st.session_state.sample_data = sample_data
                st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error al cargar el archivo: {str(e)}")
    
    # Mostrar análisis si hay datos
    if 'sample_data' in st.session_state:
        data = st.session_state.sample_data
        
        # Mostrar datos
        st.subheader("📋 Vista Previa de Datos")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Estadísticas básicas
        st.subheader("📊 Estadísticas Descriptivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Información General**")
            st.write(f"Registros totales: {len(data):,}")
            st.write(f"Variables: {len(data.columns)}")
            st.write(f"Memoria usada: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with col2:
            st.write("**Tipos de Datos**")
            dtype_counts = data.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count}")
        
        # Visualizaciones
        st.subheader("📈 Visualizaciones")
        
        # Seleccionar variables para visualizar
        available_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if available_columns:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                chart_type = st.selectbox(
                    "Tipo de gráfico:",
                    ["Barras", "Torta", "Conteo"]
                )
                
                x_axis = st.selectbox(
                    "Variable para análisis:",
                    available_columns
                )
            
            with viz_col2:
                if chart_type == "Barras":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    data[x_axis].value_counts().plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title(f'Distribución de {x_axis}')
                    ax.set_ylabel('Frecuencia')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                elif chart_type == "Torta":
                    fig, ax = plt.subplots(figsize=(8, 8))
                    counts = data[x_axis].value_counts()
                    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
                    ax.set_title(f'Distribución de {x_axis}')
                    st.pyplot(fig)
                
                else:  # Conteo
                    st.write(f"**Distribución de {x_axis}:**")
                    counts = data[x_axis].value_counts()
                    st.dataframe(counts)
            
            # Análisis cruzado
            if len(available_columns) > 1:
                st.subheader("🔍 Análisis Cruzado")
                
                col_x = st.selectbox("Variable X:", available_columns, key='x_var')
                col_y = st.selectbox("Variable Y:", available_columns, key='y_var')
                
                if col_x != col_y:
                    crosstab = pd.crosstab(data[col_x], data[col_y], normalize='index') * 100
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    crosstab.plot(kind='bar', ax=ax, stacked=True)
                    ax.set_title(f'Relación entre {col_x} y {col_y}')
                    ax.set_ylabel('Porcentaje (%)')
                    plt.xticks(rotation=45)
                    plt.legend(title=col_y, bbox_to_anchor=(1.05, 1), loc='upper left')
                    st.pyplot(fig)

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
        """)
        st.stop()
    
    # Formulario para entrada de datos
    with st.form("prediction_form"):
        st.subheader("📝 Ingrese los datos para la predicción")
        
        col1, col2 = st.columns(2)
        
        with col1:
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
                        
                        # Mostrar resultados
                        st.success(f"✅ Predicción completada en {response_time:.0f}ms")
                        
                        # Layout de resultados
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.subheader("🎯 Resultado de la Predicción")
                            
                            prediction = result['predictions'][0] if 'predictions' in result else None
                            
                            if prediction is not None:
                                if prediction == 1:
                                    st.markdown('<div class="prediction-high">', unsafe_allow_html=True)
                                    st.error("🔴 **CLASIFICACIÓN: ALTO RIESGO**")
                                    st.write("Este caso requiere atención prioritaria y seguimiento cercano.")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="prediction-low">', unsafe_allow_html=True)
                                    st.success("🟢 **CLASIFICACIÓN: BAJO RIESGO**")
                                    st.write("Este caso se encuentra dentro de los parámetros normales.")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Mostrar probabilidades si están disponibles
                                if 'probabilities' in result:
                                    probs = result['probabilities'][0]
                                    prob_low = probs[0] * 100
                                    prob_high = probs[1] * 100
                                    
                                    st.metric("Probabilidad Bajo Riesgo", f"{prob_low:.1f}%")
                                    st.metric("Probabilidad Alto Riesgo", f"{prob_high:.1f}%")
                                    
                                    # Gráfico de probabilidades
                                    fig, ax = plt.subplots(figsize=(8, 3))
                                    bars = ax.bar(['Bajo Riesgo', 'Alto Riesgo'], [prob_low, prob_high], 
                                                 color=['#4CAF50', '#F44336'])
                                    ax.set_ylabel('Probabilidad (%)')
                                    ax.set_ylim(0, 100)
                                    
                                    # Agregar valores en las barras
                                    for bar, value in zip(bars, [prob_low, prob_high]):
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                                f'{value:.1f}%', ha='center', va='bottom')
                                    
                                    st.pyplot(fig)
                            
                            else:
                                st.warning("No se pudo obtener una predicción válida")
                        
                        with res_col2:
                            st.subheader("📋 Datos Ingresados")
                            st.json(input_data)
                            
                            # Opción para guardar la predicción
                            if st.button("💾 Guardar Predicción"):
                                if 'saved_predictions' not in st.session_state:
                                    st.session_state.saved_predictions = []
                                
                                saved_pred = {
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'input': input_data,
                                    'prediction': prediction,
                                    'probabilities': result.get('probabilities', [None])[0] if 'probabilities' in result else None
                                }
                                st.session_state.saved_predictions.append(saved_pred)
                                st.success("Predicción guardada en sesión")
                    
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

# Página de Predicción por Lotes
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
    
    # Plantilla de datos
    with st.expander("📥 Descargar Plantilla de Datos"):
        template_data = pd.DataFrame({
            'Genero': ['Masculino', 'Femenino'],
            'Grupo_etario': ['19 a 45', '45 a 50'],
            'Tipo_afiliado': ['COTIZANTE', 'BENEFICIARIO'],
            'Departamento': ['BOGOTA D.C.', 'ANTIOQUIA'],
            'Municipio': ['BOGOTA', 'MEDELLIN'],
            'Zona': ['Urbana', 'Urbana'],
            'Nivel_Sisben': ['1', '2']
        })
        
        csv = template_data.to_csv(index=False)
        st.download_button(
            label="📋 Descargar Plantilla CSV",
            data=csv,
            file_name="plantilla_datos_modelo.csv",
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
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Registros", f"{len(batch_data):,}")
            
            with col2:
                st.metric("Columnas", len(batch_data.columns))
            
            with col3:
                file_size = uploaded_file.size / 1024  # KB
                st.metric("Tamaño", f"{file_size:.1f} KB")
            
            # Mostrar vista previa
            st.subheader("👀 Vista Previa del Archivo")
            st.dataframe(batch_data.head(10), use_container_width=True)
            
            # Validar datos antes de procesar
            st.subheader("🔍 Validación de Datos")
            
            required_columns = ['Genero', 'Grupo_etario', 'Tipo_afiliado', 'Departamento', 'Municipio', 'Zona', 'Nivel_Sisben']
            missing_columns = [col for col in required_columns if col not in batch_data.columns]
            
            if missing_columns:
                st.error(f"❌ Faltan columnas requeridas: {', '.join(missing_columns)}")
                st.info("Por favor, asegúrese de que su archivo contenga todas las columnas necesarias")
            else:
                st.success("✅ Todas las columnas requeridas están presentes")
                
                # Mostrar resumen de datos
                st.write("**Resumen por columna:**")
                for col in required_columns:
                    unique_vals = batch_data[col].nunique()
                    sample_vals = batch_data[col].head(3).tolist()
                    st.write(f"- **{col}**: {unique_vals} valores únicos (ej: {', '.join(map(str, sample_vals))})")
            
            # Procesar predicción
            if st.button("🚀 Ejecutar Predicción por Lotes", type="primary", disabled=bool(missing_columns)):
                with st.spinner(f"📊 Procesando {len(batch_data):,} registros..."):
                    try:
                        # Convertir a formato JSON para la API - CORREGIDO
                        records = batch_data.to_dict('records')
                        
                        start_time = time.time()
                        response = requests.post(
                            f"{API_BASE_URL}/batch_predict",
                            json={"records": records},  # CORREGIDO: Cambiado de "data" a "records"
                            timeout=60
                        )
                        processing_time = (time.time() - start_time)
                        
                        if response.status_code == 200:
                            results = response.json()
                            predictions = results.get('predictions', [])
                            
                            # Agregar predicciones al DataFrame
                            batch_data['prediccion'] = predictions
                            batch_data['prediccion_texto'] = batch_data['prediccion'].map({0: 'Bajo Riesgo', 1: 'Alto Riesgo'})
                            
                            # Agregar timestamp
                            batch_data['fecha_procesamiento'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            st.success(f"✅ {len(predictions):,} predicciones completadas en {processing_time:.1f} segundos")
                            
                            # Mostrar resumen
                            st.subheader("📈 Resumen de Predicciones")
                            
                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                            
                            total = len(batch_data)
                            alto_riesgo = batch_data['prediccion'].sum()
                            bajo_riesgo = total - alto_riesgo
                            tasa_alto_riesgo = (alto_riesgo / total) * 100
                            
                            with summary_col1:
                                st.metric("Total Procesado", f"{total:,}")
                            
                            with summary_col2:
                                st.metric("Alto Riesgo", f"{alto_riesgo:,}")
                            
                            with summary_col3:
                                st.metric("Bajo Riesgo", f"{bajo_riesgo:,}")
                            
                            with summary_col4:
                                st.metric("Tasa Alto Riesgo", f"{tasa_alto_riesgo:.1f}%")
                            
                            # Visualización rápida
                            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Gráfico de torta
                            labels = ['Bajo Riesgo', 'Alto Riesgo']
                            sizes = [bajo_riesgo, alto_riesgo]
                            colors = ['#4CAF50', '#F44336']
                            
                            ax[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                            ax[0].set_title('Distribución de Predicciones')
                            
                            # Gráfico de barras
                            ax[1].bar(labels, sizes, color=colors)
                            ax[1].set_title('Conteo de Predicciones')
                            ax[1].set_ylabel('Número de Registros')
                            
                            for i, v in enumerate(sizes):
                                ax[1].text(i, v + max(sizes)*0.01, f'{v:,}', ha='center')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Mostrar resultados detallados
                            st.subheader("📋 Resultados Detallados")
                            st.dataframe(batch_data, use_container_width=True)
                            
                            # Descargar resultados
                            st.subheader("💾 Descargar Resultados")
                            
                            output_col1, output_col2 = st.columns(2)
                            
                            with output_col1:
                                # CSV
                                csv = batch_data.to_csv(index=False)
                                st.download_button(
                                    label="📥 Descargar CSV Completo",
                                    data=csv,
                                    file_name=f"resultados_prediccion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with output_col2:
                                # Solo alto riesgo
                                alto_riesgo_data = batch_data[batch_data['prediccion'] == 1]
                                if not alto_riesgo_data.empty:
                                    csv_alto = alto_riesgo_data.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Descargar Solo Alto Riesgo",
                                        data=csv_alto,
                                        file_name=f"alto_riesgo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                            
                            # Guardar en session state para análisis
                            st.session_state.batch_results = batch_data
                            st.session_state.last_batch_file = uploaded_file.name
                            
                            # Resumen ejecutivo
                            with st.expander("📊 Resumen Ejecutivo"):
                                st.write(f"""
                                **Resumen del Procesamiento por Lotes**
                                
                                - **Archivo procesado**: {uploaded_file.name}
                                - **Total de registros**: {total:,}
                                - **Registros de alto riesgo**: {alto_riesgo:,} ({tasa_alto_riesgo:.1f}%)
                                - **Registros de bajo riesgo**: {bajo_riesgo:,} ({(100-tasa_alto_riesgo):.1f}%)
                                - **Tiempo de procesamiento**: {processing_time:.1f} segundos
                                - **Fecha y hora**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                """)
                        
                        else:
                            st.error(f"❌ Error en la API: {response.status_code}")
                            try:
                                error_detail = response.json()
                                st.write("Detalles del error:", error_detail)
                            except:
                                st.write("Respuesta:", response.text)
                    
                    except requests.exceptions.Timeout:
                        st.error("⏰ Timeout - La API no respondió en 60 segundos")
                    except Exception as e:
                        st.error(f"❌ Error procesando el lote: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error leyendo el archivo: {str(e)}")
            st.info("Asegúrese de que el archivo sea un CSV válido y esté correctamente formateado")

# Página de Resultados (mantiene la misma estructura pero mejorada)
elif app_mode == "📈 Resultados":
    st.header("Análisis de Resultados")
    
    if 'batch_results' not in st.session_state:
        st.info("""
        ℹ️ **No hay resultados de predicción disponibles**
        
        Para ver análisis de resultados:
        1. Vaya a **📁 Predicción por Lotes**
        2. Suba un archivo CSV y procese las predicciones
        3. Los resultados estarán disponibles para análisis en esta sección
        """)
        st.stop()
    
    results = st.session_state.batch_results
    
    # Estadísticas de resultados
    st.subheader("📊 Estadísticas de Predicciones")
    
    total = len(results)
    alto_riesgo = results['prediccion'].sum() if results['prediccion'].dtype != 'object' else len(results[results['prediccion'] == 1])
    bajo_riesgo = total - alto_riesgo
    tasa_alto_riesgo = (alto_riesgo / total) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predicciones", f"{total:,}")
    
    with col2:
        st.metric("Alto Riesgo", f"{alto_riesgo:,}")
    
    with col3:
        st.metric("Bajo Riesgo", f"{bajo_riesgo:,}")
    
    with col4:
        st.metric("Tasa Alto Riesgo", f"{tasa_alto_riesgo:.1f}%")
    
    # Visualizaciones mejoradas
    st.subheader("📈 Visualizaciones de Resultados")
    
    viz_type = st.selectbox(
        "Tipo de visualización:",
        ["Distribución General", "Análisis por Característica", "Comparativa Detallada"]
    )
    
    if viz_type == "Distribución General":
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de torta mejorado
            fig, ax = plt.subplots(figsize=(8, 8))
            labels = ['Bajo Riesgo', 'Alto Riesgo']
            sizes = [bajo_riesgo, alto_riesgo]
            colors = ['#4CAF50', '#F44336']
            explode = (0.05, 0.05)  # resaltar las porciones
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                  shadow=True, startangle=90)
            ax.axis('equal')
            ax.set_title('Distribución de Predicciones', fontsize=16, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            # Gráfico de barras horizontal
            fig, ax = plt.subplots(figsize=(10, 4))
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, sizes, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Número de Registros')
            ax.set_title('Conteo de Predicciones por Categoría')
            
            # Agregar valores en las barras
            for i, v in enumerate(sizes):
                ax.text(v + max(sizes)*0.01, i, f'{v:,}', va='center')
            
            st.pyplot(fig)
    
    elif viz_type == "Análisis por Característica":
        # Seleccionar característica para análisis
        available_features = [col for col in results.columns if col not in ['prediccion', 'prediccion_texto', 'fecha_procesamiento']]
        
        if available_features:
            selected_feature = st.selectbox("Seleccione característica para análisis:", available_features)
            
            # Crosstab mejorado
            crosstab = pd.crosstab(results[selected_feature], results['prediccion'], normalize='index') * 100
            
            fig, ax = plt.subplots(figsize=(12, 8))
            crosstab.plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336'])
            ax.set_ylabel('Porcentaje (%)')
            ax.set_title(f'Distribución de Predicciones por {selected_feature}', fontweight='bold')
            ax.legend(['Bajo Riesgo', 'Alto Riesgo'])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabla detallada
            st.subheader("📋 Tabla de Distribución")
            count_table = pd.crosstab(results[selected_feature], results['prediccion'])
            count_table['Total'] = count_table.sum(axis=1)
            count_table['% Alto Riesgo'] = (count_table[1] / count_table['Total'] * 100).round(1)
            st.dataframe(count_table.style.background_gradient(subset=['% Alto Riesgo'], cmap='Reds'))
    
    else:  # Comparativa Detallada
        st.subheader("🔍 Análisis Comparativo Detallado")
        
        # Seleccionar dos características para comparar
        available_features = [col for col in results.columns if col not in ['prediccion', 'prediccion_texto', 'fecha_procesamiento']]
        
        if len(available_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                feature1 = st.selectbox("Primera característica:", available_features, key='feat1')
            
            with col2:
                feature2 = st.selectbox("Segunda característica:", available_features, key='feat2')
            
            if feature1 != feature2:
                # Heatmap de correlación
                pivot_table = results.pivot_table(
                    index=feature1, 
                    columns=feature2, 
                    values='prediccion', 
                    aggfunc='mean'
                ) * 100
                
                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(pivot_table.values, cmap='Reds', aspect='auto')
                
                # Etiquetas
                ax.set_xticks(np.arange(len(pivot_table.columns)))
                ax.set_yticks(np.arange(len(pivot_table.index)))
                ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
                ax.set_yticklabels(pivot_table.index)
                ax.set_xlabel(feature2)
                ax.set_ylabel(feature1)
                ax.set_title(f'Porcentaje de Alto Riesgo por {feature1} y {feature2}')
                
                # Barra de color
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Porcentaje de Alto Riesgo (%)')
                
                # Texto en las celdas
                for i in range(len(pivot_table.index)):
                    for j in range(len(pivot_table.columns)):
                        text = ax.text(j, i, f'{pivot_table.iloc[i, j]:.1f}%',
                                      ha="center", va="center", color="black", fontweight='bold')
                
                st.pyplot(fig)

# Página Acerca del Modelo (mejorada)
elif app_mode == "ℹ️ Acerca del Modelo":
    st.header("ℹ️ Información del Modelo")
    
    st.subheader("🎯 Descripción del Sistema")
    st.markdown("""
    Este sistema de clasificación utiliza **machine learning** para analizar datos del sistema de salud colombiano
    y realizar predicciones basadas en características demográficas y de afiliación.
    
    ### Objetivos Principales
    - 🔍 **Identificar** patrones en los datos de afiliación al sistema de salud
    - 📈 **Clasificar** casos según nivel de riesgo
    - 🎯 **Optimizar** la asignación de recursos en salud
    - 📊 **Proporcionar** insights para la toma de decisiones
    """)
    
    # Características técnicas en pestañas
    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Arquitectura", "📈 Métricas", "🔧 Tecnologías", "👥 Desarrollo"])
    
    with tab1:
        st.markdown("""
        ### Arquitectura del Sistema
        
        ```mermaid
        graph TD
            A[Streamlit UI] --> B[Flask/FastAPI];
            B --> C[Modelo ML];
            C --> D[Base de Datos];
            B --> E[Resultados];
            E --> A;
        ```
        
        **Componentes principales:**
        - **Frontend**: Interfaz Streamlit para interacción con usuarios
        - **Backend**: API REST con Flask/FastAPI para procesamiento
        - **ML Engine**: Modelos de Scikit-learn/XGBoost
        - **Data Layer**: Bases de datos BDUA y EPSS
        """)
    
    with tab2:
        st.markdown("""
        ### Métricas de Rendimiento
        
        | Métrica | Valor | Descripción |
        |---------|-------|-------------|
        | Precisión | > 85% | Exactitud general del modelo |
        | Recall | > 80% | Capacidad de detectar casos positivos |
        | F1-Score | > 82% | Balance entre precisión y recall |
        | AUC-ROC | > 0.88 | Capacidad de discriminación |
        | Tiempo Inferencia | < 100ms | Velocidad de predicción |
        
        **Nota**: Las métricas pueden variar según los datos de entrada y configuración del modelo.
        """)
    
    with tab3:
        st.markdown("""
        ### Stack Tecnológico
        
        **Machine Learning & Data Science**
        - Scikit-learn
        - XGBoost
        - Pandas / NumPy
        - Joblib (serialización)
        
        **Backend & API**
        - Flask / FastAPI
        - REST API design
        - JSON serialization
        
        **Frontend & UI**
        - Streamlit
        - Matplotlib / Seaborn
        - Plotly (visualizaciones)
        
        **Despliegue & Infraestructura**
        - Docker (containerización)
        - Streamlit Sharing
        - Python 3.8+
        """)
    
    with tab4:
        st.markdown("""
        ### Información de Desarrollo
        
        **Equipo de Desarrollo**
        - **Líder de Proyecto**: Equipo de Ciencia de Datos
        - **Desarrolladores Backend**: Especialistas en ML y APIs
        - **Desarrolladores Frontend**: Especialistas en UI/UX
        - **Analistas de Datos**: Expertos en dominio de salud
        
        **Detalles del Proyecto**
        - **Versión**: 1.0.0
        - **Última actualización**: Julio 2025
        - **Licencia**: Creative Commons Attribution Share Alike 4.0 International
        - **Repositorio**: [Enlace al repositorio]()
        
        **Ciclo de Desarrollo**
        - 📋 **Análisis de Requerimientos**
        - 🏗️ **Diseño de Arquitectura**
        - 🔧 **Desarrollo del Modelo**
        - 🧪 **Validación y Testing**
        - 🚀 **Despliegue y Monitoreo**
        """)
    
    # Endpoints de la API
    st.subheader("🌐 Endpoints de la API")
    
    if api_healthy:
        try:
            response = requests.get(f"{API_BASE_URL}")
            endpoints = response.json().get('endpoints', {})
            
            for endpoint, description in endpoints.items():
                st.code(f"{endpoint}: {description}", language='http')
        except:
            st.info("No se pudieron cargar los endpoints de la API")
    else:
        st.info("La API no está disponible para mostrar endpoints")
        
        # Mostrar endpoints esperados
        st.markdown("""
        **Endpoints esperados cuando la API esté disponible:**
        ```
        GET  /          - Información general de la API
        GET  /health    - Estado del servicio y modelo
        POST /predict   - Predicciones individuales
        POST /batch_predict - Predicciones por lotes
        ```
        """)

# Footer mejorado
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown(
        "**Sistema de Clasificación - Modelo de Salud Colombia** | "
        "Desarrollado con Streamlit 🚀"
    )

with footer_col2:
    if api_healthy:
        st.markdown(f"🟢 API: {API_BASE_URL}")
    else:
        st.markdown(f"🔴 API: {API_BASE_URL}")

with footer_col3:
    st.markdown(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")