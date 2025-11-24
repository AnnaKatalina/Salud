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
import base64
from PIL import Image

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
    .projection-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
    }
    .metric-low {
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
    ["🏠 Inicio", "🔮 Predicción Individual", "📁 Predicción por Lotes", "📈 Proyecciones Futuras", "📊 Análisis de Resultados", "⚙️ Configuración"]
)

# Configuración API
API_BASE_URL = st.sidebar.text_input("URL de la API:", "http://localhost:5000")

# Función para verificar estado de la API
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return True, data, response.elapsed.total_seconds() * 1000
        else:
            return False, {"error": f"Error {response.status_code}"}, 0
    except Exception as e:
        return False, {"error": str(e)}, 0

# Verificar estado
api_healthy, api_status, response_time = check_api_health()

# Mostrar estado en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Estado del Sistema")

if api_healthy:
    st.sidebar.success("✅ API Conectada")
    st.sidebar.metric("Tiempo Respuesta", f"{response_time:.0f} ms")
    if api_status.get('model_loaded'):
        st.sidebar.success("✅ Modelo Cargado")
        st.sidebar.info(f"Tipo: {api_status.get('model_type', 'N/A')}")
    else:
        st.sidebar.error("❌ Modelo No Cargado")
else:
    st.sidebar.error("❌ API No Disponible")
    st.sidebar.error(f"Error: {api_status.get('error', 'Desconocido')}")

# Página de Proyecciones Futuras
if app_mode == "📈 Proyecciones Futuras":
    st.header("📈 Proyecciones Futuras del Sistema de Salud")
    
    if not api_healthy:
        st.error("La API no está disponible para generar proyecciones.")
        st.stop()
    
    # Obtener proyecciones
    with st.spinner("🔄 Cargando proyecciones futuras..."):
        try:
            response = requests.get(f"{API_BASE_URL}/projections", timeout=30)
            
            if response.status_code == 200:
                projections_data = response.json()
                
                # Mostrar gráfico de proyecciones
                if 'plot' in projections_data:
                    plot_data = projections_data['plot'].split(',')[1]  # Remover 'data:image/png;base64,'
                    image = Image.open(io.BytesIO(base64.b64decode(plot_data)))
                    st.image(image, use_column_width=True, caption="Proyecciones de Mortalidad y Probabilidad de Subsidiado")
                
                # Mostrar datos tabulares
                st.subheader("📊 Datos de Proyección")
                projections_df = pd.DataFrame(projections_data['projections'])
                
                # Formatear columnas
                projections_df['Tasa_mortalidad_pred'] = projections_df['Tasa_mortalidad_pred'].round(3)
                projections_df['Prob_Subsidiado'] = (projections_df['Prob_Subsidiado'] * 100).round(2)
                projections_df['Prob_Contributivo'] = (projections_df['Prob_Contributivo'] * 100).round(2)
                
                # Mostrar métricas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_year = datetime.now().year
                    st.metric("Año Actual", current_year)
                
                with col2:
                    latest_mortality = projections_df['Tasa_mortalidad_pred'].iloc[0]
                    st.metric("Tasa Mortalidad 2024", f"{latest_mortality}‰")
                
                with col3:
                    prob_subsidiado_2024 = projections_df['Prob_Subsidiado'].iloc[0]
                    st.metric("Prob Subsidiado 2024", f"{prob_subsidiado_2024}%")
                
                with col4:
                    prob_contributivo_2024 = projections_df['Prob_Contributivo'].iloc[0]
                    st.metric("Prob Contributivo 2024", f"{prob_contributivo_2024}%")
                
                # Mostrar tabla detallada
                st.dataframe(projections_df.style.format({
                    'Tasa_mortalidad_pred': '{:.3f}',
                    'Prob_Subsidiado': '{:.2f}%',
                    'Prob_Contributivo': '{:.2f}%'
                }), use_container_width=True)
                
                # Análisis de tendencias
                st.subheader("📈 Análisis de Tendencias")
                
                trend_col1, trend_col2 = st.columns(2)
                
                with trend_col1:
                    mortality_trend = projections_df['Tasa_mortalidad_pred'].pct_change().iloc[1:].mean() * 100
                    trend_icon = "📈" if mortality_trend > 0 else "📉"
                    st.metric(
                        "Tendencia Mortalidad", 
                        f"{mortality_trend:+.2f}%", 
                        delta=f"{mortality_trend:+.2f}%",
                        delta_color="inverse"
                    )
                
                with trend_col2:
                    subsidio_trend = projections_df['Prob_Subsidiado'].pct_change().iloc[1:].mean()
                    st.metric(
                        "Tendencia Subsidiado", 
                        f"{subsidio_trend:+.2f}%", 
                        delta=f"{subsidio_trend:+.2f}%"
                    )
                
                # Recomendaciones basadas en proyecciones
                st.subheader("💡 Recomendaciones Estratégicas")
                
                latest_probs = projections_df.iloc[-1]
                if latest_probs['Prob_Subsidiado'] > 60:
                    st.warning("""
                    **Alerta:** Alta probabilidad de crecimiento en régimen subsidiado.
                    **Recomendaciones:**
                    - Fortalecer programas de prevención en zonas rurales
                    - Aumentar capacidad de atención primaria
                    - Revisar asignación presupuestal
                    """)
                else:
                    st.success("""
                    **Situación estable:** Las proyecciones indican un equilibrio entre regímenes.
                    **Recomendaciones:**
                    - Mantener programas actuales
                    - Monitorear tendencias trimestralmente
                    - Fortalecer la atención preventiva
                    """)
                
                # Descargar proyecciones
                st.subheader("💾 Descargar Proyecciones")
                
                csv = projections_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Proyecciones CSV",
                    data=csv,
                    file_name=f"proyecciones_salud_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.error(f"Error obteniendo proyecciones: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error cargando proyecciones: {str(e)}")
    
    # Datos de mortalidad histórica
    with st.expander("📊 Ver Datos de Mortalidad Histórica"):
        try:
            response = requests.get(f"{API_BASE_URL}/mortality", timeout=10)
            if response.status_code == 200:
                mortality_data = response.json()
                historical_df = pd.DataFrame(mortality_data['historical'])
                
                st.write("**Datos de Mortalidad Histórica (2018-2023):**")
                st.dataframe(historical_df, use_container_width=True)
                
                # Mostrar ecuación de regresión
                st.write("**Ecuación de Regresión:**")
                st.latex(f"y = {mortality_data['regression_coef']:.4f}x + {mortality_data['regression_intercept']:.4f}")
                
        except Exception as e:
            st.error(f"Error cargando datos históricos: {e}")

# Página de Predicción Individual Mejorada
elif app_mode == "🔮 Predicción Individual":
    st.header("🔮 Predicción Individual con Modelo Avanzado")
    
    if not api_healthy:
        st.error("La API no está disponible para realizar predicciones.")
        st.stop()
    
    with st.form("advanced_prediction_form"):
        st.subheader("📝 Datos para Predicción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Variables categóricas del modelo
            zona = st.selectbox("Zona de Afiliación *", [
                "Urbana", "Rural", "Urbana-Cabecera Municipal",
                "Rural - Dispersal", "Rural - Resto Rural", "_OTRA_"
            ])
            
            departamento = st.selectbox("Departamento *", [
                "BOGOTA D.C.", "ANTIOQUIA", "VALLE", "CUNDINAMARCA",
                "ATLANTICO", "SANTANDER", "BOLIVAR", "NARIÑO",
                "BOYACA", "CORDOBA", "META", "TOLIMA", "_OTRA_"
            ])
            
            municipio = st.text_input("Municipio *", "BOGOTA")
        
        with col2:
            # Variable numérica clave
            st.markdown("**Variable Epidemiológica**")
            tasa_mortalidad = st.slider(
                "Tasa de Mortalidad (‰) *", 
                min_value=0.0, 
                max_value=15.0, 
                value=5.5, 
                step=0.1,
                help="Tasa de mortalidad por cada 1000 habitantes"
            )
            
            # Información contextual
            st.info("""
            **Referencias de Tasa de Mortalidad:**
            - 🔵 **Baja**: < 5‰
            - 🟡 **Media**: 5-7‰  
            - 🔴 **Alta**: > 7‰
            """)
        
        submitted = st.form_submit_button("🎯 Realizar Predicción Avanzada", type="primary")
        
        if submitted:
            # Preparar datos para el modelo
            input_data = {
                "Tasa_mortalidad": tasa_mortalidad,
                "Zona de Afiliación": zona,
                "Departamento": departamento,
                "Municipio": municipio.upper()
            }
            
            with st.spinner("🔍 Ejecutando modelo de clasificación..."):
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
                        
                        # Mostrar resultados
                        prediction = result['predictions'][0]
                        probabilities = result['probabilities'][0]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🎯 Resultado de Clasificación")
                            
                            if prediction == 1:
                                st.markdown('<div class="metric-high">', unsafe_allow_html=True)
                                st.error("🔴 **ALTO RIESGO - RÉGIMEN SUBSIDIADO**")
                                st.write("""
                                **Características identificadas:**
                                - Perfil epidemiológico complejo
                                - Posible necesidad de intervención
                                - Seguimiento recomendado
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="metric-low">', unsafe_allow_html=True)
                                st.success("🟢 **BAJO RIESGO - RÉGIMEN CONTRIBUTIVO**")
                                st.write("""
                                **Características identificadas:**
                                - Perfil epidemiológico estable
                                - Situación dentro de parámetros normales
                                - Monitoreo rutinario
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.subheader("📊 Probabilidades del Modelo")
                            
                            prob_contributivo = probabilities[0] * 100
                            prob_subsidiado = probabilities[1] * 100
                            
                            # Mostrar métricas
                            st.metric(
                                "Probabilidad Contributivo", 
                                f"{prob_contributivo:.1f}%",
                                delta=f"{(prob_contributivo-50):+.1f}%" if prob_contributivo != 50 else None,
                                delta_color="inverse" if prob_contributivo < 50 else "normal"
                            )
                            
                            st.metric(
                                "Probabilidad Subsidiado", 
                                f"{prob_subsidiado:.1f}%", 
                                delta=f"{(prob_subsidiado-50):+.1f}%" if prob_subsidiado != 50 else None,
                                delta_color="normal" if prob_subsidiado > 50 else "inverse"
                            )
                            
                            # Gráfico de probabilidades
                            fig, ax = plt.subplots(figsize=(8, 3))
                            categories = ['Contributivo', 'Subsidiado']
                            probabilities_pct = [prob_contributivo, prob_subsidiado]
                            colors = ['#4CAF50', '#F44336']
                            
                            bars = ax.bar(categories, probabilities_pct, color=colors, alpha=0.8)
                            ax.set_ylabel('Probabilidad (%)')
                            ax.set_ylim(0, 100)
                            ax.set_title('Distribución de Probabilidades')
                            
                            # Agregar valores en las barras
                            for bar, value in zip(bars, probabilities_pct):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
                            
                            st.pyplot(fig)
                        
                        # Análisis detallado
                        st.subheader("🔍 Análisis Detallado")
                        
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            st.write("**Factores de Influencia:**")
                            if tasa_mortalidad > 7:
                                st.write("📍 **Tasa de mortalidad alta** - Factor de riesgo significativo")
                            elif tasa_mortalidad < 5:
                                st.write("📍 **Tasa de mortalidad baja** - Factor protector")
                            else:
                                st.write("📍 **Tasa de mortalidad media** - Impacto moderado")
                            
                            if "Rural" in zona:
                                st.write("📍 **Zona rural** - Mayor probabilidad de subsidiado")
                            else:
                                st.write("📍 **Zona urbana** - Mayor probabilidad de contributivo")
                        
                        with analysis_col2:
                            st.write("**Datos Utilizados:**")
                            st.json(input_data)
                    
                    else:
                        st.error(f"Error en la API: {response.status_code}")
                        try:
                            error_detail = response.json()
                            st.write("Detalles:", error_detail)
                        except:
                            st.write("Respuesta:", response.text)
                
                except Exception as e:
                    st.error(f"Error en predicción: {str(e)}")

# Página de Inicio Mejorada
elif app_mode == "🏠 Inicio":
    st.header("Bienvenido al Sistema Avanzado de Clasificación")
    
    if not api_healthy:
        st.error("La API del modelo no está disponible. Verifica que esté ejecutándose.")
    else:
        st.success("✅ Sistema operativo con modelo avanzado de machine learning")
    
    # Resumen del sistema con características del modelo
    st.subheader("🎯 Características del Modelo Implementado")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Algoritmo", "Logistic Regression")
    
    with col2:
        st.metric("Variables", "4+")
    
    with col3:
        st.metric("Preprocesamiento", "One-Hot Encoding")
    
    with col4:
        st.metric("Proyecciones", "Hasta 2026")
    
    # Descripción técnica
    st.subheader("🔧 Arquitectura Técnica")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **📈 Modelo de Clasificación:**
        - Logistic Regression con regularización
        - Variables: Tasa mortalidad + categóricas
        - Preprocesamiento: One-Hot Encoding
        - Solver: SAGA para grandes datasets
        """)
    
    with tech_col2:
        st.markdown("""
        **📊 Modelo de Proyección:**
        - Regresión Lineal para mortalidad
        - Proyección años 2024-2026
        - Integración con modelo de clasificación
        - Gráficos automáticos
        """)
    
    # Ejemplo de uso rápido
    st.subheader("🚀 Flujo de Trabajo Recomendado")
    
    steps_col1
