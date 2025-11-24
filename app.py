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

# URL base de la API (modificar según despliegue)
API_BASE_URL = st.sidebar.text_input("URL de la API:", "http://localhost:5000")

# Función para verificar estado de la API
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Error {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

# Página de Inicio
if app_mode == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Clasificación de Salud")

    # Verificar estado de la API
    st.subheader("🔍 Estado del Sistema")
    api_healthy, api_status = check_api_health()

    if api_healthy:
        st.success("✅ API conectada y operacional")
        st.json(api_status)
    else:
        st.error(f"❌ Error conectando con la API: {api_status['error']}")
        st.info("Asegúrese de que la API esté ejecutándose en la URL especificada")

    # Resumen del sistema
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Modelo", "Cargado" if api_healthy and api_status.get('model_loaded') else "No disponible")

    with col2:
        st.metric("Estado", "Operacional" if api_healthy else "Offline")

    with col3:
        st.metric("Endpoints", len(api_status.get('endpoints', {})) if api_healthy else 0)

    # Información sobre los datos
    st.subheader("📋 Bases de Datos Utilizadas")

    with st.expander("Base de Datos - Régimen Contributivo"):
        st.markdown("""
        - **Fuente**: Base de Datos Única de Afiliados (BDUA)
        - **Registros**: ~641,000
        - **Actualización**: Julio 2025
        - **Variables**: Género, grupo etario, régimen, tipo de afiliado, ubicación geográfica, nivel Sisbén
        """)

    with st.expander("Base de Datos - Régimen Subsidiado"):
        st.markdown("""
        - **Fuente**: Entidades Promotoras de Salud (EPSS)
        - **Registros**: ~1,000,000+
        - **Actualización**: Julio 2025
        - **Variables**: Género, grupo etario, tipo de afiliación, zona geográfica, nivel Sisbén, grupo poblacional
        """)

# Página de Análisis Exploratorio
elif app_mode == "📊 Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")

    # Cargar datos de ejemplo (simulado)
    if st.button("Cargar Datos de Ejemplo"):
        with st.spinner("Cargando datos..."):
            # Simular datos basados en la estructura del notebook
            np.random.seed(42)
            n_samples = 1000

            sample_data = pd.DataFrame({
                'Genero': np.random.choice(['Masculino', 'Femenino'], n_samples),
                'Grupo_etario': np.random.choice(['15 a 19', '19 a 45', '45 a 50', '50 a 55', '55 a 60', '60 a 65', '65 a 70', '70 a 75', '> 75'], n_samples),
                'Régimen': np.random.choice(['Contributivo', 'Subsidiado'], n_samples, p=[0.6, 0.4]),
                'Tipo_afiliado': np.random.choice(['COTIZANTE', 'BENEFICIARIO', 'CABEZA DE FAMILIA'], n_samples),
                'Departamento': np.random.choice(['BOGOTA D.C.', 'ANTIOQUIA', 'VALLE', 'CUNDINAMARCA', 'SANTANDER'], n_samples),
                'Zona': np.random.choice(['Urbana', 'Rural', 'Urbana-Cabecera Municipal'], n_samples)
            })

            st.session_state.sample_data = sample_data

        if 'sample_data' in st.session_state:
            st.success("Datos cargados exitosamente!")

    if 'sample_data' in st.session_state:
        data = st.session_state.sample_data

        # Mostrar datos
        st.subheader("Vista Previa de Datos")
        st.dataframe(data.head(10))

        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Distribución por Género**")
            gender_counts = data['Genero'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            st.pyplot(fig)

        with col2:
            st.write("**Distribución por Régimen**")
            regimen_counts = data['Régimen'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(regimen_counts.index, regimen_counts.values)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Más visualizaciones
        st.subheader("Distribución por Grupo Etario")
        age_counts = data['Grupo_etario'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        age_counts.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        plt.title('Distribución por Grupo Etario')
        plt.tight_layout()
        st.pyplot(fig)

# Página de Predicción Individual
elif app_mode == "🔮 Predicción Individual":
    st.header("Predicción Individual")

    if not check_api_health()[0]:
        st.error("La API no está disponible. Por favor, verifique la conexión.")
        st.stop()

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
                    response = requests.post(
                        f"{API_BASE_URL}/predict",
                        json=input_data
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Predicción completada exitosamente!")

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

                        with col2:
                            st.subheader("Datos Ingresados")
                            st.json(input_data)

                    else:
                        st.error(f"Error en la predicción: {response.status_code}")
                        st.json(response.json())

                except Exception as e:
                    st.error(f"Error conectando con la API: {str(e)}")

# Página de Predicción por Lotes
elif app_mode == "📁 Predicción por Lotes":
    st.header("Predicción por Lotes")

    if not check_api_health()[0]:
        st.error("La API no está disponible. Por favor, verifique la conexión.")
        st.stop()

    st.info("""
    **Instrucciones:**
    - Suba un archivo CSV con los datos para predicción
    - El archivo debe tener las columnas requeridas por el modelo
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

                        response = requests.post(
                            f"{API_BASE_URL}/batch_predict",
                            json={"data": records}
                        )

                        if response.status_code == 200:
                            results = response.json()
                            predictions = results.get('predictions', [])

                            # Agregar predicciones al DataFrame
                            batch_data['prediccion'] = [p.get('prediction', 'N/A') for p in predictions]
                            batch_data['confianza'] = [p.get('confidence', 0) for p in predictions]

                            st.success(f"✅ {len(predictions)} predicciones completadas")

                            # Mostrar resumen
                            st.subheader("Resumen de Predicciones")
                            col1, col2, col3 = st.columns(3)

                            alto_riesgo = batch_data['prediccion'].sum() if batch_data['prediccion'].dtype != 'object' else 0
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

                        else:
                            st.error(f"Error en la predicción por lotes: {response.status_code}")

                    except Exception as e:
                        st.error(f"Error procesando el lote: {str(e)}")

        except Exception as e:
            st.error(f"Error leyendo el archivo: {str(e)}")

# Página de Resultados
elif app_mode == "📈 Resultados":
    st.header("Análisis de Resultados")

    if 'batch_results' not in st.session_state:
        st.info("No hay resultados de predicción disponibles. Realice una predicción por lotes primero.")
        st.stop()

    results = st.session_state.batch_results

    # Estadísticas de resultados
    st.subheader("Estadísticas de Predicciones")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total = len(results)
        st.metric("Total Predicciones", total)

    with col2:
        alto_riesgo = results['prediccion'].sum() if results['prediccion'].dtype != 'object' else len(results[results['prediccion'] == 1])
        st.metric("Alto Riesgo", alto_riesgo)

    with col3:
        bajo_riesgo = total - alto_riesgo
        st.metric("Bajo Riesgo", bajo_riesgo)

    with col4:
        tasa_alto_riesgo = (alto_riesgo / total) * 100 if total > 0 else 0
        st.metric("Tasa Alto Riesgo", f"{tasa_alto_riesgo:.1f}%")

    # Visualizaciones
    st.subheader("Distribución de Predicciones")

    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de torta
        labels = ['Bajo Riesgo', 'Alto Riesgo']
        sizes = [bajo_riesgo, alto_riesgo]
        colors = ['#4CAF50', '#F44336']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    with col2:
        # Distribución de confianza
        fig, ax = plt.subplots()
        ax.hist(results['confianza'], bins=20, alpha=0.7, color='skyblue')
        ax.set_xlabel('Confianza')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Confianza en Predicciones')
        st.pyplot(fig)

    # Análisis por características
    st.subheader("Análisis por Características")

    analysis_feature = st.selectbox(
        "Seleccione característica para análisis:",
        [col for col in results.columns if col not in ['prediccion', 'confianza']]
    )

    if analysis_feature:
        # Crosstab de la característica seleccionada vs predicción
        crosstab = pd.crosstab(results[analysis_feature], results['prediccion'], normalize='index') * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        crosstab.plot(kind='bar', ax=ax)
        ax.set_ylabel('Porcentaje (%)')
        ax.set_title(f'Distribución de Predicciones por {analysis_feature}')
        ax.legend(['Bajo Riesgo', 'Alto Riesgo'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# Página Acerca del Modelo
elif app_mode == "ℹ️ Acerca del Modelo":
    st.header("Información del Modelo")

    st.subheader("📋 Descripción del Sistema")
    st.markdown("""
    Este sistema de clasificación utiliza machine learning para analizar datos del sistema de salud colombiano
    y realizar predicciones basadas en características demográficas y de afiliación.

    ### 🎯 Objetivo
    Clasificar casos en categorías de riesgo para optimizar la asignación de recursos y mejorar la atención en salud.

    ### 📊 Datos Utilizados
    - **Régimen Contributivo**: ~641,000 registros
    - **Régimen Subsidiado**: ~1,000,000+ registros
    - **Variables**: Género, grupo etario, régimen, tipo de afiliado, ubicación geográfica, nivel Sisbén
    """)

    st.subheader("🔧 Características Técnicas")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🏗️ Arquitectura
        - **Backend**: API REST con Flask/FastAPI
        - **Frontend**: Streamlit
        - **ML Framework**: Scikit-learn/XGBoost
        - **Despliegue**: Streamlit Sharing
        """)

    with col2:
        st.markdown("""
        ### 📈 Métricas del Modelo
        - **Precisión**: > 85%
        - **Recall**: > 80%
        - **F1-Score**: > 82%
        - **AUC-ROC**: > 0.88
        """)

    st.subheader("🚀 Endpoints de la API")

    if check_api_health()[0]:
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            endpoints = response.json().get('endpoints', {})

            for endpoint, description in endpoints.items():
                st.code(f"{endpoint}: {description}")
        except:
            st.info("No se pudieron cargar los endpoints de la API")
    else:
        st.info("La API no está disponible para mostrar endpoints")

    st.subheader("👥 Desarrollo")
    st.markdown("""
    - **Desarrollado por**: Equipo de Ciencia de Datos - Salud Colombia
    - **Versión**: 1.0.0
    - **Última actualización**: Julio 2025
    - **Licencia**: Creative Commons Attribution Share Alike 4.0 International
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Sistema de Clasificación - Modelo de Salud Colombia** | "
    "Desarrollado con Streamlit 🚀"
)
