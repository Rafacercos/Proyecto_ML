import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


model = joblib.load("modelo_viernes.pkl")
st.set_page_config(page_title="FiscoGuard", layout="wide")
st.title("FiscoGuard")

imagen_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Seal_of_the_United_States_Internal_Revenue_Service.svg/1200px-Seal_of_the_United_States_Internal_Revenue_Service.svg.png"


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
    
    html, body, [class*="st-"], h1, h2, h3, h4, h5, h6 {{
        font-family: 'Roboto', sans-serif !important;
    }}

    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.9)), 
                          url("{imagen_url}");
        background-size: 650px;
        background-repeat: no-repeat;
        background-position: center center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

logo = Image.open("LogoML.pgn")
st.image(logo, width=150)

menu = st.sidebar.radio("Navegación", ["Inicio", "Información relevante", "Ayuda"])

if menu == "Inicio":
    st.subheader("Herramienta de Predicción de Ingreso - Análisis de Riesgo Fiscal")
    st.markdown("---")
    st.markdown("Ingrese los datos del individuo para predecir si su ingreso anual es superior o inferior a 50.000 dólares.")

    col1, col2 = st.columns(2)
    with col1:
        sexo = st.selectbox("Sexo", ["Masculino", "Femenino"])
        edad = st.slider("Edad", min_value=17, max_value=90, value=35)
        horas_semana = st.slider("Horas trabajadas por semana", min_value=1, max_value=99, value=40)
        educacion = st.selectbox("Nivel educativo", [
             "Guardería", "1st-4th", "5th-6th", "7th-8th", "9th",
            "10th", "11th", "12th", "Graduado de escuela secundaria", 
            "Algunos estudios universitarios", "Asociado académico", "Asociado técnico", 
            "Licenciatura", "Maestría", "Doctorado", "Formación profesional"])
    with col2:
        estado_civil = st.selectbox("Estado civil", ['Nunca casado/a', 'Casado/a (civil)', 'Divorciado/a', 'Cónyuge ausente',
            'Separado/a', 'Casado/a (Fuerzas Armadas)', 'Viudo/a'])
        puesto = st.selectbox("Tipo de puesto laboral", [
            'Autónomo incorporado', 'Autónomo no incorporado', 'Gobierno local', 'Gobierno federal',
            'Gobierno estatal', 'Sector privado', 'Desconocido', 'Sin salario', 'Nunca ha trabajado'])
        ocupacion = st.selectbox("Tipo de ocupación", [
            "Profesional / Especialista", "Ejecutivo / Gerencial", "Ventas", "Servicios de protección / Seguridad",
            "Soporte técnico", "Oficios / Reparaciones", "Desconocido / No especificado", "Agricultura / Pesca",
            "Administrativo / Oficina", "Transporte / Movimiento", "Operador / Inspector de máquinas",
            "Servicio doméstico", "Manipuladores / Limpiadores", "Otros servicios", "Fuerzas armadas"])

    
    sex_male = 1 if sexo == "Masculino" else 0
    sex_female = 1 if sexo == "Femenino" else 0

    educ_map = {
         "Guardería": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
            "10th": 6, "11th": 7, "12th": 8, "Graduado de escuela secundaria": 9, 
            "Algunos estudios universitarios": 10, "Asociado académico": 11, "Asociado técnico": 12, 
            "Licenciatura": 13, "Maestría": 14, "Doctorado": 16, "Formación profesional": 15
    }
    education_num = educ_map[educacion]
    educacion_superior = 1 if education_num > 13 else 0

    marital_casado = 1 if estado_civil == "Casado/a (civil)" else 0
    marital_nunca = 1 if estado_civil == "Nunca casado/a" else 0

    mapa2 = {
        'Autónomo incorporado': 10, 'Autónomo no incorporado': 5, 'Gobierno local': 3,
        'Gobierno federal': 2, 'Gobierno estatal': 2, 'Sector privado': 2,
        'Desconocido': 1, 'Sin salario': 1, 'Nunca ha trabajado': 1
    }
    puesto_ord = mapa2[puesto]

    oc_map = {
        "Profesional / Especialista":17, "Ejecutivo / Gerencial":15, "Ventas":10,
        "Servicios de protección / Seguridad":7, "Soporte técnico":7, "Oficios / Reparaciones":6,
        "Desconocido / No especificado":5, "Agricultura / Pesca":5, "Administrativo / Oficina":4,
        "Transporte / Movimiento":4, "Operador / Inspector de máquinas":3, "Servicio doméstico":2,
        "Manipuladores / Limpiadores":2, "Otros servicios":1, "Fuerzas armadas":0
    }
    occ_ord = oc_map[ocupacion]


    rango_edad = pd.cut([edad], bins=[0, 25, 35, 45, 55, 65, 100], labels=False)[0]
    edad_ajustada = edad * education_num
    horas_puesto = horas_semana * puesto_ord

    input_data = pd.DataFrame({
        "puesto_ord": [puesto_ord],
        "sex_ Male": [sex_male],
        "sex_ Female": [sex_female],
        "age": [edad],
        "rango_edad": [rango_edad],
        "hours-per-week": [horas_semana],
        "educacion_superior": [educacion_superior],
        "education-num": [education_num],
        "marital-status_Casado/a (civil)": [marital_casado],
        "occ_ord": [occ_ord],
        "edad_ajustada": [edad_ajustada],
        "horas_puesto": [horas_puesto],
        "marital-status_Nunca casado/a": [marital_nunca]
    })

    if st.button("Predecir ingreso"):
        
        pred = model.predict(input_data)[0]
        probas = model.predict_proba(input_data)[0]
        resultado = ">50K" if pred == 1 else "<=50K"
        
        
        st.success(f"Predicción: El ingreso estimado es {resultado} dólares anuales")
        st.markdown("**Probabilidades:**")
        st.markdown(f"- Probabilidad de ingreso <= 50K: {probas[0]*100:.2f}%")
        st.markdown(f"- Probabilidad de ingreso > 50K: {probas[1]*100:.2f}%")

        
        st.subheader("Importancia de las características")
        feature_names = input_data.columns[1:10]
        feature_values = input_data.iloc[0].values[1:10]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(["Trabajo","Estado Civil","Nivel Académico","Educación Superior","Horas Laborales (Semana)","Rango de Edad","Edad","Sexo"], feature_values[1:10])
        ax.set_xlabel("Valor de la característica")
        ax.set_title("Importancia de las características en la predicción")
        st.pyplot(fig)
    
        with st.expander("Descripción variables"):
            st.markdown("""
            ## Descripción de las características más importantes:
                        
        - **Sexo** : Sexo del sujeto (Hombre,Mujer)
        - **Edad** : Edad del sujeto (Rango 17-90)                
        - **Rango de Edad** : Variable ordinal discretizada que divide a los sujetos en distintos tramos de edad
        - **Horas Laborales (Semana)** : Estimación del número de horas trabajadas en una semana por el sujeto
        - **Educación Superior** : Variable dicotómica que divide a los sujetos según si se han formado en educación superio o no (A partir de universidad)
        - **Nivel Académico** : Nivel académico más alto cursado por el sujeto
        - **Estado Civil** : Estado civil del sujeto (casado, soltero,divorciado...)
        - **Trabajo** : Profesión a la que se dedica el sujeto   
            """)
       


   
    with st.expander("Recordatorio: Tramos y Porcentajes de Impuestos en EE.UU."):
        st.markdown("""
        **Tramos federales de impuestos sobre la renta para personas individuales (2024):**

        - Hasta $11,000 → 10%
        - Rango (11,001 - $44,725) → 12%
        - Rango (44,726 - $95,375) → 22%
        - Rango (95,376 - $182,100) → 24%
        - Rango (182,101 - $231,250) → 32%
        - Rango (231,251 - $578,125) → 35%
        - Más de $578,125 → 37%

        Los impuestos en EE.UU. son progresivos. Es decir, cada tramo se grava con su tipo impositivo específico.
        [Fuente oficial - IRS](https://www.irs.gov/)
        """)

elif menu == "Información relevante":
    
    st.title("Información Legal y Fiscal Relevante")
    st.markdown("""
    Esta herramienta ha sido desarrollada con el objetivo de ayudar a las instituciones fiscales a identificar perfiles
    con riesgo fiscal potencial, basándose en patrones de ingreso y características sociodemográficas y laborales.

    ### Leyes y Reglamentos Relevantes:
    - **[Internal Revenue Code (IRC)](https://www.law.cornell.edu/uscode/text/26)**: Título 26 del Código de EE.UU.
    - **[FATCA (Foreign Account Tax Compliance Act)](https://www.irs.gov/businesses/corporations/foreign-account-tax-compliance-act-fatca)**: Ley de cumplimiento tributario de cuentas extranjeras.
    - **[Ley de Transparencia y Responsabilidad Fiscal](https://www.congress.gov/bill/113th-congress/house-bill/2531)**.
    - **[Formulario 1040 (Declaración de la renta individual)](https://www.irs.gov/forms-pubs/about-form-1040)**: incluye anexos relevantes según ingresos y ocupación.
    - **[Publication 505](https://www.irs.gov/forms-pubs/about-publication-505)**: Guía oficial sobre retenciones e impuestos estimados.
                
   ### Datos Analizados:
    - Nivel educativo
    - Tipo de ocupación
    - Sector de trabajo (público, privado, autónomo)
    - Horas trabajadas
    - Edad y estado civil

   Este sistema no sustituye la auditoría ni decisiones legales. Es un **apoyo para análisis exploratorio** en fiscalización.
""")

elif menu == "Ayuda":
    st.title("Centro de Ayuda")
    st.markdown("""Para obtener más ayuda o información sobre sus obligaciones fiscales:
                🔗 [Sitio oficial del IRS - Individuals](https://www.irs.gov/individuals)
                🔗 [Preguntas frecuentes - IRS](https://www.irs.gov/help/ita)
                🔗 [Publicaciones fiscales en español](https://www.irs.gov/es/forms-pubs)

También puede contactar con asesores fiscales certificados o consultar a su representante de Hacienda local.
""")


