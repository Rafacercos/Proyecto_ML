import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Evaluación de Riesgo Fiscal",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_viernes.pkl") 

modelo = cargar_modelo()

st.markdown("""
    <style>
        .title {
            font-size: 30px;
            font-weight: bold;
            color: #1e3d58;
            text-align: center;
            margin-bottom: 30px;
        }
        .subheader {
            font-size: 20px;
            font-weight: bold;
            color: #1e3d58;
            margin-bottom: 20px;
        }
        .description {
            font-size: 16px;
            color: #4b5d67;
            margin-bottom: 30px;
        }
        .section {
            padding: 20px;
            background-color: #f5f7fa;
            border-radius: 10px;
            margin-bottom: 40px;
        }
        .info-box {
            padding: 10px;
            background-color: #e1f0f7;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Sistema de Evaluación de Riesgo Fiscal</div>', unsafe_allow_html=True)

st.markdown("""
Este sistema estima la probabilidad de que un individuo perciba ingresos anuales superiores a 50.000 USD,
basado en características laborales, educativas y demográficas. Esta herramienta es útil para detectar patrones de riesgo fiscal en base a datos declarados.
""", unsafe_allow_html=True)

with st.form("formulario_datos"):

    st.markdown('<div class="subheader">Datos del contribuyente</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        sexo = st.selectbox("Sexo", options=["Male", "Female"])
        
        
        edad = st.slider("Edad", min_value=17, max_value=90, step=1, value=30)
        
        
        education_options = {
            "Guardería": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
            "10th": 6, "11th": 7, "12th": 8, "Graduado de escuela secundaria": 9, 
            "Algunos estudios universitarios": 10, "Asociado académico": 11, "Asociado técnico": 12, 
            "Licenciatura": 13, "Maestría": 14, "Doctorado": 16, "Formación profesional": 15
        }
        education_level = st.selectbox("Nivel educativo", options=list(education_options.keys()))
        education_num = education_options[education_level]

        
        horas_por_semana = st.slider("Horas trabajadas por semana", min_value=1, max_value=100, step=1, value=40)
    
    with col2:
        estado_civil = st.selectbox("Estado civil", [
            'Nunca casado/a', 'Casado/a (civil)', 'Divorciado/a', 'Cónyuge ausente',
            'Separado/a', 'Casado/a (Fuerzas Armadas)', 'Viudo/a'
        ])
        
        puesto_label = st.selectbox("Tipo de empleo", [
            'Autónomo incorporado', 'Autónomo no incorporado', 'Gobierno local', 
            'Gobierno federal', 'Gobierno estatal', 'Sector privado', 
            'Desconocido', 'Sin salario', 'Nunca ha trabajado'
        ])
        
        ocupacion_label = st.selectbox("Ocupación", [
            "Profesional / Especialista", "Ejecutivo / Gerencial", "Ventas",
            "Servicios de protección / Seguridad", "Soporte técnico",
            "Oficios / Reparaciones", "Desconocido / No especificado",
            "Agricultura / Pesca", "Administrativo / Oficina", "Transporte / Movimiento",
            "Operador / Inspector de máquinas", "Servicio doméstico",
            "Manipuladores / Limpiadores", "Otros servicios", "Fuerzas armadas"
        ])

    submitted = st.form_submit_button("Evaluar")

if submitted:
    puesto_mapping = {
        'Autónomo incorporado': 10,
        'Autónomo no incorporado': 5,
        'Gobierno local': 3,
        'Gobierno federal': 2,
        'Gobierno estatal': 2,
        'Sector privado': 2,
        'Desconocido': 1,
        'Sin salario': 1,
        'Nunca ha trabajado': 1
    }
    ocupacion_mapping = {
        "Profesional / Especialista": 17,
        "Ejecutivo / Gerencial": 15,
        "Ventas": 10,
        "Servicios de protección / Seguridad": 7,
        "Soporte técnico": 7,
        "Oficios / Reparaciones": 6,
        "Desconocido / No especificado": 5,
        "Agricultura / Pesca": 5,
        "Administrativo / Oficina": 4,
        "Transporte / Movimiento": 4,
        "Operador / Inspector de máquinas": 3,
        "Servicio doméstico": 2,
        "Manipuladores / Limpiadores": 2,
        "Otros servicios": 1,
        "Fuerzas armadas": 0
    }

    puesto_ord = puesto_mapping[puesto_label]
    occ_ord = ocupacion_mapping[ocupacion_label]

    sex_male = 1 if sexo == "Male" else 0
    sex_female = 1 if sexo == "Female" else 0
    marital_casado = 1 if estado_civil == "Casado/a (civil)" else 0
    marital_soltero = 1 if estado_civil == "Nunca casado/a" else 0

    rango_edad = pd.cut([edad], bins=[0, 25, 35, 45, 55, 65, 100], labels=False)[0]
    educacion_superior = 1 if education_num > 13 else 0
    edad_ajustada = edad * education_num
    horas_puesto = horas_por_semana * puesto_ord

    columnas_esperadas = [
        "puesto_ord", "sex_ Male", "sex_ Female", "age", "rango_edad", "hours-per-week",
        "educacion_superior", "education-num", "marital-status_Casado/a (civil)",
        "occ_ord", "edad_ajustada", "horas_puesto", "marital-status_Nunca casado/a"
    ]

    valores = [[
        puesto_ord, sex_male, sex_female, edad, rango_edad, horas_por_semana,
        educacion_superior, education_num, marital_casado,
        occ_ord, edad_ajustada, horas_puesto, marital_soltero
    ]]

    df_input = pd.DataFrame(valores, columns=columnas_esperadas)

    prediccion = modelo.predict(df_input)[0]
    probas = modelo.predict_proba(df_input)[0]

    st.markdown('<div class="subheader">Resultado de la evaluación</div>', unsafe_allow_html=True)
    if prediccion == 1:
        st.markdown("**Clasificación: Ingreso estimado superior a 50.000 USD.**")
    else:
        st.markdown("**Clasificación: Ingreso estimado igual o inferior a 50.000 USD.**")

    st.markdown(f"**Probabilidad de ingreso > 50K:** {probas[1]:.2%}")
    st.markdown(f"**Probabilidad de ingreso ≤ 50K:** {probas[0]:.2%}")

    with st.expander("Información sobre impuestos en EE. UU."):
        st.markdown("""
        A continuación se muestran los rangos de ingresos y los porcentajes de impuestos aplicables en EE. UU.
        <table>
            <thead>
                <tr>
                    <th>Ingreso Anual</th>
                    <th>Porcentaje de Impuesto</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Hasta $9,875</td>
                    <td>10%</td>
                </tr>
                <tr>
                    <td>$9,876 - $40,125</td>
                    <td>12%</td>
                </tr>
                <tr>
                    <td>$40,126 - $85,525</td>
                    <td>22%</td>
                </tr>
                <tr>
                    <td>$85,526 - $163,300</td>
                    <td>24%</td>
                </tr>
                <tr>
                    <td>$163,301 - $207,350</td>
                    <td>32%</td>
                </tr>
                <tr>
                    <td>$207,351 - $518,400</td>
                    <td>35%</td>
                </tr>
                <tr>
                    <td>Más de $518,400</td>
                    <td>37%</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)


