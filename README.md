# 🧾 Herramienta de Predicción de Ingresos Anuales

## 📘 Descripción del proyecto

Esta herramienta permite estimar si una persona tendrá un ingreso anual superior a 50.000 dólares. Está basada en un modelo de aprendizaje automático entrenado con datos reales del censo de Estados Unidos y tiene como objetivo facilitar el análisis de perfiles socioeconómicos.

Puede resultar útil para organismos públicos como agencias tributarias o entidades de control económico, ayudando a identificar patrones, facilitar estudios estadísticos y aportar un criterio adicional para la detección de inconsistencias fiscales.

---

## 🧠 ¿Qué datos se usan?

La aplicación solicita información relevante del perfil del contribuyente:

- Edad
- Nivel educativo
- Horas trabajadas por semana
- Tipo de ocupación
- Tipo de puesto o régimen laboral
- Estado civil
- Sexo

La combinación de estas variables permite realizar una estimación sobre el nivel de ingresos.

---

## 🖥️ ¿Qué hace la herramienta?

Al introducir los datos, el sistema genera una predicción indicando si los ingresos estimados son mayores o menores a 50.000 dólares anuales. También muestra la probabilidad de que una persona pertenezca a uno u otro grupo.

Además, la interfaz incluye:

- Información sobre los tramos de impuestos federales en EE.UU.
- Enlaces a leyes, formularios y guías fiscales oficiales.
- Una sección de ayuda con acceso a sitios gubernamentales.

---

## 🧾 Legislación y referencias relevantes

- [Internal Revenue Code (IRC)](https://www.law.cornell.edu/uscode/text/26): Código federal de impuestos de EE.UU.
- [FATCA (Foreign Account Tax Compliance Act)](https://home.treasury.gov/policy-issues/tax-policy/foreign-account-tax-compliance-act): Ley para el control de cuentas extranjeras.
- [Formulario 1040](https://www.irs.gov/forms-pubs/about-form-1040): Declaración de la renta para individuos.
- [Publication 505](https://www.irs.gov/publications/p505): Guía sobre retenciones e impuestos estimados.
- [Ley de Transparencia y Responsabilidad Fiscal](https://www.congress.gov/bill/116th-congress/house-bill/5933)

---

## 📌 Aplicaciones posibles

- Apoyo a la detección de fraudes fiscales
- Estudios socioeconómicos y demográficos
- Mejora de sistemas de asesoría tributaria
- Priorización de auditorías basadas en perfiles de riesgo

---

## ▶️ Cómo ejecutar la aplicación

1. Asegúrate de tener Python instalado.
2. Instala las dependencias necesarias.
3. Ejecuta el siguiente comando en la terminal:

```bash
streamlit run app.py
```

Esto abrirá la aplicación en tu navegador web.

---

## 🏛️ Uso responsable

Este sistema es una herramienta de apoyo. No debe sustituir el análisis profesional ni utilizarse como única base para decisiones legales o fiscales. El respeto a la privacidad de los datos y la interpretación ética de los resultados son fundamentales.

---
