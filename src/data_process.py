import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('../data/raw/adult.data',header=None)

df.rename(columns= {
    0: "age",
    1: "workclass",
    2: "fnlwgt",
    3: "education",
    4: "education-num",
    5: "marital-status",
    6: "occupation",
    7: "relationship",
    8: "race",
    9: "sex",
    10: "capital-gain",
    11: "capital-loss",
    12: "hours-per-week",
    13: "native-country",
    14: "income"
    
},inplace=True)

valores=[]
for x in df["income"]:
    valores.append(x.strip())

df["income"] = valores
df["income"] = np.where(df["income"]=="<=50K",0,1)

df["extras"] = df["capital-gain"]- df["capital-loss"]

valores2=[]
for x in df["workclass"]:
    valores2.append(x.strip())

df["workclass"]=valores2

mapa_trabajo = {
    'State-gov': 'Gobierno estatal',
    'Self-emp-not-inc': 'Autónomo no incorporado',
    'Private': 'Sector privado',
    'Federal-gov': 'Gobierno federal',
    'Local-gov': 'Gobierno local',
    '?': 'Desconocido',
    'Self-emp-inc': 'Autónomo incorporado',
    'Without-pay': 'Sin salario',
    'Never-worked': 'Nunca ha trabajado'
}
 

df["workclass"] = df["workclass"].map(mapa_trabajo)

mapa2 = {
    'Autónomo incorporado':10,
    'Autónomo no incorporado':5,
    'Gobierno local':3,
    'Gobierno federal':2,
    'Gobierno estatal':2,
    'Sector privado':2,
    'Desconocido':1,
    'Sin salario':1,
    'Nunca ha trabajado':1
}
 
df["puesto_ord"] = df["workclass"].map(mapa2)

valores3=[]
for x in df["marital-status"]:
    valores3.append(x.strip())

df["marital-status"]= valores3

marit_map = {
    'Never-married': 'Nunca casado/a',
    'Married-civ-spouse': 'Casado/a (civil)',
    'Divorced': 'Divorciado/a',
    'Married-spouse-absent': 'Cónyuge ausente',
    'Separated': 'Separado/a',
    'Married-AF-spouse': 'Casado/a (Fuerzas Armadas)',
    'Widowed': 'Viudo/a'
}

df["marital-status"] = df["marital-status"].map(marit_map)

df = pd.get_dummies(df,columns= ["marital-status"])
df = pd.get_dummies(df,columns=["race"])

valores4=[]
for x in df["occupation"]:
    valores4.append(x.strip())

df["occupation"]= valores4

occupation_map = {
    'Adm-clerical': 'Administrativo / Oficina',
    'Exec-managerial': 'Ejecutivo / Gerencial',
    'Handlers-cleaners': 'Manipuladores / Limpiadores',
    'Prof-specialty': 'Profesional / Especialista',
    'Other-service': 'Otros servicios',
    'Sales': 'Ventas',
    'Craft-repair': 'Oficios / Reparaciones',
    'Transport-moving': 'Transporte / Movimiento',
    'Farming-fishing': 'Agricultura / Pesca',
    'Machine-op-inspct': 'Operador / Inspector de máquinas',
    'Tech-support': 'Soporte técnico',
    '?': 'Desconocido / No especificado',
    'Protective-serv': 'Servicios de protección / Seguridad',
    'Armed-Forces': 'Fuerzas armadas',
    'Priv-house-serv': 'Servicio doméstico'
}

df["occupation"]= df["occupation"].map(occupation_map)

oc_map = {
    "Profesional / Especialista":17,
    "Ejecutivo / Gerencial":15,
    "Ventas":10,
    "Servicios de protección / Seguridad":7,
    "Soporte técnico":7,
    "Oficios / Reparaciones":6,
    "Desconocido / No especificado":5,
    "Agricultura / Pesca":5,
    "Administrativo / Oficina":4,
    "Transporte / Movimiento":4,
    "Operador / Inspector de máquinas":3,
    "Servicio doméstico":2,
    "Manipuladores / Limpiadores":2,
    "Otros servicios":1,
    "Fuerzas armadas":0
}


df["occ_ord"] = df["occupation"].map(oc_map)

df= pd.get_dummies(df,columns=["sex"])

df["educacion_superior"] = np.where(df["education-num"]>=13,1,0)
df["educacion_inferior"] = np.where(df["education-num"]<=9,1,0)
df["edad_ajustada"] = df["age"]*df["education-num"]
df["horas_puesto"] = df["hours-per-week"] * df["occ_ord"]
df["edu_oc"] = df["education-num"] * df["puesto_ord"]
df["rango_edad"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 65,100], labels=False)

df["horas_ord"] = pd.cut(df["hours-per-week"], bins=[0, 25, 40, 60], labels=["Jornada_corta", "Jornada_media", "Jornada_larga"])
df = pd.get_dummies(df, columns=["horas_ord"])

df_mod = df[['puesto_ord',"sex_ Male","sex_ Female","age","rango_edad","hours-per-week","educacion_superior","education-num","marital-status_Casado/a (civil)","occ_ord","edad_ajustada","horas_puesto","marital-status_Nunca casado/a","income"]]

df_mod.to_csv("C:/Users/rafac/Proyecto_ML/Data/Processed/ds_limpio.csv",index=False,encoding='utf-8')