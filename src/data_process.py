import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


df = pd.read_csv("../Data/Raw/adult.data",header=None)

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

valores=[]
for x in df["workclass"]:
    valores.append(x.strip())

df["workclass"]=valores

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
    'Autónomo incorporado':6,
    'Gobierno federal':4,
    'Gobierno local':3,
    'Autónomo no incorporado':3,
    'Gobierno estatal':2,
    'Sector privado':2,
    'Desconocido':1,
    'Sin salario':0,
    'Nunca ha trabajado':0
}
 
df["puesto_ord"] = df["workclass"].map(mapa2)

valores=[]
for x in df["marital-status"]:
    valores.append(x.strip())

df["marital-status"]= valores

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

valores=[]
for x in df["occupation"]:
    valores.append(x.strip())

df["occupation"]= valores

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
    "Ejecutivo / Gerencial":5,
    "Profesional / Especialista":4,
    "Ventas":3,
    "Servicios de protección / Seguridad":3,
    "Soporte técnico":3,
    "Oficios / Reparaciones":2,
    "Transporte / Movimiento":2,
    "Desconocido / No especificado":1,
    "Agricultura / Pesca":1,
    "Administrativo / Oficina":1,
    "Operador / Inspector de máquinas":1,
    "Fuerzas armadas":1,
    "Servicio doméstico":0,
    "Manipuladores / Limpiadores":0,
    "Otros servicios":0,
    
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

df["native-country"] = df["native-country"].str.strip()
df["Extranjero"] = np.where(df["native-country"]=='United-States',0,1)

df['age'] = np.log1p(df['age'])
df['capital-gain'] = np.log1p(df['capital-gain'])
df['capital-loss'] = np.log1p(df['capital-loss'])
df['hours-per-week'] = np.log1p(df['hours-per-week'])
df['edad_ajustada'] = np.log1p(df['edad_ajustada'])

df["relationship"] = df["relationship"].str.strip()
mapa_r = {
    'Husband':6,
    'Wife':5,
    'Unmarried':4,
    'Own-child':3,
    'Not-in-family': 2,
    'Other-relative':1
}

df["rel_ord"] = df["relationship"].map(mapa_r)


X = df[[ 'age', 'education-num',
       'puesto_ord', 'marital-status_Casado/a (civil)', 'occ_ord', 'sex_ Female', 'sex_ Male', 'edad_ajustada', 'edu_oc', 'rel_ord']]
  

scaler = StandardScaler()
X_scal = scaler.fit_transform(X)

km = KMeans(n_clusters=3,random_state=42)
km.fit(X_scal)

clusters = km.labels_
df["cluster"] = clusters

X = df.drop(columns=["income","native-country","relationship","occupation","education","workclass"])
y = df["income"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

generador = SMOTE(sampling_strategy='auto', random_state=42) 
X_trainR, y_trainR = generador.fit_resample(X_train, y_train)

df_train = pd.concat((X_trainR,y_trainR),axis=1)
df_test = pd.concat((X_test,y_test),axis= 1)
df_entero = pd.concat((df_train,df_test),axis=0)

df_entero.to_csv('../Data/Processed/Datos.csv')