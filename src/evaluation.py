import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest,VarianceThreshold
from catboost import CatBoostClassifier
from utils import evaluar_modelo
import pickle

filename = "../models/modelo_CAT.pkl"

with open(filename, "rb") as archivo:
    modelo = pickle.load(archivo)

PRUEBA = pd.read_csv('../Data/Test/Test.csv',index_col=0)
x_prueba = PRUEBA.drop(columns=["income"])
y_prueba = PRUEBA["income"]

y_pred = modelo.predict(x_prueba)
evaluar_modelo(y_prueba,y_pred)
