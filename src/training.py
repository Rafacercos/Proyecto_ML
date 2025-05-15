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
import pickle


df = pd.read_csv("../Data/Train/Train.csv",index_col=0)
X = df.drop(columns=["income"])
y = df["income"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

pipe4 = Pipeline(steps=[
    ("scaler",StandardScaler()),
    ("class",CatBoostClassifier())
])

params = {
    'scaler':[StandardScaler(),MinMaxScaler(),'passthrough'],
    'class__depth': np.arange(3,12),
    'class__iterations': [100, 300, 500],
    'class__learning_rate':  [0.01, 0.05, 0.1, 0.2],
    'class__l2_leaf_reg':[1, 3, 5, 7, 10],
    'class__bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
    'class__subsample': [0.6, 0.8, 1.0],
    'class__auto_class_weights': ['Balanced', None]
}

Rs2 = RandomizedSearchCV(n_iter=50,estimator=pipe4,param_distributions=params,cv=10,verbose=2,scoring='recall',n_jobs=-1)
Rs2.fit(x_train,y_train)

modeloCAT = Rs2.best_estimator_
modeloCAT.fit(x_train,y_train)

filename = "../models/modeloFINAL.pkl"
with open(filename,"wb")as archivo:
    pickle.dump(modeloCAT,archivo)