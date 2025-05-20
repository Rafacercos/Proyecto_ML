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

def evaluar_modelo (y_test,y_pred):
    print("acc score:",accuracy_score(y_test,y_pred))
    print("recallscore",recall_score(y_test,y_pred))
    print("precision score:",precision_score(y_test,y_pred))
    labels = np.unique(y_test)
    return plt.figure(figsize=(6, 5)),sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels),plt.xlabel('Prediccion'),plt.ylabel('Verdaderos'),plt.title('Confusion Matrix'),plt.tight_layout(),plt.show()

def ver_importancias (modelo):
    importancias = np.round(modelo.named_steps['class'].feature_importances_,2)
    nombres = X.columns
    df_imp = pd.DataFrame({
    "variable": nombres,
    "Importancia": importancias})
    return df_imp.sort_values(by="Importancia",ascending=False).head(10).set_index("variable")