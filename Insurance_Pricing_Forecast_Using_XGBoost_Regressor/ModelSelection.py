
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score



models = [
    RandomForestRegressor(),
    SVR(),
    GradientBoostingRegressor(),
    XGBRegressor(),
    LinearRegression(),
    Lasso(),
    Ridge()
]




# Apply GridSearchCV
def model_selection(X, y, list_of_models, hyperparameters_dict):
    model_keys = list(hyperparameters_dict.keys())
    results = []
    
    i = 0
    
    for model in list_of_models:
        key = model_keys[i]
        
        params = hyperparameters_dict[key]
        i += 1
        print(model)
        print(params)
        print(f"Performing Grid Search for {model.__class__.__name__} with hyperparameters: {params}")
        classifier = GridSearchCV(model, params, cv=5)
        classifier.fit(X, y)
        
        results.append({
            'model_used' : model,
            'highest_score' :classifier.best_score_,
            'best_hyperparameters' : classifier.best_params_
        })
        

    result_dataframe = pd.DataFrame(results, columns = ['model used', 'highest score', 'best param'])
    return result_dataframe