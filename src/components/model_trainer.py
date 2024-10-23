from constants import CONFIG_FILE_PATH
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import sys
import os
import mlflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.components.model_evaluation import ModelEvaluation
from src.utils import *
import pickle
from src.constants import *
import configparser
config = configparser.RawConfigParser()
config.read(r'C:\Users\pavan\OneDrive\Desktop\HomePrice_predection (1)\HomePrice\src\config\config.ini')
import pandas as pd
mlflow.set_tracking_uri(uri='http://127.0.0.1:5000')
mlflow.set_experiment('House_Prediction')
import pickle
model_dir = config.get('DATA', 'model_dir')
class ModelTrainer:
    def _init_(self):
        self.config = config.read(CONFIG_FILE_PATH)
        # self.model_dir = config.get('DATA', 'model_dir')

    def _train_and_log_model(self, model_name, model, X_train, y_train, X_test, y_test, best_model_params):
        with mlflow.start_run() as run:
            # Fit the model
            model.set_params(**best_model_params)
            model.fit(X_train, y_train)
            
            # Make predictions and calculate score
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            
            # Log parameters and metrics
            mlflow.log_params(best_model_params)
            mlflow.log_metric("train_r2", train_score)
            mlflow.log_metric("test_r2", test_score)
            mlflow.set_tag('Training info',f'{model_name} model for House Prediction')
            
            # Save and log the model
            mlflow.sklearn.log_model(model, "model") if model_name != 'xgboost' else mlflow.xgboost.log_model(model, "model")
            
            # Save model to disk
            pickle.dump(model, open(os.path.join(model_dir, 'final_model.pkl'), 'wb'))
            
            print(f"{model_name} model saved and logged with R2-Score: Train : {train_score}, Test : {test_score}")

    def final_model(self, X_train, X_test, y_train, y_test, best_model_name, best_model_params):
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(),
            'decision_tree': DecisionTreeRegressor(),
            'lasso': Lasso(max_iter=25000),
            'svr': SVR(),
            'random_forest': RandomForestRegressor(),
            'knn': KNeighborsRegressor(),
            'xgboost': XGBRegressor()
        }
        
        if best_model_name in models:
            model = models[best_model_name]
            self._train_and_log_model(best_model_name, model, X_train, y_train, X_test, y_test, best_model_params)
        else:
            print("Invalid model type")

    def find_best_model(self, X_train, y_train, X_test, y_test):
        models = {
            'linear_regression': {
                'model': LinearRegression(),
                'parameters': {}
            },
            'lasso': {
                'model': Lasso(max_iter=25000),
                'parameters': {
                    'alpha': np.logspace(-1, 1, 10),
                    'selection': ['random', 'cyclic'],
                    'tol': [0.001, 0.01],
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            },
            'ridge': {
                'model': Ridge(),
                'parameters': [
                    {
                        'alpha': np.logspace(-1, 1, 10),
                        'solver': ['auto', 'svd'],
                        'positive': [False],
                        'fit_intercept': [True, False],
                        'tol': [0.001]
                    },
                    {
                        'alpha': np.logspace(-1, 1, 10),
                        'solver': ['lbfgs'],
                        'positive': [True],
                        'fit_intercept': [True, False],
                        'tol': [0.001]
                    }
                ]
            },
            'svr': {
                'model': SVR(),
                'parameters': {
                    'gamma': ['scale'],
                    'epsilon': [0.01, 0.1, 0.5],
                    'degree': [2, 3],
                    'kernel': ['linear', 'rbf'],
                    'C': [0.1, 10, 100]
                }
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(),
                'parameters': {
                    'criterion': ['squared_error', 'absolute_error'],
                    'max_depth': [10, 15],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(),
                'parameters': {
                    'n_estimators': [10, 30],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 3],
                    'min_samples_leaf': [1, 2]
                }
            },
           
            'knn': {
                'model': KNeighborsRegressor(),
                'parameters': {
                    'n_neighbors': [7, 9, 10, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree', 'brute']
                }
            }
        }

        scores = []
        for model_name, model_params in models.items():
            gs = GridSearchCV(model_params['model'], model_params['parameters'], cv=7, scoring='r2', return_train_score=False)
            try:
                gs.fit(X_train, y_train)
                y_pred = gs.best_estimator_.predict(X_test)
                test_score = r2_score(y_test, y_pred)
                scores.append({
                    'model': model_name,
                    'best_parameters': gs.best_params_,
                    'score': gs.best_score_,
                    'test_r2': test_score
                })
                print('Completed', model_name)
            except Exception as e:
                print(f"Error encountered with {model_name}: {e}")
        
        return pd.DataFrame(scores, columns=['model', 'best_parameters', 'score', 'test_r2'])