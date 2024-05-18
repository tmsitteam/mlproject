import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    model_obj_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def model_trainer(self, transformed_train_data, transformed_test_data):
        try:
            X_train, y_train, X_test, y_test = transformed_train_data[:, :-1], transformed_train_data[:, -1], transformed_test_data[:, :-1], transformed_test_data[:, -1]

            models = {
                        'Linear Regression': LinearRegression(),
                        'Decision Tree': DecisionTreeRegressor(), 
                        'Random Forest': RandomForestRegressor(),
                        'GBT': GradientBoostingRegressor()
                     }
            
            logging.info('Models initialized')

            params = {
                        'Linear Regression': {

                                             },
                        'Decision Tree': {
                                            'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                                         }, 
                        'Random Forest': {
                                            'n_estimators': [8, 16, 32, 64, 128, 256]
                                         },
                        'GBT': {
                                    'learning_rate':[.1 ,.01, .05, .001],
                                    'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                                    'n_estimators': [8, 16, 32, 64, 128, 256]
                               }
                     }
            
            logging.info('Hyperparameters initialized')
            
            model_report : dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            logging.info('Model report generated')

            best_model_score =  max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model  =  models[best_model_name]

            logging.info('Best model selected')

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            save_object(file_path=self.model_trainer_config.model_obj_file_path, obj=best_model)

            logging.info('Trained model object saved')
            return  r2_score(y_test, best_model.predict(X_test))
            
        except Exception as e:
            raise CustomException(e, sys)