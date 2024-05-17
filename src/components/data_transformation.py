import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path =  os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            num_columns = ['writing_score', 'reading_score']
            cat_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                                        steps=[
                                                ('imputer', SimpleImputer(strategy='median')),
                                                ('scaler', StandardScaler())
                                              ]
                                   )
            
            logging.info('Numerical columns pipeline initialized')

            cat_pipeline = Pipeline(
                                        steps=[
                                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                                ('one_hot_encoder', OneHotEncoder()),
                                                ('scaler', StandardScaler(with_mean=False))
                                              ]
                                   )
            
            logging.info('Categorical columns pipeline initialized')

            preprocessor = ColumnTransformer(
                                                [
                                                    ('num_pipeline', num_pipeline, num_columns),
                                                    ('cat_pipeline', cat_pipeline, cat_columns)
                                                ]
                                            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data successfully')

            preprocessing_obj = self.get_data_transformation_obj()

            logging.info('Obtained preprocessing object')

            target_column = 'math_score'

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logging.info('X_train, y_train, X_test, and y_test split is done')

            X_train_transformed =  preprocessing_obj.fit_transform(X_train)
            X_test_transformed  = preprocessing_obj.fit_transform(X_test)

            logging.info('Transformation stage 1 done')
            
            transformed_train_array = np.c_[X_train_transformed, np.array(y_train)]
            transformed_test_array = np.c_[X_test_transformed, np.array(y_test)]

            logging.info('Transformation stage 2 done')

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path, obj = preprocessing_obj)

            logging.info('Transformation object saved')

            return transformed_train_array, transformed_test_array, self.data_transformation_config.preprocessor_obj_file_path


        except Exception as e:
            raise CustomException(e, sys)