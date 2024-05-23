import os
import sys
from src.exception import CustomException
from src.utils import load_object

import pandas as pd

class PredictPipeline:
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self, gender:str, race_ethnicity:str, parental_level_of_education:str, lunch:str, test_preparation_course:str, reading_score:int, writing_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def convert_to_df(self):
        try:
            input_data_dict = {
                                'gender': [self.gender],
                                'race_ethnicity': [self.race_ethnicity],
                                'parental_level_of_education': [self.parental_level_of_education],
                                'lunch': [self.lunch],
                                'test_preparation_course': [self.test_preparation_course],
                                'reading_score': [self.reading_score],
                                'writing_score': [self.writing_score]
                              }
            
            input_df = pd.DataFrame(input_data_dict)
            return input_df
        
        except Exception as e:
            raise CustomException(e, sys)