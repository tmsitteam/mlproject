import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


#class level variables
@dataclass
class DataIngestionConfig:
    source_data_path : str = os.path.join('artifacts', 'data.csv')
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion initiated')
        try:
            df = pd.read_csv('data_source/stud.csv')
            logging.info('Read source data as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.source_data_path),exist_ok=True)
            logging.info('Artifacts  directory created')

            df.to_csv(self.ingestion_config.source_data_path, index = False, header = True)
            logging.info('Source data read')

            train_data, test_data =  train_test_split(df, test_size = 0.2, random_state = 42)
            logging.info('Train-Test split is done')

            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info('Data ingestion completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__  == '__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    tansformed_train_data, transformed_test_data, _ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    best_r2_score = modeltrainer.model_trainer(tansformed_train_data, transformed_test_data)