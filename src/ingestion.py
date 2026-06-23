from pathlib import Path
import sys
import pandas as pd 

from src.logger.logger import logging
from src.exceptions.exceptionz import CustomException

class DataIngestion:
    def load_data(self,file_path)->pd.DataFrame:
        try:
            logging.info("Data Ingestion started")
            new_path=self.validate_file(file_path)
            df=pd.read_csv(new_path)
            logging.info("Data loaded successfully")
            logging.info(f"Shape of data is {df.shape}")            

            return df
        except Exception as e:
            raise CustomException(e,sys)
    def validate_file(self,file_path):
        try:
            path = Path(file_path)
            if not path.is_file():
                raise ValueError("file does not exist")
            
            if path.suffix !=".csv":
                raise ValueError("file is not of correct type")

            return path
        except Exception as e:
            raise CustomException(e,sys)
