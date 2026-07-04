import pandas as pd
import sys
from pathlib import Path
from src.logger.logger import logging
from src.exceptions.exceptions import CustomException
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeatureEng:
    def __init__(self):
        self.fe_dir=Path("artifacts")/"FE"
        self.fe_dir.mkdir(parents=True,exist_ok=True)
        self.preprocessor_path=self.fe_dir/"preprocessor.pkl"
    def perform_feature_engineering(self,df,target):
        try:
            logging.info("Feature Engineering started")
            X,y=self.split_features_target(df,target)
            preprocessor=self.build_preprocessor(X)
            X_preprocessed=self.fit_transform(preprocessor,X)
            self.save_preprocessor(preprocessor)
            return X_preprocessed,y,preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def split_features_target(self,df,target):
        try:
            logging.info("Splitting features and target")
            X=df.drop(columns=[target])
            y=df[target]
            return X,y
        except Exception as e:
            raise CustomException(e,sys)
    def build_preprocessor(self,X):
        try:
            logging.info("Building preprocessing pipeline")
            num_cols=X.select_dtypes(include=["number"]).columns.tolist()
            cat_cols=X.select_dtypes(include=["object"]).columns.tolist()
            num_pipeline=Pipeline(
                steps=[
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
                ]
            )            
            preprocessor = ColumnTransformer(
                [
                     ("categorical", cat_pipeline, cat_cols),
                      ("numerical", num_pipeline, num_cols)
                ],
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def fit_transform(self,preprocessor,X):
        try:
            logging.info("Applying feature transformations")
            X_preprocessed=preprocessor.fit_transform(X)
            return X_preprocessed
        except Exception as e:
            raise CustomException(e,sys)
    def save_preprocessor(self,preprocessor):
        try:
            logging.info("Saving preprocessor artifact")
        except Exception as e:
            raise CustomException(e,sys)