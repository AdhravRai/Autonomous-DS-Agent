import pandas as pd
import sys
from pathlib import Path
from src.logger.logger import logging
from src.exceptions.exceptions import CustomException

class EDA:
    def __init__(self):
        self.eda_dir=Path("artifacts")/"eda"
        self.num_dir=self.eda_dir/"numerical"
        self.cat_dir=self.eda_dir/"categorical"
        self.eda_dir.mkdir(parents=True,exist_ok=True)
        self.num_dir.mkdir(parents=True,exist_ok=True)
        self.cat_dir.mkdir(parents=True,exist_ok=True)
    def perform_eda(self,df,target):
        try:
            summary=self.generate_summary(df)
            self.analyze_num_cols(df,target)
            self.analyze_cat_cols(df,target)
            corr_matrix=self.analyze_correlations(df,target)
            
            return {
                summary,
                corr_matrix
            }
        except Exception as e:
            raise CustomException(e,sys)
    def get_numerical_cols(self,df):
         try:
            num_cols=df.select_dtypes(include=["number"]).columns.tolist()
            return num_cols
         except Exception as e:
            raise CustomException(e,sys)
    def get_categorical_cols(self,df):
        try:
            cat_cols=df.select_dtypes(include=["object"]).columns.tolist()
            return cat_cols
        except Exception as e:
            raise CustomException(e,sys)
    def generate_summary(self,df):
        try:
            num_cols=self.get_numerical_cols(df)
            cat_cols=self.get_categorical_cols(df)
            summary={
                "rows":len(df),
                "columns": len(df.columns),
                "description" : df.describe(),
                "numerical_columns_count":len(num_cols),
                "categorical_columns_count":len(cat_cols)
            }
            return summary
        except Exception as e:
            raise CustomException(e,sys)
    def analyze_correlations(self,df,target):
        try:
            corr_matrix=df.corr(method="pearson")
            if pd.api.types.is_numeric_dtype(df[target]) :
                target_corr=corr_matrix[target]
                target_corr=target_corr.drop(target)
                target_corr=target_corr.sort_values(key=abs,ascending=False)
                message = None
                return {
                    "corr_matrix": corr_matrix,
                    "target_corr": target_corr,
                    "message": message
                }
            else :
                target_corr =None
                message = "Target is categorical. Pearson correlation skipped"
                return{
                    "corr_matrix": corr_matrix,
                    "target_corr": target_corr,
                    "message": message                    
                }

        except Exception as e:
            raise CustomException(e,sys)

    def analyze_num_cols(self,df,target):
         try:
            pass
         except Exception as e:
            raise CustomException(e,sys)
    def analyze_cat_cols(self,df,target):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)

    
