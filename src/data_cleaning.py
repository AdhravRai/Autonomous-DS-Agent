import pandas as pd
import sys
from src.logger.logger import logging
from src.exceptions.exceptions import CustomException

class DataCleaning:
    def clean_data(self,df):
        try:
            logging.info("Data Cleaning started")
            logging.info("Standardizing column names")
            df=self.standardize_column_names(df)

            logging.info("Removing  duplicates rows")
            df=self.remove_duplicates(df)
            logging.info("Handling missing values")
            df=self.handle_missing_values(df)
            logging.info("Correcting data types")
            df=self.correct_data_types(df)
            logging.info("Data cleaning completed successfully")
            return df
        except Exception as e:
            raise CustomException(e,sys)
    def standardize_column_names(self,df):
        try:
            df_cols=[]
            for col in df.columns:
                col=col.strip().lower().replace(" ","_")
                df_cols.append(col)
            df.columns=df_cols
            return df
        except Exception as e:
            raise CustomException(e,sys)
    def remove_duplicates(self,df):
        try:
            initial_rows=len(df)
            df=df.drop_duplicates(keep="first")
            final_rows=len(df)
            removed = initial_rows-final_rows
            logging.info(f"Removed {removed} duplicate rows")
            return df
        except Exception as e:
            raise CustomException(e,sys)
    def handle_missing_values(self,df):
        try:
            df_cleaned=df.loc[:,df.isnull().mean() <0.7]
            logging.info(f"Removing {len(df.columns)-len(df_cleaned.columns)} colums due to 70 % or more missing values")
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == object:
                    df_cleaned[col]=df_cleaned[col].fillna(df_cleaned[col].mode()[0])
                else :
                    df_cleaned[col]=df_cleaned[col].fillna(df_cleaned[col].median())
            return df_cleaned
        except Exception as e:
            raise CustomException(e,sys)
    def correct_data_types(self,df):
        try:
            for col in df.columns:
                if df[col].dtype == object:
                        ratio1=pd.to_numeric(df[col],errors="coerce").notna().mean()
                        ratio2=pd.to_datetime(df[col],errors="coerce").notna().mean()
                        if ratio1 >0.9:
                             df[col]=pd.to_numeric(df[col],errors="coerce")
                        elif ratio2 >0.9:
                            df[col]=pd.to_datetime(df[col],errors="coerce")
                        else :
                            continue
 
            return df
        except Exception as e:
            raise CustomException(e,sys)
