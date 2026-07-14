import pandas as pd
import sys
from src.logger.logger import logging
from src.exceptions.exceptions import CustomException
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso

class ModelTraining:
    def __init__(self):
        pass
    def perform_model_training(self,X_preprocessed,y,feature_names):
        try:
            logging.info("Model Training started")
            model_type=self.detect_problem_type(y)
            X_train,X_test,y_train,y_test=self.split_data(X_preprocessed,y)
            trained_models=self.train_models(X_train,y_train,model_type)
            return trained_models,X_train,X_test,y_train,y_test,model_type,feature_names
        except Exception as e:
            raise CustomException(e,sys)
    def detect_problem_type(self,y):
        try:
            logging.info("detecting problem type")
            model_type=""
            if pd.api.types.is_numeric_dtype(y) and y.nunique() >5:
                model_type="regression"
            else:
                model_type="classification"
            return model_type
        except Exception as e:
            raise CustomException(e,sys)
    def split_data(self,X_preprocessed,y):
        try:
            logging.info("Dividing into training and testing data")
            X_train,X_test,y_train,y_test=train_test_split(X_preprocessed,y,test_size=0.2,random_state=42)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            raise CustomException(e,sys)
    def train_models(self,X_train,y_train,model_type):
        try:
            logging.info("training the models")
            if model_type == 'classification' :
                models={
                    "Logistic Regression":LogisticRegression(),
                    "Decision Tree":DecisionTreeClassifier(),
                    "Random Forest":RandomForestClassifier(),
                    "Gradient Boost":GradientBoostingClassifier()
                }
                for model in models.values():
                    model.fit(X_train, y_train) 
            else :
                models = {
                    "Linear Regression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boost":GradientBoostingRegressor(),
                    "Random Forest Regressor": RandomForestRegressor()                  
                }                
                for model in models.values():
                    model.fit(X_train, y_train) 
            return models
        except Exception as e:
            raise CustomException(e,sys)
