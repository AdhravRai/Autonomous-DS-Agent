import sys
import numpy as np
import pandas as pd
from src.logger.logger import logging
from src.exceptions.exceptions import CustomException
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

class Evaluation:

    def __init__(self):
        logging.info("Evaluation module initialized.")
    def evaluate_model(self, y_true, y_pred, model_type):
        try:
            if model_type == "regression":
                metrics = {
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "R2": r2_score(y_true, y_pred)
                }
            else:    
                metrics = {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall": recall_score(y_true, y_pred, zero_division=0),
                    "F1": f1_score(y_true, y_pred, zero_division=0),
                    "ROC_AUC": roc_auc_score(y_true, y_pred)
                }           
            return metrics
        except Exception as e:
            raise CustomException(e,sys)    
    def evaluate_all_models(
        self,
        trained_models,
        X_test,
        y_test,
        model_type
    ):
        try:
            results = {}
            for model_name, model in trained_models.items():
                logging.info(f"Evaluating {model_name}")
                y_pred = model.predict(X_test)
    
                metrics = self.evaluate_model(
                    y_true=y_test,
                    y_pred=y_pred,
                    model_type=model_type
                )
                results[model_name] = metrics
                logging.info(f"{model_name} evaluated successfully.")
            return results
        except Exception as e:
            raise CustomException(e, sys)
    def compare_models(self, evaluation_results):

        try:
            comparison_df = pd.DataFrame.from_dict(
                evaluation_results,
                orient="index"
            )
    
            comparison_df.index.name = "Model"
            comparison_df.reset_index(inplace=True)
    
            logging.info("Model comparison dataframe created successfully.")
    
            return comparison_df
    
        except Exception as e:
            raise CustomException(e, sys)


    def select_best_model(
            self,
            comparison_df,
            trained_models,
            model_type
    ):
        """
        Select the best performing model.
        """
    
        try:
    
            if model_type == "regression":
    
                best_model_row = comparison_df.loc[
                    comparison_df["R2"].idxmax()
                ]
    
            else:
    
                best_model_row = comparison_df.loc[
                    comparison_df["F1"].idxmax()
                ]
    
            best_model_name = best_model_row["Model"]
    
            best_model = trained_models[best_model_name]
    
            logging.info(
                f"Best model selected : {best_model_name}"
            )
    
            return best_model_name, best_model, best_model_row
    
        except Exception as e:
            raise CustomException(e, sys)

