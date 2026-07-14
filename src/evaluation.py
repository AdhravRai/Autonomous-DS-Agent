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
import shap
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path

class Evaluation:

    def __init__(self):
        self.eval_dir = Path("artifacts") / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Evaluation module initialized.")
    def perform_evaluation(
        self,
        trained_models,
        X_train,
        X_test,
        y_train,
        y_test,
        model_type,
        feature_names
        ):
        try:
            logging.info("Evaluation started.")
            evaluation_results = self.evaluate_all_models(trained_models,X_test,y_test,model_type)
            comparison_df = self.compare_models(evaluation_results)
    
            csv_path, json_path = self.save_metrics(comparison_df)
            (best_model_name,best_model,best_metrics) = self.select_best_model(comparison_df,trained_models, model_type)
            model_path = self.save_best_model(best_model)
            feature_importance_path = (
                self.generate_feature_importance(
                    best_model,
                    feature_names
                )
            )
            shap_paths = self.generate_shap(best_model,X_train,feature_names)
            logging.info("Evaluation completed successfully.")
    
            return {
                "comparison_results": comparison_df,
                "problem_type": model_type,
                "best_model_name": best_model_name,
                "best_metrics": best_metrics,
                "best_model_path": model_path,
                "metrics_csv": csv_path,
                "metrics_json": json_path,
                "feature_importance": feature_importance_path,
                "shap_paths": shap_paths
             }

        except Exception as e:
            raise CustomException(e, sys)
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
    def evaluate_all_models(self,trained_models,X_test,y_test,model_type):
        try:
            results = {}
            for model_name, model in trained_models.items():
                logging.info(f"Evaluating {model_name}")
                y_pred = model.predict(X_test)
    
                metrics = self.evaluate_model(y_true=y_test,y_pred=y_pred,model_type=model_type)
                results[model_name] = metrics
                logging.info(f"{model_name} evaluated successfully.")
            return results
        except Exception as e:
            raise CustomException(e, sys)
    def compare_models(self, evaluation_results):
        try:
            comparison_df = pd.DataFrame.from_dict(evaluation_results,orient="index" )    
            comparison_df.index.name = "Model"
            comparison_df.reset_index(inplace=True)
    
            logging.info("Model comparison dataframe created successfully.")    
            return comparison_df    
        except Exception as e:
            raise CustomException(e, sys)
    def select_best_model(self,comparison_df,trained_models, model_type):
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
    def save_best_model(self, best_model):
        try:    
            model_path = self.eval_dir / "best_model.pkl"    
            joblib.dump(best_model, model_path)    
            logging.info(f"Best model saved at {model_path}")    
            return model_path    
        except Exception as e:
            raise CustomException(e, sys)
    def save_metrics(self, comparison_df):
        try:
            csv_path = self.eval_dir / "model_comparison.csv"
            json_path = self.eval_dir / "model_comparison.json"
            comparison_df.to_csv(
                csv_path,
                index=False
            )
            comparison_df.to_json(
                json_path,
                orient="records",
                indent=4
            )
            logging.info("Evaluation metrics saved successfully.")   
            return csv_path, json_path

        except Exception as e:
            raise CustomException(e, sys)
    def generate_shap(self,best_model,X_train,feature_names):
        try:    
            if not hasattr(best_model, "feature_importances_"):    
                logging.info(
                    "SHAP generation skipped for non-tree model."
                )    
                return None    
            explainer = shap.TreeExplainer(best_model)    
            shap_values = explainer(X_train)   
            shap.summary_plot(
                shap_values,
                X_train,
                feature_names=feature_names,
                show=False
            )    
            summary_path = self.eval_dir / "shap_summary.png"    
            plt.savefig( summary_path,bbox_inches="tight" )    
            plt.close()    
            shap.plots.waterfall(
                 shap_values[0],
                 show=False
             )    
            waterfall_path = self.eval_dir / "shap_waterfall.png"    
            plt.savefig(waterfall_path,bbox_inches="tight")    
            plt.close()    
            logging.info("SHAP plots generated successfully.")    
            return {
                    "summary_plot": str(summary_path),
                    "waterfall_plot": str(waterfall_path)
            } 
        except Exception as e:
            raise CustomException(e, sys)
    def generate_feature_importance(self,best_model,feature_names):
        try:
            if not hasattr(best_model, "feature_importances_"):
                logging.info(
                    "Feature importance not available for this model."
                )
                return None
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": best_model.feature_importances_
            })
            importance_df.sort_values(by="Importance",ascending=False,inplace=True)
            feature_importance_path = (self.eval_dir / "feature_importance.csv")    
            importance_df.to_csv(feature_importance_path,index=False)
    
            logging.info("Feature importance generated successfully.")
    
            return feature_importance_path
    
        except Exception as e:
            raise CustomException(e, sys)
        
        