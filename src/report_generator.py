import sys  
from pathlib import Path
from src.exceptions.exceptions import CustomException
from src.logger.logger import logging
import json
from datetime import datetime

class ReportGenerator:
    def __init__(self):
        self.reports_dir = Path("artifacts")/"reports"
        self.report_json= self.reports_dir/"report.json"
        self.report_txt= self.reports_dir/"report.txt"
        self.reports_dir.mkdir(parents=True,exist_ok=True)

    def build_report(self,eda_results,evaluation_results,target):
        logging.info("Building report.")
        return {    
        "Dataset Summary":eda_results["summary"],
        "Problem Type":evaluation_results["problem_type"],
        "Target Column":target,
        "Best Model":evaluation_results["best_model_name"],
        "Performance":
            evaluation_results["best_metrics"],
        "Feature Importance":
            evaluation_results["feature_importance"],
        "SHAP":
            evaluation_results["shap_paths"],
        "Generated on : ":datetime.now().isoformat()
        }
    def save_report(self,report):
        with open(self.report_json, "w") as f:
            json.dump(report, f, indent=4, default=str)
        with open(self.report_txt,"w") as f:
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
        return {
        "json_report": str(self.report_json),
        "text_report": str(self.report_txt)
        }
    def generate_report(self,eda_results,evaluation_results,target):
        report =self.build_report(eda_results,evaluation_results,target)
        report_paths = self.save_report(report)
        return {
        "report": report,
        "report_paths": report_paths
        }
