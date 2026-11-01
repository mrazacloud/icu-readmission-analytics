import os
import json
import logging
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Setup structured logging to output clean JSON arrays for AWS CloudWatch parsing
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ICU_Pipeline")

class ICUDataPipeline:
    def __init__(self, db_url: str = None):
        """Initializes database connectivity engine."""
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.engine = create_engine(self.db_url) if self.db_url else None

    def log_metric(self, stage: str, metric_name: str, value: float, unit: str = "Count"):
        """Outputs a structured JSON telemetry log for FinOps tracking."""
        telemetry = {
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "Stage": stage,
            "MetricName": metric_name,
            "Value": value,
            "Unit": unit
        }
        logger.info(json.dumps(telemetry))

    def extract_ehr_data(self) -> pd.DataFrame:
        """Extracts electronic health records or builds a synthetic fallback matrix."""
        start_time = time.time()
        
        if self.engine:
            query = """
            SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
                   v.heart_rate, v.sys_bp, v.temperature, v.glucose
            FROM admissions a
            JOIN vitalsigns v ON a.hadm_id = v.hadm_id
            """
            df = pd.read_sql(query, self.engine)
        else:
            logger.warning("No database connected. Injecting synthetic MIMIC-IV schema.")
            np.random.seed(42)
            records = 2000
            df = pd.DataFrame({
                "subject_id": np.random.randint(10000, 99999, size=records),
                "hadm_id": np.random.randint(100000, 199999, size=records),
                "heart_rate": np.random.choice([np.nan, 72, 85, 90, 110, 60], size=records, p=[0.1, 0.3, 0.3, 0.1, 0.1, 0.1]),
                "sys_bp": np.random.choice([np.nan, 120, 135, 140, 95, 160], size=records, p=[0.1, 0.3, 0.3, 0.1, 0.1, 0.1]),
                "temperature": np.random.normal(37.0, 0.8, size=records),
                "glucose": np.random.normal(120, 30, size=records),
                "readmitted_30d": np.random.choice([0, 1], size=records, p=[0.82, 0.18])
            })
            df.loc[df["temperature"] < 35.5, "temperature"] = np.nan

        duration = time.time() - start_time
        self.log_metric("Extraction", "RecordsExtracted", float(len(df)))
        self.log_metric("Extraction", "ExecutionTimeSec", duration, "Seconds")
        return df

    def transform_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates multi-hour patient windows and handles missing clinical biomarkers."""
        start_time = time.time()
        
        # Aggregate features per unique hospital admission
        processed_df = df.groupby(["subject_id", "hadm_id"]).agg({
            "heart_rate": ["mean", "max"],
            "sys_bp": ["mean", "min"],
            "temperature": ["mean"],
            "glucose": ["mean"]
        })
        
        processed_df.columns = ['_'.join(col).strip() for col in processed_df.columns.values]
        processed_df = processed_df.reset_index()

        # Median Imputation Strategy for missing health vitals
        clinical_biomarkers = ["heart_rate_mean", "heart_rate_max", "sys_bp_mean", "sys_bp_min", "temperature_mean", "glucose_mean"]
        for marker in clinical_biomarkers:
            missing_count = processed_df[marker].isnull().sum()
            if missing_count > 0:
                median_val = processed_df[marker].median()
                processed_df[marker] = processed_df[marker].fillna(median_val)
                self.log_metric("Transformation", f"ImputedValues_{marker}", float(missing_count))

        if "readmitted_30d" in df.columns:
            labels = df.groupby(["subject_id", "hadm_id"])["readmitted_30d"].first().reset_index()
            processed_df = pd.merge(processed_df, labels, on=["subject_id", "hadm_id"])

        duration = time.time() - start_time
        self.log_metric("Transformation", "ExecutionTimeSec", duration, "Seconds")
        return processed_df

if __name__ == "__main__":
    pipeline = ICUDataPipeline()
    raw_data = pipeline.extract_ehr_data()
    clean_features = pipeline.transform_and_impute(raw_data)
    print(f"\n✅ ETL Complete! Extracted and cleaned matrix shape: {clean_features.shape}")