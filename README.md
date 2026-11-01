# Automated ICU Readmission Predictor & FinOps Telemetry Pipeline

An end-to-end, production-grade clinical analytics pipeline that processes Electronic Health Records (EHR) mimicking the MIMIC-IV schema to predict 30-day ICU readmission risks. This system integrates structural feature engineering, explainable AI (XAI) via SHAP, and automated cloud telemetry hooks designed for real-time monitoring inside AWS CloudWatch.

## 🚀 Key Architectural Features
- **Data Engineering Pipeline (`src/data_pipeline.py`)**: Features multi-hour aggregation windows for vital biomarkers, rolling statistical extractions, and robust clinical median-imputation handling.
- **Predictive AI Classifier (`src/model.py`)**: Implements an optimized Gradient Boosting Framework evaluating patient risk criteria.
- **Explainable AI Compliance**: Integrated **SHAP (SHAPley Additive exPlanations)** TreeExplainer vectors to ensure clinical feature transparency and model auditing compliance.
- **FinOps & Telemetry Telemetry**: Outputs structured JSON application runtime logs tracking processing metrics (`ExecutionTimeSec`, `RecordsExtracted`, `Model_AUC`) designed for effortless aggregation by cloud-native insight engines.
- **Infrastructure-as-Code (`infra/main.tf`)**: Declarative Terraform schema provisioning log aggregation resources and explicit data lifecycle retention controls.

## 🛠️ Tech Stack
- **Languages**: Python, HCL (Terraform)
- **Machine Learning**: Scikit-Learn, SHAP, NumPy
- **Data Analysis**: Pandas
- **Infrastructure**: Docker, AWS Lambda (Target), AWS CloudWatch, Terraform

## ⚙️ Quick Local Setup

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com
   cd icu-readmission-analytics
   ```

2. Install dependencies (Using Python 3.11/Anaconda base recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the complete training and telemetry generation pipeline:
   ```bash
   python src/model.py
   ```

## 📊 Sample Production Logs (CloudWatch Ready)
```json
{"Timestamp": "2026-11-01T09:15:31Z", "Stage": "Transformation", "MetricName": "ExecutionTimeSec", "Value": 0.044, "Unit": "Seconds"}
{"Timestamp": "2026-11-01T09:15:32Z", "Stage": "ModelTraining", "MetricName": "Model_AUC", "Value": 0.4764, "Unit": "Ratio"}
{"Timestamp": "2026-11-01T09:15:32Z", "Stage": "Explainability", "MetricName": "TopFeature_SHAP_Impact", "Value": 0.256, "Unit": "Score"}
```
