import time
import json
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import shap
from data_pipeline import ICUDataPipeline

class ICUModelTrainer:
    def __init__(self):
        self.model = None
        self.explainer = None

    def train_predictive_pipeline(self, data, feature_names: list):
        """Trains an enterprise Gradient Boosting model and logs FinOps telemetry."""
        pipeline = ICUDataPipeline()
        start_time = time.time()

        # Isolate features and target safely
        X = data[feature_names].values
        y = data["readmitted_30d"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a robust Scikit-Learn Gradient Boosting Classifier
        self.model = GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Evaluate predictive performance matrices (AUC)
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_prob)
        
        duration = time.time() - start_time
        pipeline.log_metric("ModelTraining", "Model_AUC", float(auc_score), "Ratio")
        pipeline.log_metric("ModelTraining", "TrainingDurationSec", duration, "Seconds")

        print(f"\n⚡ Model Training Complete. Test Set AUC: {auc_score:.4f}")
        
        # Healthcare Explainability Integration using SHAP explainer objects
        print("Computing SHAP values for clinical validation...")
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer(X_test[:100])
        
        mean_shap = np.abs(shap_values.values).mean(axis=0)
        pipeline.log_metric("Explainability", "TopFeature_SHAP_Impact", float(np.max(mean_shap)), "Score")

if __name__ == "__main__":
    # Execute full data-to-model sequence pipeline locally
    pipeline = ICUDataPipeline()
    features = pipeline.transform_and_impute(pipeline.extract_ehr_data())
    
    target_features = ["heart_rate_mean", "heart_rate_max", "sys_bp_mean", "sys_bp_min", "temperature_mean", "glucose_mean"]
    
    trainer = ICUModelTrainer()
    trainer.train_predictive_pipeline(features, target_features)
