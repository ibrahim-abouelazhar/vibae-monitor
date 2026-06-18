import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

CSV_PATH = "data/feature_time_48k_2048_load_1.csv"
MODEL_SAVE_PATH = "data/processed/baseline_model.pkl"

def clean_label(label_str):
    if label_str.startswith("Normal"):
        return "Normal"
    parts = label_str.split("_")
    # e.g., 'Ball_007_1' -> 'Ball_007', 'OR_007_6_1' -> 'OR_007'
    return f"{parts[0]}_{parts[1]}"

def train_baseline_rf():
    print("Training Baseline Random Forest model...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Baseline CSV dataset not found at: {CSV_PATH}")
        
    # 1. Load data
    df = pd.read_csv(CSV_PATH)
    
    # 2. Extract features and labels
    feature_cols = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
    X = df[feature_cols].values
    y_raw = df['fault'].values
    y = np.array([clean_label(lbl) for lbl in y_raw])
    
    # 3. Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Create Pipeline with Scaling and RF
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    mlflow.set_experiment("VibAE-Monitor-2D")
    
    with mlflow.start_run(run_name="Baseline_RandomForest") as run:
        mlflow.log_param("model_family", "baseline_classifier")
        mlflow.log_param("model_type", "random_forest_1d")
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Baseline Random Forest Accuracy: {acc:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        # Log individual class metrics to MLflow
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"f1_score_{label}", metrics["f1-score"])
                
        # Save model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        with open(MODEL_SAVE_PATH, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"Baseline model saved to {MODEL_SAVE_PATH}")
        
        # Log artifact to MLflow
        mlflow.sklearn.log_model(pipeline, "baseline_rf_model")
        print("Baseline model successfully tracked in MLflow.")
        
    return pipeline

if __name__ == "__main__":
    train_baseline_rf()
