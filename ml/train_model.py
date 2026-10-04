"""
OrbitRoute AI - Machine Learning Model Trainer
Trains a Random Forest Classifier to evaluate inter-satellite route viability.
Computes real classification metrics, confusion matrix, and feature importances.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_routing_model(
    dataset_path: str = "data/orbitroute_dataset.csv",
    model_save_path: str = "models/random_forest.pkl",
    metrics_save_path: str = "results/model_metrics.json"
):
    print(f"[Training] Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        from ml.data_generator import save_dataset
        save_dataset(dataset_path)

    df = pd.read_csv(dataset_path)

    # Clean dataset & check missing values
    df = df.dropna()

    # Feature definitions
    feature_cols = [
        "distance_km",
        "signal_strength",
        "bandwidth_mbps",
        "latency_ms",
        "packet_loss_percent",
        "energy_consumption",
        "link_quality",
        "satellite_load",
        "queue_length",
        "relative_velocity",
        "connection_stability",
        "number_of_hops"
    ]

    # Categorical encoders for source and destination satellite IDs
    src_encoder = LabelEncoder()
    dst_encoder = LabelEncoder()
    df["source_encoded"] = src_encoder.fit_transform(df["source_satellite"])
    df["destination_encoded"] = dst_encoder.fit_transform(df["destination_satellite"])

    all_features = ["source_encoded", "destination_encoded"] + feature_cols
    X = df[all_features]
    y = df["optimal_route"]

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[Training] Train set size: {len(X_train)} samples, Test set size: {len(X_test)} samples")

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=160,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Evaluate on Test Set
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature Importance analysis
    importances = rf_model.feature_importances_
    feature_importance_dict = {
        feat: round(float(imp), 4)
        for feat, imp in zip(all_features, importances)
    }
    # Sort descending
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Confusion Matrix: {cm}")

    # Prepare package with encoders and model
    model_package = {
        "model": rf_model,
        "src_encoder": src_encoder,
        "dst_encoder": dst_encoder,
        "feature_names": all_features,
        "numerical_features": feature_cols,
        "score_min": float(df["routing_score"].min()),
        "score_max": float(df["routing_score"].max())
    }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model_package, model_save_path)
    print(f"[Model] Saved trained Random Forest artifact to {model_save_path}")

    # Prepare metrics JSON
    metrics = {
        "model_name": "Random Forest Classifier",
        "dataset_total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "feature_importance": dict(sorted_features),
        "features_list": all_features,
        "hyperparameters": {
            "n_estimators": 160,
            "max_depth": 12,
            "criterion": "gini",
            "min_samples_split": 4,
            "min_samples_leaf": 2
        }
    }

    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved evaluation metrics to {metrics_save_path}")

    return model_package, metrics

if __name__ == "__main__":
    train_routing_model()
