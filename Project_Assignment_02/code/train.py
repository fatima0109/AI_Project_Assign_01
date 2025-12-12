"""
Unified Training Script
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from model_baseline_A import build_baseline_a
from model_baseline_C import build_baseline_c
from model_proposed import build_proposed_model

def train_sklearn_model(build_fn, X_train, y_train, model_path):
    """Train sklearn model."""
    print(f"Training {model_path.stem}...")
    model = build_fn()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f" Saved to {model_path}")
    return model

def train_all_models(splits, output_dir="models"):
    """Train all models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    models = {}
    
    # Train Baseline A
    baseline_a = train_sklearn_model(
        build_baseline_a,
        splits["X_train"],
        splits["y_train"],
        output_dir / "baseline_a.joblib"
    )
    models["Baseline A"] = baseline_a
    
    # Train Baseline C
    baseline_c = train_sklearn_model(
        build_baseline_c,
        splits["X_train"],
        splits["y_train"],
        output_dir / "baseline_c.joblib"
    )
    models["Baseline C"] = baseline_c
    
    # Train Proposed Model
    proposed = train_sklearn_model(
        build_proposed_model,
        splits["X_train"],
        splits["y_train"],
        output_dir / "proposed.joblib"
    )
    models["Proposed"] = proposed
    
    # Evaluate on validation set
    print("\n Validation Performance:")
    for name, model in models.items():
        y_pred = model.predict(splits["X_val"])
        acc = accuracy_score(splits["y_val"], y_pred)
        f1 = f1_score(splits["y_val"], y_pred, average='weighted')
        print(f"{name:15} Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return models