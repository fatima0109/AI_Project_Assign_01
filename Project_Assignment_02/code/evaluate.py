"""
Unified Evaluation Script
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from pathlib import Path

def evaluate_models(models, X_test, y_test, output_dir="results"):
    """Evaluate all models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    results = []
    predictions = {}
    
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n📈 Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1_Score": round(f1, 4)
        })
        
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        
        # Save detailed report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(output_dir / f"{name.lower().replace(' ', '_')}_report.csv")
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(output_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.csv")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Generate plots
    generate_plots(predictions, y_test, results_df, plots_dir)
    
    return results_df, predictions

def generate_plots(predictions, y_test, results_df, plots_dir):
    """Generate all required plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df["Accuracy"], width, label="Accuracy", alpha=0.8)
    plt.bar(x + width/2, results_df["F1_Score"], width, label="F1 Score", alpha=0.8)
    
    plt.xlabel("Models", fontsize=12, fontweight='bold')
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.title("Model Performance Comparison - Assignment 3", fontsize=14, fontweight='bold')
    plt.xticks(x, results_df["Model"], rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison.pdf", format='pdf', dpi=300)
    plt.close()
    
    # Plot 2: Confusion Matrices
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues')
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f'{name}\nAccuracy: {acc:.3f}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices.pdf", format='pdf', dpi=300)
    plt.close()
    
    print(f"\n✅ Plots saved to {plots_dir}/")