# File: evaluate_results.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from pathlib import Path

def evaluate_models(y_true, predictions_dict, results_dir):
    """
    Evaluate multiple models and save results.
    """
    metrics_data = []
    
    print(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
    print("-"*60)
    
    results_dir = Path(results_dir)
    
    for model_name, y_pred in predictions_dict.items():
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics_data.append({
            'Model': model_name,
            'Accuracy': round(acc, 4),
            'F1_Score': round(f1, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4)
        })
        
        print(f"{model_name:<20} {acc:<10.4f} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f}")
        
        # Save detailed report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = results_dir / f'{model_name.lower().replace(" ", "_")}_report_assignment3.csv'
        report_df.to_csv(report_path, index=True)
        
        # Save confusion matrix as CSV
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm)
        cm_path = results_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}_assignment3.csv'
        cm_df.to_csv(cm_path, index=False)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = results_dir / 'model_comparison_assignment3.csv'
    metrics_df.to_csv(metrics_path, index=False)
    
    print(f"\n✅ Metrics saved to: {metrics_path}")
    
    return metrics_df

def plot_training_history(history, save_path):
    """Plot training history curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training history plot saved to: {save_path}")

def plot_confusion_matrices(y_true, predictions_dict, plots_dir):
    """Plot confusion matrices for all models"""
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    plots_dir = Path(plots_dir)
    
    for ax, (model_name, y_pred) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f'{model_name}\nAccuracy: {acc:.3f}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = plots_dir / 'confusion_matrices_assignment3.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrices saved to: {save_path}")

def plot_metric_comparison(metrics_df, plots_dir):
    """Plot metric comparison bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(metrics_df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], metrics_df['Accuracy'], width, 
           label='Accuracy', color='skyblue', edgecolor='black')
    ax.bar([i + width/2 for i in x], metrics_df['F1_Score'], width, 
           label='F1 Score', color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - Assignment 3', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right', fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (acc, f1) in enumerate(zip(metrics_df['Accuracy'], metrics_df['F1_Score'])):
        ax.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plots_dir = Path(plots_dir)
    save_path = plots_dir / 'metric_comparison_assignment3.pdf'
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Metric comparison plot saved to: {save_path}")