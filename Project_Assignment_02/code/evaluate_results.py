# File: evaluate_results.py
"""
Comprehensive evaluation and visualization system.
Student 3's responsibility.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve, auc,
    precision_recall_curve
)
import json
import joblib
from pathlib import Path
import torch
from torch.utils.data import DataLoader

class ResultAnalyzer:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
    def generate_comprehensive_report(self, y_true, y_preds_dict, model_names):
        """Generate comprehensive evaluation report"""
        
        report_data = []
        
        for name, y_pred in y_preds_dict.items():
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            report_data.append({
                'Model': name,
                'Accuracy': round(acc, 4),
                'F1-Macro': round(f1_macro, 4),
                'F1-Weighted': round(f1_weighted, 4),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4)
            })
            
            # Save detailed classification report
            cls_report = classification_report(
                y_true, y_pred, 
                output_dict=True,
                zero_division=0
            )
            
            with open(self.out_dir / f'report_{name}.json', 'w') as f:
                json.dump(cls_report, f, indent=4)
        
        # Create comparison dataframe
        df_comparison = pd.DataFrame(report_data)
        df_comparison.to_csv(self.out_dir / 'model_comparison.csv', index=False)
        
        return df_comparison
    
    def plot_confusion_matrices(self, y_true, y_preds_dict, class_names=None):
        """Plot confusion matrices for all models"""
        n_models = len(y_preds_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, y_pred) in zip(axes, y_preds_dict.items()):
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
            ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_true, y_pred):.3f}')
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                           ha="center", va="center",
                           color="white" if cm_normalized[i, j] > thresh else "black")
            
            if class_names:
                tick_marks = np.arange(len(class_names))
                ax.set_xticks(tick_marks)
                ax.set_xticklabels(class_names, rotation=45, ha='right')
                ax.set_yticks(tick_marks)
                ax.set_yticklabels(class_names)
            
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
        
        plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(self.out_dir / 'confusion_matrices.pdf', format='pdf', dpi=300)
        plt.close()
    
    def plot_metric_comparison(self, comparison_df):
        """Plot bar chart comparing all metrics"""
        metrics = ['Accuracy', 'F1-Macro', 'Precision', 'Recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=self.colors[:len(comparison_df)])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.out_dir / 'metric_comparison.pdf', format='pdf', dpi=300)
        plt.close()
    
    def plot_training_history(self, history_dict):
        """Plot training history curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax = axes[0]
        for model_name, history in history_dict.items():
            if 'train_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                ax.plot(epochs, history['train_loss'], 'o-', 
                       label=f'{model_name} Train', alpha=0.7)
            if 'val_loss' in history:
                epochs = range(1, len(history['val_loss']) + 1)
                ax.plot(epochs, history['val_loss'], 's--',
                       label=f'{model_name} Val', alpha=0.7)
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax = axes[1]
        for model_name, history in history_dict.items():
            if 'train_accuracy' in history:
                epochs = range(1, len(history['train_accuracy']) + 1)
                ax.plot(epochs, history['train_accuracy'], 'o-',
                       label=f'{model_name} Train', alpha=0.7)
            if 'val_accuracy' in history:
                epochs = range(1, len(history['val_accuracy']) + 1)
                ax.plot(epochs, history['val_accuracy'], 's--',
                       label=f'{model_name} Val', alpha=0.7)
        ax.set_title('Training and Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.out_dir / 'training_history.pdf', format='pdf', dpi=300)
        plt.close()
    
    def analyze_error_patterns(self, X_test, y_true, y_preds_dict, model_names):
        """Analyze error patterns across models"""
        error_analysis = {}
        
        for name, y_pred in y_preds_dict.items():
            # Find misclassified samples
            misclassified_idx = np.where(y_true != y_pred)[0]
            
            if len(misclassified_idx) > 0:
                error_samples = []
                for idx in misclassified_idx[:10]:  # First 10 errors
                    error_samples.append({
                        'text': X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx],
                        'true_label': y_true[idx],
                        'pred_label': y_pred[idx]
                    })
                
                # Calculate per-class error rates
                unique_classes = np.unique(y_true)
                class_errors = {}
                for cls in unique_classes:
                    cls_idx = np.where(y_true == cls)[0]
                    if len(cls_idx) > 0:
                        cls_errors[cls] = np.sum(y_pred[cls_idx] != cls) / len(cls_idx)
                
                error_analysis[name] = {
                    'total_errors': len(misclassified_idx),
                    'error_rate': len(misclassified_idx) / len(y_true),
                    'error_samples': error_samples,
                    'class_error_rates': class_errors
                }
        
        # Save error analysis
        with open(self.out_dir / 'error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=4, default=str)
        
        return error_analysis
      
if __name__ == "__main__":
    print("evaluate_results.py executed successfully!")
