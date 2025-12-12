# File: run_experiments.py
"""
Assignment 3
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("ASSIGNMENT 3 - FINAL CORRECTED SOLUTION")

# STEP 1: LOAD AND EXPLORE DATA 
print("\n STEP 1: EXPLORING DATA COLUMNS")

train_path = Path("QEvasion/data/train-00000-of-00001.parquet")
train_df = pd.read_parquet(train_path)

print(f"Train shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")

# Check text columns more carefully
print("\n TEXT COLUMNS ANALYSIS:")
text_columns = ['affirmative_questions', 'question', 'interview_question', 'interview_answer', 'gpt3.5_summary']
for col in text_columns:
    if col in train_df.columns:
        sample = train_df[col].iloc[0]
        length = len(str(sample))
        print(f"  {col:25s}: Sample length = {length:4d} chars, Sample preview: '{str(sample)[:100]}...'")

# Choose the best text column
text_col = None
for col in ['question', 'interview_question', 'interview_answer', 'gpt3.5_summary']:
    if col in train_df.columns:
        avg_len = train_df[col].astype(str).str.len().mean()
        if avg_len > 20:  # Reasonable text length
            text_col = col
            print(f"\n Selected text column: '{text_col}' (avg length: {avg_len:.0f} chars)")
            break

if text_col is None:
    text_col = 'question' if 'question' in train_df.columns else train_df.columns[0]
    print(f"\n  Using fallback text column: '{text_col}'")

label_col = 'evasion_label'
print(f"Using label column: '{label_col}'")

# Remove empty labels
train_df = train_df[train_df[label_col].notna() & (train_df[label_col].astype(str).str.strip() != '')].copy()

# Encode labels
le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df[label_col])

print(f"\n DATASET SUMMARY:")
print(f"  Samples: {len(train_df)}")
print(f"  Classes: {len(le.classes_)}")
print(f"  Class distribution:")
for i, cls in enumerate(le.classes_):
    count = (train_df['label_encoded'] == i).sum()
    percentage = count / len(train_df) * 100
    print(f"    {cls:25s}: {count:4d} ({percentage:5.1f}%)")

# STEP 2: PREPROCESS TEXT
print("\n STEP 2: PREPROCESSING TEXT")

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = ' '.join(text.split())
    return text

train_df['text_clean'] = train_df[text_col].astype(str).apply(clean_text)

print(f"Sample cleaned text: '{train_df['text_clean'].iloc[0][:150]}...'")
print(f"Text length stats:")
print(f"  Min: {train_df['text_clean'].str.len().min()} chars")
print(f"  Max: {train_df['text_clean'].str.len().max()} chars")
print(f"  Mean: {train_df['text_clean'].str.len().mean():.0f} chars")

# STEP 3: SPLIT DATA
print("\n STEP 3: SPLITTING DATA")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    train_df['text_clean'],
    train_df['label_encoded'],
    test_size=0.2,
    random_state=42,
    stratify=train_df['label_encoded']
)

# Further split train into train/validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

print(f"Training samples: {len(X_train_final)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# STEP 4: CREATE MODELS
print("\n STEP 4: CREATING MODELS")

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_final),
    y=y_train_final
)
class_weight_dict = dict(zip(np.unique(y_train_final), class_weights))

print("Class weights for balancing:")
for i, cls in enumerate(le.classes_):
    if i in class_weight_dict:
        print(f"  {cls:25s}: weight = {class_weight_dict[i]:.2f}")

# Model 1: Baseline A
baseline_a = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 1)
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='lbfgs'
    ))
])

# Model 2: Baseline C
baseline_c = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2)
    )),
    ('clf', LinearSVC(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ))
])

# Model 3: Proposed (Enhanced)
proposed = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        max_iter=2000,
        C=0.5,
        class_weight=class_weight_dict,
        solver='lbfgs',
        random_state=42
    ))
])

# Model 4: Random Forest (for comparison)
rf_model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2)
    )),
    ('clf', RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

models = {
    'Baseline_A': baseline_a,
    'Baseline_C': baseline_c,
    'Proposed': proposed,
    'RandomForest': rf_model
}

print(f"\nCreated {len(models)} models")

# STEP 5: TRAIN MODELS
print("\n STEP 5: TRAINING MODELS")

validation_results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    try:
        model.fit(X_train_final, y_train_final)
        
        # Validation performance
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        validation_results.append({
            'Model': name,
            'Val_Accuracy': val_acc,
            'Val_F1': val_f1
        })
        
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation F1 Score: {val_f1:.4f}")
        
    except Exception as e:
        print(f"  ❌ Training failed: {e}")

# STEP 6: TEST EVALUATION
print("\n STEP 6: TEST EVALUATION")

test_results = []
predictions = {}
detailed_reports = {}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    try:
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        test_results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'F1_Score': round(f1, 4)
        })
        
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test F1 Score: {f1:.4f}")
        
        # Get detailed report
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        detailed_reports[name] = report
        
        # Print top 3 classes by F1
        class_f1_scores = []
        for i, cls in enumerate(le.classes_):
            if cls in report:
                class_f1_scores.append((cls, report[cls]['f1-score']))
        
        class_f1_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top 3 classes by F1:")
        for cls, f1_score_val in class_f1_scores[:3]:
            print(f"    {cls:25s}: {f1_score_val:.4f}")
            
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")

# STEP 7: CREATE SIMPLE ENSEMBLE
print("\n STEP 7: CREATING ENSEMBLE")

if len(predictions) >= 2:
    # Simple majority voting
    pred_arrays = list(predictions.values())
    ensemble_preds = []
    
    for i in range(len(pred_arrays[0])):
        votes = [pred[i] for pred in pred_arrays]
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]
        ensemble_preds.append(winner)
    
    ensemble_preds = np.array(ensemble_preds)
    predictions['Ensemble'] = ensemble_preds
    
    acc = accuracy_score(y_test, ensemble_preds)
    f1 = f1_score(y_test, ensemble_preds, average='weighted')
    
    test_results.append({
        'Model': 'Ensemble',
        'Accuracy': round(acc, 4),
        'F1_Score': round(f1, 4)
    })
    
    print(f"Ensemble Accuracy: {acc:.4f}")
    print(f"Ensemble F1 Score: {f1:.4f}")

# STEP 8: SAVE RESULTS
print("\n STEP 8: SAVING RESULTS")

output_dir = Path("assignment3_final_corrected")
output_dir.mkdir(exist_ok=True)

# Save models
models_dir = output_dir / "models"
models_dir.mkdir(exist_ok=True)

for name, model in models.items():
    try:
        joblib.dump(model, models_dir / f"{name}.joblib")
        print(f"   Saved {name} model")
    except Exception as e:
        print(f"   Failed to save {name}: {e}")

# Save results
results_df = pd.DataFrame(test_results)
results_df.to_csv(output_dir / "model_comparison.csv", index=False)
print(f"   Saved model comparison")

# Save predictions
for name, pred in predictions.items():
    try:
        pred_df = pd.DataFrame({
            'text': X_test.tolist(),
            'true_label': y_test.tolist(),
            'true_label_name': le.inverse_transform(y_test),
            'predicted_label': pred.tolist(),
            'predicted_label_name': le.inverse_transform(pred),
            'correct': (y_test == pred).astype(int).tolist()
        })
        pred_df.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        print(f"   Saved {name} predictions")
    except Exception as e:
        print(f"   Failed to save {name} predictions: {e}")

# STEP 9: CREATE VISUALIZATIONS
print("\n STEP 9: CREATING VISUALIZATIONS")

plots_dir = output_dir / "plots"
plots_dir.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')

# Plot 1: Model Comparison
if len(results_df) > 0:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#73C6B6', '#C73E1D']
    
    bars1 = plt.bar(x - width/2, results_df['Accuracy'], width, 
                   label='Accuracy', color=colors[0], edgecolor='black', alpha=0.8)
    bars2 = plt.bar(x + width/2, results_df['F1_Score'], width, 
                   label='F1 Score', color=colors[1], edgecolor='black', alpha=0.8)
    
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison - Assignment 3', fontsize=14, fontweight='bold')
    plt.xticks(x, results_df['Model'], rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison.pdf", format='pdf', dpi=300)
    plt.close()
    print(f" Created model comparison plot")

# Plot 2: Confusion Matrix for Best Model
if predictions and len(y_test) > 0:
    # Find best model
    if len(results_df) > 0:
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        if best_model_name in predictions:
            y_pred_best = predictions[best_model_name]
            
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred_best)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(cmap='Blues', xticks_rotation=45)
            acc = accuracy_score(y_test, y_pred_best)
            plt.title(f'Confusion Matrix: {best_model_name}\nAccuracy: {acc:.3f}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(plots_dir / "confusion_matrix_best.pdf", format='pdf', dpi=300)
            plt.close()
            print(f" Created confusion matrix for {best_model_name}")

# Plot 3: Class Distribution
plt.figure(figsize=(10, 6))
class_counts = Counter(y_test)
classes = le.classes_
counts = [class_counts.get(i, 0) for i in range(len(classes))]

bars = plt.bar(range(len(classes)), counts, color=colors[:len(classes)], 
               edgecolor='black', alpha=0.8)

plt.xlabel('Classes', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Test Set Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(range(len(classes)), classes, rotation=45, ha='right')

# Add count labels
for bar, count in zip(bars, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, str(count), 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(plots_dir / "class_distribution.pdf", format='pdf', dpi=300)
plt.close()
print(f" Created class distribution plot")

# STEP 10: CREATE SUMMARY
print("\n STEP 10: CREATING SUMMARY")

if len(results_df) > 0:
    best_acc_idx = results_df['Accuracy'].idxmax()
    best_f1_idx = results_df['F1_Score'].idxmax()
    
    summary = {
        'experiment': {
            'name': 'Assignment 3 - QEvasion Text Classification',
            'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'text_column_used': text_col,
            'label_column_used': label_col,
            'note': 'Test set had empty labels - used 80/20 split from training data'
        },
        'dataset': {
            'total_samples': len(train_df),
            'train_samples': len(X_train_final),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'n_classes': len(le.classes_),
            'classes': le.classes_.tolist(),
            'class_imbalance_ratio': max(Counter(y_test).values()) / min(Counter(y_test).values())
        },
        'models': list(models.keys()),
        'results': results_df.to_dict('records'),
        'best_models': {
            'by_accuracy': {
                'model': results_df.loc[best_acc_idx, 'Model'],
                'accuracy': float(results_df.loc[best_acc_idx, 'Accuracy']),
                'f1_score': float(results_df.loc[best_acc_idx, 'F1_Score'])
            },
            'by_f1_score': {
                'model': results_df.loc[best_f1_idx, 'Model'],
                'accuracy': float(results_df.loc[best_f1_idx, 'Accuracy']),
                'f1_score': float(results_df.loc[best_f1_idx, 'F1_Score'])
            }
        },
        'key_insights': [
            f"Dataset has significant class imbalance (ratio: {max(Counter(y_test).values()) / min(Counter(y_test).values()):.1f}:1)",
            f"Best model achieves {results_df['Accuracy'].max():.1%} accuracy",
            f"F1 scores range from {results_df['F1_Score'].min():.1%} to {results_df['F1_Score'].max():.1%}",
            f"Ensemble method {'improved' if 'Ensemble' in results_df['Model'].values and results_df.loc[results_df['Model'] == 'Ensemble', 'Accuracy'].values[0] > results_df.loc[results_df['Model'] != 'Ensemble', 'Accuracy'].max() else 'did not significantly improve'} performance"
        ]
    }
    
    with open(output_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f" Created experiment summary")

# TEP 11: FINAL OUTPUT
print(" ASSIGNMENT 3 - FINAL CORRECTED SOLUTION COMPLETE")

if len(results_df) > 0:
    print(f"\n FINAL RESULTS:")
    print(results_df.to_string(index=False))
    
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"\n BEST MODEL: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   F1 Score: {best_model['F1_Score']:.4f}")

print(f"\n PERFORMANCE ANALYSIS:")
print(f"  1. Used '{text_col}' column for text features (avg length: {train_df['text_clean'].str.len().mean():.0f} chars)")
print(f"  2. Dataset has {len(le.classes_)} classes with significant imbalance")
print(f"  3. All models trained with class balancing techniques")
print(f"  4. Created ensemble for improved performance")
