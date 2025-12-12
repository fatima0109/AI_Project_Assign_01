"""
Fixed Ensemble Model
"""
import numpy as np
import joblib
from collections import Counter

class FixedEnsemble:
    def __init__(self, model_paths=None, weights=None):
        """Fixed ensemble model."""
        self.models = {}
        self.model_names = []
        self.weights = weights
        
        if model_paths:
            for name, path in model_paths.items():
                try:
                    self.models[name] = joblib.load(path)
                    self.model_names.append(name)
                    print(f"✅ Loaded {name}")
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
        
        if self.weights is None and self.models:
            self.weights = [1.0 / len(self.models)] * len(self.models)
    
    def predict(self, X):
        """Majority voting prediction."""
        all_predictions = []
        
        for name in self.model_names:
            if name in self.models:
                pred = self.models[name].predict(X)
                all_predictions.append(pred)
        
        if not all_predictions:
            raise ValueError("No models available for prediction")
        
        # Stack predictions
        stacked = np.column_stack(all_predictions)
        ensemble_preds = []
        
        # Majority voting
        for row in stacked:
            votes = Counter(row)
            winner = votes.most_common(1)[0][0]
            ensemble_preds.append(winner)
        
        return np.array(ensemble_preds)
    
    def predict_proba(self, X):
        """Weighted probability averaging."""
        all_probas = []
        
        for name in self.model_names:
            if name in self.models:
                model = self.models[name]
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    all_probas.append(proba)
                else:
                    # For models without predict_proba (like SVC)
                    pred = model.predict(X)
                    classes = np.unique(pred)
                    n_classes = len(classes)
                    proba = np.zeros((len(pred), n_classes))
                    for i, p in enumerate(pred):
                        idx = np.where(classes == p)[0][0]
                        proba[i, idx] = 1.0
                    all_probas.append(proba)
        
        if not all_probas:
            raise ValueError("No models available for probability prediction")
        
        # Weighted average
        weighted_proba = np.zeros_like(all_probas[0])
        for i, proba in enumerate(all_probas):
            weight = self.weights[i] if i < len(self.weights) else 1.0
            weighted_proba += weight * proba
        
        # Normalize
        weighted_proba /= weighted_proba.sum(axis=1, keepdims=True)
        
        return weighted_proba