"""
Simplified ensemble model combining all baselines + proposed transformer.
Student 1's responsibility.
"""
import numpy as np
import joblib

class SimpleEnsemble:
    def __init__(self, model_paths, weights=None):
        """
        Simple ensemble of baseline models
        """
        self.models = {}
        self.weights = weights or [0.25, 0.25, 0.25, 0.25]

        try:
            self.models['A'] = joblib.load(model_paths['A'])
            print("Loaded Model A")
        except (FileNotFoundError, KeyError):
            print("Could not load Model A")

        try:
            self.models['C'] = joblib.load(model_paths['C'])
            print("Loaded Model C")
        except (FileNotFoundError, KeyError):
            print("Could not load Model C")

    def predict(self, texts):
        """Simple majority voting prediction"""
        all_predictions = []

        if 'A' in self.models:
            all_predictions.append(self.models['A'].predict(texts))
        if 'C' in self.models:
            all_predictions.append(self.models['C'].predict(texts))

        if not all_predictions:
            raise ValueError("No models loaded for ensemble")

        stacked_preds = np.column_stack(all_predictions)
        final_predictions = []

        for preds in stacked_preds:
            unique, counts = np.unique(preds, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])

        return np.array(final_predictions)

    def predict_proba_weighted(self, texts):
        """Weighted probability averaging"""
        all_probs = []

        if 'A' in self.models and hasattr(self.models['A'], 'predict_proba'):
            all_probs.append(self.models['A'].predict_proba(texts))
        if 'C' in self.models and hasattr(self.models['C'], 'predict_proba'):
            all_probs.append(self.models['C'].predict_proba(texts))

        if not all_probs:
            preds = self.predict(texts)
            unique_preds = np.unique(preds)
            n_classes = len(unique_preds)
            probs = np.zeros((len(preds), n_classes))
            for i, pred in enumerate(preds):
                idx = np.where(unique_preds == pred)[0][0]
                probs[i, idx] = 1.0
            return probs

        weighted_probs = np.zeros_like(all_probs[0], dtype=float)
        for i, prob in enumerate(all_probs):
            weight = self.weights[i] if i < len(self.weights) else 1.0 / len(all_probs)
            weighted_probs += weight * prob

        return weighted_probs / weighted_probs.sum(axis=1, keepdims=True)
