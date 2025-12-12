"""
Baseline A: TF-IDF + Logistic Regression
"""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_baseline_a(max_features=5000):
    """Build Baseline A pipeline."""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 1)
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    return pipeline