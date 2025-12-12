"""
Baseline C: TF-IDF + LinearSVC
"""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def build_baseline_c(max_features=5000):
    """Build Baseline C pipeline."""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)
        )),
        ('clf', LinearSVC(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    return pipeline