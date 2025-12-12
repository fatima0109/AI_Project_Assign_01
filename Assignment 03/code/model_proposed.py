"""
Proposed Model: Enhanced TF-IDF + Logistic Regression
"""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_proposed_model(max_features=5000):
    """Build Proposed Model pipeline."""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight='balanced',
            solver='liblinear',
            random_state=42,
            penalty='l2'
        ))
    ])
    return pipeline