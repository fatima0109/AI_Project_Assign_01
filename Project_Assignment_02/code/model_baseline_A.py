from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline(max_features=10000):
    """
    Build a TF-IDF + Logistic Regression pipeline for text classification.
    
    Args:
        max_features: Maximum number of features for TF-IDF vectorizer
    
    Returns:
        Fitted sklearn Pipeline object
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga'))
    ])
