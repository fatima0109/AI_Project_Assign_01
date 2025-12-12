# Student C: TF-IDF + LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def build_pipeline(max_features=10000):
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1,2))),
        ('svc', LinearSVC(max_iter=10000))
    ])
