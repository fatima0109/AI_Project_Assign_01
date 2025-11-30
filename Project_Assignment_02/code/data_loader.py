import pandas as pd
from pathlib import Path

def load_parquet_dataset(train_path, test_path, text_col=None, label_col=None):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    # If columns not specified, try to auto-detect
    if text_col is None:
        for c in train.columns:
            if any(x in c.lower() for x in ['question','text','utter','sentence']):
                text_col = c; break
    if label_col is None:
        for c in train.columns:
            if any(x in c.lower() for x in ['label','labels','target','class','is_evasive','evasive']):
                label_col = c; break
    if text_col is None or label_col is None:
        # fallback: pick first object column as text and a small-unique column as label
        obj_cols = [c for c in train.columns if train[c].dtype == object]
        if obj_cols:
            text_col = text_col or obj_cols[0]
        label_col = label_col or min(train.columns.tolist(), key=lambda c: train[c].nunique())
    train = train[[text_col, label_col]].dropna().rename(columns={text_col:'text', label_col:'label'})
    test = test[[text_col, label_col]].dropna().rename(columns={text_col:'text', label_col:'label'})
    return train, test
