from sklearn.model_selection import train_test_split

def split_train_val(df, val_size=0.15, random_state=42):
    X = df['text'].values; y = df['label'].values
    return train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=y)
