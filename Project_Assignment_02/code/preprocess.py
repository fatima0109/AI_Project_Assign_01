"""
Unified text preprocessing.
"""
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = ' '.join(text.split())
    
    return text

def prepare_splits(train_df, test_df, val_size=0.15):
    """Prepare train/val/test splits."""
    # Preprocess text
    train_df["text_clean"] = train_df["text"].apply(clean_text)
    test_df["text_clean"] = test_df["text"].apply(clean_text)
    
    # Split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_df["text_clean"],
        train_df["label"],
        test_size=val_size,
        random_state=42,
        stratify=train_df["label"]
    )
    
    X_test = test_df["text_clean"]
    y_test = test_df["label"]
    
    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return splits