"""
Unified data loader for QEvasion dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class QEvasionDataLoader:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.label_encoder = LabelEncoder()
        
    def find_data_files(self):
        """Find QEvasion data files automatically."""
        possible_locations = [
            self.base_path / "QEvasion/data/train-00000-of-00001.parquet",
            self.base_path / "QEvasion/data/test-00000-of-00001.parquet",
            self.base_path / "train-00000-of-00001.parquet",
            self.base_path / "test-00000-of-00001.parquet",
            self.base_path / "data/train-00000-of-00001.parquet",
            self.base_path / "data/test-00000-of-00001.parquet",
        ]
        
        train_path = None
        test_path = None
        
        for path in possible_locations:
            if path.exists():
                if "train" in str(path):
                    train_path = path
                elif "test" in str(path):
                    test_path = path
        
        return train_path, test_path
    
    def load_data(self, train_path=None, test_path=None):
        """Load and standardize QEvasion data."""
        if train_path is None or test_path is None:
            train_path, test_path = self.find_data_files()
        
        if train_path is None or test_path is None:
            raise FileNotFoundError("Could not find QEvasion data files")
        
        print(f"Loading train: {train_path}")
        print(f"Loading test: {test_path}")
        
        # Load parquet files
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # Detect text and label columns
        text_col, label_col = self._detect_columns(train_df)
        
        # Standardize column names
        train_df = train_df.rename(columns={text_col: "text", label_col: "label"})
        test_df = test_df.rename(columns={text_col: "text", label_col: "label"})
        
        # Keep only needed columns
        train_df = train_df[["text", "label"]].copy()
        test_df = test_df[["text", "label"]].copy()
        
        # Clean text
        train_df["text"] = train_df["text"].astype(str).fillna("")
        test_df["text"] = test_df["text"].astype(str).fillna("")
        
        # Encode labels
        train_df["label"] = self.label_encoder.fit_transform(train_df["label"])
        test_df["label"] = self.label_encoder.transform(test_df["label"])
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return train_df, test_df
    
    def _detect_columns(self, df):
        """Automatically detect text and label columns."""
        text_candidates = ["question", "text", "utterance", "content", "interview_question"]
        label_candidates = ["label", "labels", "evasion_label", "is_evasive", "target"]
        
        text_col = None
        label_col = None
        
        # Look for text column
        for col in df.columns:
            col_lower = col.lower()
            if any(candidate in col_lower for candidate in text_candidates):
                text_col = col
                break
        
        # If not found, use first string column
        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
        
        # Look for label column
        for col in df.columns:
            col_lower = col.lower()
            if col != text_col and any(candidate in col_lower for candidate in label_candidates):
                label_col = col
                break
        
        # If not found, use column with fewest unique values
        if label_col is None:
            for col in df.columns:
                if col != text_col:
                    label_col = col
                    break
        
        return text_col, label_col

def load_qevasion_data(train_path=None, test_path=None):
    """Convenience function for loading data."""
    loader = QEvasionDataLoader()
    return loader.load_data(train_path, test_path)