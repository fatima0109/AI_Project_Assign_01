# File: data_loader.py
"""
Enhanced data loader for Assignment 3 that extends your existing loader.
Specifically designed for QEvasion dataset with your file structure.
"""
import pandas as pd
import json
from pathlib import Path
from datasets import load_dataset, load_from_disk
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Your existing functions - I'll keep them and add new ones
def load_arrow_or_parquet(path):
    """
    Load a dataset from either:
    - Parquet file
    - Arrow file (.arrow)
    - CSV file (.csv)
    - HuggingFace `load_from_disk` directory
    Returns a pandas DataFrame.
    """
    p = Path(path)

    # Case 1: Directory → load_from_disk
    if p.is_dir():
        ds = load_from_disk(str(p))
        return ds.to_pandas()

    # Case 2: Single parquet file
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(str(p))

    # Case 3: Single arrow file
    if p.suffix.lower() == ".arrow":
        ds = load_dataset("arrow", data_files={"data": str(p)})["data"]
        return ds.to_pandas()

    # Case 4: Single CSV file
    if p.suffix.lower() == ".csv":
        return pd.read_csv(str(p))

    # Unsupported file type
    raise ValueError(f"Unsupported file type for: {path}")

def auto_detect_text_label_cols(df):
    """
    Automatically detect text and label columns.
    Enhanced for QEvasion dataset.
    """
    text_col = None
    label_col = None

    # First, check for specific QEvasion column names
    qevasion_text_cols = ['question', 'interview_question', 'interview_answer', 
                         'combined_text', 'text', 'utterance']
    
    qevasion_label_cols = ['label', 'labels', 'is_evasive', 'evasive', 
                          'target', 'class', 'category']
    
    # Check for QEvasion specific columns first
    for c in df.columns:
        if c in qevasion_text_cols:
            text_col = c
            break
    
    # If not found, use your original logic
    if text_col is None:
        for c in df.columns:
            if any(x in c.lower() for x in ["question", "text", "utter", "sentence", "interview"]):
                text_col = c
                break

    # Check for QEvasion specific label columns
    for c in df.columns:
        if c in qevasion_label_cols:
            label_col = c
            break
    
    # If not found, use your original logic
    if label_col is None:
        for c in df.columns:
            if any(x in c.lower() for x in ["label", "labels", "target", "class", "is_evasive", "evasive"]):
                label_col = c
                break

    if text_col is None:
        # fallback: any object column
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        text_col = obj_cols[0] if obj_cols else df.columns[0]

    if label_col is None:
        # fallback: column with lowest number of unique values
        label_col = min(df.columns, key=lambda c: df[c].nunique())

    return text_col, label_col

def load_dataset_flexible(train_path, test_path, text_col=None, label_col=None):
    """
    Load train/test datasets, regardless of format (arrow/parquet/load_from_disk).
    Standardizes them to df with ['text','label'].
    """
    train_df = load_arrow_or_parquet(train_path)
    test_df = load_arrow_or_parquet(test_path)

    # Auto-detect columns if not specified
    if text_col is None or label_col is None:
        t_text, t_label = auto_detect_text_label_cols(train_df)
        text_col = text_col or t_text
        label_col = label_col or t_label

    # Keep only required columns
    train_df = train_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    test_df  = test_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    return train_df, test_df

# ===== NEW FUNCTIONS FOR ASSIGNMENT 3 =====

def load_qevasion_data(train_path=None, test_path=None, auto_find=True):
    """
    Load QEvasion dataset with automatic file discovery.
    
    Args:
        train_path: Explicit path to training data
        test_path: Explicit path to test data
        auto_find: If True, automatically search for files if paths not provided
    
    Returns:
        train_df, test_df: Standardized DataFrames
    """
    # If paths not provided and auto_find is True, search for files
    if auto_find and (train_path is None or test_path is None):
        base_dirs = [
            Path("."),
            Path("QEvasion"),
            Path("QEvasion/data"),
            Path("QEvasion/full_train_set/train"),
            Path("QEvasion/full_test_set/train"),
            Path("QEvasion/train/train"),
            Path("QEvasion/test_set/train")
        ]
        
        if train_path is None:
            train_path = find_qevasion_file(base_dirs, "train")
        
        if test_path is None:
            test_path = find_qevasion_file(base_dirs, "test")
    
    if train_path is None or test_path is None:
        raise FileNotFoundError("Could not find QEvasion data files")
    
    print(f"Loading training data from: {train_path}")
    print(f"Loading test data from: {test_path}")
    
    # Use your existing function
    train_df, test_df = load_dataset_flexible(train_path, test_path)
    
    # Additional processing for Assignment 3
    train_df = preprocess_for_assignment3(train_df, is_train=True)
    test_df = preprocess_for_assignment3(test_df, is_train=False)
    
    return train_df, test_df

def find_qevasion_file(search_dirs, file_type="train"):
    """
    Find QEvasion data files in common locations.
    
    Args:
        search_dirs: List of directories to search
        file_type: "train" or "test"
    
    Returns:
        Path to found file or None
    """
    # File patterns to search for
    patterns = {
        "train": [
            "train-00000-of-00001.parquet",
            "data-00000-of-00001.arrow",
            "train.parquet",
            "train.csv",
            "data.parquet"
        ],
        "test": [
            "test-00000-of-00001.parquet",
            "data-00000-of-00001.arrow",
            "test.parquet",
            "test.csv"
        ]
    }
    
    for base_dir in search_dirs:
        if base_dir.exists():
            for pattern in patterns[file_type]:
                file_path = base_dir / pattern
                if file_path.exists():
                    return str(file_path)
    
    return None

def preprocess_for_assignment3(df, is_train=True):
    """
    Preprocess DataFrame for Assignment 3 requirements.
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data
    
    Returns:
        Processed DataFrame
    """
    df = df.copy()
    
    # Ensure text is string
    if 'text' in df.columns:
        df['text'] = df['text'].astype(str).fillna('')
    
    # Ensure label is numeric
    if 'label' in df.columns:
        if df['label'].dtype == 'object' or df['label'].dtype.name == 'category':
            le = LabelEncoder()
            df['label'] = le.fit_transform(df['label']) if is_train else le.transform(df['label'])
        elif df['label'].dtype == 'bool':
            df['label'] = df['label'].astype(int)
    
    # Add text length features
    if 'text' in df.columns:
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
    
    return df

def explore_dataset_qevasion(df, name="Dataset"):
    """
    Enhanced exploration for QEvasion dataset.
    
    Args:
        df: DataFrame to explore
        name: Dataset name for display
    
    Returns:
        Dictionary with exploration results
    """
    print(f"\n{'='*60}")
    print(f"EXPLORING {name.upper()}")
    print('='*60)
    
    results = {
        'name': name,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'text' in df.columns:
        print(f"\n📊 TEXT ANALYSIS:")
        print(f"  First sample: {df['text'].iloc[0][:150]}...")
        
        # Text length statistics
        text_lengths = df['text'].str.len()
        print(f"  Text length - Mean: {text_lengths.mean():.0f} chars")
        print(f"  Text length - Min: {text_lengths.min()} chars")
        print(f"  Text length - Max: {text_lengths.max()} chars")
        print(f"  Text length - Std: {text_lengths.std():.0f} chars")
        
        # Word count statistics
        word_counts = df['text'].str.split().str.len()
        print(f"  Word count - Mean: {word_counts.mean():.0f} words")
        print(f"  Word count - Min: {word_counts.min()} words")
        print(f"  Word count - Max: {word_counts.max()} words")
        
        results['text_stats'] = {
            'mean_length': text_lengths.mean(),
            'min_length': text_lengths.min(),
            'max_length': text_lengths.max(),
            'mean_words': word_counts.mean()
        }
    
    if 'label' in df.columns:
        print(f"\n🎯 LABEL ANALYSIS:")
        unique_labels = df['label'].unique()
        print(f"  Unique labels: {len(unique_labels)} → {sorted(unique_labels)}")
        
        # Label distribution
        label_dist = df['label'].value_counts().sort_index()
        print(f"  Label distribution:")
        for label, count in label_dist.items():
            percentage = (count / len(df)) * 100
            print(f"    Label {label}: {count} samples ({percentage:.1f}%)")
        
        # Class imbalance
        if len(label_dist) > 1:
            imbalance_ratio = label_dist.max() / label_dist.min()
            print(f"  Class imbalance ratio: {imbalance_ratio:.2f}:1")
        
        results['label_stats'] = {
            'n_unique': len(unique_labels),
            'distribution': label_dist.to_dict(),
            'imbalance_ratio': imbalance_ratio if len(label_dist) > 1 else 1.0
        }
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  MISSING VALUES:")
        for col, count in missing[missing > 0].items():
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count} missing ({percentage:.1f}%)")
    else:
        print(f"\n✓ No missing values found")
    
    # Data types
    print(f"\n📝 DATA TYPES:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    results['missing_values'] = missing[missing > 0].to_dict()
    
    print(f"\n{'='*60}")
    
    return results

def create_text_features(df):
    """
    Create additional text features for analysis.
    
    Args:
        df: DataFrame with 'text' column
    
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    if 'text' in df.columns:
        # Basic features
        df['char_count'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['avg_word_length'] = df['char_count'] / df['word_count'].replace(0, 1)
        
        # Special character counts
        df['has_question'] = df['text'].str.contains(r'\?').astype(int)
        df['has_exclamation'] = df['text'].str.contains(r'\!').astype(int)
        df['has_number'] = df['text'].str.contains(r'\d').astype(int)
        df['has_uppercase'] = df['text'].str.contains(r'[A-Z]').astype(int)
        
        # Sentence-like features
        df['sentence_count'] = df['text'].str.split(r'[.!?]+').str.len() - 1
        
        # Cleanliness metrics
        df['whitespace_ratio'] = df['text'].str.count(r'\s') / df['char_count'].replace(0, 1)
        df['punctuation_ratio'] = df['text'].str.count(r'[^\w\s]') / df['char_count'].replace(0, 1)
    
    return df

def analyze_class_characteristics(df, label_col='label', text_col='text'):
    """
    Analyze text characteristics by class.
    
    Args:
        df: DataFrame with text and labels
        label_col: Name of label column
        text_col: Name of text column
    
    Returns:
        Dictionary with analysis by class
    """
    analysis = {}
    
    if label_col in df.columns and text_col in df.columns:
        unique_labels = df[label_col].unique()
        
        for label in unique_labels:
            label_df = df[df[label_col] == label]
            
            # Text statistics for this class
            texts = label_df[text_col]
            analysis[label] = {
                'count': len(label_df),
                'avg_text_length': texts.str.len().mean(),
                'avg_word_count': texts.str.split().str.len().mean(),
                'common_words': get_common_words(texts, n=10),
                'unique_words_ratio': get_unique_words_ratio(texts)
            }
    
    return analysis

def get_common_words(texts, n=10):
    """Get most common words in a series of texts"""
    from collections import Counter
    import re
    
    all_words = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    return dict(Counter(all_words).most_common(n))

def get_unique_words_ratio(texts):
    """Calculate ratio of unique words to total words"""
    import re
    
    all_words = []
    unique_words = set()
    
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
        unique_words.update(words)
    
    if len(all_words) == 0:
        return 0
    
    return len(unique_words) / len(all_words)

def prepare_data_for_training(train_df, test_df, 
                             text_col='text', 
                             label_col='label',
                             validation_size=0.15):
    """
    Prepare data for training with validation split.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        text_col: Text column name
        label_col: Label column name
        validation_size: Size of validation split
    
    Returns:
        Dictionary with all splits
    """
    from sklearn.model_selection import train_test_split
    
    # Extract features and labels
    X_train = train_df[text_col]
    y_train = train_df[label_col]
    
    X_test = test_df[text_col]
    y_test = test_df[label_col]
    
    # Create validation split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_size,
        random_state=42,
        stratify=y_train
    )
    
    print(f"\n📊 DATA SPLITS:")
    print(f"  Training: {len(X_train_split)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    splits = {
        'X_train': X_train_split,
        'y_train': y_train_split,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Add dataset info
    splits['info'] = {
        'n_classes': len(y_train.unique()),
        'classes': sorted(y_train.unique()),
        'class_distribution_train': y_train.value_counts().to_dict(),
        'class_distribution_test': y_test.value_counts().to_dict()
    }
    
    return splits

# ===== MAIN FUNCTION FOR ASSIGNMENT 3 =====

def main_assignment3_data_loading():
    """
    Main function for Assignment 3 data loading.
    Run this to load and explore QEvasion data.
    """
    print("="*70)
    print("ASSIGNMENT 3: QEVASION DATA LOADER")
    print("="*70)
    
    try:
        # Try to auto-discover data
        print("\n🔍 Searching for QEvasion data files...")
        train_df, test_df = load_qevasion_data(auto_find=True)
        
        # Explore datasets
        print("\n" + "="*70)
        print("DATASET EXPLORATION")
        print("="*70)
        
        train_results = explore_dataset_qevasion(train_df, "TRAINING DATA")
        test_results = explore_dataset_qevasion(test_df, "TEST DATA")
        
        # Compare train and test distributions
        print("\n" + "="*70)
        print("DATASET COMPARISON")
        print("="*70)
        
        train_labels = train_df['label'].value_counts().sort_index()
        test_labels = test_df['label'].value_counts().sort_index()
        
        print("\nLabel distribution comparison:")
        print(f"{'Label':<10} {'Train':<10} {'Test':<10} {'Ratio (Train/Test)':<20}")
        print("-"*50)
        
        for label in sorted(set(train_labels.index) | set(test_labels.index)):
            train_count = train_labels.get(label, 0)
            test_count = test_labels.get(label, 0)
            ratio = train_count / test_count if test_count > 0 else float('inf')
            print(f"{label:<10} {train_count:<10} {test_count:<10} {ratio:<20.2f}")
        
        # Prepare for training
        print("\n" + "="*70)
        print("PREPARING DATA FOR TRAINING")
        print("="*70)
        
        splits = prepare_data_for_training(train_df, test_df)
        
        # Create text features for analysis
        print("\n📈 Creating text features for analysis...")
        train_df_features = create_text_features(train_df)
        test_df_features = create_text_features(test_df)
        
        # Analyze class characteristics
        print("\n🎯 Analyzing class characteristics...")
        class_analysis = analyze_class_characteristics(train_df_features)
        
        for label, stats in class_analysis.items():
            print(f"\nClass {label}:")
            print(f"  Samples: {stats['count']}")
            print(f"  Avg text length: {stats['avg_text_length']:.0f} chars")
            print(f"  Avg word count: {stats['avg_word_count']:.1f} words")
            print(f"  Unique words ratio: {stats['unique_words_ratio']:.3f}")
            print(f"  Top 5 words: {list(stats['common_words'].keys())[:5]}")
        
        print("\n" + "="*70)
        print("✅ DATA LOADING COMPLETE")
        print("="*70)
        
        # Return everything for use in experiments
        return {
            'train_df': train_df,
            'test_df': test_df,
            'train_df_features': train_df_features,
            'test_df_features': test_df_features,
            'splits': splits,
            'class_analysis': class_analysis,
            'exploration_results': {
                'train': train_results,
                'test': test_results
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        print("\nTrying alternative loading method...")
        
        # Try specific known paths
        known_paths = [
            ("QEvasion/data/train-00000-of-00001.parquet", "QEvasion/data/test-00000-of-00001.parquet"),
            ("QEvasion/full_train_set/train/data-00000-of-00001.arrow", "QEvasion/full_test_set/train/data-00000-of-00001.arrow")
        ]
        
        for train_path, test_path in known_paths:
            try:
                print(f"\nTrying: {train_path}, {test_path}")
                train_df, test_df = load_dataset_flexible(train_path, test_path)
                train_df = preprocess_for_assignment3(train_df, is_train=True)
                test_df = preprocess_for_assignment3(test_df, is_train=False)
                
                print(f"Successfully loaded {len(train_df)} training and {len(test_df)} test samples")
                return {
                    'train_df': train_df,
                    'test_df': test_df,
                    'splits': prepare_data_for_training(train_df, test_df)
                }
            except Exception as e2:
                print(f"Failed: {e2}")
                continue
        
        raise FileNotFoundError("Could not load QEvasion data from any known location")

if __name__ == "__main__":
    # Run the data loading when executed directly
    data = main_assignment3_data_loading()
    print(f"\nData loaded successfully!")
    print(f"Training samples: {len(data['train_df'])}")
    print(f"Test samples: {len(data['test_df'])}")