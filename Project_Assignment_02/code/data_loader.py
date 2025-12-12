import pandas as pd
from pathlib import Path
from datasets import load_dataset, load_from_disk

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
    """
    text_col = None
    label_col = None

    for c in df.columns:
        if any(x in c.lower() for x in ["question", "text", "utter", "sentence", "interview"]):
            text_col = c
            break

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
