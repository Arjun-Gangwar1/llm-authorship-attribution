"""
src/data/loader.py — Unified data loader for all dataset variants.

Handles both column-naming conventions present in this project:
  - Full RAID dataset:  columns  model, generation
  - Demo dataset:       columns  generated_by, text

Usage:
    from src.data.loader import load_split, CLASSES

    # Full dataset
    X_train, y_train = load_split("LLM_12_class/raid/train/train.parquet")

    # Demo data (auto-detects column names)
    X_val, y_val = load_split("data/demo/small/val/val.parquet")

    # Mini subset — 1000 rows per class for fast debugging
    X_mini, y_mini = load_split("LLM_12_class/raid/train/train.parquet", subset=1000)
"""

import pandas as pd
import numpy as np

CLASSES = [
    "chatgpt", "cohere", "cohere-chat", "gpt2", "gpt3", "gpt4",
    "human", "llama-chat", "mistral", "mistral-chat", "mpt", "mpt-chat",
]


def load_split(path: str, subset: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a parquet split and return (texts, labels) as numpy arrays.

    Parameters
    ----------
    path   : path to the .parquet file
    subset : if set, sample this many rows per class (for fast debugging)

    Returns
    -------
    texts  : np.ndarray of str, shape (N,)
    labels : np.ndarray of str, shape (N,)  — values in CLASSES
    """
    df = pd.read_parquet(path)

    # Normalise column names
    if "generated_by" in df.columns:
        df = df.rename(columns={"generated_by": "model", "text": "generation"})

    # Keep only the 12 known classes
    df = df[df["model"].isin(CLASSES)].reset_index(drop=True)

    # Optional mini-subset (stratified by class)
    if subset is not None:
        df = (
            df.groupby("model", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), subset), random_state=42))
              .reset_index(drop=True)
        )

    return df["generation"].values, df["model"].values


def load_full_dataset(
    train_path: str = "LLM_12_class/raid/train/train.parquet",
    val_path:   str = "LLM_12_class/raid/val/val.parquet",
    test_path:  str = "LLM_12_class/raid/test/test.parquet",
    subset: int = None,
):
    """Load all three splits at once. Returns (X_train, y_train, X_val, y_val, X_test, y_test)."""
    X_train, y_train = load_split(train_path, subset=subset)
    X_val,   y_val   = load_split(val_path)
    X_test,  y_test  = load_split(test_path)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_demo_dataset(size: str = "small", subset: int = None):
    """
    Load the demo (smoke-test) dataset.

    Parameters
    ----------
    size   : 'small' or 'medium'
    subset : rows per class (None = use all demo rows)
    """
    base = f"data/demo/{size}"
    prefix = "mini" if size == "small" else "medium"
    X_train, y_train = load_split(f"{base}/train/{prefix}_train.parquet", subset=subset)
    X_val,   y_val   = load_split(f"{base}/val/{prefix}_val.parquet")
    X_test,  y_test  = load_split(f"{base}/test/{prefix}_test.parquet")
    return X_train, y_train, X_val, y_val, X_test, y_test
