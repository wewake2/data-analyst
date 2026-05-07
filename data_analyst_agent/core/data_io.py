"""
Large CSV ingest helpers.

For bigger CSVs :
  1. Loading speed and memory: use pyarrow engine when available, downcast
     numeric dtypes, treat object cols with low cardinality as 'category'.
  2. Profiling cost: profile in O(rows * cols) with vectorised pandas, never
     ship the full frame to the LLM. (DataInsightAgent already only ships a
     short profile string, but `nunique` etc. can be slow on 5M+ rows;)
  3. Plot speed: matplotlib chokes on millions of points. The plot agent
     should sample. We expose `safe_sample()` for that purpose.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def load_csv(path: str, *, low_memory_threshold_mb: int = 100) -> pd.DataFrame:
    """
    Load a CSV with sensible defaults for big files.

    - Uses the pyarrow engine when installed (much faster on big files).
    - Downcasts numeric columns to save RAM.
    - Converts low-cardinality string columns to 'category' dtype.
    """
    size_mb = os.path.getsize(path) / (1024 * 1024)

    read_kwargs: dict = {}
    try:
        import pyarrow
        read_kwargs["engine"] = "pyarrow"
    except ImportError:
        # Fall back to the C engine; still fine, just slower.
        read_kwargs["engine"] = "c"
        read_kwargs["low_memory"] = False

    df = pd.read_csv(path, **read_kwargs)

    # For big frames, optimise dtypes to reduce footprint.
    if size_mb >= low_memory_threshold_mb:
        df = _optimise_dtypes(df)
    return df


def _optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics; categorise low-cardinality strings."""
    for col in df.select_dtypes(include="integer").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include="float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include="object").columns:
        # Categorise if uniques < 50% of rows AND fewer than 10k uniques.
        nu = df[col].nunique(dropna=True)
        if nu < min(len(df) * 0.5, 10_000):
            df[col] = df[col].astype("category")
    return df


def memory_footprint_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def safe_sample(df: pd.DataFrame, max_rows: int = 50_000,
                random_state: int = 0) -> pd.DataFrame:
    """Sample a frame down to a plot-friendly size if needed."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)
