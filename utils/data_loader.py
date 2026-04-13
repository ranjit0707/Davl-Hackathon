"""
Data Loader — handles CSV/Excel ingestion and dataset meta-info extraction.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st


WEATHER_TARGET_KEYWORDS = [
    "weather", "type", "condition", "label", "class", "category",
    "status", "description", "summary", "forecast", "pattern", "season",
]

WEATHER_NUM_KEYWORDS = [
    "temp", "temperature", "humidity", "pressure", "wind", "rain",
    "rainfall", "precipitation", "visibility", "cloud", "dew", "uv",
    "solar", "evaporation", "sunshine", "snow", "hail", "fog",
]


@st.cache_data(show_spinner=False)
def load_dataset(file) -> pd.DataFrame | None:
    """Load CSV or Excel into a DataFrame."""
    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
    return None


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return a metadata dict used across all modules."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    null_counts = df.isnull().sum().to_dict()
    null_pct    = {c: round(v / len(df) * 100, 2) for c, v in null_counts.items() if v > 0}
    unique_counts = {c: df[c].nunique() for c in df.columns}
    memory_mb   = round(df.memory_usage(deep=True).sum() / 1e6, 3)

    return dict(
        rows=len(df),
        cols=len(df.columns),
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        null_counts=null_counts,
        null_pct=null_pct,
        unique_counts=unique_counts,
        memory_mb=memory_mb,
        dtypes=df.dtypes.astype(str).to_dict(),
    )


def detect_target_column(df: pd.DataFrame) -> str | None:
    """Detect the most likely weather-type target column."""
    cols_lower = {c.lower(): c for c in df.columns}
    for kw in WEATHER_TARGET_KEYWORDS:
        for lc, orig in cols_lower.items():
            if kw in lc:
                if df[orig].dtype == object or df[orig].nunique() <= 30:
                        return orig
    # fall-back: first categorical column with 2–20 unique values
    for c in df.select_dtypes(exclude=np.number).columns:
        if 2 <= df[c].nunique() <= 20:
            return c
    return None


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")
