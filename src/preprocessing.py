"""dtype auditing and cleanup shared by both notebooks.

`pd.read_csv(path, sep=';', decimal=',')` correctly parses most numeric
columns, but ~33 columns (mostly BKI features) stay `object` because their
values are already plain-dot floats like '52800.0' with no comma to trigger
the decimal parser. We detect those programmatically (regex + >99% match
rate among non-null values) instead of hardcoding a column list, since the
same detection must hold for both train and test.
"""
import re

import pandas as pd

from src.config import CATEGORICAL_COLS, DATE_COL, FEATURE_FAMILY_PREFIXES

NUMERIC_STRING_PATTERN = re.compile(r"^-?\d+\.?\d*$")
NUMERIC_MATCH_THRESHOLD = 0.99


def detect_misclassified_numeric_columns(df, exclude_cols=None, threshold=NUMERIC_MATCH_THRESHOLD):
    """Return object columns whose non-null values are actually plain numbers.

    A column qualifies when the fraction of non-null values matching
    ``^-?\\d+\\.?\\d*$`` exceeds `threshold`. `exclude_cols` (e.g. the known
    categorical columns) is never considered, since some of those, like
    `dp_address_unique_regions`, are mostly-numeric-looking but genuinely
    categorical.
    """
    exclude_cols = set(exclude_cols or [])
    candidates = [c for c in df.columns if df[c].dtype == object and c not in exclude_cols]

    misclassified = []
    for col in candidates:
        non_null = df[col].dropna().astype(str)
        if len(non_null) == 0:
            continue
        match_frac = non_null.str.match(NUMERIC_STRING_PATTERN).mean()
        if match_frac > threshold:
            misclassified.append(col)
    return misclassified


def fix_numeric_dtypes(df, columns):
    """Coerce the given columns to numeric in place (returns the same df)."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def feature_family(col):
    """Map a feature name to its source-system family prefix, or 'other'."""
    for prefix in FEATURE_FAMILY_PREFIXES:
        if col.startswith(prefix):
            return prefix.rstrip("_")
    return "other"


def preprocess(df, is_train):
    """Apply the shared dtype fixes to a raw train/test dataframe.

    - Parses `dt` to datetime.
    - Detects and fixes misclassified numeric (object) columns, excluding
      the known categorical columns.

    `is_train` is accepted for interface symmetry with train/test-specific
    steps added later (e.g. dropping target/w handling is done by callers,
    not here) and to keep both notebooks calling the same signature.
    """
    df = df.copy()

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    misclassified_cols = detect_misclassified_numeric_columns(df, exclude_cols=CATEGORICAL_COLS)
    df = fix_numeric_dtypes(df, misclassified_cols)

    return df, misclassified_cols
