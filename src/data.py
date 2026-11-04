import pandas as pd
import numpy as np
from typing import Tuple, Generator
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from src.config import (
    PROCESSED_DATA_PATH,
    TARGET_COL,
    COUNTRY_COL,
    YEAR_COL,
    FEATURE_COLS,
    RANDOM_STATE,
    TEMPORAL_SPLIT_YEAR,
)


def load_dataset(filepath: str = None) -> pd.DataFrame:
    """Load cleaned dataset and ensure required columns are present."""
    path = filepath or PROCESSED_DATA_PATH
    df = pd.read_csv(path)
    
    # Filter out known anomaly countries if present (as in original exploratory analysis)
    df = df[df[COUNTRY_COL] != "ARE"].copy()
    
    # Ensure numeric columns are properly cast
    for col in FEATURE_COLS + [TARGET_COL, YEAR_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, COUNTRY_COL, YEAR_COL])
    return df.reset_index(drop=True)


def get_random_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standard random shuffle train/test split (contains data leakage across country-years)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_temporal_split(
    df: pd.DataFrame, split_year: int = TEMPORAL_SPLIT_YEAR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/test split.
    Train on historical years (<= split_year), test on future out-of-time years (> split_year).
    Tests model's ability to extrapolate into the future.
    """
    train_df = df[df[YEAR_COL] <= split_year].copy()
    test_df = df[df[YEAR_COL] > split_year].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_group_kfold_splits(
    df: pd.DataFrame, n_splits: int = 5
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    GroupKFold partitioned by country.
    Guarantees that no country in the test fold appears in the training fold.
    Tests spatial generalization to unseen nations.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[COUNTRY_COL].values
    return gkf.split(df, groups=groups)


def get_kfold_splits(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = RANDOM_STATE
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Standard K-Fold (for comparison against GroupKFold to quantify data leakage)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return kf.split(df)

