import pytest
import pandas as pd
from src.data import (
    load_dataset,
    get_random_split,
    get_temporal_split,
    get_group_kfold_splits,
)
from src.config import SELECTED_FEATURES, TARGET_COL, COUNTRY_COL, YEAR_COL


def test_load_dataset():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert TARGET_COL in df.columns
    for feat in SELECTED_FEATURES:
        assert feat in df.columns
    # ARE should be excluded
    assert "ARE" not in df[COUNTRY_COL].values


def test_temporal_split():
    df = load_dataset()
    split_year = 2003
    train_df, test_df = get_temporal_split(df, split_year=split_year)

    assert not train_df.empty
    assert not test_df.empty
    assert train_df[YEAR_COL].max() <= split_year
    assert test_df[YEAR_COL].min() > split_year
    # No row overlap
    assert len(train_df) + len(test_df) == len(df)


def test_group_kfold_no_country_leakage():
    df = load_dataset()
    splits = list(get_group_kfold_splits(df, n_splits=5))
    assert len(splits) == 5

    for train_idx, val_idx in splits:
        train_countries = set(df.iloc[train_idx][COUNTRY_COL].unique())
        val_countries = set(df.iloc[val_idx][COUNTRY_COL].unique())
        # Zero intersection: no country should be in both train and validation
        assert train_countries.isdisjoint(val_countries)

