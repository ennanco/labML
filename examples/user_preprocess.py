"""Example preprocessing hook used by `labml prepare`."""

from __future__ import annotations

import pandas as pd


def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return a cleaned dataframe according to user-defined parameters."""
    output = df.copy()
    target_column = str(params.get("target_column", "target"))
    if params.get("drop_na_target") and target_column in output.columns:
        output = output.dropna(subset=[target_column])
    return output.reset_index(drop=True)
