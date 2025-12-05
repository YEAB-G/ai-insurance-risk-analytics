"""Feature engineering helpers.

These functions can be imported in notebooks or training scripts to keep
your feature logic in one place.
"""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure key risk features exist in the dataframe.

    Adds:
    - LossRatio = TotalClaims / TotalPremium
    - Margin = TotalPremium - TotalClaims
    - HasClaim = 1 if TotalClaims > 0 else 0
    """
    df = df.copy()

    if {"TotalPremium", "TotalClaims"}.issubset(df.columns):
        df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, pd.NA)
        df["LossRatio"] = df["LossRatio"].clip(upper=5)
        df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
        df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    return df


def get_default_feature_lists(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    """Return (numeric_features, categorical_features) lists.

    This is a starting point; you can adjust it based on EDA.
    """
    numeric_candidates = [
        "TotalPremium",
        "TotalClaims",
        "SumInsured",
        "Cubiccapacity",
        "Kilowatts",
        "CapitalOutstanding",
        "CustomValueEstimate",
        "NumberOfVehiclesInFleet",
    ]

    categorical_candidates = [
        "Province",
        "PostalCode",
        "VehicleType",
        "Make",
        "Model",
        "Gender",
        "MaritalStatus",
        "CoverType",
        "CoverCategory",
        "CoverGroup",
        "Product",
    ]

    numeric_features = [c for c in numeric_candidates if c in df.columns]
    categorical_features = [c for c in categorical_candidates if c in df.columns]

    return numeric_features, categorical_features
