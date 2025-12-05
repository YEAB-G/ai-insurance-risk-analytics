"""Script to prepare the raw insurance data for analysis and modeling.

Run this script from the project root:

    python src/data/prepare_data.py

It will read `data/raw/insurance.csv` and write a cleaned version to
`data/processed/insurance_clean.csv`.

Because this environment does not have access to your actual data,
the cleaning steps below are deliberately simple and safe.
You should customise them once you inspect your dataset locally.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .load_data import load_raw_data





RAW_PATH = os.path.join("data", "raw", "MachineLearningRating_v3.txt")


PROCESSED_PATH = os.path.join("data", "processed", "insurance_clean.csv")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light cleaning and create core risk features.

    This is a good starting point. Adjust it to match your data quality.
    """
    df = df.copy()

    # Standardise column names (optional)
    df.columns = [c.strip() for c in df.columns]

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Basic numeric cleaning: replace negative premiums/claims with NaN
    for col in ["TotalPremium", "TotalClaims"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Create core metrics if the columns exist
    if {"TotalPremium", "TotalClaims"}.issubset(df.columns):
        df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, np.nan)
        df["LossRatio"] = df["LossRatio"].clip(upper=5)  # cap extreme ratios
        df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
        df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    return df


def main() -> None:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}. Please place your dataset there first."
        )

   
    df_raw = load_raw_data(RAW_PATH, sep=",")  # or sep="\t"

    df_clean = basic_cleaning(df_raw)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
