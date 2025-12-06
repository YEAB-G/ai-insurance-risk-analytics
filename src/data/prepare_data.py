"""Script to prepare the raw insurance data for analysis and modeling.

Run this script from the project root:

    python -m src.data.prepare_data

It will read `data/raw/MachineLearningRating_v3.txt` (pipe-delimited)
and write a cleaned version to `data/processed/insurance_clean.csv`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Paths relative to the project root
RAW_PATH = os.path.join("data", "raw", "MachineLearningRating_v3.txt")
PROCESSED_PATH = os.path.join("data", "processed", "insurance_clean.csv")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light cleaning and create core risk features."""
    df = df.copy()

    # Standardise column names (remove leading/trailing spaces)
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
    print("=== prepare_data.main() started ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"RAW_PATH: {RAW_PATH}")
    print(f"PROCESSED_PATH: {PROCESSED_PATH}")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}. "
            "Please place MachineLearningRating_v3.txt there."
        )

    print("Loading raw data with pipe separator...")
    df_raw = pd.read_csv(
        RAW_PATH,
        sep="|",            # your file is pipe-delimited
        engine="python",    # more flexible parser
        on_bad_lines="skip" # skip malformed lines instead of crashing
    )
    print(f"Raw data loaded: shape = {df_raw.shape}")

    print("Applying basic cleaning...")
    df_clean = basic_cleaning(df_raw)
    print(f"Cleaned data shape: {df_clean.shape}")

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_PATH}")
    print("=== prepare_data.main() finished ===")


if __name__ == "__main__":
    main()
