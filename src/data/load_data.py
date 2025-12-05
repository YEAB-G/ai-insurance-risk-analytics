"""Utility functions for loading raw and processed insurance data."""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date-like columns if they exist."""
    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"])
    return df


def load_raw_data(path: str = os.path.join("data", "raw", "insurance.csv")) -> pd.DataFrame:
    """Load the raw insurance dataset.

    Parameters
    ----------
    path : str
        File path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with parsed date columns (if present).
    """
    df = pd.read_csv(path)
    return _parse_dates(df)


def load_processed_data(
    path: str = os.path.join("data", "processed", "insurance_clean.csv")
) -> pd.DataFrame:
    """Load the processed insurance dataset.

    Parameters
    ----------
    path : str
        File path to the processed CSV file.

    Returns
    -------
    pd.DataFrame
        Processed dataset with parsed date columns (if present).
    """
    df = pd.read_csv(path)
    return _parse_dates(df)
