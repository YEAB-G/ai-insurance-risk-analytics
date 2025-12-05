"""Train a claim severity regression model.

This script trains a tree-based regression model (RandomForestRegressor)
to predict `TotalClaims` on records where a claim occurred.

Run from project root:

    python src/models/train_severity_model.py
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.load_data import load_processed_data
from src.features.build_features import add_core_features, get_default_feature_lists


def main() -> None:
    df = load_processed_data()
    df = add_core_features(df)

    # Use only rows with a positive claim amount
    df_sev = df[df.get("HasClaim", 0) == 1].copy()

    numeric_features, categorical_features = get_default_feature_lists(df_sev)

    X = df_sev[numeric_features + categorical_features]
    y = df_sev["TotalClaims"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"Severity model RMSE: {rmse:.2f}")
    print(f"Severity model R^2:  {r2:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, os.path.join("models", "severity_model.pkl"))
    print("Saved model to models/severity_model.pkl")


if __name__ == "__main__":
    main()
