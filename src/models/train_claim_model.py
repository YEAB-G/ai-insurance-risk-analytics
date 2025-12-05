"""Train a binary classification model for claim probability.

This script trains a RandomForestClassifier to predict whether a policy
will have a claim (`HasClaim`).

Run from project root:

    python src/models/train_claim_model.py
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.load_data import load_processed_data
from src.features.build_features import add_core_features, get_default_feature_lists


def main() -> None:
    df = load_processed_data()
    df = add_core_features(df)

    if "HasClaim" not in df.columns:
        raise ValueError("Expected 'HasClaim' column in processed data.")

    numeric_features, categorical_features = get_default_feature_lists(df)

    X = df[numeric_features + categorical_features]
    y = df["HasClaim"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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

    model = RandomForestClassifier(
        n_estimators=400,
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
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    print(f"Classification model AUC: {auc:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, os.path.join("models", "claim_model.pkl"))
    print("Saved model to models/claim_model.pkl")


if __name__ == "__main__":
    main()
