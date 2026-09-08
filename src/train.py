"""
Train the STEM base-salary regression model.

This script consolidates the exploration done in
notebooks/base_salary_prediction.ipynb into a single reproducible pipeline:

1. Load the raw survey data.
2. Clean it (drop zero-salary rows, engineer `experience_before_company`).
3. Fit a preprocessing + XGBoost pipeline (the model selected in the notebook
   after comparing Linear/Ridge/RandomForest/XGBoost).
4. Evaluate on a held-out test split.
5. Save the fitted pipeline (model/salary_model.pkl) and a metadata file
   (model/metadata.json) that the frontend uses to populate its form fields
   and to show the model's expected error range.

Usage:
    python src/train.py --data data/stem-4_25-update-1.xlsx
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["yearsofexperience", "yearsatcompany", "experience_before_company"]
CATEGORICAL_FEATURES = ["company", "title", "nation", "gender", "race", "education"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "basesalary"

# How many of the most frequent values to offer in the frontend dropdowns for
# high-cardinality fields (company, title). Keeps the UI usable without
# shipping every one of the thousands of raw values.
TOP_N_OPTIONS = {
    "company": 60,
    "title": 30,
    "nation": 30,
    "gender": 10,
    "race": 10,
    "education": 10,
}


def load_and_clean(data_path: Path) -> pd.DataFrame:
    df = pd.read_excel(data_path)

    # Same decision as the notebook: a base salary of 0 isn't a meaningful
    # regression target, so those rows are dropped rather than imputed.
    df = df[df["basesalary"] > 0].copy()

    # Feature engineering: experience accumulated before the current job.
    df["experience_before_company"] = (
        df["yearsofexperience"] - df["yearsatcompany"]
    ).clip(lower=0)

    return df


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_metadata(df: pd.DataFrame, metrics: dict) -> dict:
    dropdowns = {}
    for col, top_n in TOP_N_OPTIONS.items():
        counts = df[col].dropna().astype(str).value_counts()
        dropdowns[col] = counts.head(top_n).index.tolist()

    numeric_ranges = {}
    for col in ["yearsofexperience", "yearsatcompany"]:
        numeric_ranges[col] = {
            "min": 0,
            "max": int(np.ceil(df[col].quantile(0.99))),
            "median": float(df[col].median()),
        }

    return {
        "dropdowns": dropdowns,
        "numeric_ranges": numeric_ranges,
        "metrics": metrics,
        "currency": "USD",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE_DIR / "data" / "stem-4_25-update-1.xlsx",
        help="Path to the raw salary survey Excel file.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Could not find dataset at {args.data}\n"
            "Download 'Data Science and STEM Salaries' from Kaggle and place "
            "the .xlsx file at that path (see data/README.md)."
        )

    print(f"Loading data from {args.data} ...")
    df = load_and_clean(args.data)
    print(f"Rows after cleaning: {len(df):,}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    print("Training XGBoost pipeline...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, predictions)

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(rmse, 2),
        "r2": round(float(r2), 4),
        "test_rows": int(len(y_test)),
    }
    print("Test set performance:", json.dumps(metrics, indent=2))

    # Refit on all available data before shipping the model, now that we've
    # measured its honest performance on the held-out split above.
    print("Refitting on the full dataset for the final artifact...")
    pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "salary_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

    metadata = build_metadata(df, metrics)
    metadata_path = MODEL_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
