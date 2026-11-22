"""Data loading, EDA summary, leakage-free splitting, and sklearn preprocessing pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET_COL = "Delivery_Time_min"
ID_COL = "Order_ID"

CORE_FEATURES = [
    "Distance_km",
    "Traffic_Level",
    "Weather",
    "Preparation_Time_min",
]

FULL_FEATURES = [
    "Distance_km",
    "Traffic_Level",
    "Weather",
    "Preparation_Time_min",
    "Time_of_Day",
    "Vehicle_Type",
    "Courier_Experience_yrs",
]

CORE_NUMERIC_FEATURES = ["Distance_km", "Preparation_Time_min"]
FULL_NUMERIC_FEATURES = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

ORDINAL_FEATURES = ["Traffic_Level"]
TRAFFIC_ORDER = ["Low", "Medium", "High"]

CORE_NOMINAL_FEATURES = ["Weather"]
FULL_NOMINAL_FEATURES = ["Weather", "Time_of_Day", "Vehicle_Type"]

VALID_WEATHER = ["Clear", "Rainy", "Foggy", "Snowy", "Windy"]
VALID_TRAFFIC = ["Low", "Medium", "High"]
VALID_TIME_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]
VALID_VEHICLE_TYPE = ["Bike", "Scooter", "Car"]


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load Food_Delivery_Times.csv, clean whitespace, and drop rows with missing target."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    # Normalize string columns: strip whitespace and treat empty strings as NaN
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        df[col] = df[col].astype("object").apply(
            lambda v: v.strip() if isinstance(v, str) else v
        )
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "NaN": np.nan})

    # Drop rows only if target is missing or non-positive
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df = df.dropna(subset=[TARGET_COL])
        df = df[df[TARGET_COL] > 0].reset_index(drop=True)

    return df


def compute_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute structured EDA statistics for reporting and visualization."""
    n_rows, n_cols = df.shape

    missing_values: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        miss_cnt = int(df[col].isna().sum())
        missing_values[col] = {
            "count": miss_cnt,
            "pct": round(100.0 * miss_cnt / n_rows, 2) if n_rows > 0 else 0.0,
        }

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != ID_COL
    ]
    numeric_summary: dict[str, dict[str, Any]] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        q25 = float(s.quantile(0.25))
        q50 = float(s.quantile(0.50))
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        outlier_cnt = int(((s < lower_fence) | (s > upper_fence)).sum())
        numeric_summary[col] = {
            "count": int(s.count()),
            "missing": int(df[col].isna().sum()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "q25": round(q25, 4),
            "median": round(q50, 4),
            "q75": round(q75, 4),
            "max": round(float(s.max()), 4),
            "iqr": round(iqr, 4),
            "outlier_count_iqr": outlier_cnt,
            "skewness": round(float(s.skew()), 4),
        }

    categorical_cols = [
        c for c in ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
        if c in df.columns
    ]
    categorical_summary: dict[str, dict[str, Any]] = {}
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=True).to_dict()
        categorical_summary[col] = {
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique(dropna=True)),
            "value_counts": {str(k): int(v) for k, v in vc.items()},
        }

    # Anomalies check
    delivery_lt_prep = df[df[TARGET_COL] < df["Preparation_Time_min"]]
    target_s = df[TARGET_COL].dropna()
    t_q25, t_q75 = float(target_s.quantile(0.25)), float(target_s.quantile(0.75))
    t_iqr = t_q75 - t_q25
    t_upper_fence = t_q75 + 1.5 * t_iqr
    t_lower_fence = t_q25 - 1.5 * t_iqr

    anomalies = {
        "delivery_lt_prep_count": int(len(delivery_lt_prep)),
        "delivery_lt_prep_order_ids": (
            [int(x) for x in delivery_lt_prep[ID_COL].tolist()]
            if ID_COL in df.columns
            else []
        ),
        "non_positive_distance_count": int((df["Distance_km"] <= 0).sum()),
        "non_positive_delivery_count": int((df[TARGET_COL] <= 0).sum()),
        "target_iqr_lower_fence": round(t_lower_fence, 4),
        "target_iqr_upper_fence": round(t_upper_fence, 4),
        "target_outlier_count": int(
            ((target_s < t_lower_fence) | (target_s > t_upper_fence)).sum()
        ),
    }

    # Correlations with target
    corr_df = df[numeric_cols].copy()
    if "Traffic_Level" in df.columns:
        traffic_map = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
        corr_df["Traffic_Level_Ord"] = df["Traffic_Level"].map(traffic_map)

    correlations_with_target: dict[str, dict[str, float]] = {}
    for col in corr_df.columns:
        if col == TARGET_COL:
            continue
        valid = corr_df[[col, TARGET_COL]].dropna()
        pearson = float(valid[col].corr(valid[TARGET_COL], method="pearson"))
        spearman = float(valid[col].corr(valid[TARGET_COL], method="spearman"))
        correlations_with_target[col] = {
            "pearson": round(pearson, 4),
            "spearman": round(spearman, 4),
        }

    def _group_target_stats(group_col: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        temp = df[[group_col, TARGET_COL]].copy()
        temp[group_col] = temp[group_col].fillna("Missing")
        for grp, sub in temp.groupby(group_col):
            vals = sub[TARGET_COL]
            out[str(grp)] = {
                "count": int(len(vals)),
                "mean": round(float(vals.mean()), 2),
                "median": round(float(vals.median()), 2),
                "std": round(float(vals.std()), 2) if len(vals) > 1 else 0.0,
            }
        return out

    return {
        "dataset_shape": {"rows": int(n_rows), "columns": int(n_cols)},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": missing_values,
        "total_missing_cells": int(df.isna().sum().sum()),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_order_ids": (
            int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else 0
        ),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "anomalies": anomalies,
        "correlations_with_target": correlations_with_target,
        "target_by_weather": _group_target_stats("Weather"),
        "target_by_traffic": _group_target_stats("Traffic_Level"),
    }


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into Train (70%), Validation (15%), and Test (15%) strictly before fitting."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=round(1.0 - train_size, 10),
        random_state=random_state,
        shuffle=True,
    )
    relative_test_size = round(test_size / (val_size + test_size), 10)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_feature_columns(mode: str = "core") -> list[str]:
    """Return ordered feature list for 'core' or 'full' mode."""
    if mode == "core":
        return list(CORE_FEATURES)
    if mode == "full":
        return list(FULL_FEATURES)
    raise ValueError(f"Unsupported mode '{mode}'. Expected 'core' or 'full'.")


def build_preprocessor(mode: str = "core") -> ColumnTransformer:
    """Build a leakage-free ColumnTransformer fitted strictly on training data.

    - Numeric features: median imputation (+ missing indicator in full mode) + StandardScaler
    - Traffic_Level: most-frequent imputation -> OrdinalEncoder(['Low', 'Medium', 'High']) -> StandardScaler
    - Nominal features: constant 'Unknown' imputation -> OneHotEncoder(handle_unknown='ignore')
    """
    if mode == "core":
        numeric_cols = CORE_NUMERIC_FEATURES
        nominal_cols = CORE_NOMINAL_FEATURES
        add_num_indicator = False
    elif mode == "full":
        numeric_cols = FULL_NUMERIC_FEATURES
        nominal_cols = FULL_NOMINAL_FEATURES
        add_num_indicator = True
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'core' or 'full'.")

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=add_num_indicator),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[TRAFFIC_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("ord", ordinal_pipeline, ORDINAL_FEATURES),
            ("cat", nominal_pipeline, nominal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
