"""End-to-end training, model comparison, evaluation, and artifact generation pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    CORE_FEATURES,
    FULL_FEATURES,
    ID_COL,
    TARGET_COL,
    build_preprocessor,
    compute_eda_summary,
    load_data,
    split_data,
)

DATA_PATH = PROJECT_ROOT / "Food_Delivery_Times.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RANDOM_STATE = 42


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute MAE, RMSE, R2, and MAPE (in percent)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    mape = float(mean_absolute_percentage_error(y_t, y_p) * 100.0)
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape_pct": round(mape, 4),
    }


def bootstrap_test_cis(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Compute percentile bootstrap confidence intervals for test MAE, RMSE, and R2."""
    rng = np.random.default_rng(random_state)
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    n = len(y_t)

    maes = np.empty(n_bootstrap, dtype=float)
    rmses = np.empty(n_bootstrap, dtype=float)
    r2s = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt_b = y_t[idx]
        yp_b = y_p[idx]
        maes[i] = mean_absolute_error(yt_b, yp_b)
        rmses[i] = root_mean_squared_error(yt_b, yp_b)
        r2s[i] = r2_score(yt_b, yp_b)

    alpha = (1.0 - confidence) / 2.0
    lo_q = alpha * 100.0
    hi_q = (1.0 - alpha) * 100.0

    def _ci(arr: np.ndarray) -> dict[str, float]:
        return {
            "lower": round(float(np.percentile(arr, lo_q)), 4),
            "upper": round(float(np.percentile(arr, hi_q)), 4),
        }

    return {
        "confidence_level": confidence,
        "n_bootstrap": n_bootstrap,
        "mae": _ci(maes),
        "rmse": _ci(rmses),
        "r2": _ci(r2s),
    }


def compute_slice_errors(
    df_test: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    slice_cols: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute MAE, RMSE, mean bias, and sample count per category slice on test set."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_t - y_p

    slice_results: dict[str, dict[str, dict[str, Any]]] = {}
    for col in slice_cols:
        col_vals = df_test[col].fillna("Missing").astype(str)
        col_dict: dict[str, dict[str, Any]] = {}
        for grp in sorted(col_vals.unique()):
            mask = (col_vals == grp).to_numpy()
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            col_dict[grp] = {
                "count": cnt,
                "mae": round(float(mean_absolute_error(y_t[mask], y_p[mask])), 4),
                "rmse": round(float(root_mean_squared_error(y_t[mask], y_p[mask])), 4),
                "mean_bias": round(float(np.mean(residuals[mask])), 4),
            }
        slice_results[col] = col_dict
    return slice_results


def compute_permutation_importances(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    random_state: int = RANDOM_STATE,
) -> list[dict[str, Any]]:
    """Compute permutation importance on test set in raw feature space (MAE & RMSE increase)."""
    perm_mae = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=30,
        random_state=random_state,
        n_jobs=1,
    )
    perm_rmse = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=30,
        random_state=random_state,
        n_jobs=1,
    )

    items: list[dict[str, Any]] = []
    for i, feat in enumerate(feature_names):
        items.append(
            {
                "feature": feat,
                "mae_increase_mean": round(float(perm_mae.importances_mean[i]), 4),
                "mae_increase_std": round(float(perm_mae.importances_std[i]), 4),
                "rmse_increase_mean": round(float(perm_rmse.importances_mean[i]), 4),
                "rmse_increase_std": round(float(perm_rmse.importances_std[i]), 4),
            }
        )
    items.sort(key=lambda d: d["mae_increase_mean"], reverse=True)
    return items


def get_candidate_models() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Define candidate regression estimators and their hyperparameter grids."""
    return {
        "DummyRegressor": (
            DummyRegressor(strategy="mean"),
            {},
        ),
        "LinearRegression": (
            LinearRegression(),
            {},
        ),
        "Ridge": (
            Ridge(random_state=RANDOM_STATE),
            {"model__alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]},
        ),
        "Lasso": (
            Lasso(random_state=RANDOM_STATE, max_iter=10000),
            {"model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
        ),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [4, 6, 8, 10],
                "model__min_samples_leaf": [3, 5, 10],
            },
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [5, 10],
            },
        ),
        "HistGradientBoostingRegressor": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__max_iter": [100, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [3, 4, 6],
                "model__min_samples_leaf": [10, 20],
            },
        ),
        "SVR_RBF": (
            SVR(kernel="rbf"),
            {
                "model__C": [1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
                "model__epsilon": [0.1, 0.5, 1.0, 2.0],
                "model__gamma": ["scale", 0.05, 0.1],
            },
        ),
    }


def fit_and_evaluate_candidate(
    name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    mode: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv: KFold,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit a candidate pipeline on train set with 5-fold CV tuning and evaluate on train & val."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(mode=mode)),
            ("model", clone(estimator)),
        ]
    )

    if param_grid:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        best_params = {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        }
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = {}

    # 5-fold CV metrics on train set using best hyperparameters
    cv_scores = cross_validate(
        best_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=1,
    )

    train_pred = best_pipeline.predict(X_train)
    val_pred = best_pipeline.predict(X_val)

    train_metrics = compute_regression_metrics(y_train, train_pred)
    val_metrics = compute_regression_metrics(y_val, val_pred)

    record = {
        "model_name": name,
        "best_params": best_params,
        "cv_5fold": {
            "mae_mean": round(float(-np.mean(cv_scores["test_mae"])), 4),
            "mae_std": round(float(np.std(-cv_scores["test_mae"])), 4),
            "rmse_mean": round(float(-np.mean(cv_scores["test_rmse"])), 4),
            "rmse_std": round(float(np.std(-cv_scores["test_rmse"])), 4),
            "r2_mean": round(float(np.mean(cv_scores["test_r2"])), 4),
            "r2_std": round(float(np.std(cv_scores["test_r2"])), 4),
        },
        "train": train_metrics,
        "val": val_metrics,
        "overfitting_check": {
            "rmse_gap_val_minus_train": round(
                val_metrics["rmse"] - train_metrics["rmse"], 4
            ),
            "r2_gap_train_minus_val": round(
                train_metrics["r2"] - val_metrics["r2"], 4
            ),
        },
    }
    return best_pipeline, record


def run_training_pipeline(
    data_path: Path = DATA_PATH,
    models_dir: Path = MODELS_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Any]:
    """Execute the full ML pipeline: EDA, split, train/compare, evaluate, and save artifacts."""
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & EDA
    df = load_data(data_path)
    eda_summary = compute_eda_summary(df)
    eda_path = artifacts_dir / "eda_summary.json"
    with eda_path.open("w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    # 2. Strict Train (70%) / Val (15%) / Test (15%) split BEFORE any fitting
    train_df, val_df, test_df = split_data(df, random_state=RANDOM_STATE)

    X_train_core = train_df[CORE_FEATURES]
    y_train = train_df[TARGET_COL]
    X_val_core = val_df[CORE_FEATURES]
    y_val = val_df[TARGET_COL]
    X_test_core = test_df[CORE_FEATURES]
    y_test = test_df[TARGET_COL]

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = get_candidate_models()

    # 3. Train & compare all candidate models on Core-4 features
    core_results: list[dict[str, Any]] = []
    fitted_core_models: dict[str, Pipeline] = {}

    for name, (estimator, param_grid) in candidates.items():
        fitted_pipe, record = fit_and_evaluate_candidate(
            name=name,
            estimator=estimator,
            param_grid=param_grid,
            mode="core",
            X_train=X_train_core,
            y_train=y_train,
            X_val=X_val_core,
            y_val=y_val,
            cv=cv,
        )
        fitted_core_models[name] = fitted_pipe
        core_results.append(record)

    # Sort Core-4 models by validation RMSE (ascending)
    core_results.sort(key=lambda r: r["val"]["rmse"])
    best_core_record = core_results[0]
    best_core_name = best_core_record["model_name"]
    best_core_model = fitted_core_models[best_core_name]

    # 4. Evaluate best Core-4 model on holdout test set
    core_val_pred = best_core_model.predict(X_val_core)
    core_test_pred = best_core_model.predict(X_test_core)
    core_test_metrics = compute_regression_metrics(y_test, core_test_pred)
    core_test_cis = bootstrap_test_cis(y_test, core_test_pred)
    core_perm_imp = compute_permutation_importances(
        best_core_model, X_test_core, y_test, CORE_FEATURES
    )
    core_slice_errors = compute_slice_errors(
        test_df, y_test, core_test_pred, ["Weather", "Traffic_Level"]
    )

    # Conformal 90% residual quantile on validation set for prediction intervals
    core_val_abs_res = np.abs(np.asarray(y_val, dtype=float) - core_val_pred)
    core_pi_half_width_90 = round(float(np.quantile(core_val_abs_res, 0.90)), 4)

    # 5. Benchmark: Train & evaluate the best model family on Full-7 features
    X_train_full = train_df[FULL_FEATURES]
    X_val_full = val_df[FULL_FEATURES]
    X_test_full = test_df[FULL_FEATURES]

    best_family_estimator, best_family_grid = candidates[best_core_name]
    best_full_model, full_record = fit_and_evaluate_candidate(
        name=best_core_name,
        estimator=best_family_estimator,
        param_grid=best_family_grid,
        mode="full",
        X_train=X_train_full,
        y_train=y_train,
        X_val=X_val_full,
        y_val=y_val,
        cv=cv,
    )
    full_val_pred = best_full_model.predict(X_val_full)
    full_test_pred = best_full_model.predict(X_test_full)
    full_test_metrics = compute_regression_metrics(y_test, full_test_pred)
    full_test_cis = bootstrap_test_cis(y_test, full_test_pred)
    full_perm_imp = compute_permutation_importances(
        best_full_model, X_test_full, y_test, FULL_FEATURES
    )
    full_slice_errors = compute_slice_errors(
        test_df, y_test, full_test_pred, ["Weather", "Traffic_Level"]
    )
    full_val_abs_res = np.abs(np.asarray(y_val, dtype=float) - full_val_pred)
    full_pi_half_width_90 = round(float(np.quantile(full_val_abs_res, 0.90)), 4)

    # 6. Save trained sklearn Pipelines via joblib
    core_model_path = models_dir / "core_model.joblib"
    full_model_path = models_dir / "full_model.joblib"
    joblib.dump(best_core_model, core_model_path)
    joblib.dump(best_full_model, full_model_path)

    # 7. Save predictions.csv across train/val/test splits for downstream visualization
    split_frames = []
    for split_name, sub_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        out_df = sub_df.copy()
        pred_c = best_core_model.predict(sub_df[CORE_FEATURES])
        pred_f = best_full_model.predict(sub_df[FULL_FEATURES])
        actual = sub_df[TARGET_COL].to_numpy(dtype=float)

        out_df.insert(0, "split", split_name)
        out_df["actual"] = actual
        out_df["predicted_core"] = np.round(pred_c, 4)
        out_df["residual_core"] = np.round(actual - pred_c, 4)
        out_df["abs_error_core"] = np.round(np.abs(actual - pred_c), 4)
        out_df["predicted_full"] = np.round(pred_f, 4)
        out_df["residual_full"] = np.round(actual - pred_f, 4)
        out_df["abs_error_full"] = np.round(np.abs(actual - pred_f), 4)
        split_frames.append(out_df)

    predictions_df = pd.concat(split_frames, ignore_index=True)
    # Reorder columns cleanly
    ordered_cols = [
        "split",
        ID_COL,
        "actual",
        "predicted_core",
        "residual_core",
        "abs_error_core",
        "predicted_full",
        "residual_full",
        "abs_error_full",
        *FULL_FEATURES,
    ]
    predictions_df = predictions_df[ordered_cols]
    predictions_path = artifacts_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # 8. Save metrics.json
    metrics_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dataset": {
            "path": str(data_path.name),
            "total_rows": int(len(df)),
            "splits": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "core_features": CORE_FEATURES,
            "full_features": FULL_FEATURES,
            "target": TARGET_COL,
        },
        "core_model_comparison": core_results,
        "best_core_model": {
            "model_name": best_core_name,
            "mode": "core",
            "features": CORE_FEATURES,
            "best_params": best_core_record["best_params"],
            "cv_5fold": best_core_record["cv_5fold"],
            "train": best_core_record["train"],
            "val": best_core_record["val"],
            "test": core_test_metrics,
            "test_bootstrap_95ci": core_test_cis,
            "overfitting_check": best_core_record["overfitting_check"],
            "prediction_interval_90_half_width": core_pi_half_width_90,
            "permutation_importance_test": core_perm_imp,
            "slice_errors_test": core_slice_errors,
            "artifact_path": "models/core_model.joblib",
        },
        "full_benchmark_model": {
            "model_name": best_core_name,
            "mode": "full",
            "features": FULL_FEATURES,
            "best_params": full_record["best_params"],
            "cv_5fold": full_record["cv_5fold"],
            "train": full_record["train"],
            "val": full_record["val"],
            "test": full_test_metrics,
            "test_bootstrap_95ci": full_test_cis,
            "overfitting_check": full_record["overfitting_check"],
            "prediction_interval_90_half_width": full_pi_half_width_90,
            "permutation_importance_test": full_perm_imp,
            "slice_errors_test": full_slice_errors,
            "artifact_path": "models/full_model.joblib",
        },
        "core_vs_full_test_delta": {
            "mae_diff_full_minus_core": round(
                full_test_metrics["mae"] - core_test_metrics["mae"], 4
            ),
            "rmse_diff_full_minus_core": round(
                full_test_metrics["rmse"] - core_test_metrics["rmse"], 4
            ),
            "r2_diff_full_minus_core": round(
                full_test_metrics["r2"] - core_test_metrics["r2"], 4
            ),
        },
    }

    metrics_path = artifacts_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


def main() -> None:
    metrics = run_training_pipeline()
    core = metrics["best_core_model"]
    full = metrics["full_benchmark_model"]
    print(f"Best Core-4 Model: {core['model_name']} | params={core['best_params']}")
    print(
        f"  Val  -> MAE: {core['val']['mae']:.3f}, RMSE: {core['val']['rmse']:.3f}, R2: {core['val']['r2']:.4f}"
    )
    print(
        f"  Test -> MAE: {core['test']['mae']:.3f}, RMSE: {core['test']['rmse']:.3f}, R2: {core['test']['r2']:.4f}, MAPE: {core['test']['mape_pct']:.2f}%"
    )
    print(f"Full-7 Benchmark ({full['model_name']}):")
    print(
        f"  Test -> MAE: {full['test']['mae']:.3f}, RMSE: {full['test']['rmse']:.3f}, R2: {full['test']['r2']:.4f}, MAPE: {full['test']['mape_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
