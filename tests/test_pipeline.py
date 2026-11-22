"""Automated QA and verification test suite for Food Delivery Time Prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    CORE_FEATURES,
    FULL_FEATURES,
    ID_COL,
    TARGET_COL,
    TRAFFIC_ORDER,
    build_preprocessor,
    compute_eda_summary,
    load_data,
    split_data,
)
from src.predict import predict_delivery_time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Food_Delivery_Times.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

EXPECTED_PLOTS = [
    "01_target_distribution.png",
    "02_missing_values.png",
    "03_feature_relationships.png",
    "04_correlation_heatmap.png",
    "05_model_comparison.png",
    "06_actual_vs_predicted.png",
    "07_residual_analysis.png",
    "08_feature_importance.png",
    "09_slice_error_analysis.png",
]


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def splits(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return split_data(raw_df, random_state=42)


# ---------------------------------------------------------------------------
# 1. Data Loading & Schema Validation
# ---------------------------------------------------------------------------
class TestDataLoadingAndSchema:
    def test_dataset_exists_and_loads(self, raw_df: pd.DataFrame) -> None:
        assert DATA_PATH.exists()
        assert len(raw_df) == 1000
        assert raw_df.shape[1] == 9

    def test_expected_columns_present(self, raw_df: pd.DataFrame) -> None:
        expected_cols = {ID_COL, TARGET_COL, *FULL_FEATURES}
        assert expected_cols.issubset(set(raw_df.columns))

    def test_target_is_valid_and_positive(self, raw_df: pd.DataFrame) -> None:
        assert raw_df[TARGET_COL].isna().sum() == 0
        assert (raw_df[TARGET_COL] > 0).all()

    def test_eda_summary_structure(self, raw_df: pd.DataFrame) -> None:
        summary = compute_eda_summary(raw_df)
        for key in [
            "dataset_shape",
            "columns",
            "missing_values",
            "numeric_summary",
            "categorical_summary",
            "anomalies",
            "correlations_with_target",
            "target_by_weather",
            "target_by_traffic",
        ]:
            assert key in summary
        assert summary["dataset_shape"]["rows"] == len(raw_df)


# ---------------------------------------------------------------------------
# 2. Train / Validation / Test Splitting
# ---------------------------------------------------------------------------
class TestDataSplitting:
    def test_split_proportions(
        self,
        raw_df: pd.DataFrame,
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        train_df, val_df, test_df = splits
        n = len(raw_df)
        assert len(train_df) + len(val_df) + len(test_df) == n
        assert abs(len(train_df) / n - 0.70) < 0.01
        assert abs(len(val_df) / n - 0.15) < 0.01
        assert abs(len(test_df) / n - 0.15) < 0.01

    def test_splits_are_disjoint(
        self,
        raw_df: pd.DataFrame,
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        train_df, val_df, test_df = splits
        train_ids = set(train_df[ID_COL])
        val_ids = set(val_df[ID_COL])
        test_ids = set(test_df[ID_COL])

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert len(train_ids | val_ids | test_ids) == len(raw_df)

    def test_split_reproducibility(self, raw_df: pd.DataFrame) -> None:
        tr1, va1, te1 = split_data(raw_df, random_state=42)
        tr2, va2, te2 = split_data(raw_df, random_state=42)
        pd.testing.assert_frame_equal(tr1, tr2)
        pd.testing.assert_frame_equal(va1, va2)
        pd.testing.assert_frame_equal(te1, te2)


# ---------------------------------------------------------------------------
# 3. Leakage-Free Preprocessing Pipeline
# ---------------------------------------------------------------------------
class TestPreprocessingPipeline:
    @pytest.mark.parametrize("mode,features", [("core", CORE_FEATURES), ("full", FULL_FEATURES)])
    def test_fit_transform_no_nans_and_valid_shape(
        self,
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        mode: str,
        features: list[str],
    ) -> None:
        train_df, val_df, test_df = splits
        preprocessor = build_preprocessor(mode=mode)

        X_train_t = preprocessor.fit_transform(train_df[features])
        X_val_t = preprocessor.transform(val_df[features])
        X_test_t = preprocessor.transform(test_df[features])

        assert X_train_t.shape[0] == len(train_df)
        assert X_val_t.shape[0] == len(val_df)
        assert X_test_t.shape[0] == len(test_df)
        assert X_train_t.shape[1] == X_val_t.shape[1] == X_test_t.shape[1]
        assert X_train_t.shape[1] >= len(features)

        assert np.isfinite(X_train_t).all()
        assert np.isfinite(X_val_t).all()
        assert np.isfinite(X_test_t).all()

    def test_handles_all_missing_and_unseen_categories(
        self,
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        train_df, _, _ = splits
        preprocessor = build_preprocessor(mode="full")
        preprocessor.fit(train_df[FULL_FEATURES])

        edge_df = pd.DataFrame(
            [
                {
                    "Distance_km": np.nan,
                    "Traffic_Level": None,
                    "Weather": None,
                    "Preparation_Time_min": np.nan,
                    "Time_of_Day": None,
                    "Vehicle_Type": None,
                    "Courier_Experience_yrs": np.nan,
                },
                {
                    "Distance_km": 10.0,
                    "Traffic_Level": "Extreme",
                    "Weather": "Hurricane",
                    "Preparation_Time_min": 15.0,
                    "Time_of_Day": "Midnight",
                    "Vehicle_Type": "Drone",
                    "Courier_Experience_yrs": 2.0,
                },
            ]
        )
        transformed = preprocessor.transform(edge_df)
        assert transformed.shape[0] == 2
        assert np.isfinite(transformed).all()

    def test_ordinal_encoding_preserves_traffic_order(
        self,
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        train_df, _, _ = splits
        preprocessor = build_preprocessor(mode="core")
        preprocessor.fit(train_df[CORE_FEATURES])

        ord_pipe = preprocessor.named_transformers_["ord"]
        ord_encoder = ord_pipe.named_steps["ordinal"]
        assert list(ord_encoder.categories_[0]) == TRAFFIC_ORDER

        sample = pd.DataFrame(
            [
                {"Distance_km": 10.0, "Traffic_Level": lvl, "Weather": "Clear", "Preparation_Time_min": 15.0}
                for lvl in TRAFFIC_ORDER
            ]
        )
        encoded = ord_pipe.transform(sample[["Traffic_Level"]]).ravel()
        assert encoded[0] < encoded[1] < encoded[2]

    def test_no_data_leakage_from_val_or_test(
        self,
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        train_df, val_df, test_df = splits
        preprocessor = build_preprocessor(mode="core")
        preprocessor.fit(train_df[CORE_FEATURES])

        num_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
        expected_mean = train_df[["Distance_km", "Preparation_Time_min"]].mean().to_numpy()
        np.testing.assert_allclose(num_scaler.mean_, expected_mean, rtol=1e-6)

        # Transforming val/test must not mutate fitted scaler statistics
        mean_before = num_scaler.mean_.copy()
        _ = preprocessor.transform(val_df[CORE_FEATURES])
        _ = preprocessor.transform(test_df[CORE_FEATURES])
        np.testing.assert_array_equal(num_scaler.mean_, mean_before)


# ---------------------------------------------------------------------------
# 4. Saved Models & Prediction API Sanity / Monotonicity
# ---------------------------------------------------------------------------
class TestModelsAndPredictionAPI:
    @pytest.mark.parametrize(
        "model_file,features",
        [("core_model.joblib", CORE_FEATURES), ("full_model.joblib", FULL_FEATURES)],
    )
    def test_saved_models_load_and_predict(
        self,
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        model_file: str,
        features: list[str],
    ) -> None:
        _, _, test_df = splits
        model_path = MODELS_DIR / model_file
        assert model_path.exists() and model_path.stat().st_size > 0

        model = joblib.load(model_path)
        preds = model.predict(test_df[features])
        assert preds.shape == (len(test_df),)
        assert np.isfinite(preds).all()
        assert (preds > 0).all()

    def test_predict_delivery_time_core_and_full_contract(self) -> None:
        core_res = predict_delivery_time(
            distance_km=10.0,
            traffic_level="Medium",
            weather="Rainy",
            preparation_time_min=15.0,
            mode="core",
        )
        full_res = predict_delivery_time(
            distance_km=10.0,
            traffic_level="medium",
            weather="rainy",
            preparation_time_min=15.0,
            time_of_day="Evening",
            vehicle_type="Bike",
            courier_experience_yrs=4.0,
            mode="full",
        )
        for res, expected_mode in [(core_res, "core"), (full_res, "full")]:
            assert set(res.keys()) == {
                "predicted_minutes",
                "lower_bound",
                "upper_bound",
                "model_name",
                "mode",
            }
            assert res["mode"] == expected_mode
            assert np.isfinite(res["predicted_minutes"])
            assert 0 < res["lower_bound"] <= res["predicted_minutes"] <= res["upper_bound"]

    def test_predict_delivery_time_monotonic_distance(self) -> None:
        p_short = predict_delivery_time(2.0, "Medium", "Clear", 15.0)["predicted_minutes"]
        p_mid = predict_delivery_time(8.0, "Medium", "Clear", 15.0)["predicted_minutes"]
        p_long = predict_delivery_time(18.0, "Medium", "Clear", 15.0)["predicted_minutes"]
        assert p_short < p_mid < p_long

    def test_predict_delivery_time_monotonic_prep_time(self) -> None:
        p_fast = predict_delivery_time(10.0, "Medium", "Clear", 5.0)["predicted_minutes"]
        p_med = predict_delivery_time(10.0, "Medium", "Clear", 15.0)["predicted_minutes"]
        p_slow = predict_delivery_time(10.0, "Medium", "Clear", 28.0)["predicted_minutes"]
        assert p_fast < p_med < p_slow

    def test_predict_delivery_time_monotonic_traffic(self) -> None:
        p_low = predict_delivery_time(10.0, "Low", "Clear", 15.0)["predicted_minutes"]
        p_med = predict_delivery_time(10.0, "Medium", "Clear", 15.0)["predicted_minutes"]
        p_high = predict_delivery_time(10.0, "High", "Clear", 15.0)["predicted_minutes"]
        assert p_low <= p_med <= p_high
        assert p_high > p_low

    def test_predict_delivery_time_handles_missing_and_validation(self) -> None:
        res_missing = predict_delivery_time(10.0, None, None, 15.0, mode="core")
        assert np.isfinite(res_missing["predicted_minutes"])
        assert res_missing["predicted_minutes"] > 0

        with pytest.raises(ValueError):
            predict_delivery_time(-1.0, "Low", "Clear", 15.0)
        with pytest.raises(ValueError):
            predict_delivery_time(5.0, "Low", "Clear", -5.0)
        with pytest.raises(ValueError):
            predict_delivery_time(5.0, "Low", "Clear", 10.0, mode="invalid")


# ---------------------------------------------------------------------------
# 5. Artifacts & Visualization Plots Verification
# ---------------------------------------------------------------------------
class TestArtifactsAndPlots:
    def test_metrics_json_valid(self) -> None:
        path = ARTIFACTS_DIR / "metrics.json"
        assert path.exists() and path.stat().st_size > 0
        data = json.loads(path.read_text(encoding="utf-8"))

        for key in [
            "dataset",
            "core_model_comparison",
            "best_core_model",
            "full_benchmark_model",
            "core_vs_full_test_delta",
        ]:
            assert key in data

        assert len(data["core_model_comparison"]) >= 7
        assert data["best_core_model"]["test"]["r2"] > 0.70
        assert data["best_core_model"]["test"]["mae"] < 10.0
        assert data["full_benchmark_model"]["test"]["r2"] > 0.70

    def test_predictions_csv_valid(self) -> None:
        path = ARTIFACTS_DIR / "predictions.csv"
        assert path.exists() and path.stat().st_size > 0
        df = pd.read_csv(path)
        assert len(df) == 1000
        for col in [
            "split",
            "Order_ID",
            "actual",
            "predicted_core",
            "residual_core",
            "predicted_full",
            "residual_full",
        ]:
            assert col in df.columns
        counts = df["split"].value_counts().to_dict()
        assert counts == {"train": 700, "val": 150, "test": 150}

    def test_eda_summary_json_valid(self) -> None:
        path = ARTIFACTS_DIR / "eda_summary.json"
        assert path.exists() and path.stat().st_size > 0
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["dataset_shape"]["rows"] == 1000
        assert data["dataset_shape"]["columns"] == 9

    @pytest.mark.parametrize("plot_name", EXPECTED_PLOTS)
    def test_all_nine_plots_exist_and_non_empty(self, plot_name: str) -> None:
        plot_path = PLOTS_DIR / plot_name
        assert plot_path.exists(), f"Missing plot: {plot_path}"
        assert plot_path.stat().st_size > 1024, f"Plot file too small: {plot_path}"
