"""Export trained model weights, metrics, predictions, and plots for a Hugging Face Static Space."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_delivery_time

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPACE_DIR = PROJECT_ROOT / "hf_space"


def _extract_linear_model(mode: str, metrics: dict[str, Any]) -> dict[str, Any]:
    pipe = joblib.load(MODELS_DIR / f"{mode}_model.joblib")
    pre = pipe.named_steps["preprocessor"]
    lr = pipe.named_steps["model"]

    feature_names = list(pre.get_feature_names_out())
    coefs = {name: float(val) for name, val in zip(feature_names, lr.coef_)}

    num_pipe = pre.named_transformers_["num"]
    ord_pipe = pre.named_transformers_["ord"]
    cat_pipe = pre.named_transformers_["cat"]

    meta_key = "best_core_model" if mode == "core" else "full_benchmark_model"
    pi_hw = float(metrics[meta_key]["prediction_interval_90_half_width"])

    return {
        "model_name": metrics[meta_key]["model_name"],
        "mode": mode,
        "intercept": float(lr.intercept_),
        "coefs": coefs,
        "pi_90_half_width": pi_hw,
        "num": {
            "columns": list(pre.transformers_[0][2]),
            "medians": [float(x) for x in num_pipe.named_steps["imputer"].statistics_],
            "means": [float(x) for x in num_pipe.named_steps["scaler"].mean_],
            "scales": [float(x) for x in num_pipe.named_steps["scaler"].scale_],
        },
        "ord": {
            "categories": [str(x) for x in ord_pipe.named_steps["ordinal"].categories_[0]],
            "mode_value": str(ord_pipe.named_steps["imputer"].statistics_[0]),
            "mean": float(ord_pipe.named_steps["scaler"].mean_[0]),
            "scale": float(ord_pipe.named_steps["scaler"].scale_[0]),
        },
        "cat": {
            "columns": list(pre.transformers_[2][2]),
            "categories": [
                [str(v) for v in cats]
                for cats in cat_pipe.named_steps["onehot"].categories_
            ],
        },
    }


def _js_equivalent_predict(spec: dict[str, Any], inputs: dict[str, Any]) -> float:
    """Pure-math mirror of the client-side JS predictor to verify exact numerical parity."""
    total = spec["intercept"]
    coefs = spec["coefs"]

    # Numeric features
    num_cols = spec["num"]["columns"]
    for idx, col in enumerate(num_cols):
        raw = inputs.get(col)
        is_missing = raw is None
        val = spec["num"]["medians"][idx] if is_missing else float(raw)
        z = (val - spec["num"]["means"][idx]) / spec["num"]["scales"][idx]
        total += coefs[f"num__{col}"] * z

    # Optional missing indicator for Courier_Experience_yrs in full mode
    miss_key = "num__missingindicator_Courier_Experience_yrs"
    if miss_key in coefs:
        idx_miss = len(num_cols)
        miss_val = 1.0 if inputs.get("Courier_Experience_yrs") is None else 0.0
        z_miss = (miss_val - spec["num"]["means"][idx_miss]) / spec["num"]["scales"][idx_miss]
        total += coefs[miss_key] * z_miss

    # Ordinal Traffic_Level
    traffic = inputs.get("Traffic_Level") or spec["ord"]["mode_value"]
    ord_cats = spec["ord"]["categories"]
    ord_idx = float(ord_cats.index(traffic)) if traffic in ord_cats else 1.0
    z_ord = (ord_idx - spec["ord"]["mean"]) / spec["ord"]["scale"]
    total += coefs["ord__Traffic_Level"] * z_ord

    # One-hot nominal features
    for col, cats in zip(spec["cat"]["columns"], spec["cat"]["categories"]):
        val = inputs.get(col) or "Unknown"
        key = f"cat__{col}_{val}"
        if key in coefs:
            total += coefs[key]

    return max(1.0, total)


def export() -> None:
    SPACE_DIR.mkdir(parents=True, exist_ok=True)
    plots_dst = SPACE_DIR / "plots"
    plots_dst.mkdir(parents=True, exist_ok=True)

    with (ARTIFACTS_DIR / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    with (ARTIFACTS_DIR / "eda_summary.json").open("r", encoding="utf-8") as f:
        eda = json.load(f)

    core_spec = _extract_linear_model("core", metrics)
    full_spec = _extract_linear_model("full", metrics)

    # Verify exact numerical parity against scikit-learn joblib pipelines
    sample_cases = [
        dict(distance_km=10.0, traffic_level="Medium", weather="Rainy", preparation_time_min=15.0),
        dict(distance_km=3.5, traffic_level="Low", weather="Clear", preparation_time_min=8.0),
        dict(distance_km=18.2, traffic_level="High", weather="Snowy", preparation_time_min=27.0),
    ]
    for case in sample_cases:
        py_core = predict_delivery_time(**case, mode="core")["predicted_minutes"]
        js_core = round(_js_equivalent_predict(core_spec, {
            "Distance_km": case["distance_km"],
            "Traffic_Level": case["traffic_level"],
            "Weather": case["weather"],
            "Preparation_Time_min": case["preparation_time_min"],
        }), 2)
        assert abs(py_core - js_core) <= 0.01, f"Core mismatch: {py_core} vs {js_core}"

        py_full = predict_delivery_time(
            **case, time_of_day="Evening", vehicle_type="Scooter", courier_experience_yrs=4.0, mode="full"
        )["predicted_minutes"]
        js_full = round(_js_equivalent_predict(full_spec, {
            "Distance_km": case["distance_km"],
            "Traffic_Level": case["traffic_level"],
            "Weather": case["weather"],
            "Preparation_Time_min": case["preparation_time_min"],
            "Time_of_Day": "Evening",
            "Vehicle_Type": "Scooter",
            "Courier_Experience_yrs": 4.0,
        }), 2)
        assert abs(py_full - js_full) <= 0.01, f"Full mismatch: {py_full} vs {js_full}"

    preds_df = pd.read_csv(ARTIFACTS_DIR / "predictions.csv")
    preds_records = preds_df[
        [
            "split",
            "Order_ID",
            "actual",
            "predicted_core",
            "residual_core",
            "Distance_km",
            "Traffic_Level",
            "Weather",
            "Preparation_Time_min",
        ]
    ].fillna("Unknown").to_dict(orient="records")

    bundle = {
        "models": {"core": core_spec, "full": full_spec},
        "metrics": metrics,
        "eda": eda,
        "predictions": preds_records,
    }
    out_json = SPACE_DIR / "data_bundle.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"))

    for png in sorted((ARTIFACTS_DIR / "plots").glob("*.png")):
        shutil.copy2(png, plots_dst / png.name)

    print(f"Exported static Space bundle to {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")
    print(f"Copied {len(list(plots_dst.glob('*.png')))} plots to {plots_dst}")
    print("Parity check PASSED (exact match with joblib models).")


if __name__ == "__main__":
    export()
