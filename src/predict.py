"""Prediction API and CLI for Food Delivery Time Prediction."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    CORE_FEATURES,
    FULL_FEATURES,
    VALID_TIME_OF_DAY,
    VALID_TRAFFIC,
    VALID_VEHICLE_TYPE,
    VALID_WEATHER,
)

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

_TRAFFIC_LOOKUP = {v.lower(): v for v in VALID_TRAFFIC}
_WEATHER_LOOKUP = {v.lower(): v for v in VALID_WEATHER}
_TIME_LOOKUP = {v.lower(): v for v in VALID_TIME_OF_DAY}
_VEHICLE_LOOKUP = {v.lower(): v for v in VALID_VEHICLE_TYPE}


def _normalize_category(value: str | None, lookup: dict[str, str]) -> Any:
    if value is None:
        return np.nan
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return np.nan
    return lookup.get(s.lower(), s)


@lru_cache(maxsize=4)
def _load_model(mode: str) -> Any:
    filename = "core_model.joblib" if mode == "core" else "full_model.joblib"
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python -m src.train_evaluate` first."
        )
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def _load_metrics_metadata() -> dict[str, Any]:
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def predict_delivery_time(
    distance_km: float,
    traffic_level: str | None,
    weather: str | None,
    preparation_time_min: float,
    time_of_day: str | None = None,
    vehicle_type: str | None = None,
    courier_experience_yrs: float | None = None,
    mode: str = "core",
) -> dict[str, Any]:
    """Predict delivery time in minutes with a 90% empirical prediction interval.

    Parameters
    ----------
    distance_km : float
        Delivery distance in kilometers (>= 0).
    traffic_level : str | None
        One of 'Low', 'Medium', 'High' (case-insensitive) or None.
    weather : str | None
        One of 'Clear', 'Rainy', 'Foggy', 'Snowy', 'Windy' (case-insensitive) or None.
    preparation_time_min : float
        Restaurant preparation time in minutes (>= 0).
    time_of_day : str | None, optional
        Used when mode='full'. One of 'Morning', 'Afternoon', 'Evening', 'Night'.
    vehicle_type : str | None, optional
        Used when mode='full'. One of 'Bike', 'Scooter', 'Car'.
    courier_experience_yrs : float | None, optional
        Used when mode='full'. Courier years of experience (>= 0).
    mode : str, default='core'
        'core' (4 features) or 'full' (all 7 features).

    Returns
    -------
    dict[str, Any]
        {
            "predicted_minutes": float,
            "lower_bound": float,
            "upper_bound": float,
            "model_name": str,
            "mode": str,
        }
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"core", "full"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'core' or 'full'.")

    dist_val = float(distance_km)
    prep_val = float(preparation_time_min)
    if dist_val < 0:
        raise ValueError("distance_km must be non-negative.")
    if prep_val < 0:
        raise ValueError("preparation_time_min must be non-negative.")

    row: dict[str, Any] = {
        "Distance_km": dist_val,
        "Traffic_Level": _normalize_category(traffic_level, _TRAFFIC_LOOKUP),
        "Weather": _normalize_category(weather, _WEATHER_LOOKUP),
        "Preparation_Time_min": prep_val,
    }

    if mode_norm == "full":
        exp_val: Any = np.nan
        if courier_experience_yrs is not None:
            exp_val = float(courier_experience_yrs)
            if exp_val < 0:
                raise ValueError("courier_experience_yrs must be non-negative.")
        row["Time_of_Day"] = _normalize_category(time_of_day, _TIME_LOOKUP)
        row["Vehicle_Type"] = _normalize_category(vehicle_type, _VEHICLE_LOOKUP)
        row["Courier_Experience_yrs"] = exp_val
        input_df = pd.DataFrame([row], columns=FULL_FEATURES)
    else:
        input_df = pd.DataFrame([row], columns=CORE_FEATURES)

    model = _load_model(mode_norm)
    raw_pred = float(model.predict(input_df)[0])
    predicted_minutes = round(max(1.0, raw_pred), 2)

    metrics = _load_metrics_metadata()
    section_key = "best_core_model" if mode_norm == "core" else "full_benchmark_model"
    section = metrics.get(section_key, {})

    model_name = section.get("model_name", type(model.named_steps["model"]).__name__)
    half_width = section.get("prediction_interval_90_half_width")
    if half_width is None:
        val_rmse = section.get("val", {}).get("rmse", 10.0)
        half_width = 1.645 * float(val_rmse)

    lower_bound = round(max(1.0, predicted_minutes - float(half_width)), 2)
    upper_bound = round(predicted_minutes + float(half_width), 2)

    return {
        "predicted_minutes": predicted_minutes,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "model_name": str(model_name),
        "mode": mode_norm,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict food delivery time in minutes."
    )
    parser.add_argument(
        "--distance",
        type=float,
        required=True,
        help="Delivery distance in kilometers (e.g. 8.5).",
    )
    parser.add_argument(
        "--traffic",
        type=str,
        required=True,
        help="Traffic level: Low, Medium, or High.",
    )
    parser.add_argument(
        "--weather",
        type=str,
        required=True,
        help="Weather condition: Clear, Rainy, Foggy, Snowy, or Windy.",
    )
    parser.add_argument(
        "--prep-time",
        type=float,
        required=True,
        help="Restaurant preparation time in minutes (e.g. 15).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="core",
        choices=["core", "full"],
        help="Model mode: 'core' (4 features) or 'full' (all 7 features).",
    )
    parser.add_argument(
        "--time-of-day",
        type=str,
        default=None,
        help="(Full mode) Time of day: Morning, Afternoon, Evening, Night.",
    )
    parser.add_argument(
        "--vehicle-type",
        type=str,
        default=None,
        help="(Full mode) Vehicle type: Bike, Scooter, Car.",
    )
    parser.add_argument(
        "--experience",
        type=float,
        default=None,
        help="(Full mode) Courier experience in years.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print prediction output as formatted JSON.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = predict_delivery_time(
        distance_km=args.distance,
        traffic_level=args.traffic,
        weather=args.weather,
        preparation_time_min=args.prep_time,
        time_of_day=args.time_of_day,
        vehicle_type=args.vehicle_type,
        courier_experience_yrs=args.experience,
        mode=args.mode,
    )
    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Predicted Delivery Time: {result['predicted_minutes']:.1f} min "
            f"(90% interval: {result['lower_bound']:.1f} - {result['upper_bound']:.1f} min) "
            f"[model={result['model_name']}, mode={result['mode']}]"
        )


if __name__ == "__main__":
    main()
