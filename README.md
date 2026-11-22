# Food Delivery Time Prediction

> **Live Demo (Hugging Face Spaces):** https://huggingface.co/spaces/pcsekhar/food-delivery-time-predictor

An end-to-end Machine Learning regression application that predicts exact food delivery time (`Delivery_Time_min`) from delivery distance, traffic level, weather conditions, and restaurant preparation time — complete with a Full-7 feature benchmark, publication-quality diagnostic plots, CLI predictor, interactive Streamlit dashboard, and automated test suite.

## Project Overview

- **Target**: `Delivery_Time_min` (continuous regression, in minutes)
- **Core-4 Features**: `Distance_km`, `Traffic_Level`, `Weather`, `Preparation_Time_min`
- **Full-7 Benchmark Features**: Core-4 + `Time_of_Day`, `Vehicle_Type`, `Courier_Experience_yrs`
- **Dataset**: `Food_Delivery_Times.csv` (1,000 orders, 9 columns, 120 missing cells across 4 columns, 0 duplicate rows, 0 missing targets)
- **Split Strategy**: Strict 70% Train (700) / 15% Validation (150) / 15% Holdout Test (150) with `random_state=42` performed **before** any preprocessing to prevent data leakage.
- **Preprocessing (`sklearn.pipeline.Pipeline` + `ColumnTransformer`)**:
  - **Numeric features**: Median imputation (+ missingness indicator in Full mode) + `StandardScaler`
  - **Ordinal feature (`Traffic_Level`)**: Most-frequent imputation -> `OrdinalEncoder(["Low", "Medium", "High"])` -> `StandardScaler`
  - **Nominal features (`Weather`, and `Time_of_Day`, `Vehicle_Type` in Full mode)**: Constant `"Unknown"` imputation -> `OneHotEncoder(handle_unknown="ignore")`

## Architecture

```text
DeliveryTiming/
├── Food_Delivery_Times.csv       # Raw dataset (1,000 rows × 9 columns)
├── requirements.txt              # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading, EDA stats, 70/15/15 split, leakage-free ColumnTransformer
│   ├── train_evaluate.py         # 5-fold CV tuning, 8-model comparison, holdout test evaluation, CIs, slices
│   ├── predict.py                # Python API (`predict_delivery_time`) & CLI predictor
│   └── visualizations.py         # Generates 9 publication-quality EDA & evaluation PNG charts
├── app.py                        # 4-tab interactive Streamlit dashboard (single & batch predictions + analytics)
├── models/
│   ├── core_model.joblib         # Best Core-4 sklearn Pipeline
│   └── full_model.joblib         # Full-7 benchmark sklearn Pipeline
├── artifacts/
│   ├── eda_summary.json          # Structured dataset & EDA statistics
│   ├── metrics.json              # Full CV, train/val/test metrics, 95% CIs, permutation importance, slice errors
│   ├── predictions.csv           # Split-tagged actual vs predicted values & residuals
│   └── plots/                    # 9 diagnostic & EDA PNG plots (01_... to 09_...)
└── tests/
    └── test_pipeline.py          # Automated pytest suite (schema, leakage, monotonicity, models, artifacts)
```

## Model Results Summary

Eight regression models were compared on Core-4 features using 5-fold cross-validation on the training set (with `GridSearchCV` tuning) and selected by validation RMSE:

| Model (Core-4 Features) | 5-Fold CV RMSE | Train RMSE | Val MAE | Val RMSE | Val R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| **LinearRegression** *(Selected)* | 11.24 | 11.21 | **6.83** | **10.24** | **0.8207** |
| Ridge (`alpha=5.0`) | 11.23 | 11.21 | 6.89 | 10.27 | 0.8197 |
| Lasso (`alpha=0.05`) | 11.23 | 11.22 | 6.89 | 10.28 | 0.8193 |
| HistGradientBoostingRegressor | 11.91 | 10.31 | 7.00 | 10.44 | 0.8138 |
| GradientBoostingRegressor | 11.88 | 10.43 | 7.13 | 10.64 | 0.8066 |
| SVR (RBF) | 11.42 | 11.38 | 6.92 | 10.74 | 0.8029 |
| RandomForestRegressor | 12.20 | 9.54 | 7.27 | 10.76 | 0.8023 |
| DummyRegressor (Mean Baseline) | 21.96 | 21.90 | 19.86 | 24.22 | -0.0026 |

### Holdout Test Performance (Core-4 vs Full-7 Benchmark)

| Configuration | Model | Test MAE (95% CI) | Test RMSE (95% CI) | Test R² (95% CI) | Test MAPE |
| --- | --- | --- | --- | --- | ---: |
| **Core-4** | LinearRegression | **5.85 min** (5.01 – 6.94) | **8.41 min** (6.68 – 10.40) | **0.8296** (0.7449 – 0.8865) | 10.54% |
| **Full-7 Benchmark** | LinearRegression | **5.66 min** (4.81 – 6.81) | **8.34 min** (6.66 – 10.30) | **0.8324** (0.7463 – 0.8877) | 10.05% |

**Key Insights**:
- `Distance_km` is by far the strongest driver of delivery time (Pearson *r* = 0.781; permutation MAE increase +13.90 min), followed by `Preparation_Time_min` (+4.03 min), `Traffic_Level` (+1.56 min), and `Weather` (+0.86 min).
- Adding the 3 extra features (`Courier_Experience_yrs`, `Vehicle_Type`, `Time_of_Day`) improves test MAE by only `0.19 min` and test R² by `+0.0028`, confirming that the Core-4 features capture almost all predictive signal.
- `LinearRegression` generalizes cleanly with zero overfitting (Val RMSE 10.24 < Train RMSE 11.21; Test RMSE 8.41).

## Setup Instructions

Using `uv` (recommended) on Windows PowerShell:

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Or using standard `venv` + `pip`:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

### 1. Train & Evaluate Models

Runs EDA, 70/15/15 split, 5-fold CV tuning across 8 models, holdout test evaluation, and saves models and artifacts:

```powershell
.venv\Scripts\python.exe -m src.train_evaluate
```

### 2. Generate Visualizations

Creates all 9 EDA and model diagnostic PNG plots in `artifacts/plots/`:

```powershell
.venv\Scripts\python.exe src\visualizations.py
```

### 3. CLI Delivery Time Predictions

Core-4 mode:

```powershell
.venv\Scripts\python.exe src\predict.py --distance 10 --traffic Medium --weather Rainy --prep-time 15
```

Full-7 mode with JSON output:

```powershell
.venv\Scripts\python.exe src\predict.py --distance 10 --traffic High --weather Snowy --prep-time 20 --mode full --time-of-day Evening --vehicle-type Bike --experience 3 --json
```

### 4. Launch the Streamlit Web App

```powershell
.venv\Scripts\streamlit.exe run app.py
```

### 5. Run Automated Tests

```powershell
.venv\Scripts\python.exe -m pytest -v
```
