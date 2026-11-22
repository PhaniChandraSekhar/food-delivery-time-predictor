---
title: Food Delivery Time Predictor
emoji: 🛵
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
---

# Food Delivery Time Prediction (ML Regression)

Interactive Machine Learning regression dashboard predicting exact food delivery time (`Delivery_Time_min`) from distance, traffic level, weather, and restaurant preparation time (plus an optional Full-7 feature benchmark).

- **Core-4 Production Model (`LinearRegression`)**: Holdout Test MAE **5.85 min** (95% CI: 5.01–6.94), RMSE **8.41 min**, R² **0.8296**, MAPE **10.54%**.
- **Full-7 Benchmark (`LinearRegression`)**: Holdout Test MAE **5.66 min**, RMSE **8.34 min**, R² **0.8324**.
- Trained with a strict leak-free 70/15/15 Train/Validation/Test split (`scikit-learn` `Pipeline` + `ColumnTransformer`) and exported with exact model parity for instant client-side inference.
