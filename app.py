"""Interactive Streamlit Dashboard for Food Delivery Time Prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    VALID_TIME_OF_DAY,
    VALID_TRAFFIC,
    VALID_VEHICLE_TYPE,
    VALID_WEATHER,
)
from src.predict import predict_delivery_time

DATA_PATH = PROJECT_ROOT / "Food_Delivery_Times.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
EDA_SUMMARY_PATH = ARTIFACTS_DIR / "eda_summary.json"

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #1D4ED8 100%);
        color: #F8FAFC;
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .hero-card h1 {
        margin: 0 0 0.35rem 0;
        font-size: 1.9rem;
        color: #FFFFFF;
    }
    .hero-card p {
        margin: 0;
        color: #CBD5E1;
        font-size: 0.98rem;
    }
    .prediction-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: #FFFFFF;
        padding: 1.35rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.25);
        margin-bottom: 1rem;
    }
    .prediction-banner .pred-value {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.2rem 0;
    }
    .prediction-banner .pred-sub {
        font-size: 0.92rem;
        color: #DBEAFE;
    }
    .insight-box {
        background-color: rgba(37, 99, 235, 0.08);
        border-left: 4px solid #2563EB;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_eda_summary() -> dict[str, Any]:
    with EDA_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _render_sidebar(metrics: dict[str, Any], eda: dict[str, Any]) -> None:
    best = metrics["best_core_model"]
    full = metrics["full_benchmark_model"]
    with st.sidebar:
        st.title("🛵 DeliveryTiming ML")
        st.caption("End-to-End Regression Dashboard")
        st.divider()
        st.subheader("Production Model Summary")
        st.markdown(
            f"- **Best Core-4**: `{best['model_name']}`\n"
            f"- **Test MAE**: `{best['test']['mae']:.2f}` min\n"
            f"- **Test RMSE**: `{best['test']['rmse']:.2f}` min\n"
            f"- **Test R²**: `{best['test']['r2']:.4f}`\n"
            f"- **Full-7 Test MAE**: `{full['test']['mae']:.2f}` min"
        )
        st.divider()
        st.subheader("Dataset Snapshot")
        splits = metrics["dataset"]["splits"]
        st.markdown(
            f"- **Total Orders**: `{eda['dataset_shape']['rows']:,}`\n"
            f"- **Split**: `{splits['train']}` Train / `{splits['val']}` Val / `{splits['test']}` Test\n"
            f"- **Leakage-Free**: `Pipeline` + `ColumnTransformer` fit strictly on Train\n"
            f"- **Missing Values**: `{eda['total_missing_cells']}` cells imputed in-pipeline"
        )


def _render_prediction_tab(df: pd.DataFrame, metrics: dict[str, Any], eda: dict[str, Any]) -> None:
    st.subheader("Live Delivery Time Estimator")
    st.caption(
        "Configure order parameters below to predict exact delivery time in minutes with a 90% empirical prediction interval."
    )

    col_inputs, col_output = st.columns([1.05, 1.35], gap="large")

    with col_inputs:
        full_mode = st.toggle(
            "Enable Full-Feature Mode (7 features)",
            value=False,
            help="Includes Time of Day, Vehicle Type, and Courier Experience alongside the Core-4 features.",
        )

        distance_km = st.slider(
            "Delivery Distance (km)",
            min_value=0.5,
            max_value=25.0,
            value=10.0,
            step=0.1,
        )
        prep_time_min = st.slider(
            "Restaurant Preparation Time (min)",
            min_value=5,
            max_value=45,
            value=17,
            step=1,
        )

        c1, c2 = st.columns(2)
        with c1:
            traffic_level = st.selectbox("Traffic Level", options=VALID_TRAFFIC, index=1)
        with c2:
            weather = st.selectbox("Weather Condition", options=VALID_WEATHER, index=0)

        time_of_day = None
        vehicle_type = None
        courier_experience_yrs = None

        if full_mode:
            st.markdown("**Additional Context (Full-7 Benchmark)**")
            c3, c4 = st.columns(2)
            with c3:
                time_of_day = st.selectbox("Time of Day", options=VALID_TIME_OF_DAY, index=2)
            with c4:
                vehicle_type = st.selectbox("Vehicle Type", options=VALID_VEHICLE_TYPE, index=0)
            courier_experience_yrs = st.slider(
                "Courier Experience (years)",
                min_value=0.0,
                max_value=15.0,
                value=4.0,
                step=0.5,
            )

    mode = "full" if full_mode else "core"
    result = predict_delivery_time(
        distance_km=distance_km,
        traffic_level=traffic_level,
        weather=weather,
        preparation_time_min=prep_time_min,
        time_of_day=time_of_day,
        vehicle_type=vehicle_type,
        courier_experience_yrs=courier_experience_yrs,
        mode=mode,
    )

    pred_min = result["predicted_minutes"]
    low_min = result["lower_bound"]
    high_min = result["upper_bound"]
    dataset_mean = float(eda["numeric_summary"]["Delivery_Time_min"]["mean"])
    delta_vs_avg = pred_min - dataset_mean

    with col_output:
        st.markdown(
            f"""
            <div class="prediction-banner">
                <div class="pred-sub">Predicted Delivery Time ({result['mode'].upper()} · {result['model_name']})</div>
                <div class="pred-value">{pred_min:.1f} min</div>
                <div class="pred-sub">Estimated 90% Range: <strong>{low_min:.1f} – {high_min:.1f} min</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Lower Bound (90% PI)", f"{low_min:.1f} min")
        m2.metric("Upper Bound (90% PI)", f"{high_min:.1f} min")
        m3.metric(
            "vs Dataset Average",
            f"{pred_min:.1f} min",
            delta=f"{delta_vs_avg:+.1f} min",
            delta_color="inverse",
        )

        # Plotly Gauge Chart
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pred_min,
                delta={"reference": dataset_mean, "suffix": " min", "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
                title={"text": "Predicted ETA vs Dataset Mean (56.7 min)", "font": {"size": 14}},
                number={"suffix": " min", "font": {"size": 30}},
                gauge={
                    "axis": {"range": [0, 140], "tickwidth": 1},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(16, 185, 129, 0.25)"},
                        {"range": [40, 75], "color": "rgba(245, 158, 11, 0.25)"},
                        {"range": [75, 140], "color": "rgba(239, 68, 68, 0.25)"},
                    ],
                    "threshold": {
                        "line": {"color": "#0F172A", "width": 3},
                        "thickness": 0.75,
                        "value": dataset_mean,
                    },
                },
            )
        )
        gauge_fig.update_layout(height=270, margin=dict(l=20, r=20, t=45, b=10))
        st.plotly_chart(gauge_fig, width="stretch")

        # Comparison Bar Chart against Dataset Averages
        weather_avg = float(
            eda.get("target_by_weather", {}).get(weather, {}).get("mean", dataset_mean)
        )
        traffic_avg = float(
            eda.get("target_by_traffic", {}).get(traffic_level, {}).get("mean", dataset_mean)
        )
        comp_df = pd.DataFrame(
            {
                "Benchmark": [
                    "This Prediction",
                    f"{weather} Weather Avg",
                    f"{traffic_level} Traffic Avg",
                    "Overall Dataset Avg",
                ],
                "Minutes": [pred_min, weather_avg, traffic_avg, dataset_mean],
                "Color": ["#2563EB", "#0EA5E9", "#F59E0B", "#64748B"],
            }
        )
        bar_fig = px.bar(
            comp_df,
            x="Minutes",
            y="Benchmark",
            orientation="h",
            text=comp_df["Minutes"].map(lambda v: f"{v:.1f} min"),
            color="Benchmark",
            color_discrete_map=dict(zip(comp_df["Benchmark"], comp_df["Color"])),
            title="Prediction vs Historical Benchmarks",
        )
        bar_fig.update_traces(textposition="outside", showlegend=False)
        bar_fig.update_layout(
            height=240,
            margin=dict(l=10, r=40, t=40, b=10),
            xaxis_title="Delivery Time (min)",
            yaxis_title="",
            xaxis_range=[0, max(comp_df["Minutes"].max() * 1.22, 80)],
        )
        st.plotly_chart(bar_fig, width="stretch")

    st.divider()
    st.subheader("Batch CSV Prediction")
    st.caption(
        "Upload a CSV containing `Distance_km`, `Traffic_Level`, `Weather`, and `Preparation_Time_min` "
        "(plus optional `Time_of_Day`, `Vehicle_Type`, `Courier_Experience_yrs` for Full mode)."
    )

    sample_template = df[
        [
            "Distance_km",
            "Traffic_Level",
            "Weather",
            "Preparation_Time_min",
            "Time_of_Day",
            "Vehicle_Type",
            "Courier_Experience_yrs",
        ]
    ].head(5)

    dl_col, up_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            label="Download Sample CSV Template",
            data=sample_template.to_csv(index=False).encode("utf-8"),
            file_name="sample_delivery_orders.csv",
            mime="text/csv",
        )
    with up_col:
        uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = {"Distance_km", "Traffic_Level", "Weather", "Preparation_Time_min"}
            missing_cols = required_cols - set(batch_df.columns)
            if missing_cols:
                st.error(f"Uploaded CSV is missing required columns: {sorted(missing_cols)}")
            else:
                batch_preds = []
                for _, row in batch_df.iterrows():
                    res = predict_delivery_time(
                        distance_km=float(row["Distance_km"]),
                        traffic_level=None if pd.isna(row["Traffic_Level"]) else str(row["Traffic_Level"]),
                        weather=None if pd.isna(row["Weather"]) else str(row["Weather"]),
                        preparation_time_min=float(row["Preparation_Time_min"]),
                        time_of_day=(
                            None
                            if "Time_of_Day" not in batch_df.columns or pd.isna(row["Time_of_Day"])
                            else str(row["Time_of_Day"])
                        ),
                        vehicle_type=(
                            None
                            if "Vehicle_Type" not in batch_df.columns or pd.isna(row["Vehicle_Type"])
                            else str(row["Vehicle_Type"])
                        ),
                        courier_experience_yrs=(
                            None
                            if "Courier_Experience_yrs" not in batch_df.columns or pd.isna(row["Courier_Experience_yrs"])
                            else float(row["Courier_Experience_yrs"])
                        ),
                        mode=mode,
                    )
                    batch_preds.append(res)
                pred_out = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(batch_preds)], axis=1)
                st.success(f"Generated {len(pred_out):,} predictions using `{mode}` mode.")
                st.dataframe(pred_out, width="stretch")
                st.download_button(
                    label="Download Predictions CSV",
                    data=pred_out.to_csv(index=False).encode("utf-8"),
                    file_name=f"delivery_predictions_{mode}.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not process uploaded CSV: {exc}")


def _render_eda_tab(df: pd.DataFrame, eda: dict[str, Any]) -> None:
    st.subheader("Dataset Overview & Exploratory Data Analysis")

    target_stats = eda["numeric_summary"]["Delivery_Time_min"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Orders", f"{eda['dataset_shape']['rows']:,}")
    k2.metric("Mean Delivery Time", f"{target_stats['mean']:.1f} min")
    k3.metric("Median Delivery Time", f"{target_stats['median']:.1f} min")
    k4.metric("Missing Cells", f"{eda['total_missing_cells']} ({eda['rows_with_any_missing']} rows)")
    k5.metric("IQR Target Outliers", f"{eda['anomalies']['target_outlier_count']}")

    with st.expander("Preview Raw Dataset (`Food_Delivery_Times.csv`)", expanded=False):
        n_rows = st.slider("Rows to display", min_value=10, max_value=200, value=25, step=5)
        st.dataframe(df.head(n_rows), width="stretch")

    # Missing Values Analysis
    st.markdown("### Missing Values Analysis")
    miss_rows = [
        {"Column": col, "Missing Count": info["count"], "Missing (%)": info["pct"]}
        for col, info in eda["missing_values"].items()
    ]
    miss_df = pd.DataFrame(miss_rows).sort_values("Missing Count", ascending=False)
    mc1, mc2 = st.columns([1, 1.5])
    with mc1:
        st.dataframe(miss_df, width="stretch", hide_index=True)
    with mc2:
        miss_fig = px.bar(
            miss_df[miss_df["Missing Count"] > 0],
            x="Column",
            y="Missing (%)",
            text="Missing Count",
            color="Missing (%)",
            color_continuous_scale="Reds",
            title="Columns with Missing Values (Imputed Strictly Inside Pipeline)",
        )
        miss_fig.update_traces(texttemplate="%{text} rows", textposition="outside")
        miss_fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False)
        st.plotly_chart(miss_fig, width="stretch")

    # Interactive EDA Plotly Charts
    st.markdown("### Interactive Feature Relationships")
    plot_df = df.copy()
    plot_df["Traffic_Level_Clean"] = plot_df["Traffic_Level"].fillna("Missing")
    plot_df["Weather_Clean"] = plot_df["Weather"].fillna("Missing")

    e1, e2 = st.columns(2)
    with e1:
        scatter_fig = px.scatter(
            plot_df,
            x="Distance_km",
            y="Delivery_Time_min",
            color="Traffic_Level_Clean",
            color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444", "Missing": "#94A3B8"},
            hover_data=["Order_ID", "Weather", "Preparation_Time_min", "Vehicle_Type"],
            opacity=0.72,
            title="Distance vs Delivery Time (by Traffic Level)",
        )
        scatter_fig.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(scatter_fig, width="stretch")

    with e2:
        box_weather = px.box(
            df.dropna(subset=["Weather"]),
            x="Weather",
            y="Delivery_Time_min",
            color="Weather",
            title="Delivery Time Distribution by Weather",
        )
        box_weather.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(box_weather, width="stretch")

    e3, e4 = st.columns(2)
    with e3:
        box_traffic = px.box(
            df.dropna(subset=["Traffic_Level"]),
            x="Traffic_Level",
            y="Delivery_Time_min",
            category_orders={"Traffic_Level": VALID_TRAFFIC},
            color="Traffic_Level",
            color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"},
            title="Delivery Time Distribution by Traffic Level",
        )
        box_traffic.update_layout(height=390, showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(box_traffic, width="stretch")

    with e4:
        corr_df = df[["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs", "Delivery_Time_min"]].copy()
        corr_df["Traffic_Level_Ord"] = df["Traffic_Level"].map({"Low": 0, "Medium": 1, "High": 2})
        corr_mat = corr_df.corr().round(2)
        heat_fig = px.imshow(
            corr_mat,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Matrix (Numerical + Ordinal Traffic)",
        )
        heat_fig.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(heat_fig, width="stretch")

    # Static Publication-Quality EDA Plots
    with st.expander("Static Publication-Quality EDA Plots (`artifacts/plots/`)"):
        static_eda = [
            ("01_target_distribution.png", "Target Distribution & Outliers"),
            ("02_missing_values.png", "Missing Values Profile"),
            ("03_feature_relationships.png", "Feature Relationships"),
            ("04_correlation_heatmap.png", "Correlation Heatmap"),
        ]
        for row_idx in range(0, len(static_eda), 2):
            cols = st.columns(2)
            for col_idx, (fname, caption) in enumerate(static_eda[row_idx : row_idx + 2]):
                fpath = PLOTS_DIR / fname
                if fpath.exists():
                    cols[col_idx].image(str(fpath), caption=caption, width="stretch")


def _render_evaluation_tab(preds: pd.DataFrame, metrics: dict[str, Any]) -> None:
    st.subheader("Model Evaluation, Comparison & Diagnostics")

    best = metrics["best_core_model"]
    full = metrics["full_benchmark_model"]
    delta = metrics["core_vs_full_test_delta"]
    ci = best["test_bootstrap_95ci"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Core-4 Test MAE",
        f"{best['test']['mae']:.2f} min",
        help=f"95% Bootstrap CI: [{ci['mae']['lower']:.2f}, {ci['mae']['upper']:.2f}] min",
    )
    c2.metric(
        "Core-4 Test RMSE",
        f"{best['test']['rmse']:.2f} min",
        help=f"95% Bootstrap CI: [{ci['rmse']['lower']:.2f}, {ci['rmse']['upper']:.2f}] min",
    )
    c3.metric(
        "Core-4 Test R²",
        f"{best['test']['r2']:.4f}",
        help=f"95% Bootstrap CI: [{ci['r2']['lower']:.4f}, {ci['r2']['upper']:.4f}]",
    )
    c4.metric(
        "Full-7 vs Core-4 Test MAE",
        f"{full['test']['mae']:.2f} min",
        delta=f"{delta['mae_diff_full_minus_core']:+.2f} min",
        delta_color="inverse",
    )

    st.caption(
        f"**95% Bootstrap CIs (1,000 resamples, Holdout Test n=150)** — "
        f"MAE: `[{ci['mae']['lower']:.2f}, {ci['mae']['upper']:.2f}]` min | "
        f"RMSE: `[{ci['rmse']['lower']:.2f}, {ci['rmse']['upper']:.2f}]` min | "
        f"R²: `[{ci['r2']['lower']:.4f}, {ci['r2']['upper']:.4f}]`"
    )

    # Full Comparison Table
    st.markdown("### Model Comparison Table (8 Core-4 Candidates + Full-7 Benchmark)")
    table_rows = []
    for item in metrics["core_model_comparison"]:
        is_best = item["model_name"] == best["model_name"]
        table_rows.append(
            {
                "Model": f"{item['model_name']} (Best)" if is_best else item["model_name"],
                "Feature Set": "Core-4",
                "CV RMSE (5-fold)": f"{item['cv_5fold']['rmse_mean']:.2f} ± {item['cv_5fold']['rmse_std']:.2f}",
                "Train MAE": round(item["train"]["mae"], 2),
                "Val MAE": round(item["val"]["mae"], 2),
                "Val RMSE": round(item["val"]["rmse"], 2),
                "Val R²": round(item["val"]["r2"], 4),
                "Val MAPE (%)": round(item["val"]["mape_pct"], 2),
                "Test MAE": round(best["test"]["mae"], 2) if is_best else np.nan,
                "Test RMSE": round(best["test"]["rmse"], 2) if is_best else np.nan,
                "Test R²": round(best["test"]["r2"], 4) if is_best else np.nan,
            }
        )
    table_rows.append(
        {
            "Model": f"{full['model_name']} (Benchmark)",
            "Feature Set": "Full-7",
            "CV RMSE (5-fold)": f"{full['cv_5fold']['rmse_mean']:.2f} ± {full['cv_5fold']['rmse_std']:.2f}",
            "Train MAE": round(full["train"]["mae"], 2),
            "Val MAE": round(full["val"]["mae"], 2),
            "Val RMSE": round(full["val"]["rmse"], 2),
            "Val R²": round(full["val"]["r2"], 4),
            "Val MAPE (%)": round(full["val"]["mape_pct"], 2),
            "Test MAE": round(full["test"]["mae"], 2),
            "Test RMSE": round(full["test"]["rmse"], 2),
            "Test R²": round(full["test"]["r2"], 4),
        }
    )
    comp_table_df = pd.DataFrame(table_rows).sort_values("Val RMSE", ascending=True)
    st.dataframe(comp_table_df, width="stretch", hide_index=True)

    # Interactive Actual vs Predicted
    st.markdown("### Actual vs Predicted Delivery Time (Interactive)")
    split_filter = st.multiselect(
        "Filter Splits",
        options=["train", "val", "test"],
        default=["val", "test"],
    )
    filtered_preds = preds[preds["split"].isin(split_filter)] if split_filter else preds
    avp_fig = px.scatter(
        filtered_preds,
        x="actual",
        y="predicted_core",
        color="split",
        color_discrete_map={"train": "#94A3B8", "val": "#0EA5E9", "test": "#2563EB"},
        hover_data=["Order_ID", "Distance_km", "Traffic_Level", "Weather", "Preparation_Time_min", "residual_core"],
        opacity=0.78,
        title="Actual vs Predicted Delivery Time (Core-4 LinearRegression)",
    )
    lo = float(min(filtered_preds["actual"].min(), filtered_preds["predicted_core"].min())) - 3
    hi = float(max(filtered_preds["actual"].max(), filtered_preds["predicted_core"].max())) + 3
    avp_fig.add_shape(
        type="line",
        x0=lo,
        y0=lo,
        x1=hi,
        y1=hi,
        line=dict(color="#EF4444", width=2, dash="dash"),
    )
    avp_fig.update_layout(
        height=430,
        xaxis_title="Actual Delivery Time (min)",
        yaxis_title="Predicted Delivery Time (min)",
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(avp_fig, width="stretch")

    # Residual Diagnostics
    st.markdown("### Residual Diagnostics (Holdout Test Set)")
    test_preds = preds[preds["split"] == "test"]
    r1, r2 = st.columns(2)
    with r1:
        res_scatter = px.scatter(
            test_preds,
            x="predicted_core",
            y="residual_core",
            color="Traffic_Level",
            hover_data=["Order_ID", "actual", "Distance_km", "Weather"],
            title="Test Residuals vs Predicted",
        )
        res_scatter.add_hline(y=0, line_dash="dash", line_color="#EF4444")
        res_scatter.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(res_scatter, width="stretch")
    with r2:
        res_hist = px.histogram(
            test_preds,
            x="residual_core",
            nbins=24,
            marginal="box",
            color_discrete_sequence=["#0EA5E9"],
            title="Test Residual Distribution (Actual − Predicted)",
        )
        res_hist.add_vline(x=0, line_dash="dash", line_color="#EF4444")
        res_hist.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(res_hist, width="stretch")

    # Slice-Based Error Analysis
    st.markdown("### Slice-Based Error Analysis (Holdout Test Set)")
    s1, s2 = st.columns(2)
    for col_widget, slice_key in [(s1, "Weather"), (s2, "Traffic_Level")]:
        slice_dict = best["slice_errors_test"][slice_key]
        sdf = (
            pd.DataFrame.from_dict(slice_dict, orient="index")
            .reset_index()
            .rename(columns={"index": "Slice"})
            .sort_values("mae", ascending=True)
        )
        sdf["Label"] = sdf.apply(lambda r: f"{r['Slice']} (n={int(r['count'])})", axis=1)
        fig = px.bar(
            sdf,
            x="mae",
            y="Label",
            orientation="h",
            text=sdf["mae"].map(lambda v: f"{v:.2f} min"),
            color="mae",
            color_continuous_scale="Blues",
            title=f"Test MAE by {slice_key.replace('_', ' ')}",
        )
        fig.add_vline(
            x=best["test"]["mae"],
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Overall ({best['test']['mae']:.2f})",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=340,
            coloraxis_showscale=False,
            xaxis_title="Test MAE (min)",
            yaxis_title="",
            xaxis_range=[0, sdf["mae"].max() * 1.25],
            margin=dict(l=10, r=25, t=45, b=10),
        )
        col_widget.plotly_chart(fig, width="stretch")

    with st.expander("Static Publication-Quality Evaluation Plots (`artifacts/plots/`)"):
        eval_plots = [
            ("05_model_comparison.png", "Model Comparison"),
            ("06_actual_vs_predicted.png", "Actual vs Predicted"),
            ("07_residual_analysis.png", "Residual Diagnostics & Q-Q Plot"),
            ("09_slice_error_analysis.png", "Slice-Based Error Analysis"),
        ]
        for row_idx in range(0, len(eval_plots), 2):
            cols = st.columns(2)
            for col_idx, (fname, caption) in enumerate(eval_plots[row_idx : row_idx + 2]):
                fpath = PLOTS_DIR / fname
                if fpath.exists():
                    cols[col_idx].image(str(fpath), caption=caption, width="stretch")


def _render_insights_tab(metrics: dict[str, Any]) -> None:
    st.subheader("Permutation Feature Importance & Data-Driven Insights")

    best = metrics["best_core_model"]
    full = metrics["full_benchmark_model"]
    delta = metrics["core_vs_full_test_delta"]

    imp_mode = st.radio(
        "Inspect Permutation Importance for:",
        options=["Core-4 Production Model", "Full-7 Benchmark Model"],
        horizontal=True,
    )
    imp_list = (
        best["permutation_importance_test"]
        if imp_mode == "Core-4 Production Model"
        else full["permutation_importance_test"]
    )
    imp_df = pd.DataFrame(imp_list).sort_values("mae_increase_mean", ascending=True)

    imp_fig = go.Figure(
        go.Bar(
            x=imp_df["mae_increase_mean"],
            y=imp_df["feature"],
            orientation="h",
            error_x=dict(type="data", array=imp_df["mae_increase_std"], visible=True),
            marker_color="#2563EB" if "Core-4" in imp_mode else "#10B981",
            text=imp_df["mae_increase_mean"].map(lambda v: f"+{v:.2f} min"),
            textposition="outside",
        )
    )
    imp_fig.update_layout(
        title=f"{imp_mode} — Permutation Importance (Holdout Test Set)",
        xaxis_title="Increase in Test MAE When Feature Is Shuffled (minutes)",
        yaxis_title="Feature",
        height=390,
        xaxis_range=[min(imp_df["mae_increase_mean"].min() - 0.5, -0.4), (imp_df["mae_increase_mean"] + imp_df["mae_increase_std"]).max() * 1.22],
        margin=dict(l=10, r=40, t=50, b=10),
    )
    st.plotly_chart(imp_fig, width="stretch")

    st.markdown("### Key Takeaways & Story")
    st.markdown(
        f"""
        <div class="insight-box">
            <strong>1. Distance Dominates Delivery Time (+13.90 min MAE impact):</strong>
            <code>Distance_km</code> has a strong linear correlation with delivery duration (Pearson r = 0.78) and is by far the most influential predictor. Every additional kilometer directly scales travel time.
        </div>
        <div class="insight-box">
            <strong>2. Restaurant Preparation Time Is #2 (+4.03 min MAE impact):</strong>
            <code>Preparation_Time_min</code> adds directly to total fulfillment time (r = 0.31). Reducing kitchen bottlenecks offers the clearest operational lever for faster deliveries.
        </div>
        <div class="insight-box">
            <strong>3. Traffic & Weather Drive Contextual Delays (+1.56 & +0.86 min MAE impact):</strong>
            High traffic increases average delivery time substantially over Low traffic, while Snowy and Rainy conditions add consistent weather penalties. High-traffic slices also exhibit the widest residual variance (RMSE 11.86 min).
        </div>
        <div class="insight-box">
            <strong>4. Core-4 Simplicity Wins in Production:</strong>
            Expanding from the 4 core features to all 7 features (adding <code>Time_of_Day</code>, <code>Vehicle_Type</code>, and <code>Courier_Experience_yrs</code>) improves holdout test MAE by only <strong>{abs(delta['mae_diff_full_minus_core']):.2f} min</strong> and R² by <strong>{delta['r2_diff_full_minus_core']:+.4f}</strong>. Simple linear regression outperforms tree ensembles and SVR because the underlying delivery process is predominantly additive.
        </div>
        """,
        unsafe_allow_html=True,
    )

    static_imp = PLOTS_DIR / "08_feature_importance.png"
    if static_imp.exists():
        with st.expander("View Static Feature Importance Plot (`08_feature_importance.png`)"):
            st.image(str(static_imp), caption="Permutation Feature Importance (Core-4 vs Full-7)", width="stretch")


def main() -> None:
    df = load_dataset()
    preds = load_predictions()
    metrics = load_metrics()
    eda = load_eda_summary()

    _render_sidebar(metrics, eda)

    st.markdown(
        """
        <div class="hero-card">
            <h1>🛵 Food Delivery Time Prediction Dashboard</h1>
            <p>Predict exact delivery times in minutes using distance, traffic, weather, and restaurant preparation time — backed by leakage-free ML pipelines, bootstrap confidence intervals, and slice diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🎯 Predict Delivery Time",
            "📊 Data & EDA",
            "📈 Model Evaluation & Comparison",
            "💡 Feature Importance & Insights",
        ]
    )

    with tab1:
        _render_prediction_tab(df, metrics, eda)
    with tab2:
        _render_eda_tab(df, eda)
    with tab3:
        _render_evaluation_tab(preds, metrics)
    with tab4:
        _render_insights_tab(metrics)


if __name__ == "__main__":
    main()
