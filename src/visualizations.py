"""Generate publication-quality ML and EDA visualizations for Food Delivery Time Prediction.

Saves 9 PNG figures into ``artifacts/plots/``:
1. 01_target_distribution.png
2. 02_missing_values.png
3. 03_feature_relationships.png
4. 04_correlation_heatmap.png
5. 05_model_comparison.png
6. 06_actual_vs_predicted.png
7. 07_residual_analysis.png
8. 08_feature_importance.png
9. 09_slice_error_analysis.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Food_Delivery_Times.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
EDA_SUMMARY_PATH = ARTIFACTS_DIR / "eda_summary.json"

PALETTE_PRIMARY = "#2563EB"
PALETTE_SECONDARY = "#0EA5E9"
PALETTE_ACCENT = "#F59E0B"
PALETTE_DANGER = "#EF4444"
PALETTE_SUCCESS = "#10B981"
PALETTE_SLATE = "#64748B"

TRAFFIC_ORDER = ["Low", "Medium", "High"]
TRAFFIC_COLORS = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
SPLIT_COLORS = {"train": "#94A3B8", "val": "#0EA5E9", "test": "#2563EB"}


def _configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "font.family": "sans-serif",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#E2E8F0",
        }
    )


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    df = pd.read_csv(DATA_PATH)
    preds = pd.read_csv(PREDICTIONS_PATH)
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    with EDA_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        eda = json.load(f)
    return df, preds, metrics, eda


def plot_01_target_distribution(df: pd.DataFrame, eda: dict[str, Any]) -> Path:
    """Histogram + KDE + Boxplot of Delivery_Time_min."""
    target = df["Delivery_Time_min"].dropna()
    mean_val = float(target.mean())
    median_val = float(target.median())
    upper_fence = float(eda.get("anomalies", {}).get("target_iqr_upper_fence", 116.0))
    n_outliers = int(eda.get("anomalies", {}).get("target_outlier_count", (target > upper_fence).sum()))

    fig, (ax_box, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(10, 6.2),
        gridspec_kw={"height_ratios": [1, 3.6]},
        sharex=True,
    )

    sns.boxplot(
        x=target,
        ax=ax_box,
        color=PALETTE_SECONDARY,
        width=0.5,
        flierprops={"marker": "o", "markerfacecolor": PALETTE_DANGER, "markersize": 5, "alpha": 0.8},
    )
    ax_box.axvline(upper_fence, color=PALETTE_DANGER, linestyle=":", linewidth=1.5)
    ax_box.set(xlabel="", yticks=[])
    ax_box.set_title("Delivery Time Distribution & Outlier Diagnostics (n = 1,000)", pad=10)

    sns.histplot(
        target,
        bins=30,
        kde=True,
        ax=ax_hist,
        color=PALETTE_PRIMARY,
        alpha=0.55,
        edgecolor="white",
        line_kws={"linewidth": 2.2},
    )
    ax_hist.axvline(mean_val, color=PALETTE_ACCENT, linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f} min")
    ax_hist.axvline(median_val, color=PALETTE_SUCCESS, linestyle="-", linewidth=2, label=f"Median: {median_val:.1f} min")
    ax_hist.axvline(
        upper_fence,
        color=PALETTE_DANGER,
        linestyle=":",
        linewidth=1.8,
        label=f"IQR Upper Fence: {upper_fence:.0f} min ({n_outliers} outliers)",
    )
    ax_hist.set_xlabel("Delivery Time (minutes)")
    ax_hist.set_ylabel("Order Count")
    ax_hist.legend(loc="upper right")

    stats_text = (
        f"Std: {target.std():.1f} min\n"
        f"Range: {target.min():.0f}–{target.max():.0f} min\n"
        f"Skewness: {target.skew():.2f}"
    )
    ax_hist.text(
        0.98,
        0.62,
        stats_text,
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
    )

    fig.tight_layout()
    return _save_fig(fig, "01_target_distribution.png")


def plot_02_missing_values(df: pd.DataFrame, eda: dict[str, Any]) -> Path:
    """Missing value counts and percentages across columns."""
    missing_info = eda.get("missing_values", {})
    rows = []
    for col in df.columns:
        info = missing_info.get(col, {})
        cnt = int(info.get("count", df[col].isna().sum()))
        pct = float(info.get("pct", 100.0 * cnt / len(df)))
        rows.append({"column": col, "count": cnt, "pct": pct})
    miss_df = pd.DataFrame(rows).sort_values(["count", "column"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = [PALETTE_DANGER if c > 0 else "#CBD5E1" for c in miss_df["count"]]
    bars = ax.barh(miss_df["column"], miss_df["pct"], color=colors, edgecolor="white", height=0.62)

    for bar, cnt, pct in zip(bars, miss_df["count"], miss_df["pct"]):
        label = f" {cnt} rows ({pct:.1f}%)" if cnt > 0 else " 0 (Complete)"
        ax.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold" if cnt > 0 else "normal",
            color="#1E293B" if cnt > 0 else PALETTE_SLATE,
        )

    max_pct = max(miss_df["pct"].max(), 1.0)
    ax.set_xlim(0, max_pct * 1.42)
    ax.set_xlabel("Missing Values (% of 1,000 rows)")
    ax.set_ylabel("Dataset Column")
    total_cells = int(eda.get("total_missing_cells", df.isna().sum().sum()))
    rows_any = int(eda.get("rows_with_any_missing", df.isna().any(axis=1).sum()))
    ax.set_title(
        f"Missing Value Profile ({total_cells} missing cells across {rows_any} rows; target 100% complete)",
        pad=12,
    )
    fig.tight_layout()
    return _save_fig(fig, "02_missing_values.png")


def plot_03_feature_relationships(df: pd.DataFrame) -> Path:
    """2x2 multi-panel feature relationship plots."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))

    # Panel 1: Distance vs Delivery Time colored by Traffic Level
    ax = axes[0, 0]
    plot_df = df.dropna(subset=["Distance_km", "Delivery_Time_min"]).copy()
    plot_df["Traffic_Plot"] = plot_df["Traffic_Level"].fillna("Missing")
    hue_order = ["Low", "Medium", "High", "Missing"]
    palette = {**TRAFFIC_COLORS, "Missing": "#94A3B8"}
    sns.scatterplot(
        data=plot_df,
        x="Distance_km",
        y="Delivery_Time_min",
        hue="Traffic_Plot",
        hue_order=hue_order,
        palette=palette,
        alpha=0.7,
        s=36,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    sns.regplot(
        data=plot_df,
        x="Distance_km",
        y="Delivery_Time_min",
        scatter=False,
        color="#0F172A",
        line_kws={"linewidth": 2, "linestyle": "--"},
        ax=ax,
    )
    r_dist = plot_df["Distance_km"].corr(plot_df["Delivery_Time_min"])
    ax.set_title(f"Distance vs Delivery Time (r = {r_dist:.2f})")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery Time (min)")
    ax.legend(title="Traffic Level", loc="upper left", fontsize=8.5, title_fontsize=9)

    # Panel 2: Preparation Time vs Delivery Time
    ax = axes[0, 1]
    sns.regplot(
        data=df,
        x="Preparation_Time_min",
        y="Delivery_Time_min",
        scatter_kws={"alpha": 0.55, "s": 32, "color": PALETTE_SECONDARY, "edgecolor": "white"},
        line_kws={"color": PALETTE_DANGER, "linewidth": 2.2},
        ax=ax,
    )
    r_prep = df["Preparation_Time_min"].corr(df["Delivery_Time_min"])
    ax.set_title(f"Restaurant Prep Time vs Delivery Time (r = {r_prep:.2f})")
    ax.set_xlabel("Preparation Time (min)")
    ax.set_ylabel("Delivery Time (min)")

    # Panel 3: Delivery Time by Weather
    ax = axes[1, 0]
    weather_df = df.dropna(subset=["Weather", "Delivery_Time_min"])
    weather_order = (
        weather_df.groupby("Weather")["Delivery_Time_min"]
        .median()
        .sort_values()
        .index.tolist()
    )
    sns.boxplot(
        data=weather_df,
        x="Weather",
        y="Delivery_Time_min",
        order=weather_order,
        hue="Weather",
        palette="Blues",
        legend=False,
        width=0.58,
        ax=ax,
    )
    ax.set_title("Delivery Time by Weather Condition (Sorted by Median)")
    ax.set_xlabel("Weather")
    ax.set_ylabel("Delivery Time (min)")

    # Panel 4: Delivery Time by Traffic Level
    ax = axes[1, 1]
    traffic_df = df.dropna(subset=["Traffic_Level", "Delivery_Time_min"])
    sns.boxplot(
        data=traffic_df,
        x="Traffic_Level",
        y="Delivery_Time_min",
        order=TRAFFIC_ORDER,
        hue="Traffic_Level",
        palette=TRAFFIC_COLORS,
        legend=False,
        width=0.52,
        ax=ax,
    )
    ax.set_title("Delivery Time by Traffic Level")
    ax.set_xlabel("Traffic Level")
    ax.set_ylabel("Delivery Time (min)")

    fig.suptitle("Key Feature Relationships with Food Delivery Time", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "03_feature_relationships.png")


def plot_04_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Correlation heatmap of numerical features + ordinal-encoded Traffic_Level."""
    corr_df = df[
        ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs", "Delivery_Time_min"]
    ].copy()
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    corr_df["Traffic_Level (Ord)"] = df["Traffic_Level"].map(traffic_map)
    corr_df = corr_df.rename(
        columns={
            "Distance_km": "Distance (km)",
            "Preparation_Time_min": "Prep Time (min)",
            "Courier_Experience_yrs": "Courier Exp (yrs)",
            "Delivery_Time_min": "Delivery Time (min)",
        }
    )
    cols_ordered = [
        "Delivery Time (min)",
        "Distance (km)",
        "Prep Time (min)",
        "Traffic_Level (Ord)",
        "Courier Exp (yrs)",
    ]
    corr_matrix = corr_df[cols_ordered].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.82, "label": "Pearson Correlation (r)"},
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Numerical + Ordinal Traffic)", pad=14)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return _save_fig(fig, "04_correlation_heatmap.png")


def plot_05_model_comparison(metrics: dict[str, Any]) -> Path:
    """Grouped bar chart comparing Validation/Test MAE, RMSE, and R² across all models."""
    rows = []
    best_core_name = metrics["best_core_model"]["model_name"]
    for item in metrics["core_model_comparison"]:
        name = item["model_name"]
        label = f"{name} (Core-4, Best)" if name == best_core_name else f"{name} (Core-4)"
        rows.append(
            {
                "model": label,
                "val_mae": item["val"]["mae"],
                "val_rmse": item["val"]["rmse"],
                "val_r2": item["val"]["r2"],
                "test_mae": metrics["best_core_model"]["test"]["mae"] if name == best_core_name else np.nan,
                "test_rmse": metrics["best_core_model"]["test"]["rmse"] if name == best_core_name else np.nan,
                "test_r2": metrics["best_core_model"]["test"]["r2"] if name == best_core_name else np.nan,
                "is_highlight": name == best_core_name,
            }
        )

    full = metrics["full_benchmark_model"]
    rows.append(
        {
            "model": f"{full['model_name']} (Full-7 Benchmark)",
            "val_mae": full["val"]["mae"],
            "val_rmse": full["val"]["rmse"],
            "val_r2": full["val"]["r2"],
            "test_mae": full["test"]["mae"],
            "test_rmse": full["test"]["rmse"],
            "test_r2": full["test"]["r2"],
            "is_highlight": True,
        }
    )

    comp_df = pd.DataFrame(rows).sort_values("val_rmse", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(comp_df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.4), sharey=True)

    # Panel 1: Validation MAE & RMSE
    bar_h = 0.36
    ax1.barh(y_pos - bar_h / 2, comp_df["val_mae"], height=bar_h, color=PALETTE_SECONDARY, label="Val MAE (min)")
    ax1.barh(y_pos + bar_h / 2, comp_df["val_rmse"], height=bar_h, color=PALETTE_PRIMARY, label="Val RMSE (min)")

    for i, row in comp_df.iterrows():
        ax1.text(row["val_mae"] + 0.25, i - bar_h / 2, f"{row['val_mae']:.2f}", va="center", fontsize=8.5)
        ax1.text(row["val_rmse"] + 0.25, i + bar_h / 2, f"{row['val_rmse']:.2f}", va="center", fontsize=8.5)
        if not np.isnan(row["test_mae"]):
            ax1.scatter(
                row["test_mae"],
                i - bar_h / 2,
                color=PALETTE_ACCENT,
                edgecolor="#0F172A",
                zorder=5,
                s=45,
                label="Test MAE" if i == 0 else "",
            )
            ax1.scatter(
                row["test_rmse"],
                i + bar_h / 2,
                color=PALETTE_DANGER,
                edgecolor="#0F172A",
                zorder=5,
                s=45,
                label="Test RMSE" if i == 0 else "",
            )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(comp_df["model"])
    ax1.invert_yaxis()
    ax1.set_xlabel("Error (minutes — lower is better)")
    ax1.set_title("Validation MAE & RMSE (with Holdout Test Markers)")
    ax1.set_xlim(0, comp_df["val_rmse"].max() * 1.18)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel 2: Validation & Test R²
    bar_colors = [
        PALETTE_SUCCESS if "Full-7" in m else (PALETTE_PRIMARY if "Best" in m else "#94A3B8")
        for m in comp_df["model"]
    ]
    ax2.barh(y_pos, comp_df["val_r2"], height=0.55, color=bar_colors, alpha=0.88, label="Validation R²")
    for i, row in comp_df.iterrows():
        x_txt = max(row["val_r2"], 0.0) + 0.015
        ax2.text(x_txt, i, f"{row['val_r2']:.3f}", va="center", fontsize=9, fontweight="bold")
        if not np.isnan(row["test_r2"]):
            ax2.scatter(
                row["test_r2"],
                i,
                color=PALETTE_ACCENT,
                edgecolor="#0F172A",
                zorder=5,
                s=55,
                label="Test R²" if i == 0 else "",
            )

    ax2.axvline(0, color="#334155", linewidth=1)
    ax2.set_xlabel("R² Score (higher is better)")
    ax2.set_title("Validation R² (and Holdout Test R²)")
    ax2.set_xlim(-0.08, 1.0)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Model Comparison: Core-4 Candidates vs Full-7 Benchmark", fontsize=14.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "05_model_comparison.png")


def plot_06_actual_vs_predicted(preds: pd.DataFrame, metrics: dict[str, Any]) -> Path:
    """Actual vs Predicted scatter across Train, Val, and Test splits."""
    fig, ax = plt.subplots(figsize=(8.8, 7.2))

    split_meta = [
        ("train", "Train (n=700)", 0.35, 26),
        ("val", "Validation (n=150)", 0.70, 42),
        ("test", "Test Holdout (n=150)", 0.90, 56),
    ]
    for split_key, label, alpha, size in split_meta:
        sub = preds[preds["split"] == split_key]
        ax.scatter(
            sub["actual"],
            sub["predicted_core"],
            label=label,
            color=SPLIT_COLORS[split_key],
            alpha=alpha,
            s=size,
            edgecolor="white",
            linewidth=0.45,
        )

    lo = min(preds["actual"].min(), preds["predicted_core"].min()) - 4
    hi = max(preds["actual"].max(), preds["predicted_core"].max()) + 4
    ax.plot([lo, hi], [lo, hi], color=PALETTE_DANGER, linestyle="--", linewidth=2, label="Ideal Prediction (y = x)")

    best = metrics["best_core_model"]
    annotation = (
        f"Best Core-4 Model: {best['model_name']}\n"
        f"Train  → MAE: {best['train']['mae']:.2f} min | R²: {best['train']['r2']:.3f}\n"
        f"Val    → MAE: {best['val']['mae']:.2f} min | R²: {best['val']['r2']:.3f}\n"
        f"Test   → MAE: {best['test']['mae']:.2f} min | RMSE: {best['test']['rmse']:.2f} | R²: {best['test']['r2']:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.94},
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual Delivery Time (min)")
    ax.set_ylabel("Predicted Delivery Time (min)")
    ax.set_title("Actual vs Predicted Delivery Time (Core-4 LinearRegression)", pad=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save_fig(fig, "06_actual_vs_predicted.png")


def plot_07_residual_analysis(preds: pd.DataFrame) -> Path:
    """3-panel residual diagnostics on the Holdout Test set (Residuals vs Predicted, Histogram, Q-Q)."""
    test_df = preds[preds["split"] == "test"].copy()
    residuals = test_df["residual_core"]
    predicted = test_df["predicted_core"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # Panel 1: Residuals vs Predicted
    ax = axes[0]
    ax.scatter(predicted, residuals, color=PALETTE_PRIMARY, alpha=0.78, s=42, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color=PALETTE_DANGER, linestyle="--", linewidth=1.8)
    ax.set_xlabel("Predicted Delivery Time (min)")
    ax.set_ylabel("Residual (Actual − Predicted, min)")
    ax.set_title("Test Residuals vs Predicted")

    # Panel 2: Residual Distribution Histogram + KDE
    ax = axes[1]
    sns.histplot(residuals, bins=22, kde=True, color=PALETTE_SECONDARY, edgecolor="white", alpha=0.65, ax=ax)
    ax.axvline(0, color=PALETTE_DANGER, linestyle="--", linewidth=1.6, label="Zero Error")
    ax.axvline(
        residuals.mean(),
        color=PALETTE_ACCENT,
        linestyle="-",
        linewidth=1.8,
        label=f"Mean: {residuals.mean():.2f} min",
    )
    ax.set_xlabel("Residual (min)")
    ax.set_ylabel("Frequency")
    ax.set_title("Test Residual Distribution")
    ax.legend(fontsize=8.5)

    # Panel 3: Normal Q-Q Plot
    ax = axes[2]
    (osm, osr), (slope, intercept, r_qq) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, color=PALETTE_PRIMARY, alpha=0.78, s=34, edgecolor="white", linewidth=0.3)
    ax.plot(osm, slope * osm + intercept, color=PALETTE_DANGER, linestyle="--", linewidth=1.8, label=f"Normal fit (r={r_qq:.3f})")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Ordered Residuals (min)")
    ax.set_title("Normal Q-Q Plot of Test Residuals")
    ax.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Residual Diagnostics — Holdout Test Set (n = 150)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "07_residual_analysis.png")


def plot_08_feature_importance(metrics: dict[str, Any]) -> Path:
    """Permutation feature importance bar chart with error bars (Core-4 & Full-7)."""
    core_imp = pd.DataFrame(metrics["best_core_model"]["permutation_importance_test"]).sort_values(
        "mae_increase_mean", ascending=True
    )
    full_imp = pd.DataFrame(metrics["full_benchmark_model"]["permutation_importance_test"]).sort_values(
        "mae_increase_mean", ascending=True
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4))

    for ax, imp_df, title, bar_color in [
        (ax1, core_imp, "Core-4 Model — Permutation Importance (Test)", PALETTE_PRIMARY),
        (ax2, full_imp, "Full-7 Benchmark — Permutation Importance (Test)", PALETTE_SUCCESS),
    ]:
        bars = ax.barh(
            imp_df["feature"],
            imp_df["mae_increase_mean"],
            xerr=imp_df["mae_increase_std"],
            color=bar_color,
            alpha=0.86,
            edgecolor="white",
            capsize=4,
            error_kw={"ecolor": "#334155", "elinewidth": 1.3},
            height=0.58,
        )
        for bar, mean_v, std_v in zip(bars, imp_df["mae_increase_mean"], imp_df["mae_increase_std"]):
            offset = max(mean_v, 0) + std_v + 0.25
            ax.text(
                offset,
                bar.get_y() + bar.get_height() / 2,
                f"+{mean_v:.2f} ± {std_v:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )
        ax.axvline(0, color="#475569", linewidth=1)
        ax.set_xlabel("MAE Increase When Feature Permuted (min)")
        ax.set_title(title, pad=10)
        xmax = (imp_df["mae_increase_mean"] + imp_df["mae_increase_std"]).max()
        ax.set_xlim(min(imp_df["mae_increase_mean"].min() - 0.6, -0.5), xmax * 1.28)

    fig.suptitle("Permutation Feature Importance on Holdout Test Set", fontsize=14.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "08_feature_importance.png")


def plot_09_slice_error_analysis(metrics: dict[str, Any]) -> Path:
    """Test MAE & RMSE breakdown by Weather and Traffic_Level slices."""
    slices = metrics["best_core_model"]["slice_errors_test"]
    overall_mae = float(metrics["best_core_model"]["test"]["mae"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4))

    for ax, slice_name, color in [
        (ax1, "Weather", PALETTE_PRIMARY),
        (ax2, "Traffic_Level", PALETTE_SECONDARY),
    ]:
        slice_dict = slices[slice_name]
        sdf = (
            pd.DataFrame.from_dict(slice_dict, orient="index")
            .reset_index()
            .rename(columns={"index": "slice"})
            .sort_values("mae", ascending=True)
        )
        labels = [f"{r['slice']} (n={int(r['count'])})" for _, r in sdf.iterrows()]
        y = np.arange(len(sdf))
        bar_h = 0.36
        ax.barh(y - bar_h / 2, sdf["mae"], height=bar_h, color=color, label="MAE (min)")
        ax.barh(y + bar_h / 2, sdf["rmse"], height=bar_h, color="#94A3B8", alpha=0.8, label="RMSE (min)")
        ax.axvline(
            overall_mae,
            color=PALETTE_DANGER,
            linestyle="--",
            linewidth=1.8,
            label=f"Overall Test MAE ({overall_mae:.2f})",
        )

        for i, (_, r) in enumerate(sdf.iterrows()):
            ax.text(r["mae"] + 0.18, i - bar_h / 2, f"{r['mae']:.2f}", va="center", fontsize=8.5, fontweight="bold")
            ax.text(r["rmse"] + 0.18, i + bar_h / 2, f"{r['rmse']:.2f}", va="center", fontsize=8.5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Test Error (minutes)")
        ax.set_title(f"Test Error by {slice_name.replace('_', ' ')}")
        ax.set_xlim(0, sdf["rmse"].max() * 1.22)
        ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Slice-Based Error Analysis — Holdout Test Set (Core-4 Model)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "09_slice_error_analysis.png")


def generate_all_plots() -> list[Path]:
    """Generate all 9 EDA and ML evaluation plots and return their output paths."""
    _configure_style()
    df, preds, metrics, eda = _load_artifacts()

    paths = [
        plot_01_target_distribution(df, eda),
        plot_02_missing_values(df, eda),
        plot_03_feature_relationships(df),
        plot_04_correlation_heatmap(df),
        plot_05_model_comparison(metrics),
        plot_06_actual_vs_predicted(preds, metrics),
        plot_07_residual_analysis(preds),
        plot_08_feature_importance(metrics),
        plot_09_slice_error_analysis(metrics),
    ]
    return paths


if __name__ == "__main__":
    generated = generate_all_plots()
    print(f"Generated {len(generated)} plots in {PLOTS_DIR}:")
    for p in generated:
        print(f"  - {p.name} ({p.stat().st_size:,} bytes)")
