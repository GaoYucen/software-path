#!/usr/bin/env python3
"""Analyze TTE model results and generate paper-quality figures.

Generates:
  1. Error CDF comparison: static vs dynamic models.
  2. Reliability score distribution analysis.
  3. Performance by path length bucket.
  4. Performance by hour-of-day (peak vs off-peak).
  5. Dynamic gain vs static baseline scatter.

Usage:
    python scripts/analyze_results.py \\
      --data-dir data/processed/pkdd15_grid_120k_clean \\
      --metrics-dir reports/ \\
      --output-dir reports/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Optional imports — graceful degradation if matplotlib is not installed.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}

COLORS = {
    "static": "#3498db",
    "dynamic": "#e74c3c",
    "full": "#2ecc71",
    "concat": "#f39c12",
    "simple_gate": "#9b59b6",
}


def setup_style():
    if HAS_MPL:
        for k, v in STYLE.items():
            matplotlib.rcParams[k] = v


# ---------------------------------------------------------------------------
# Figure 1: Error CDF
# ---------------------------------------------------------------------------

def plot_error_cdf(
    static_preds: np.ndarray,
    dynamic_preds: np.ndarray,
    y_true: np.ndarray,
    output_path: Path,
    static_label: str = "HGB Static",
    dynamic_label: str = "HGB DynaPath-Lite",
) -> None:
    if not HAS_MPL:
        print("matplotlib not available; skipping error CDF plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    static_err = np.abs(y_true - static_preds)
    dynamic_err = np.abs(y_true - dynamic_preds)

    for err, label, color in [
        (static_err, static_label, COLORS["static"]),
        (dynamic_err, dynamic_label, COLORS["dynamic"]),
    ]:
        sorted_err = np.sort(err)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf, label=label, color=color, linewidth=2)

    ax.set_xlabel("Absolute Error / s")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Error CDF: Static vs Dynamic Models")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Reliability distribution
# ---------------------------------------------------------------------------

def plot_reliability_distribution(
    dynamic_features: np.ndarray,
    output_path: Path,
    reliability_index: int = -1,
) -> None:
    if not HAS_MPL:
        print("matplotlib not available; skipping reliability plot.")
        return

    # dynamic_features shape: [N, L, D]
    reliability = dynamic_features[:, :, reliability_index].flatten()
    reliability = reliability[reliability > 0]  # filter out padding zeros

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(reliability, bins=50, color=COLORS["dynamic"], alpha=0.7, edgecolor="white")
    axes[0].set_xlabel("Reliability Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Reliability Score Distribution (Token-Level)")
    axes[0].axvline(np.median(reliability), color="black", linestyle="--",
                    label=f"Median: {np.median(reliability):.3f}")
    axes[0].legend()

    # CDF
    sorted_r = np.sort(reliability)
    cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    axes[1].plot(sorted_r, cdf, color=COLORS["dynamic"], linewidth=2)
    axes[1].set_xlabel("Reliability Score")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("Reliability Score CDF")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Performance by path length
# ---------------------------------------------------------------------------

def plot_by_path_length(
    y_true: np.ndarray,
    static_preds: np.ndarray,
    dynamic_preds: np.ndarray,
    row_num: np.ndarray,
    output_path: Path,
) -> None:
    if not HAS_MPL:
        print("matplotlib not available; skipping path-length plot.")
        return

    # Bin paths by length
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 64)]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-64"]

    static_mae = []
    dynamic_mae = []
    for lo, hi in bins:
        mask = (row_num >= lo) & (row_num < hi)
        if not mask.any():
            static_mae.append(0)
            dynamic_mae.append(0)
            continue
        static_mae.append(np.abs(y_true[mask] - static_preds[mask]).mean())
        dynamic_mae.append(np.abs(y_true[mask] - dynamic_preds[mask]).mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, static_mae, width, label="HGB Static", color=COLORS["static"], alpha=0.8)
    ax.bar(x + width/2, dynamic_mae, width, label="HGB DynaPath-Lite", color=COLORS["dynamic"], alpha=0.8)

    # Annotate improvement
    for i, (s, d) in enumerate(zip(static_mae, dynamic_mae)):
        if s > 0:
            gain = (s - d) / s * 100
            ax.annotate(f"-{gain:.1f}%", (i, min(s, d)), ha="center", va="bottom",
                        fontsize=8, color="green")

    ax.set_xlabel("Path Length (tokens)")
    ax.set_ylabel("MAE / s")
    ax.set_title("Performance by Path Length")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: Performance by hour-of-day
# ---------------------------------------------------------------------------

def plot_by_hour(
    y_true: np.ndarray,
    static_preds: np.ndarray,
    dynamic_preds: np.ndarray,
    departure_time: np.ndarray,
    output_path: Path,
) -> None:
    if not HAS_MPL:
        print("matplotlib not available; skipping hourly plot.")
        return

    hours = (departure_time // 3600) % 24
    hour_labels = sorted(set(hours.tolist()))

    static_mae = []
    dynamic_mae = []
    counts = []
    for h in hour_labels:
        mask = hours == h
        counts.append(mask.sum())
        static_mae.append(np.abs(y_true[mask] - static_preds[mask]).mean() if mask.any() else 0)
        dynamic_mae.append(np.abs(y_true[mask] - dynamic_preds[mask]).mean() if mask.any() else 0)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(hour_labels, static_mae, "o-", color=COLORS["static"], label="HGB Static", linewidth=2)
    ax1.plot(hour_labels, dynamic_mae, "s-", color=COLORS["dynamic"], label="HGB DynaPath-Lite", linewidth=2)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("MAE / s")
    ax1.set_title("Performance by Hour of Day")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Overlay sample counts
    ax2 = ax1.twinx()
    ax2.bar(hour_labels, counts, alpha=0.15, color="gray", label="Sample Count")
    ax2.set_ylabel("Sample Count", alpha=0.5)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 5: Dynamic gain scatter
# ---------------------------------------------------------------------------

def plot_dynamic_gain_scatter(
    y_true: np.ndarray,
    static_preds: np.ndarray,
    dynamic_preds: np.ndarray,
    reliability: np.ndarray,
    output_path: Path,
) -> None:
    if not HAS_MPL:
        print("matplotlib not available; skipping scatter plot.")
        return

    static_err = np.abs(y_true - static_preds)
    dynamic_err = np.abs(y_true - dynamic_preds)
    gain = static_err - dynamic_err  # positive = dynamic better

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gain vs true travel time
    axes[0].scatter(y_true, gain, c=gain, cmap="RdYlGn", alpha=0.3, s=5)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("True Travel Time / s")
    axes[0].set_ylabel("MAE Reduction (Static - Dynamic) / s")
    axes[0].set_title("Dynamic Gain vs. Travel Time")
    axes[0].grid(True, alpha=0.3)

    # Gain vs reliability
    axes[1].scatter(reliability, gain, c=gain, cmap="RdYlGn", alpha=0.3, s=5)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Mean Reliability Score")
    axes[1].set_ylabel("MAE Reduction (Static - Dynamic) / s")
    axes[1].set_title("Dynamic Gain vs. Reliability")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main: load data & generate all figures
# ---------------------------------------------------------------------------

def load_tabular_predictions(data_dir: Path) -> dict:
    """Load tabular model predictions from saved model files.

    Returns dict with keys: y_true, hgb_static_pred, hgb_dynamic_pred.
    """
    import joblib

    model_dir = Path("reports/pkdd15_dynapath_lite_120k_noleak_models")
    df = pd.read_csv(data_dir / "trip_features.csv")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # chrono split
    order = np.argsort(df["timestamp"].to_numpy())
    n = len(order)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    test_idx = order[n_train + n_val:]

    y = df["target_trip_time"].to_numpy(dtype=float)

    skip = {"trip_index", "trip_id", "timestamp", "target_trip_time", "num_points"}
    numeric_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    dynamic_prefixes = (
        "dyn_speed_median", "dyn_speed_std", "dyn_log_obs",
        "dyn_speed_ratio", "dyn_freshness_min", "dyn_reliability",
    )
    static_cols = [c for c in numeric_cols if not any(c.startswith(p) for p in dynamic_prefixes)]
    all_cols = static_cols + [c for c in numeric_cols if c not in static_cols]

    hgb_static_pred = None
    hgb_dynamic_pred = None

    static_path = model_dir / "hgb_static.joblib"
    if static_path.exists():
        model = joblib.load(static_path)
        hgb_static_pred = model.predict(df[static_cols].to_numpy(dtype=float)[test_idx])

    dynamic_path = model_dir / "hgb_dynapath_lite.joblib"
    if dynamic_path.exists():
        model = joblib.load(dynamic_path)
        hgb_dynamic_pred = model.predict(df[all_cols].to_numpy(dtype=float)[test_idx])

    return {
        "y_true": y[test_idx],
        "hgb_static_pred": hgb_static_pred,
        "hgb_dynamic_pred": hgb_dynamic_pred,
        "test_idx": test_idx,
    }


def load_sequence_arrays(data_dir: Path, test_idx: np.ndarray) -> dict:
    """Load sequence arrays for test samples."""
    row_num = np.load(data_dir / "row_num.npy")[test_idx]
    departure_time = np.load(data_dir / "departure_time.npy")[test_idx]
    dynamic_path = np.load(data_dir / "dynamic_path.npy")[test_idx]

    # Compute mean reliability per path
    reliability = dynamic_path[:, :, -1]  # last dim is reliability
    valid = np.arange(dynamic_path.shape[1])[None, :] < row_num[:, None]
    path_reliability = (reliability * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)

    return {
        "row_num": row_num,
        "departure_time": departure_time,
        "path_reliability": path_reliability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TTE results")
    parser.add_argument("--data-dir", default="data/processed/pkdd15_grid_120k_clean")
    parser.add_argument("--metrics-dir", default="reports/")
    parser.add_argument("--output-dir", default="reports/figures/")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    print("Loading data...")

    # Load tabular model predictions
    tab = load_tabular_predictions(data_dir)
    y_true = tab["y_true"]
    test_idx = tab["test_idx"]

    # Load sequence arrays
    seq = load_sequence_arrays(data_dir, test_idx)

    # ---- Generate figures ----
    if HAS_MPL and tab["hgb_static_pred"] is not None and tab["hgb_dynamic_pred"] is not None:
        print("\nGenerating figures...")

        plot_error_cdf(
            tab["hgb_static_pred"],
            tab["hgb_dynamic_pred"],
            y_true,
            output_dir / "error_cdf.png",
        )

        # Load full dynamic path for reliability distribution
        dynamic_path = np.load(data_dir / "dynamic_path.npy")
        plot_reliability_distribution(
            dynamic_path,
            output_dir / "reliability_distribution.png",
        )

        plot_by_path_length(
            y_true, tab["hgb_static_pred"], tab["hgb_dynamic_pred"],
            seq["row_num"], output_dir / "performance_by_length.png",
        )

        plot_by_hour(
            y_true, tab["hgb_static_pred"], tab["hgb_dynamic_pred"],
            seq["departure_time"], output_dir / "performance_by_hour.png",
        )

        plot_dynamic_gain_scatter(
            y_true, tab["hgb_static_pred"], tab["hgb_dynamic_pred"],
            seq["path_reliability"], output_dir / "dynamic_gain_scatter.png",
        )

        print(f"\nAll figures saved to: {output_dir}")
    else:
        print("matplotlib or model predictions not available; skipping figures.")
        print("To generate figures, run the tabular baselines first and install matplotlib.")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Test samples: {len(y_true)}")
    print(f"True travel time — mean: {y_true.mean():.1f}s, median: {np.median(y_true):.1f}s")
    print(f"Path reliability — mean: {seq['path_reliability'].mean():.3f}, "
          f"median: {np.median(seq['path_reliability']):.3f}")
    print(f"Path length — mean: {seq['row_num'].mean():.1f} tokens")

    if tab["hgb_static_pred"] is not None and tab["hgb_dynamic_pred"] is not None:
        static_err = np.abs(y_true - tab["hgb_static_pred"])
        dynamic_err = np.abs(y_true - tab["hgb_dynamic_pred"])
        gain = static_err - dynamic_err
        improved = (gain > 0).sum()
        print(f"\nDynamic model improved {improved}/{len(gain)} samples "
              f"({improved/len(gain)*100:.1f}%)")
        print(f"Mean improvement: {gain.mean():.2f}s")
        print(f"Median improvement: {np.median(gain):.2f}s")


if __name__ == "__main__":
    main()
