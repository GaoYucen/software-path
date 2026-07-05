#!/usr/bin/env python3
"""Run ablation experiments for the PKDD15 DynaPath-Lite tabular setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DYNAMIC_PREFIXES = (
    "dyn_speed_median",
    "dyn_speed_std",
    "dyn_log_obs",
    "dyn_speed_ratio",
    "dyn_freshness_min",
    "dyn_reliability",
)


GROUPS = {
    "speed": ("dyn_speed_median", "dyn_speed_ratio"),
    "speed_variance": ("dyn_speed_std",),
    "density": ("dyn_log_obs",),
    "freshness": ("dyn_freshness_min",),
    "reliability": ("dyn_reliability",),
    "quality": ("dyn_log_obs", "dyn_freshness_min", "dyn_reliability"),
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6)))
    mare = np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(y_true), 1e-6)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "mare": float(mare),
    }


def chronological_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(df["timestamp"].to_numpy())
    n = len(order)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    skip = {
        "trip_index",
        "trip_id",
        "timestamp",
        "target_trip_time",
        "num_points",
    }
    numeric_cols = [
        col
        for col in df.columns
        if col not in skip and pd.api.types.is_numeric_dtype(df[col])
    ]
    static_cols = [
        col
        for col in numeric_cols
        if not any(col.startswith(prefix) for prefix in DYNAMIC_PREFIXES)
    ]
    dynamic_cols = [col for col in numeric_cols if col not in static_cols]
    return static_cols, dynamic_cols, static_cols + dynamic_cols


def hgb_model(seed: int = 2026) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=seed,
    )


def evaluate(
    name: str,
    df: pd.DataFrame,
    cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    note: str,
    seed: int,
) -> dict[str, float | str | int]:
    model = hgb_model(seed)
    x = df[cols].to_numpy(dtype=float)
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    out: dict[str, float | str | int] = {
        "experiment": name,
        "num_features": len(cols),
        "note": note,
    }
    out.update(metrics(y[test_idx], pred))
    return out


def drop_group(cols: list[str], group_name: str) -> list[str]:
    prefixes = GROUPS[group_name]
    return [col for col in cols if not any(col.startswith(prefix) for prefix in prefixes)]


def dynamic_fallback_values(
    df: pd.DataFrame, dynamic_cols: list[str], train_idx: np.ndarray
) -> dict[str, float]:
    values: dict[str, float] = {}
    for col in dynamic_cols:
        if col.startswith("dyn_log_obs") or col.startswith("dyn_reliability"):
            values[col] = 0.0
        elif col.startswith("dyn_freshness_min"):
            values[col] = float(df.iloc[train_idx][col].quantile(0.95))
        elif col.startswith("dyn_speed_ratio"):
            values[col] = 1.0
        else:
            values[col] = float(df.iloc[train_idx][col].median())
    return values


def apply_dynamic_missingness(
    df: pd.DataFrame,
    dynamic_cols: list[str],
    train_idx: np.ndarray,
    keep_ratio: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fallback = dynamic_fallback_values(df, dynamic_cols, train_idx)
    masked = df.copy()
    missing_mask = rng.random(len(masked)) > keep_ratio
    for col in dynamic_cols:
        masked.loc[missing_mask, col] = fallback[col]
    return masked


def pct_gain(base_mae: float, mae: float) -> float:
    return float((base_mae - mae) / base_mae * 100.0)


def write_markdown(report: dict, path: Path) -> None:
    rows = report["results"]
    baseline = next(row for row in rows if row["experiment"] == "hgb_static")
    full = next(row for row in rows if row["experiment"] == "hgb_full_dynamic")

    lines = [
        "# PKDD15 DynaPath-Lite Ablation Report",
        "",
        "## Setup",
        "",
        f"- Data directory: `{report['data_dir']}`",
        f"- Samples: {report['num_samples']}",
        f"- Train/validation/test: {report['num_train']}/{report['num_val']}/{report['num_test']}",
        f"- Static features: {report['num_static_features']}",
        f"- Dynamic features: {report['num_dynamic_features']}",
        f"- Random seed: {report['seed']}",
        "",
        "## Results",
        "",
        "| Experiment | Features | MAE/s | RMSE/s | MAPE | MARE | MAE gain vs static | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {num_features} | {mae:.2f} | {rmse:.2f} | {mape:.4f} | "
            "{mare:.4f} | {gain:.2f}% | {note} |".format(
                **row, gain=pct_gain(float(baseline["mae"]), float(row["mae"]))
            )
        )

    lines.extend(
        [
            "",
            "## Key Comparisons",
            "",
            "- Full dynamic vs static: "
            f"{baseline['mae']:.2f}s -> {full['mae']:.2f}s, "
            f"{pct_gain(float(baseline['mae']), float(full['mae'])):.2f}% MAE gain.",
        ]
    )
    for pair in report["key_comparisons"]:
        lines.append(f"- {pair['name']}: {pair['summary']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/pkdd15_grid_120k_clean")
    parser.add_argument(
        "--report",
        default="reports/pkdd15_dynapath_lite_120k_ablation_metrics.json",
    )
    parser.add_argument(
        "--markdown",
        default="reports/pkdd15_dynapath_lite_120k_ablation_report.md",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report_path = Path(args.report)
    markdown_path = Path(args.markdown)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "trip_features.csv")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    static_cols, dynamic_cols, full_cols = feature_columns(df)
    train_idx, val_idx, test_idx = chronological_split(df)
    y = df["target_trip_time"].to_numpy(dtype=float)

    results: list[dict] = []
    results.append(
        evaluate(
            "hgb_static",
            df,
            static_cols,
            train_idx,
            test_idx,
            y,
            "Only static path and temporal-context features.",
            args.seed,
        )
    )
    results.append(
        evaluate(
            "hgb_full_dynamic",
            df,
            full_cols,
            train_idx,
            test_idx,
            y,
            "Static features plus all dynamic features.",
            args.seed,
        )
    )

    for group_name in [
        "speed",
        "speed_variance",
        "density",
        "freshness",
        "reliability",
        "quality",
    ]:
        cols = drop_group(full_cols, group_name)
        results.append(
            evaluate(
                f"hgb_wo_{group_name}",
                df,
                cols,
                train_idx,
                test_idx,
                y,
                f"Remove dynamic {group_name} feature group.",
                args.seed,
            )
        )

    sparse_df = apply_dynamic_missingness(
        df, dynamic_cols, train_idx, keep_ratio=0.30, seed=args.seed
    )
    results.append(
        evaluate(
            "hgb_sparse30_with_reliability",
            sparse_df,
            full_cols,
            train_idx,
            test_idx,
            y,
            "Keep 30% dynamic observations and replace the rest by static fallback values.",
            args.seed,
        )
    )
    results.append(
        evaluate(
            "hgb_sparse30_wo_reliability",
            sparse_df,
            drop_group(full_cols, "reliability"),
            train_idx,
            test_idx,
            y,
            "Same sparse setting, but reliability features are removed.",
            args.seed,
        )
    )
    results.append(
        evaluate(
            "hgb_sparse30_wo_quality",
            sparse_df,
            drop_group(full_cols, "quality"),
            train_idx,
            test_idx,
            y,
            "Same sparse setting, but density, freshness, and reliability features are removed.",
            args.seed,
        )
    )

    by_name = {row["experiment"]: row for row in results}
    key_comparisons = [
        {
            "name": "Reliability ablation",
            "summary": (
                f"full dynamic MAE {by_name['hgb_full_dynamic']['mae']:.2f}s, "
                f"without reliability MAE {by_name['hgb_wo_reliability']['mae']:.2f}s."
            ),
        },
        {
            "name": "Sparse dynamic reliability",
            "summary": (
                f"sparse-30 with reliability MAE "
                f"{by_name['hgb_sparse30_with_reliability']['mae']:.2f}s, "
                f"without reliability MAE "
                f"{by_name['hgb_sparse30_wo_reliability']['mae']:.2f}s."
            ),
        },
        {
            "name": "Sparse dynamic quality",
            "summary": (
                f"sparse-30 with quality signals MAE "
                f"{by_name['hgb_sparse30_with_reliability']['mae']:.2f}s, "
                f"without density/freshness/reliability MAE "
                f"{by_name['hgb_sparse30_wo_quality']['mae']:.2f}s."
            ),
        },
    ]

    report = {
        "status": "pass",
        "data_dir": str(data_dir),
        "num_samples": int(len(df)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "num_static_features": int(len(static_cols)),
        "num_dynamic_features": int(len(dynamic_cols)),
        "seed": int(args.seed),
        "static_features": static_cols,
        "dynamic_features": dynamic_cols,
        "results": results,
        "key_comparisons": key_comparisons,
        "note": (
            "Ablations reuse the no-leak chronological split and exclude num_points "
            "because it deterministically leaks the PKDD15 TTE label."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, markdown_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
