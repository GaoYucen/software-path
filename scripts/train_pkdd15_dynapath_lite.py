#!/usr/bin/env python3
"""Train quick TTE baselines on PKDD15 grid dynamic features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DYNAMIC_PREFIXES = (
    "dyn_speed_median",
    "dyn_speed_std",
    "dyn_log_obs",
    "dyn_speed_ratio",
    "dyn_freshness_min",
    "dyn_reliability",
)


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
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    return train, val, test


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    skip = {
        "trip_index",
        "trip_id",
        "timestamp",
        "target_trip_time",
        # PKDD15 trip_time is exactly (len(POLYLINE) - 1) * 15 seconds.
        # Keeping num_points would leak the label.
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
    return static_cols, static_cols + dynamic_cols


def evaluate_model(name: str, model, x_train, y_train, x_test, y_test) -> tuple[dict, object]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    result = {"model": name}
    result.update(metrics(y_test, pred))
    return result, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/pkdd15_grid")
    parser.add_argument("--report", default="reports/pkdd15_dynapath_lite_metrics.json")
    parser.add_argument("--model-dir", default="reports/pkdd15_models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report_path = Path(args.report)
    model_dir = Path(args.model_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / "trip_features.csv")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    static_cols, dyn_cols = feature_columns(df)
    train_idx, val_idx, test_idx = chronological_split(df)
    y = df["target_trip_time"].to_numpy(dtype=float)

    x_static = df[static_cols].to_numpy(dtype=float)
    x_dyn = df[dyn_cols].to_numpy(dtype=float)

    results: list[dict] = []
    saved_models: dict[str, str] = {}

    mean_pred = np.full(len(test_idx), y[train_idx].mean())
    results.append({"model": "train_mean", **metrics(y[test_idx], mean_pred)})

    models = [
        (
            "ridge_static",
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            x_static,
        ),
        (
            "ridge_static_dynamic",
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            x_dyn,
        ),
        (
            "hgb_static",
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                random_state=2026,
            ),
            x_static,
        ),
        (
            "hgb_dynapath_lite",
            HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                random_state=2026,
            ),
            x_dyn,
        ),
    ]

    if len(train_idx) <= 80_000:
        models.append(
            (
                "rf_dynapath_lite",
                RandomForestRegressor(
                    n_estimators=80,
                    random_state=2026,
                    min_samples_leaf=3,
                    n_jobs=-1,
                ),
                x_dyn,
            )
        )

    for name, model, x in models:
        result, fitted = evaluate_model(
            name, model, x[train_idx], y[train_idx], x[test_idx], y[test_idx]
        )
        results.append(result)
        model_path = model_dir / f"{name}.joblib"
        joblib.dump(fitted, model_path)
        saved_models[name] = str(model_path)

    report = {
        "status": "pass",
        "data_dir": str(data_dir),
        "num_samples": int(len(df)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "num_static_features": int(len(static_cols)),
        "num_static_dynamic_features": int(len(dyn_cols)),
        "target_seconds": {
            "min": float(y.min()),
            "mean": float(y.mean()),
            "median": float(np.median(y)),
            "max": float(y.max()),
        },
        "static_features": static_cols,
        "dynamic_features": [c for c in dyn_cols if c not in static_cols],
        "results": results,
        "saved_models": saved_models,
        "note": (
            "First-result PKDD15 grid-token experiment. It validates the model "
            "construction path before OSM map matching."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
