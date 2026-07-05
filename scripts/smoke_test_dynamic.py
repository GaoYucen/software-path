#!/usr/bin/env python3
"""Run a lightweight smoke test over generated dynamic path arrays.

This deliberately avoids torch because the current local environment does not
have it installed. The test verifies that the generated arrays are loadable,
chronologically split, and usable for a simple travel-time estimation model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return (values * mask[..., None]).sum(axis=1) / denom


def build_features(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_road = np.load(data_dir / "data_road.npy")
    row_num = np.load(data_dir / "row_num.npy")
    trip_time = np.load(data_dir / "trip_time.npy").astype(float)
    departure_time = np.load(data_dir / "departure_time.npy")
    dynamic_path = np.load(data_dir / "dynamic_path.npy").astype(float)
    pad_id = int(np.load(data_dir / "pad_id.npy")[0])

    if data_road.ndim != 2 or dynamic_path.ndim != 3:
        raise ValueError("Unexpected array dimensions")
    if data_road.shape[:2] != dynamic_path.shape[:2]:
        raise ValueError("data_road and dynamic_path length dimensions differ")
    if len(row_num) != len(trip_time) or len(row_num) != len(departure_time):
        raise ValueError("Trip-level arrays have inconsistent lengths")

    valid = (data_road != pad_id).astype(float)
    dyn_mean = masked_mean(dynamic_path, valid)
    path_len = row_num.astype(float).reshape(-1, 1)
    unique_edges = np.array(
        [
            len(set(row[data_road[i] != pad_id].tolist()))
            for i, row in enumerate(data_road)
        ],
        dtype=float,
    ).reshape(-1, 1)
    hour = ((departure_time // 3600) % 24).astype(float)
    hour_rad = 2 * np.pi * hour / 24.0
    time_feat = np.column_stack([np.sin(hour_rad), np.cos(hour_rad)])

    static_features = np.column_stack([path_len, unique_edges, time_feat])
    dynamic_features = np.column_stack([static_features, dyn_mean])
    return static_features, dynamic_features, trip_time, departure_time


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


def chronological_split(departure_time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(departure_time)
    n = len(order)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    if len(test_idx) == 0:
        test_idx = val_idx
    return train_idx, val_idx, test_idx


def fit_and_eval(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    out = {"model": name}
    out.update(metrics(y_test, pred))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/geolife_smoke")
    parser.add_argument("--report", default="reports/geolife_smoke_metrics.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    x_static, x_dynamic, y, departure_time = build_features(data_dir)
    train_idx, val_idx, test_idx = chronological_split(departure_time)

    baseline_pred = np.full(len(test_idx), y[train_idx].mean())
    results = [
        {"model": "train_mean_baseline", **metrics(y[test_idx], baseline_pred)},
        fit_and_eval(
            "ridge_static",
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            x_static[train_idx],
            y[train_idx],
            x_static[test_idx],
            y[test_idx],
        ),
        fit_and_eval(
            "ridge_static_dynamic",
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            x_dynamic[train_idx],
            y[train_idx],
            x_dynamic[test_idx],
            y[test_idx],
        ),
        fit_and_eval(
            "rf_static_dynamic",
            RandomForestRegressor(n_estimators=80, random_state=2026, min_samples_leaf=2),
            x_dynamic[train_idx],
            y[train_idx],
            x_dynamic[test_idx],
            y[test_idx],
        ),
    ]

    report = {
        "data_dir": str(data_dir),
        "num_samples": int(len(y)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "target_seconds": {
            "min": float(y.min()),
            "mean": float(y.mean()),
            "max": float(y.max()),
        },
        "results": results,
        "status": "pass",
        "note": (
            "Smoke test only. This validates generated arrays and dynamic "
            "features, not full PATH-LLM training."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
