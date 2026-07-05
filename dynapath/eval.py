#!/usr/bin/env python3
"""Unified evaluation utilities for TTE models.

Provides consistent metric computation, bootstrap confidence intervals,
and helper functions shared by all training and analysis scripts.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute standard TTE regression metrics.

    Parameters
    ----------
    y_true: Ground-truth travel times.
    y_pred: Predicted travel times.
    eps: Small constant to avoid division by zero in relative metrics.

    Returns
    -------
    dict with keys ``mae``, ``rmse``, ``mape``, ``mare``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape": float(np.mean(err / np.maximum(y_true, eps))),
        "mare": float(err.sum() / max(float(np.sum(y_true)), eps)),
    }


def quantile_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95),
) -> dict[str, float]:
    """Compute absolute error at specified quantiles."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {f"ae_q{int(q * 100)}": float(np.quantile(err, q)) for q in quantiles}


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn=regression_metrics,
    n_bootstrap: int = 1000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Bootstrap confidence intervals for regression metrics.

    Returns
    -------
    dict mapping metric name -> {``value``, ``lower``, ``upper``}.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    metrics_fn = metric_fn  # avoid shadowing module name
    point = metrics_fn(y_true, y_pred)
    bootstrap_samples: dict[str, list[float]] = {k: [] for k in point}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = metrics_fn(y_true[idx], y_pred[idx])
        for k in point:
            bootstrap_samples[k].append(sample[k])

    alpha = (1.0 - confidence) / 2.0
    result = {}
    for k in point:
        vals = np.sort(bootstrap_samples[k])
        lower = float(np.quantile(vals, alpha))
        upper = float(np.quantile(vals, 1.0 - alpha))
        result[k] = {
            "value": point[k],
            "lower": lower,
            "upper": upper,
        }
    return result


def evaluate_model(
    model,
    loader,
    device: torch.device,
    torch,
    criterion=None,
) -> dict[str, float]:
    """Run a full evaluation pass over a DataLoader.

    Parameters
    ----------
    model: nn.Module in eval mode.
    loader: DataLoader yielding batches with at least ``road_ids``,
        ``attention_mask``, ``targets``, and optionally ``dynamic_x``.
    device: torch device.
    torch: torch module reference.
    criterion: Optional loss function (defaults to MSELoss).

    Returns
    -------
    dict with ``loss`` and regression metrics.
    """
    import torch as _torch  # noqa: F811

    if criterion is None:
        criterion = _torch.nn.MSELoss()

    model.eval()
    total_loss = 0.0
    all_preds: list[float] = []
    all_trues: list[float] = []

    with _torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            has_dynamic = "dynamic_x" in batch

            # try calling model with or without dynamic_x
            try:
                if has_dynamic:
                    out = model(
                        road_ids=batch["road_ids"],
                        dynamic_x=batch["dynamic_x"],
                        attention_mask=batch["attention_mask"],
                    )
                else:
                    out = model(
                        road_ids=batch["road_ids"],
                        attention_mask=batch["attention_mask"],
                    )
            except TypeError:
                # model doesn't accept dynamic_x — fall back
                out = model(
                    road_ids=batch["road_ids"],
                    attention_mask=batch["attention_mask"],
                )

            pred = out["prediction"]
            loss = criterion(pred, batch["targets"])
            total_loss += float(loss.item()) * batch["targets"].size(0)

            all_preds.extend(pred.cpu().numpy().tolist())
            all_trues.extend(batch["targets"].cpu().numpy().tolist())

    n = len(all_trues)
    yt = np.array(all_trues, dtype=float)
    yp = np.array(all_preds, dtype=float)
    result = {"loss": total_loss / max(n, 1)}
    result.update(regression_metrics(yt, yp))
    return result


# Import torch at module level for type hints but we accept it as param
# to keep eval.py importable without torch.
try:
    import torch  # noqa: F401
except ImportError:
    pass
