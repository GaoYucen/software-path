# PKDD15 DynaPath-Lite Ablation Report

## Setup

- Data directory: `data/processed/pkdd15_grid_120k_clean`
- Samples: 113660
- Train/validation/test: 79562/17049/17049
- Static features: 20
- Dynamic features: 12
- Random seed: 2026

## Results

| Experiment | Features | MAE/s | RMSE/s | MAPE | MARE | MAE gain vs static | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| hgb_static | 20 | 88.59 | 140.34 | 0.1417 | 0.1333 | 0.00% | Only static path and temporal-context features. |
| hgb_full_dynamic | 32 | 85.22 | 137.38 | 0.1357 | 0.1283 | 3.81% | Static features plus all dynamic features. |
| hgb_wo_speed | 28 | 86.72 | 138.88 | 0.1383 | 0.1305 | 2.11% | Remove dynamic speed feature group. |
| hgb_wo_speed_variance | 30 | 85.13 | 137.12 | 0.1357 | 0.1281 | 3.91% | Remove dynamic speed_variance feature group. |
| hgb_wo_density | 30 | 85.40 | 137.36 | 0.1362 | 0.1285 | 3.60% | Remove dynamic density feature group. |
| hgb_wo_freshness | 30 | 85.24 | 137.19 | 0.1358 | 0.1283 | 3.78% | Remove dynamic freshness feature group. |
| hgb_wo_reliability | 30 | 85.11 | 137.18 | 0.1355 | 0.1281 | 3.93% | Remove dynamic reliability feature group. |
| hgb_wo_quality | 26 | 85.55 | 137.57 | 0.1365 | 0.1288 | 3.43% | Remove dynamic quality feature group. |
| hgb_sparse30_with_reliability | 32 | 88.41 | 140.72 | 0.1412 | 0.1331 | 0.20% | Keep 30% dynamic observations and replace the rest by static fallback values. |
| hgb_sparse30_wo_reliability | 30 | 88.36 | 140.67 | 0.1412 | 0.1330 | 0.26% | Same sparse setting, but reliability features are removed. |
| hgb_sparse30_wo_quality | 26 | 88.47 | 140.60 | 0.1414 | 0.1331 | 0.14% | Same sparse setting, but density, freshness, and reliability features are removed. |

## Key Comparisons

- Full dynamic vs static: 88.59s -> 85.22s, 3.81% MAE gain.
- Reliability ablation: full dynamic MAE 85.22s, without reliability MAE 85.11s.
- Sparse dynamic reliability: sparse-30 with reliability MAE 88.41s, without reliability MAE 88.36s.
- Sparse dynamic quality: sparse-30 with quality signals MAE 88.41s, without density/freshness/reliability MAE 88.47s.
