# PKDD15 Quick 5K Fair Comparison

## Setup

- Dataset: `data/processed/pkdd15_quick_5k`
- Fair comparison subset: first `5000` rows from `trip_features.csv`, matching the quick-array tensors
- Split: chronological `70/15/15`
- GPT-2 run: `gpt2`, CPU, `3` epochs, `20` train batches per epoch, `5` eval batches per pass

## Regression Baselines

Source: `reports/pkdd15_quick_5k_regression_fair_metrics.json`

| Model | MAE (s) | RMSE (s) | MAPE | MARE |
|---|---:|---:|---:|---:|
| train_mean | 276.01 | 332.20 | 0.6689 | 0.4617 |
| ridge_static | 103.71 | 134.78 | 0.2050 | 0.1735 |
| ridge_static_dynamic | 114.45 | 145.27 | 0.2305 | 0.1914 |
| hgb_static | **90.55** | 130.30 | **0.1643** | **0.1515** |
| hgb_dynapath_lite | 95.99 | 131.89 | 0.1832 | 0.1606 |
| rf_dynapath_lite | 93.89 | **129.04** | 0.1730 | 0.1571 |

## GPT-2 Engineering Version

Source: `reports/dynapath_llm_quick_budgeted/metrics.json`

| Model | MAE (s) | RMSE (s) | MAPE | MARE |
|---|---:|---:|---:|---:|
| DynaPathLLM-GPT2 budgeted | 564.43 | 626.12 | 0.9671 | 0.9730 |

## Takeaways

- On this 5K fair subset, the current GPT-2 engineering version is much worse than the simple regression baselines.
- The best regression MAE is `90.55s` (`hgb_static`), while the GPT-2 engineering version reaches `564.43s`.
- Dynamic tabular features do not help on this 5K subset as clearly as they do on the 120K full experiment.
- The GPT-2 result is still useful as an engineering validation because it proves the text-modality path and GPT-2 backbone run end-to-end on real data.

## Main Caveats

- The text modality is generated pseudo-semantic text from grid-token behavior, not real OSM road text.
- The GPT-2 result above is a budgeted CPU run, not a full unconstrained training run.
- `data/processed/pkdd15_quick_5k/trip_features.csv` still contains the full `113660` rows, while the array tensors contain only the first `5000`; therefore the fair regression comparison must explicitly subset to the first `5000` rows.
