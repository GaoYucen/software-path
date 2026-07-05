# PKDD15 DynaPath-Lite First Result

## Goal

Build a fast first model on PKDD15 Porto without waiting for OSM map matching.

This version uses:

```text
GPS polyline -> grid-cell path tokens -> static path features + dynamic traffic features -> TTE model
```

It is intended as a first-result experiment for the proposed dynamic multimodal path representation paper.

## Dataset

Input:

```text
pkdd-15-predict-taxi-service-trajectory-i/train.csv
```

Processed subset:

```text
first 120,000 rows
```

Cleaning:

- Dropped `MISSING_DATA=True`.
- Dropped empty or too short polylines.
- Kept trajectories inside Porto bounds:

```text
longitude: [-8.75, -8.45]
latitude:  [41.00, 41.30]
```

Output directory:

```text
data/processed/pkdd15_grid_120k_clean
```

Generated summary:

| Item | Value |
|---|---:|
| Raw rows read | 119,999 |
| Dropped out-of-bounds trips | 734 |
| Usable trips | 113,660 |
| Grid cells | 20,227 |
| Max path length | 64 |
| Dynamic history window | 60 minutes |

## Model Construction

### Path Tokens

Because PKDD15 provides GPS points but no road segment IDs, each GPS point is quantized into a grid cell:

```text
(longitude, latitude) -> grid_cell_id
```

Consecutive duplicate cells are collapsed:

```text
[c1, c1, c2, c2, c3] -> [c1, c2, c3]
```

This yields a PATH-LLM-style path sequence:

```text
data_road.npy
```

Important limitation:

```text
These are grid tokens, not OSM road segment IDs.
For final paper experiments, replace grid cells with map-matched road segments.
```

### Target

PKDD15 samples one GPS point every 15 seconds. Travel time is:

```text
trip_time = (len(POLYLINE) - 1) * 15
```

For modeling, `num_points` is **excluded** from the feature set because it directly leaks the label.

### Static Features

The static path features include:

- number of grid cells;
- number of unique grid cells;
- route distance;
- direct origin-destination distance;
- tortuosity;
- start/end coordinates;
- hour and day-of-week encodings;
- call type;
- day type;
- whether origin stand is known.

### Dynamic Features

For each trip departing at time `tau`, dynamic features are computed from prior trips only:

```text
history window = [tau - 60 min, tau)
```

For each grid cell on the path:

- historical median speed;
- speed standard deviation;
- observation count;
- speed ratio against historical baseline;
- freshness;
- reliability.

Trip-level dynamic features are obtained by mean/max pooling along the path.

Reliability is:

```text
reliability = min(1, obs_count / 10) * exp(-freshness_min / 60)
```

This is the first version of the proposed static-dynamic decoupled dynamic path representation.

## Generated Files

Core generated files:

| File | Meaning |
|---|---|
| `trip_features.csv` | static + dynamic tabular features |
| `data_road.npy` | grid-token path sequence |
| `dynamic_path.npy` | per-path, per-cell dynamic attributes |
| `trip_time.npy` | TTE label |
| `row_num.npy` | valid path-token length |
| `departure_time.npy` | Unix departure timestamp |
| `metadata.json` | preprocessing metadata |

Scripts:

```bash
python scripts/prepare_pkdd15_grid_dynamic.py \
  --max-rows 120000 \
  --out-dir data/processed/pkdd15_grid_120k_clean
```

```bash
python scripts/train_pkdd15_dynapath_lite.py \
  --data-dir data/processed/pkdd15_grid_120k_clean \
  --report reports/pkdd15_dynapath_lite_120k_noleak_metrics.json \
  --model-dir reports/pkdd15_dynapath_lite_120k_noleak_models
```

## Training Setup

Chronological split:

| Split | Count |
|---|---:|
| Train | 79,562 |
| Validation | 17,049 |
| Test | 17,049 |

Target statistics:

| Statistic | Seconds |
|---|---:|
| Min | 135 |
| Mean | 670.16 |
| Median | 600 |
| Max | 2,385 |

Metrics:

- MAE
- RMSE
- MAPE
- MARE

## Results

Source:

```text
reports/pkdd15_dynapath_lite_120k_noleak_metrics.json
```

| Model | MAE | RMSE | MAPE | MARE |
|---|---:|---:|---:|---:|
| Train mean | 268.14 | 353.38 | 0.5396 | 0.4036 |
| Ridge static | 97.42 | 152.09 | 0.1580 | 0.1466 |
| Ridge static + dynamic | 91.79 | 146.81 | 0.1473 | 0.1382 |
| HGB static | 88.60 | 140.46 | 0.1417 | 0.1333 |
| HGB DynaPath-Lite | **85.22** | **137.38** | **0.1357** | **0.1283** |
| RF DynaPath-Lite | 87.94 | 140.08 | 0.1404 | 0.1323 |

Dynamic-feature gains:

| Comparison | MAE Change |
|---|---:|
| Ridge static -> Ridge static + dynamic | 97.42 -> 91.79, about 5.8% better |
| HGB static -> HGB DynaPath-Lite | 88.60 -> 85.22, about 3.8% better |

## Interpretation

This first result supports the paper direction:

```text
historical dynamic traffic states improve travel-time estimation beyond static path features
```

The best current model is:

```text
HGB DynaPath-Lite
```

It uses both static path geometry and leakage-aware historical dynamic traffic states.

## Known Limitations

1. Grid cells are not road segments.
2. The model is tabular, not yet a sequence neural model.
3. The experiment uses only the first 120k training rows.
4. Dynamic features are pooled at trip level; the next model should use the full `dynamic_path.npy` sequence.
5. The official PKDD test set is for destination prediction snapshots, not ordinary full-trip TTE evaluation.

## Next Fast Steps

1. Scale from 120k rows to 500k rows.
2. Add an ablation without reliability.
3. Add dynamic sparsity experiments by randomly dropping historical observations.
4. Replace grid cells with H3/geohash or OSM map-matched road IDs.
5. Build a small sequence model when PyTorch is available:

```text
cell embedding + dynamic MLP + reliability gate + Transformer/GRU -> TTE
```

For the paper draft, this result can be used as the first experimental table while the stronger sequence model is being implemented.
