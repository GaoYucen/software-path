# GeoLife Dynamic Path Smoke Test Report

## Dataset

Downloaded a small public GeoLife demo dataset from the MovingPandas repository:

- Raw CSV: `data/raw/demodata_geolife.csv`
- Source note: `data/raw/demodata_geolife.README`
- Source URL: <https://raw.githubusercontent.com/movingpandas/movingpandas/main/tutorials/data/demodata_geolife.csv>

The README states that the data comes from the Microsoft Research Asia GeoLife project.

This is a **pipeline smoke-test dataset**, not a final paper dataset. It is small and does not include map-matched road IDs.

## Conversion Method

Because the demo CSV contains GPS points but no map matching, points were quantized into spatial grid cells. Each grid cell is treated as a pseudo road segment.

Important caveat:

```text
Grid cells are pseudo road segments for smoke testing only.
For paper experiments, replace them with real map-matched road IDs.
```

Generated with:

```bash
python scripts/prepare_geolife_smoke.py
```

## Generated Attributes

The script builds PATH-LLM-style trip arrays:

- `data_road.npy`: pseudo edge sequence
- `trip_time.npy`: trip duration in seconds
- `row_num.npy`: valid path length
- `departure_time.npy`: trip departure Unix timestamp
- `dynamic_path.npy`: per-trip, per-edge dynamic attributes
- `dynamic_edge_slot.csv`: long-form dynamic attribute table

Dynamic attributes:

- `speed_median`
- `speed_std`
- `log(1 + obs_count)`
- `speed_ratio`
- `freshness_min`

Leakage control:

```text
For each trip departing at tau, dynamic features are computed only from observations before tau.
```

## Data Summary

From `data/processed/geolife_smoke/metadata.json`:

| Item | Value |
|---|---:|
| GPS points | 5,908 |
| Valid point-to-point segments | 5,830 |
| Generated trip windows | 1,159 |
| Pseudo edges | 611 |
| Max path length | 32 |
| History window | 30 minutes |
| Grid size | 0.001 degrees |

Array shapes:

| File | Shape |
|---|---|
| `data_road.npy` | `(1159, 32)` |
| `dynamic_path.npy` | `(1159, 32, 5)` |
| `trip_time.npy` | `(1159,)` |
| `row_num.npy` | `(1159,)` |
| `departure_time.npy` | `(1159,)` |

## Smoke Test

Run command:

```bash
python scripts/smoke_test_dynamic.py
```

The smoke test uses chronological splitting:

| Split | Count |
|---|---:|
| Train | 811 |
| Validation | 173 |
| Test | 175 |

Target travel time statistics:

| Statistic | Seconds |
|---|---:|
| Min | 26.0 |
| Mean | 359.14 |
| Max | 19,825.0 |

Metrics from `reports/geolife_smoke_metrics.json`:

| Model | MAE | RMSE | MAPE | MARE |
|---|---:|---:|---:|---:|
| Train mean baseline | 588.18 | 1859.54 | 1.4310 | 1.0226 |
| Ridge static | 2938.33 | 4025.76 | 16.1226 | 5.1087 |
| Ridge static + dynamic | 1834.65 | 2739.55 | 9.5245 | 3.1898 |
| Random forest static + dynamic | 1077.36 | 1902.68 | 4.9186 | 1.8731 |

Status:

```text
PASS
```

The test proves that:

- the downloaded GPS data can be parsed;
- trajectory timestamps can be used to build leakage-aware dynamic attributes;
- PATH-LLM-style arrays can be generated;
- dynamic path tensors can be loaded into a travel-time-estimation pipeline.

The test does **not** prove final model quality.

## Next Step For Paper Experiments

Replace pseudo grid-cell edges with real map-matched road IDs, then rerun the same pipeline on a larger timestamped taxi trajectory dataset.

Minimum required real-data fields:

```text
trip_id
edge_id sequence
departure_time
travel_time
```

Better fields:

```text
trip_id
edge_id
edge_enter_time
edge_leave_time
```

With those, `dynamic_path.npy` becomes a real dynamic traffic-state tensor rather than a pseudo-grid smoke-test tensor.
