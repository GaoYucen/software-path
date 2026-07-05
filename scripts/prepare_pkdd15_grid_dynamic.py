#!/usr/bin/env python3
"""Prepare a fast PKDD15 Porto grid-token dataset with dynamic attributes.

PKDD15 provides GPS polylines but not map-matched road segment IDs. For a fast
first result, this script quantizes GPS points into grid cells and treats cells
as path tokens. Dynamic attributes are computed chronologically from prior
trips only, so the generated features avoid future leakage.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ParsedTrip:
    trip_id: str
    timestamp: int
    call_type: str
    day_type: str
    taxi_id: int
    origin_stand: float
    cells: list[int]
    points: list[tuple[float, float]]
    segment_speeds: list[tuple[int, float]]
    distance_m: float
    duration_s: float


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(a))


def fast_polyline(polyline: str) -> list[tuple[float, float]]:
    """Parse a PKDD polyline without importing json for speed."""
    if not isinstance(polyline, str) or len(polyline) < 5 or polyline == "[]":
        return []
    nums = np.fromstring(
        polyline.replace("[", " ").replace("]", " ").replace(",", " "),
        sep=" ",
        dtype=np.float64,
    )
    if len(nums) < 4 or len(nums) % 2 != 0:
        return []
    coords = nums.reshape(-1, 2)
    return [(float(lon), float(lat)) for lon, lat in coords]


def cell_for_point(
    lon: float, lat: float, lon0: float, lat0: float, grid_deg: float
) -> tuple[int, int]:
    return (
        int(math.floor((lon - lon0) / grid_deg)),
        int(math.floor((lat - lat0) / grid_deg)),
    )


def dynamic_for_cell(
    history: dict[int, deque[tuple[int, float]]],
    baseline_speed: dict[int, float],
    cell: int,
    timestamp: int,
    history_seconds: int,
) -> tuple[list[float], dict[str, float]]:
    cutoff = timestamp - history_seconds
    queue = history.get(cell, deque())
    obs = [speed for ts, speed in queue if cutoff <= ts < timestamp and speed > 0]
    baseline = baseline_speed.get(cell, 4.0)
    if obs:
        speed_median = float(np.median(obs))
        speed_std = float(np.std(obs))
        obs_count = len(obs)
        latest_ts = max(ts for ts, _ in queue if cutoff <= ts < timestamp)
        freshness_min = (timestamp - latest_ts) / 60.0
    else:
        speed_median = float(baseline)
        speed_std = 0.0
        obs_count = 0
        freshness_min = history_seconds / 60.0 * 2.0
    speed_ratio = speed_median / max(float(baseline), 1e-6)
    reliability = min(1.0, obs_count / 10.0) * math.exp(-freshness_min / 60.0)
    values = [
        speed_median,
        speed_std,
        math.log1p(obs_count),
        speed_ratio,
        freshness_min,
        reliability,
    ]
    named = {
        "speed_median": speed_median,
        "speed_std": speed_std,
        "obs_count": float(obs_count),
        "speed_ratio": speed_ratio,
        "freshness_min": freshness_min,
        "reliability": reliability,
    }
    return values, named


def static_features(trip: ParsedTrip) -> dict[str, float]:
    points = trip.points
    start_lon, start_lat = points[0]
    end_lon, end_lat = points[-1]
    direct_m = haversine_m(start_lon, start_lat, end_lon, end_lat)
    hour = (trip.timestamp // 3600) % 24
    dow = (trip.timestamp // 86400 + 3) % 7
    out = {
        "num_points": float(len(points)),
        "num_cells": float(len(trip.cells)),
        "unique_cells": float(len(set(trip.cells))),
        "distance_m": float(trip.distance_m),
        "direct_m": float(direct_m),
        "tortuosity": float(trip.distance_m / max(direct_m, 1.0)),
        "start_lon": start_lon,
        "start_lat": start_lat,
        "end_lon": end_lon,
        "end_lat": end_lat,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "origin_stand_known": float(not np.isnan(trip.origin_stand)),
        "call_A": float(trip.call_type == "A"),
        "call_B": float(trip.call_type == "B"),
        "call_C": float(trip.call_type == "C"),
        "day_A": float(trip.day_type == "A"),
        "day_B": float(trip.day_type == "B"),
        "day_C": float(trip.day_type == "C"),
    }
    return out


def parse_rows(
    csv_path: Path,
    max_rows: int,
    min_points: int,
    max_points: int,
    grid_deg: float,
    max_segment_speed: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[list[ParsedTrip], dict]:
    usecols = [
        "TRIP_ID",
        "CALL_TYPE",
        "ORIGIN_STAND",
        "TAXI_ID",
        "TIMESTAMP",
        "DAY_TYPE",
        "MISSING_DATA",
        "POLYLINE",
    ]
    raw = pd.read_csv(csv_path, usecols=usecols, nrows=max_rows)
    raw = raw[~raw["MISSING_DATA"]].copy()

    all_lons: list[float] = []
    all_lats: list[float] = []
    parsed_points: list[list[tuple[float, float]]] = []
    dropped_out_of_bounds = 0
    for poly in raw["POLYLINE"]:
        pts = fast_polyline(poly)
        if pts and any(
            lon < lon_min or lon > lon_max or lat < lat_min or lat > lat_max
            for lon, lat in pts
        ):
            dropped_out_of_bounds += 1
            pts = []
        parsed_points.append(pts)
        if min_points <= len(pts) <= max_points:
            all_lons.extend(lon for lon, _ in pts)
            all_lats.extend(lat for _, lat in pts)
    if not all_lons:
        raise RuntimeError("No usable polylines found")
    lon0 = min(all_lons)
    lat0 = min(all_lats)

    cell_lookup: dict[tuple[int, int], int] = {}
    trips: list[ParsedTrip] = []
    for row, points in zip(raw.itertuples(index=False), parsed_points):
        if len(points) < min_points or len(points) > max_points:
            continue
        cells: list[int] = []
        for lon, lat in points:
            key = cell_for_point(lon, lat, lon0, lat0, grid_deg)
            if key not in cell_lookup:
                cell_lookup[key] = len(cell_lookup)
            cell = cell_lookup[key]
            if not cells or cells[-1] != cell:
                cells.append(cell)
        if len(cells) < 2:
            continue

        distance_m = 0.0
        segment_speeds: list[tuple[int, float]] = []
        for i, ((lon1, lat1), (lon2, lat2)) in enumerate(zip(points[:-1], points[1:])):
            dist = haversine_m(lon1, lat1, lon2, lat2)
            speed = dist / 15.0
            if 0 < speed <= max_segment_speed:
                distance_m += dist
                start_cell = cell_lookup[cell_for_point(lon1, lat1, lon0, lat0, grid_deg)]
                segment_speeds.append((start_cell, speed))
        if not segment_speeds:
            continue
        duration_s = float((len(points) - 1) * 15)
        trips.append(
            ParsedTrip(
                trip_id=str(row.TRIP_ID),
                timestamp=int(row.TIMESTAMP),
                call_type=str(row.CALL_TYPE),
                day_type=str(row.DAY_TYPE),
                taxi_id=int(row.TAXI_ID),
                origin_stand=float(row.ORIGIN_STAND),
                cells=cells,
                points=points,
                segment_speeds=segment_speeds,
                distance_m=distance_m,
                duration_s=duration_s,
            )
        )
    trips.sort(key=lambda t: t.timestamp)
    meta = {
        "raw_rows": int(len(raw)),
        "dropped_out_of_bounds": int(dropped_out_of_bounds),
        "usable_trips": int(len(trips)),
        "num_grid_cells": int(len(cell_lookup)),
        "lon0": float(lon0),
        "lat0": float(lat0),
        "grid_deg": grid_deg,
    }
    return trips, meta


def compute_baseline_speeds(trips: list[ParsedTrip]) -> dict[int, float]:
    speeds: dict[int, list[float]] = defaultdict(list)
    for trip in trips:
        for cell, speed in trip.segment_speeds:
            speeds[cell].append(speed)
    return {cell: float(np.median(vals)) for cell, vals in speeds.items()}


def build_dataset(
    trips: list[ParsedTrip],
    meta: dict,
    max_len: int,
    history_minutes: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    history_seconds = history_minutes * 60
    baseline_speed = compute_baseline_speeds(trips)
    history: dict[int, deque[tuple[int, float]]] = defaultdict(deque)
    feature_rows: list[dict[str, float | str | int]] = []
    dynamic_rows: list[dict[str, float | int | str]] = []

    pad_id = int(meta["num_grid_cells"])
    data_road = np.full((len(trips), max_len), pad_id, dtype=np.int64)
    dynamic_path = np.zeros((len(trips), max_len, 6), dtype=np.float32)
    row_num = np.zeros(len(trips), dtype=np.int64)
    trip_time = np.zeros(len(trips), dtype=np.float32)
    departure_time = np.zeros(len(trips), dtype=np.int64)

    for idx, trip in enumerate(trips):
        cells = trip.cells[:max_len]
        row_num[idx] = len(cells)
        trip_time[idx] = trip.duration_s
        departure_time[idx] = trip.timestamp
        data_road[idx, : len(cells)] = np.array(cells, dtype=np.int64)

        per_cell_features = []
        per_cell_named = []
        for pos, cell in enumerate(cells):
            dyn_values, dyn_named = dynamic_for_cell(
                history, baseline_speed, cell, trip.timestamp, history_seconds
            )
            dynamic_path[idx, pos] = np.array(dyn_values, dtype=np.float32)
            per_cell_features.append(dyn_values)
            per_cell_named.append(dyn_named)
            dynamic_rows.append(
                {
                    "trip_index": idx,
                    "position": pos,
                    "cell_id": cell,
                    "timestamp": trip.timestamp,
                    **dyn_named,
                }
            )
        dyn = np.array(per_cell_features, dtype=float)
        dyn_mean = dyn.mean(axis=0)
        dyn_max = dyn.max(axis=0)
        feat = static_features(trip)
        names = [
            "dyn_speed_median",
            "dyn_speed_std",
            "dyn_log_obs",
            "dyn_speed_ratio",
            "dyn_freshness_min",
            "dyn_reliability",
        ]
        for i, name in enumerate(names):
            feat[f"{name}_mean"] = float(dyn_mean[i])
            feat[f"{name}_max"] = float(dyn_max[i])
        feat.update(
            {
                "trip_index": idx,
                "trip_id": trip.trip_id,
                "timestamp": trip.timestamp,
                "target_trip_time": trip.duration_s,
            }
        )
        feature_rows.append(feat)

        for cell, speed in trip.segment_speeds:
            history[cell].append((trip.timestamp, speed))
            cutoff = trip.timestamp - history_seconds * 4
            while history[cell] and history[cell][0][0] < cutoff:
                history[cell].popleft()

    arrays = {
        "data_road": data_road,
        "dynamic_path": dynamic_path,
        "row_num": row_num,
        "trip_time": trip_time,
        "departure_time": departure_time,
        "pad_id": np.array([pad_id], dtype=np.int64),
        "road_size": np.array([pad_id], dtype=np.int64),
    }
    return pd.DataFrame(feature_rows), arrays, pd.DataFrame(dynamic_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default="pkdd-15-predict-taxi-service-trajectory-i/train.csv",
    )
    parser.add_argument("--out-dir", default="data/processed/pkdd15_grid")
    parser.add_argument("--max-rows", type=int, default=120_000)
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=160)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--grid-deg", type=float, default=0.001)
    parser.add_argument("--history-minutes", type=int, default=60)
    parser.add_argument("--max-segment-speed", type=float, default=60.0)
    parser.add_argument("--lon-min", type=float, default=-8.75)
    parser.add_argument("--lon-max", type=float, default=-8.45)
    parser.add_argument("--lat-min", type=float, default=41.00)
    parser.add_argument("--lat-max", type=float, default=41.30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trips, meta = parse_rows(
        Path(args.train_csv),
        args.max_rows,
        args.min_points,
        args.max_points,
        args.grid_deg,
        args.max_segment_speed,
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )
    features, arrays, dynamic_rows = build_dataset(
        trips, meta, args.max_len, args.history_minutes
    )

    features.to_csv(out_dir / "trip_features.csv", index=False)
    dynamic_rows.to_csv(out_dir / "dynamic_cell_features.csv", index=False)
    for name, value in arrays.items():
        np.save(out_dir / f"{name}.npy", value)

    meta.update(
        {
            "train_csv": args.train_csv,
            "max_rows": args.max_rows,
            "min_points": args.min_points,
            "max_points": args.max_points,
            "max_len": args.max_len,
            "history_minutes": args.history_minutes,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "dynamic_dim": 6,
            "note": (
                "Grid cells are fast path tokens for first-result experiments. "
                "Use OSM map matching for final road-segment experiments."
            ),
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote PKDD15 grid dataset to {out_dir}")


if __name__ == "__main__":
    main()
