#!/usr/bin/env python3
"""Build PATH-LLM-style smoke-test arrays from a small GeoLife demo CSV.

The demo data has raw GPS points with timestamps but no map matching. For a
fast pipeline smoke test, we quantize points into spatial grid cells and use
each cell as a pseudo road segment. This is not a replacement for real map
matching; it only validates the data and dynamic-feature pipeline.
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


RAW_URL = (
    "https://raw.githubusercontent.com/movingpandas/movingpandas/main/"
    "tutorials/data/demodata_geolife.csv"
)


@dataclass
class Segment:
    traj_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    edge_id: int
    distance_m: float
    duration_s: float
    speed_mps: float


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


def parse_points(raw_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv, sep=";")
    expected = {"X", "Y", "trajectory_id", "sequence", "t"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {raw_csv}: {sorted(missing)}")

    df = df.rename(
        columns={
            "X": "lon",
            "Y": "lat",
            "trajectory_id": "traj_id",
            "t": "timestamp",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values(["traj_id", "timestamp", "sequence"]).reset_index(drop=True)
    return df[["traj_id", "sequence", "lon", "lat", "timestamp"]]


def assign_grid_edges(points: pd.DataFrame, grid_deg: float) -> tuple[pd.DataFrame, dict]:
    lat0 = float(points["lat"].min())
    lon0 = float(points["lon"].min())
    cell_keys = []
    for row in points.itertuples(index=False):
        gx = int(math.floor((row.lon - lon0) / grid_deg))
        gy = int(math.floor((row.lat - lat0) / grid_deg))
        cell_keys.append((gx, gy))

    edge_lookup: dict[tuple[int, int], int] = {}
    edge_ids = []
    for key in cell_keys:
        if key not in edge_lookup:
            edge_lookup[key] = len(edge_lookup)
        edge_ids.append(edge_lookup[key])

    points = points.copy()
    points["edge_id"] = edge_ids
    meta = {
        "grid_deg": grid_deg,
        "lat0": lat0,
        "lon0": lon0,
        "num_edges": len(edge_lookup),
    }
    return points, meta


def build_segments(points: pd.DataFrame) -> list[Segment]:
    segments: list[Segment] = []
    for traj_id, group in points.groupby("traj_id", sort=True):
        group = group.sort_values("timestamp")
        rows = list(group.itertuples(index=False))
        for cur, nxt in zip(rows[:-1], rows[1:]):
            duration_s = (nxt.timestamp - cur.timestamp).total_seconds()
            if duration_s <= 0:
                continue
            distance_m = haversine_m(cur.lon, cur.lat, nxt.lon, nxt.lat)
            speed_mps = distance_m / duration_s
            if not np.isfinite(speed_mps) or speed_mps <= 0 or speed_mps > 60:
                continue
            segments.append(
                Segment(
                    traj_id=int(traj_id),
                    start_time=cur.timestamp,
                    end_time=nxt.timestamp,
                    edge_id=int(cur.edge_id),
                    distance_m=float(distance_m),
                    duration_s=float(duration_s),
                    speed_mps=float(speed_mps),
                )
            )
    return segments


def make_trip_windows(
    segments: list[Segment], min_len: int, max_len: int, stride: int
) -> list[list[Segment]]:
    trips: list[list[Segment]] = []
    by_traj: dict[int, list[Segment]] = defaultdict(list)
    for seg in segments:
        by_traj[seg.traj_id].append(seg)

    for traj_id in sorted(by_traj):
        seq = sorted(by_traj[traj_id], key=lambda s: s.start_time)
        if len(seq) < min_len:
            continue
        for start in range(0, len(seq) - min_len + 1, stride):
            window = seq[start : min(start + max_len, len(seq))]
            if len(window) >= min_len:
                trips.append(window)
    trips.sort(key=lambda w: w[0].start_time)
    return trips


def edge_baselines(segments: list[Segment]) -> dict[int, float]:
    speeds: dict[int, list[float]] = defaultdict(list)
    for seg in segments:
        speeds[seg.edge_id].append(seg.speed_mps)
    return {
        edge_id: float(np.median(vals))
        for edge_id, vals in speeds.items()
        if len(vals) > 0
    }


def dynamic_feature_for_edge(
    history: dict[int, deque[tuple[pd.Timestamp, float]]],
    baselines: dict[int, float],
    edge_id: int,
    depart_time: pd.Timestamp,
    history_minutes: int,
) -> tuple[list[float], dict]:
    cutoff = depart_time - pd.Timedelta(minutes=history_minutes)
    obs = [
        speed
        for ts, speed in history.get(edge_id, [])
        if cutoff <= ts < depart_time and speed > 0
    ]
    baseline = baselines.get(edge_id, 1.0)
    if obs:
        speed_median = float(np.median(obs))
        speed_std = float(np.std(obs))
        obs_count = len(obs)
        latest_ts = max(ts for ts, _ in history[edge_id] if cutoff <= ts < depart_time)
        freshness = float((depart_time - latest_ts).total_seconds())
    else:
        speed_median = float(baseline)
        speed_std = 0.0
        obs_count = 0
        freshness = float(history_minutes * 60 * 2)
    speed_ratio = speed_median / max(float(baseline), 1e-6)
    feature = [
        speed_median,
        speed_std,
        math.log1p(obs_count),
        speed_ratio,
        freshness / 60.0,
    ]
    row = {
        "edge_id": edge_id,
        "slot_start": depart_time.floor("15min").isoformat(),
        "speed_median": speed_median,
        "speed_std": speed_std,
        "obs_count": obs_count,
        "speed_ratio": speed_ratio,
        "freshness_min": freshness / 60.0,
    }
    return feature, row


def build_arrays(
    trips: list[list[Segment]],
    baselines: dict[int, float],
    max_len: int,
    history_minutes: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    road_size = max(baselines) + 1
    pad_id = road_size
    history: dict[int, deque[tuple[pd.Timestamp, float]]] = defaultdict(deque)

    data_road = np.full((len(trips), max_len), pad_id, dtype=np.int64)
    dynamic_path = np.zeros((len(trips), max_len, 5), dtype=np.float32)
    row_num = np.zeros(len(trips), dtype=np.int64)
    trip_time = np.zeros(len(trips), dtype=np.float32)
    departure_time = np.zeros(len(trips), dtype=np.int64)
    dynamic_rows: list[dict] = []

    for i, trip in enumerate(trips):
        depart = trip[0].start_time
        valid_len = min(len(trip), max_len)
        row_num[i] = valid_len
        trip_time[i] = float(sum(seg.duration_s for seg in trip[:valid_len]))
        departure_time[i] = int(depart.timestamp())
        for j, seg in enumerate(trip[:valid_len]):
            data_road[i, j] = seg.edge_id
            feat, dyn_row = dynamic_feature_for_edge(
                history, baselines, seg.edge_id, depart, history_minutes
            )
            dynamic_path[i, j] = np.array(feat, dtype=np.float32)
            dyn_row["trip_index"] = i
            dyn_row["position"] = j
            dynamic_rows.append(dyn_row)

        for seg in trip:
            history[seg.edge_id].append((seg.end_time, seg.speed_mps))
            cutoff = seg.end_time - pd.Timedelta(minutes=history_minutes * 4)
            while history[seg.edge_id] and history[seg.edge_id][0][0] < cutoff:
                history[seg.edge_id].popleft()

    arrays = {
        "data_road": data_road,
        "dynamic_path": dynamic_path,
        "row_num": row_num,
        "trip_time": trip_time,
        "departure_time": departure_time,
        "road_size": np.array([road_size], dtype=np.int64),
        "pad_id": np.array([pad_id], dtype=np.int64),
    }
    return arrays, pd.DataFrame(dynamic_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", default="data/raw/demodata_geolife.csv")
    parser.add_argument("--out-dir", default="data/processed/geolife_smoke")
    parser.add_argument("--grid-deg", type=float, default=0.001)
    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--history-minutes", type=int, default=30)
    args = parser.parse_args()

    raw_csv = Path(args.raw_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = parse_points(raw_csv)
    points, grid_meta = assign_grid_edges(points, args.grid_deg)
    segments = build_segments(points)
    trips = make_trip_windows(segments, args.min_len, args.max_len, args.stride)
    if len(trips) < 10:
        raise RuntimeError(f"Too few trip windows generated: {len(trips)}")
    baselines = edge_baselines(segments)
    arrays, dynamic_rows = build_arrays(
        trips, baselines, args.max_len, args.history_minutes
    )

    for name, value in arrays.items():
        np.save(out_dir / f"{name}.npy", value)
    points.to_csv(out_dir / "points_with_pseudo_edges.csv", index=False)
    dynamic_rows.to_csv(out_dir / "dynamic_edge_slot.csv", index=False)
    meta = {
        "raw_url": RAW_URL,
        "raw_csv": str(raw_csv),
        "num_points": int(len(points)),
        "num_segments": int(len(segments)),
        "num_trips": int(len(trips)),
        "max_len": int(args.max_len),
        "min_len": int(args.min_len),
        "stride": int(args.stride),
        "history_minutes": int(args.history_minutes),
        **grid_meta,
        "note": (
            "Grid cells are pseudo road segments for smoke testing only; "
            "replace them with map-matched road IDs for paper experiments."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"Wrote arrays to {out_dir}")


if __name__ == "__main__":
    main()
