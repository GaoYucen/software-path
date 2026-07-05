#!/usr/bin/env python3
"""Prepare an OSM edge-level PKDD15 dataset with approximate map matching.

This script is the practical upgrade path from grid tokens to real OSM edges.
It downloads or loads a cached OSM drive graph for the Porto area, snaps each
GPS point to the nearest graph node, bridges consecutive snapped nodes through
shortest paths, converts the resulting node path into edge tokens, and then
builds dynamic edge features chronologically from past trips only.

The matching strategy is intentionally lightweight and fully local:

1. build/load an OSMnx drive graph for the bounding box;
2. snap every GPS point to the nearest graph node;
3. stitch node-to-node shortest paths to form a continuous route;
4. map route edges to integer edge ids and export edge metadata + text.

This is not a full HMM map matcher, but it is sufficient to produce usable
edge-level data, real OSM attributes, and text modality artifacts for the
current DynaPath pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class ParsedTrip:
    trip_id: str
    timestamp: int
    call_type: str
    day_type: str
    taxi_id: int
    origin_stand: float
    points: list[tuple[float, float]]
    duration_s: float


@dataclass
class MatchedTrip:
    trip: ParsedTrip
    edge_ids: list[int]
    edge_lengths_m: list[float]
    edge_speeds_mps: list[float]


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


def _first_value(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(v) for v in value if v not in (None, ""))
    return str(value)


def stable_index(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    idx = int.from_bytes(digest[:8], "little") % dim
    sign = 1.0 if digest[8] % 2 == 0 else -1.0
    return idx, sign


def encode_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    words = TOKEN_RE.findall(text.lower())
    if not words:
        return vec
    feats = list(words)
    feats.extend(f"{a}__{b}" for a, b in zip(words[:-1], words[1:]))
    for token in feats:
        idx, sign = stable_index(token, dim)
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def dynamic_for_edge(
    history: dict[int, deque[tuple[int, float]]],
    baseline_speed: dict[int, float],
    edge_id: int,
    timestamp: int,
    history_seconds: int,
) -> tuple[list[float], dict[str, float]]:
    cutoff = timestamp - history_seconds
    queue = history.get(edge_id, deque())
    obs = [speed for ts, speed in queue if cutoff <= ts < timestamp and speed > 0]
    baseline = baseline_speed.get(edge_id, 4.0)
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


def static_features(
    trip: ParsedTrip,
    edge_ids: list[int],
    edge_lengths_m: list[float],
) -> dict[str, float]:
    points = trip.points
    start_lon, start_lat = points[0]
    end_lon, end_lat = points[-1]
    direct_m = haversine_m(start_lon, start_lat, end_lon, end_lat)
    distance_m = float(sum(edge_lengths_m))
    hour = (trip.timestamp // 3600) % 24
    dow = (trip.timestamp // 86400 + 3) % 7
    out = {
        "num_points": float(len(points)),
        "num_edges": float(len(edge_ids)),
        "unique_edges": float(len(set(edge_ids))),
        "distance_m": distance_m,
        "direct_m": float(direct_m),
        "tortuosity": float(distance_m / max(direct_m, 1.0)),
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
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[list[ParsedTrip], dict[str, Any]]:
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

    trips: list[ParsedTrip] = []
    dropped_out_of_bounds = 0
    for row in raw.itertuples(index=False):
        points = fast_polyline(row.POLYLINE)
        if not (min_points <= len(points) <= max_points):
            continue
        if any(
            lon < lon_min or lon > lon_max or lat < lat_min or lat > lat_max
            for lon, lat in points
        ):
            dropped_out_of_bounds += 1
            continue
        trips.append(
            ParsedTrip(
                trip_id=str(row.TRIP_ID),
                timestamp=int(row.TIMESTAMP),
                call_type=str(row.CALL_TYPE),
                day_type=str(row.DAY_TYPE),
                taxi_id=int(row.TAXI_ID),
                origin_stand=float(row.ORIGIN_STAND),
                points=points,
                duration_s=float((len(points) - 1) * 15),
            )
        )
    trips.sort(key=lambda t: t.timestamp)
    meta = {
        "raw_rows": int(len(raw)),
        "usable_trips": int(len(trips)),
        "dropped_out_of_bounds": int(dropped_out_of_bounds),
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
    }
    return trips, meta


def load_or_download_graph(
    graph_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
):
    import osmnx as ox

    if graph_path.exists():
        graph = ox.load_graphml(graph_path)
    else:
        graph = ox.graph_from_bbox(
            (lon_min, lat_min, lon_max, lat_max),
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
        )
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, graph_path)
    return graph


def _choose_edge_data(graph, u: int, v: int) -> tuple[int, dict[str, Any]]:
    data = graph.get_edge_data(u, v)
    if data is None:
        raise KeyError(f"Missing edge data for ({u}, {v})")
    if "length" in data:
        return 0, data
    key, best = min(data.items(), key=lambda item: float(item[1].get("length", 1e18)))
    return int(key), best


def _append_edge(
    edge_tokens: list[tuple[int, int, int]],
    edge_lengths: list[float],
    graph,
    u: int,
    v: int,
) -> None:
    key, attrs = _choose_edge_data(graph, u, v)
    token = (int(u), int(v), key)
    length = float(attrs.get("length", 0.0))
    if length <= 0:
        return
    if edge_tokens and edge_tokens[-1] == token:
        edge_lengths[-1] += length
    else:
        edge_tokens.append(token)
        edge_lengths.append(length)


def match_trip_to_edges(
    trip: ParsedTrip,
    graph,
    graph_undirected,
    *,
    max_snap_jump_m: float,
    max_bridge_factor: float,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    import osmnx as ox

    lons = np.array([p[0] for p in trip.points], dtype=float)
    lats = np.array([p[1] for p in trip.points], dtype=float)
    nodes = ox.distance.nearest_nodes(graph, X=lons, Y=lats)
    if isinstance(nodes, (np.integer, int)):
        nodes = [int(nodes)]
    else:
        nodes = [int(n) for n in nodes]

    dedup_nodes: list[int] = []
    dedup_points: list[tuple[float, float]] = []
    for node, point in zip(nodes, trip.points):
        if not dedup_nodes or dedup_nodes[-1] != node:
            dedup_nodes.append(node)
            dedup_points.append(point)

    if len(dedup_nodes) < 2:
        return [], []

    edge_tokens: list[tuple[int, int, int]] = []
    edge_lengths: list[float] = []
    for idx in range(len(dedup_nodes) - 1):
        src = dedup_nodes[idx]
        dst = dedup_nodes[idx + 1]
        if src == dst:
            continue

        point_dist = haversine_m(
            dedup_points[idx][0],
            dedup_points[idx][1],
            dedup_points[idx + 1][0],
            dedup_points[idx + 1][1],
        )
        if point_dist > max_snap_jump_m:
            continue

        path_nodes: list[int] | None = None
        try:
            path_nodes = nx.shortest_path(graph, src, dst, weight="length")
        except Exception:
            try:
                path_nodes = nx.shortest_path(graph_undirected, src, dst, weight="length")
            except Exception:
                path_nodes = None
        if not path_nodes or len(path_nodes) < 2:
            continue

        path_len = 0.0
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            try:
                _, attrs = _choose_edge_data(graph if graph.has_edge(u, v) else graph_undirected, u, v)
            except Exception:
                continue
            path_len += float(attrs.get("length", 0.0))

        if path_len <= 0:
            continue
        if path_len > max_bridge_factor * max(point_dist, 15.0):
            continue

        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            if graph.has_edge(u, v):
                _append_edge(edge_tokens, edge_lengths, graph, u, v)
            elif graph_undirected.has_edge(u, v):
                _append_edge(edge_tokens, edge_lengths, graph_undirected, u, v)
            elif graph_undirected.has_edge(v, u):
                _append_edge(edge_tokens, edge_lengths, graph_undirected, v, u)

    return edge_tokens, edge_lengths


def build_edge_vocabulary(graph) -> tuple[dict[tuple[int, int, int], int], list[dict[str, Any]], list[str]]:
    edge_to_id: dict[tuple[int, int, int], int] = {}
    edge_records: list[dict[str, Any]] = []
    edge_texts: list[str] = []
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        token = (int(u), int(v), int(key))
        edge_id = len(edge_to_id)
        edge_to_id[token] = edge_id
        name = _normalize_text_value(_first_value(attrs.get("name")))
        ref = _normalize_text_value(_first_value(attrs.get("ref")))
        highway = _normalize_text_value(_first_value(attrs.get("highway")))
        oneway = _normalize_text_value(_first_value(attrs.get("oneway")))
        lanes = _normalize_text_value(_first_value(attrs.get("lanes")))
        maxspeed = _normalize_text_value(_first_value(attrs.get("maxspeed")))
        bridge = _normalize_text_value(_first_value(attrs.get("bridge")))
        tunnel = _normalize_text_value(_first_value(attrs.get("tunnel")))
        length = float(attrs.get("length", 0.0))
        osmid = _normalize_text_value(_first_value(attrs.get("osmid")))
        text = (
            f"osm edge {edge_id}. highway {highway or 'unknown'}. "
            f"name {name or 'unknown'}. ref {ref or 'unknown'}. "
            f"oneway {oneway or 'unknown'}. lanes {lanes or 'unknown'}. "
            f"maxspeed {maxspeed or 'unknown'}. bridge {bridge or 'no'}. "
            f"tunnel {tunnel or 'no'}. length_m {length:.1f}."
        )
        edge_records.append(
            {
                "edge_id": edge_id,
                "u": int(u),
                "v": int(v),
                "key": int(key),
                "osmid": osmid,
                "name": name,
                "ref": ref,
                "highway": highway,
                "oneway": oneway,
                "lanes": lanes,
                "maxspeed": maxspeed,
                "bridge": bridge,
                "tunnel": tunnel,
                "length_m": length,
                "geometry_wkt": attrs.get("geometry").wkt if attrs.get("geometry") is not None else "",
            }
        )
        edge_texts.append(text)
    return edge_to_id, edge_records, edge_texts


def match_all_trips(
    trips: list[ParsedTrip],
    graph,
    edge_to_id: dict[tuple[int, int, int], int],
    *,
    max_len: int,
    max_snap_jump_m: float,
    max_bridge_factor: float,
) -> tuple[list[MatchedTrip], dict[str, Any]]:
    graph_undirected = graph.to_undirected()
    matched: list[MatchedTrip] = []
    dropped_no_path = 0
    dropped_too_short = 0
    for trip in trips:
        edge_tokens, edge_lengths = match_trip_to_edges(
            trip,
            graph,
            graph_undirected,
            max_snap_jump_m=max_snap_jump_m,
            max_bridge_factor=max_bridge_factor,
        )
        if len(edge_tokens) < 2:
            dropped_too_short += 1
            continue
        edge_ids = [edge_to_id[token] for token in edge_tokens if token in edge_to_id][:max_len]
        edge_lengths = edge_lengths[: len(edge_ids)]
        if len(edge_ids) < 2:
            dropped_no_path += 1
            continue
        total_len = float(sum(edge_lengths))
        if total_len <= 0:
            dropped_no_path += 1
            continue
        duration = max(float(trip.duration_s), 1.0)
        edge_speeds = [
            length / max(duration * (length / total_len), 1e-6)
            for length in edge_lengths
        ]
        matched.append(
            MatchedTrip(
                trip=trip,
                edge_ids=edge_ids,
                edge_lengths_m=edge_lengths,
                edge_speeds_mps=edge_speeds,
            )
        )
    meta = {
        "matched_trips": int(len(matched)),
        "dropped_too_short": int(dropped_too_short),
        "dropped_no_path": int(dropped_no_path),
    }
    return matched, meta


def compute_baseline_speeds(matched_trips: list[MatchedTrip]) -> dict[int, float]:
    speeds: dict[int, list[float]] = defaultdict(list)
    for trip in matched_trips:
        for edge_id, speed in zip(trip.edge_ids, trip.edge_speeds_mps):
            speeds[edge_id].append(speed)
    return {edge_id: float(np.median(vals)) for edge_id, vals in speeds.items()}


def build_dataset(
    matched_trips: list[MatchedTrip],
    edge_records: list[dict[str, Any]],
    edge_texts: list[str],
    *,
    max_len: int,
    history_minutes: int,
    hidden_dim: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    history_seconds = history_minutes * 60
    baseline_speed = compute_baseline_speeds(matched_trips)
    history: dict[int, deque[tuple[int, float]]] = defaultdict(deque)

    road_size = len(edge_records) + 1
    pad_id = len(edge_records)
    data_road = np.full((len(matched_trips), max_len), pad_id, dtype=np.int64)
    dynamic_path = np.zeros((len(matched_trips), max_len, 6), dtype=np.float32)
    row_num = np.zeros(len(matched_trips), dtype=np.int64)
    trip_time = np.zeros(len(matched_trips), dtype=np.float32)
    departure_time = np.zeros(len(matched_trips), dtype=np.int64)
    semantic_embeddings = np.zeros((road_size, hidden_dim), dtype=np.float32)

    feature_rows: list[dict[str, float | str | int]] = []
    dynamic_rows: list[dict[str, float | int | str]] = []

    for edge_id, text in enumerate(edge_texts):
        semantic_embeddings[edge_id] = encode_text(text, hidden_dim)

    for idx, mt in enumerate(matched_trips):
        trip = mt.trip
        edges = mt.edge_ids[:max_len]
        speeds = mt.edge_speeds_mps[:max_len]
        lengths = mt.edge_lengths_m[:max_len]
        row_num[idx] = len(edges)
        trip_time[idx] = trip.duration_s
        departure_time[idx] = trip.timestamp
        data_road[idx, : len(edges)] = np.array(edges, dtype=np.int64)

        per_edge_features = []
        for pos, edge_id in enumerate(edges):
            dyn_values, dyn_named = dynamic_for_edge(
                history, baseline_speed, edge_id, trip.timestamp, history_seconds
            )
            dynamic_path[idx, pos] = np.array(dyn_values, dtype=np.float32)
            per_edge_features.append(dyn_values)
            dynamic_rows.append(
                {
                    "trip_index": idx,
                    "position": pos,
                    "edge_id": edge_id,
                    "timestamp": trip.timestamp,
                    **dyn_named,
                }
            )

        dyn = np.array(per_edge_features, dtype=float)
        dyn_mean = dyn.mean(axis=0)
        dyn_max = dyn.max(axis=0)
        feat = static_features(trip, edges, lengths)
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

        for edge_id, speed in zip(edges, speeds):
            history[edge_id].append((trip.timestamp, speed))
            cutoff = trip.timestamp - history_seconds * 4
            while history[edge_id] and history[edge_id][0][0] < cutoff:
                history[edge_id].popleft()

    arrays = {
        "data_road": data_road,
        "dynamic_path": dynamic_path,
        "row_num": row_num,
        "trip_time": trip_time,
        "departure_time": departure_time,
        "pad_id": np.array([pad_id], dtype=np.int64),
        "road_size": np.array([pad_id], dtype=np.int64),
        "semantic_embeddings": semantic_embeddings,
    }
    return pd.DataFrame(feature_rows), arrays, pd.DataFrame(dynamic_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PKDD15 OSM edge-level dynamic dataset")
    parser.add_argument(
        "--train-csv",
        default="pkdd-15-predict-taxi-service-trajectory-i/train.csv",
    )
    parser.add_argument("--out-dir", default="data/processed/pkdd15_osm_120k")
    parser.add_argument("--graph-path", default="data/raw/porto_drive.graphml")
    parser.add_argument("--max-rows", type=int, default=120_000)
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=160)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--history-minutes", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--max-snap-jump-m", type=float, default=500.0)
    parser.add_argument("--max-bridge-factor", type=float, default=4.0)
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
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )

    graph = load_or_download_graph(
        Path(args.graph_path),
        args.lon_min,
        args.lon_max,
        args.lat_min,
        args.lat_max,
    )
    edge_to_id, edge_records, edge_texts = build_edge_vocabulary(graph)
    matched_trips, match_meta = match_all_trips(
        trips,
        graph,
        edge_to_id,
        max_len=args.max_len,
        max_snap_jump_m=args.max_snap_jump_m,
        max_bridge_factor=args.max_bridge_factor,
    )

    trip_features, arrays, dynamic_rows = build_dataset(
        matched_trips,
        edge_records,
        edge_texts,
        max_len=args.max_len,
        history_minutes=args.history_minutes,
        hidden_dim=args.hidden_dim,
    )

    trip_features.to_csv(out_dir / "trip_features.csv", index=False)
    dynamic_rows.to_csv(out_dir / "dynamic_edge_features.csv", index=False)
    pd.DataFrame(edge_records).to_csv(out_dir / "edge_metadata.csv", index=False)
    pd.DataFrame(
        [{"edge_id": idx, "text": text} for idx, text in enumerate(edge_texts)]
    ).to_csv(out_dir / "edge_texts.csv", index=False)

    for name, value in arrays.items():
        np.save(out_dir / f"{name}.npy", value)

    meta.update(match_meta)
    meta.update(
        {
            "graph_path": args.graph_path,
            "num_osm_edges": int(len(edge_records)),
            "max_len": args.max_len,
            "history_minutes": args.history_minutes,
            "hidden_dim": args.hidden_dim,
            "max_snap_jump_m": args.max_snap_jump_m,
            "max_bridge_factor": args.max_bridge_factor,
            "note": (
                "Approximate local OSM map matching with nearest-node snapping "
                "and shortest-path bridging. Produces real OSM edge sequences, "
                "edge metadata, and text modality artifacts."
            ),
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote OSM edge-level dataset to {out_dir}")


if __name__ == "__main__":
    main()
