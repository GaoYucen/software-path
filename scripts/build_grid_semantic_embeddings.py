#!/usr/bin/env python3
"""Build lightweight text modality artifacts for grid-token path data.

This engineering-oriented script gives each grid token a textual description
derived from observed usage and dynamic traffic statistics, then converts the
text into a deterministic hashed embedding. The result is not a replacement
for real OSM road semantics, but it provides a real text-modality input for
the Path-LLM-style pipeline when only grid-cell tokens are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def label_quantile(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value <= low:
        return labels[0]
    if value >= high:
        return labels[2]
    return labels[1]


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

    features = list(words)
    features.extend(f"{a}__{b}" for a, b in zip(words[:-1], words[1:]))
    for token in features:
        idx, sign = stable_index(token, dim)
        vec[idx] += sign

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def summarize_cells(data_dir: Path) -> dict[str, np.ndarray]:
    road_ids = np.load(data_dir / "data_road.npy").astype(np.int64)
    dynamic_x = np.load(data_dir / "dynamic_path.npy").astype(np.float32)
    row_num = np.load(data_dir / "row_num.npy").astype(np.int64)
    departure_time = np.load(data_dir / "departure_time.npy").astype(np.int64)
    pad_id = int(np.load(data_dir / "pad_id.npy")[0])
    road_size = int(np.load(data_dir / "road_size.npy")[0]) + 1

    counts = np.zeros(road_size, dtype=np.int64)
    trip_counts = np.zeros(road_size, dtype=np.int64)
    mean_pos = np.zeros(road_size, dtype=np.float64)
    mean_speed = np.zeros(road_size, dtype=np.float64)
    mean_speed_ratio = np.zeros(road_size, dtype=np.float64)
    mean_freshness = np.zeros(road_size, dtype=np.float64)
    mean_reliability = np.zeros(road_size, dtype=np.float64)
    hour_hist = np.zeros((road_size, 4), dtype=np.int64)

    for trip_idx in range(len(road_ids)):
        valid_len = int(row_num[trip_idx])
        if valid_len <= 0:
            continue

        seen_in_trip: set[int] = set()
        dep_hour = int((departure_time[trip_idx] // 3600) % 24)
        hour_bucket = min(dep_hour // 6, 3)
        trip_cells = road_ids[trip_idx, :valid_len]
        trip_dynamic = dynamic_x[trip_idx, :valid_len]
        denom = max(valid_len - 1, 1)

        for pos, cell in enumerate(trip_cells.tolist()):
            if cell == pad_id:
                continue
            counts[cell] += 1
            mean_pos[cell] += pos / denom
            mean_speed[cell] += float(trip_dynamic[pos, 0])
            mean_speed_ratio[cell] += float(trip_dynamic[pos, 3])
            mean_freshness[cell] += float(trip_dynamic[pos, 4])
            mean_reliability[cell] += float(trip_dynamic[pos, 5])
            if cell not in seen_in_trip:
                trip_counts[cell] += 1
                hour_hist[cell, hour_bucket] += 1
                seen_in_trip.add(cell)

    valid = counts > 0
    for arr in (mean_pos, mean_speed, mean_speed_ratio, mean_freshness, mean_reliability):
        arr[valid] /= counts[valid]

    return {
        "pad_id": np.array([pad_id], dtype=np.int64),
        "road_size": np.array([road_size], dtype=np.int64),
        "counts": counts,
        "trip_counts": trip_counts,
        "mean_pos": mean_pos,
        "mean_speed": mean_speed,
        "mean_speed_ratio": mean_speed_ratio,
        "mean_freshness": mean_freshness,
        "mean_reliability": mean_reliability,
        "hour_hist": hour_hist,
    }


def build_texts(summary: dict[str, np.ndarray]) -> list[dict[str, object]]:
    counts = summary["counts"]
    trip_counts = summary["trip_counts"]
    mean_pos = summary["mean_pos"]
    mean_speed = summary["mean_speed"]
    mean_speed_ratio = summary["mean_speed_ratio"]
    mean_freshness = summary["mean_freshness"]
    mean_reliability = summary["mean_reliability"]
    hour_hist = summary["hour_hist"]
    pad_id = int(summary["pad_id"][0])

    active = counts[counts > 0]
    speed_active = mean_speed[mean_speed > 0]
    ratio_active = mean_speed_ratio[mean_speed_ratio > 0]
    freshness_active = mean_freshness[counts > 0]
    reliability_active = mean_reliability[counts > 0]

    count_low, count_high = np.quantile(active, [0.33, 0.67]) if len(active) else (0.0, 0.0)
    speed_low, speed_high = (
        np.quantile(speed_active, [0.33, 0.67]) if len(speed_active) else (0.0, 0.0)
    )
    ratio_low, ratio_high = (
        np.quantile(ratio_active, [0.33, 0.67]) if len(ratio_active) else (0.0, 0.0)
    )
    fresh_low, fresh_high = (
        np.quantile(freshness_active, [0.33, 0.67]) if len(freshness_active) else (0.0, 0.0)
    )
    rel_low, rel_high = (
        np.quantile(reliability_active, [0.33, 0.67]) if len(reliability_active) else (0.0, 0.0)
    )

    bucket_names = ("overnight", "morning", "afternoon", "evening")
    records: list[dict[str, object]] = []
    for cell_id in range(len(counts)):
        if cell_id == pad_id:
            records.append(
                {
                    "cell_id": cell_id,
                    "text": "Padding token for missing path positions.",
                    "token_count": 0,
                }
            )
            continue

        count = int(counts[cell_id])
        if count == 0:
            text = (
                f"Grid token {cell_id} in the Porto taxi path dataset. "
                "Rare or unseen token with sparse observations and unknown traffic pattern."
            )
            records.append({"cell_id": cell_id, "text": text, "token_count": 0})
            continue

        usage = label_quantile(float(count), float(count_low), float(count_high), ("rare", "moderate", "frequent"))
        speed = label_quantile(
            float(mean_speed[cell_id]),
            float(speed_low),
            float(speed_high),
            ("slow", "medium_speed", "fast"),
        )
        ratio = label_quantile(
            float(mean_speed_ratio[cell_id]),
            float(ratio_low),
            float(ratio_high),
            ("below_baseline", "near_baseline", "above_baseline"),
        )
        freshness = label_quantile(
            float(mean_freshness[cell_id]),
            float(fresh_low),
            float(fresh_high),
            ("fresh", "recent", "stale"),
        )
        reliability = label_quantile(
            float(mean_reliability[cell_id]),
            float(rel_low),
            float(rel_high),
            ("low_reliability", "medium_reliability", "high_reliability"),
        )
        if mean_pos[cell_id] < 0.34:
            path_role = "early_path_segment"
        elif mean_pos[cell_id] > 0.66:
            path_role = "late_path_segment"
        else:
            path_role = "middle_path_segment"

        peak_bucket = bucket_names[int(np.argmax(hour_hist[cell_id]))]
        text = (
            f"Grid token {cell_id} in the Porto taxi path dataset. "
            f"This token is a {usage} path location and usually appears as an {path_role}. "
            f"Observed traffic is {speed} with {ratio} behavior relative to local baseline speed. "
            f"Historical observations are {freshness} and {reliability}. "
            f"The token is most active during the {peak_bucket} period. "
            f"It appears in {int(trip_counts[cell_id])} trips and {count} valid path positions."
        )
        records.append({"cell_id": cell_id, "text": text, "token_count": count})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build text-modality embeddings for grid tokens")
    parser.add_argument("--data-dir", default="data/processed/pkdd15_grid_120k_clean")
    parser.add_argument("--output-name", default="semantic_embeddings.npy")
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--texts-name", default="semantic_texts.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    summary = summarize_cells(data_dir)
    records = build_texts(summary)

    embeddings = np.zeros((int(summary["road_size"][0]), args.hidden_dim), dtype=np.float32)
    for item in records:
        cell_id = int(item["cell_id"])
        if cell_id == int(summary["pad_id"][0]):
            continue
        embeddings[cell_id] = encode_text(str(item["text"]), args.hidden_dim)

    np.save(data_dir / args.output_name, embeddings)
    with (data_dir / args.texts_name).open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta = {
        "status": "pass",
        "data_dir": str(data_dir),
        "output_embedding": str(data_dir / args.output_name),
        "output_texts": str(data_dir / args.texts_name),
        "hidden_dim": args.hidden_dim,
        "road_size": int(summary["road_size"][0]),
        "pad_id": int(summary["pad_id"][0]),
        "note": (
            "Grid-token semantic embeddings are derived from generated text "
            "summaries of observed traffic behavior. They are suitable for "
            "engineering validation but not a substitute for real OSM road text."
        ),
    }
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
