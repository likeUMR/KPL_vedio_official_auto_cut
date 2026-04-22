#!/usr/bin/env python3
"""Analyze duration distribution of split highlight scenes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze scene durations from scene_split_manifest.json.")
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scenes/scene_split_manifest.json",
        help="Scene split manifest path.",
    )
    parser.add_argument(
        "--output",
        default="data/kpl_scene_duration_stats.json",
        help="Output JSON stats path.",
    )
    parser.add_argument(
        "--csv-output",
        default="data/kpl_scene_durations.csv",
        help="Output per-scene CSV path.",
    )
    return parser.parse_args()


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int | None]:
    values = sorted(values)
    if not values:
        return {
            "count": 0,
            "sum": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "sum": round(sum(values), 3),
        "min": values[0],
        "p25": round(percentile(values, 0.25) or 0, 3),
        "median": round(percentile(values, 0.5) or 0, 3),
        "p75": round(percentile(values, 0.75) or 0, 3),
        "p90": round(percentile(values, 0.9) or 0, 3),
        "p95": round(percentile(values, 0.95) or 0, 3),
        "max": values[-1],
        "mean": round(statistics.fmean(values), 3),
    }


def bucket_label(seconds: float) -> str:
    edges = [10, 20, 30, 45, 60, 90, 120]
    previous = 0
    for edge in edges:
        if seconds < edge:
            return f"{previous}-{edge}s"
        previous = edge
    return ">=120s"


def load_scenes(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for video_index, video in enumerate(payload.get("videos") or [], start=1):
        input_path = video.get("input_path") or ""
        for scene in video.get("scenes") or []:
            rows.append(
                {
                    "video_index": video_index,
                    "input_path": input_path,
                    "scene_index": scene.get("index"),
                    "start": scene.get("start"),
                    "end": scene.get("end"),
                    "duration": scene.get("duration"),
                    "output_path": scene.get("output_path"),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video_index", "scene_index", "duration", "start", "end", "input_path", "output_path"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    scenes = load_scenes(manifest_path)
    durations = [float(scene["duration"]) for scene in scenes if scene.get("duration") is not None]
    histogram = dict(sorted(Counter(bucket_label(value) for value in durations).items()))
    by_video: list[dict[str, Any]] = []
    for video_index in sorted({row["video_index"] for row in scenes}):
        values = [float(row["duration"]) for row in scenes if row["video_index"] == video_index]
        input_path = next(row["input_path"] for row in scenes if row["video_index"] == video_index)
        by_video.append({"video_index": video_index, "input_path": input_path, "duration_seconds": summarize(values)})

    output = {
        "metadata": {
            "manifest": str(manifest_path),
            "scene_count": len(scenes),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "duration_seconds": summarize(durations),
        "duration_histogram_seconds": histogram,
        "by_video": by_video,
        "scenes": scenes,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.csv_output), scenes)
    print(f"scenes: {len(scenes)}")
    print(f"saved: {output_path}")
    print(f"saved: {Path(args.csv_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
