#!/usr/bin/env python3
"""Analyze enriched KPL video metadata by title prefix category."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PREFIX_PATTERN = re.compile(r"^\s*【([^】]+)】")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group KPL videos by title prefix and summarize play-count/duration distributions."
    )
    parser.add_argument(
        "--input",
        default="data/kpl_programmes_enriched.json",
        help="Input enriched JSON. Default: data/kpl_programmes_enriched.json",
    )
    parser.add_argument(
        "--output",
        default="data/kpl_video_category_stats.json",
        help="Output JSON stats path. Default: data/kpl_video_category_stats.json",
    )
    parser.add_argument(
        "--csv-output",
        default="data/kpl_video_category_stats.csv",
        help="Output CSV summary path. Default: data/kpl_video_category_stats.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Store top N videos by play count per category in JSON. Default: 10",
    )
    return parser.parse_args()


def title_category(title: str) -> str:
    match = PREFIX_PATTERN.match(title or "")
    return match.group(1).strip() if match else "无前缀"


def to_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


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


def bucket_label(value: float, edges: list[float], unit: str) -> str:
    previous = 0.0
    for edge in edges:
        if value < edge:
            return f"{format_edge(previous)}-{format_edge(edge)}{unit}"
        previous = edge
    return f">={format_edge(edges[-1])}{unit}"


def format_edge(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:g}"


def histogram(values: list[float], edges: list[float], unit: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        counts[bucket_label(value, edges, unit)] += 1
    ordered: dict[str, int] = {}
    previous = 0.0
    for edge in edges:
        label = f"{format_edge(previous)}-{format_edge(edge)}{unit}"
        ordered[label] = counts.get(label, 0)
        previous = edge
    ordered[f">={format_edge(edges[-1])}{unit}"] = counts.get(f">={format_edge(edges[-1])}{unit}", 0)
    return ordered


def summarize(values: list[float]) -> dict[str, float | int | None]:
    sorted_values = sorted(values)
    if not sorted_values:
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
        "count": len(sorted_values),
        "sum": round(sum(sorted_values), 4),
        "min": sorted_values[0],
        "p25": round(percentile(sorted_values, 0.25) or 0, 4),
        "median": round(percentile(sorted_values, 0.5) or 0, 4),
        "p75": round(percentile(sorted_values, 0.75) or 0, 4),
        "p90": round(percentile(sorted_values, 0.9) or 0, 4),
        "p95": round(percentile(sorted_values, 0.95) or 0, 4),
        "max": sorted_values[-1],
        "mean": round(statistics.fmean(sorted_values), 4),
    }


def compact_video(video: dict[str, Any]) -> dict[str, Any]:
    return {
        "vfid": video.get("vfid"),
        "title": video.get("title"),
        "play_count": video.get("play_count"),
        "duration": video.get("duration"),
        "seasonid": video.get("seasonid"),
        "play_url": video.get("play_url"),
    }


def analyze(videos: list[dict[str, Any]], top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video in videos:
        groups[title_category(str(video.get("title") or ""))].append(video)

    category_stats: list[dict[str, Any]] = []
    for category, items in groups.items():
        play_counts = [
            number for item in items if (number := to_number(item.get("play_count"))) is not None
        ]
        durations = [
            number for item in items if (number := to_number(item.get("duration"))) is not None
        ]
        top_videos = sorted(
            items,
            key=lambda item: to_number(item.get("play_count")) or -1,
            reverse=True,
        )[:top_n]
        category_stats.append(
            {
                "category": category,
                "video_count": len(items),
                "play_count": summarize(play_counts),
                "duration_seconds": summarize(durations),
                "play_count_histogram": histogram(
                    play_counts,
                    [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000],
                    "",
                ),
                "duration_histogram_seconds": histogram(
                    durations,
                    [30, 60, 120, 180, 300, 600, 1_200],
                    "s",
                ),
                "top_videos_by_play_count": [compact_video(video) for video in top_videos],
            }
        )

    category_stats.sort(
        key=lambda row: (
            row["video_count"],
            row["play_count"]["sum"] if row["play_count"]["sum"] is not None else -1,
        ),
        reverse=True,
    )
    overall = {
        "total_videos": len(videos),
        "category_count": len(category_stats),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return category_stats, overall


def write_csv(path: Path, category_stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "category",
        "video_count",
        "play_count_sum",
        "play_count_min",
        "play_count_p25",
        "play_count_median",
        "play_count_p75",
        "play_count_p90",
        "play_count_p95",
        "play_count_max",
        "play_count_mean",
        "duration_sum_seconds",
        "duration_min_seconds",
        "duration_p25_seconds",
        "duration_median_seconds",
        "duration_p75_seconds",
        "duration_p90_seconds",
        "duration_p95_seconds",
        "duration_max_seconds",
        "duration_mean_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in category_stats:
            play = row["play_count"]
            duration = row["duration_seconds"]
            writer.writerow(
                {
                    "category": row["category"],
                    "video_count": row["video_count"],
                    "play_count_sum": play["sum"],
                    "play_count_min": play["min"],
                    "play_count_p25": play["p25"],
                    "play_count_median": play["median"],
                    "play_count_p75": play["p75"],
                    "play_count_p90": play["p90"],
                    "play_count_p95": play["p95"],
                    "play_count_max": play["max"],
                    "play_count_mean": play["mean"],
                    "duration_sum_seconds": duration["sum"],
                    "duration_min_seconds": duration["min"],
                    "duration_p25_seconds": duration["p25"],
                    "duration_median_seconds": duration["median"],
                    "duration_p75_seconds": duration["p75"],
                    "duration_p90_seconds": duration["p90"],
                    "duration_p95_seconds": duration["p95"],
                    "duration_max_seconds": duration["max"],
                    "duration_mean_seconds": duration["mean"],
                }
            )


def main() -> int:
    args = parse_args()
    if args.top_n < 0:
        raise SystemExit("--top-n must be >= 0")

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise SystemExit("input JSON must contain a videos list")

    category_stats, overall = analyze(videos, args.top_n)
    output = {
        "metadata": {
            **overall,
            "input": args.input,
            "category_rule": "Use leading full-width bracket prefix like 【精彩集锦】; otherwise 无前缀.",
        },
        "categories": category_stats,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.csv_output), category_stats)

    print(f"categories: {overall['category_count']}")
    print(f"videos: {overall['total_videos']}")
    print(f"saved: {output_path}")
    print(f"saved: {Path(args.csv_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
