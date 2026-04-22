#!/usr/bin/env python3
"""Plot highlight scene processing and schedule-matching results."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager


FONT_CANDIDATES = [
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
PALETTE = ["#2563EB", "#16A34A", "#F97316", "#9333EA", "#DC2626", "#0891B2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stage-10 highlight processing charts.")
    parser.add_argument("--complete-focus", default="data/complete_focus_split_analysis/complete_focus_split_analysis.json")
    parser.add_argument("--schedule-matches", default="data/side_player_schedule_matches/segment_schedule_matches.json")
    parser.add_argument("--title-ocr", default="data/segment_title_ocr/segment_title_ocr.json")
    parser.add_argument("--output-dir", default="data/plots/stage10_highlights")
    return parser.parse_args()


def configure_style() -> None:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
        }
    )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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


def draw_funnel(complete_focus: dict[str, Any], matches: dict[str, Any], title_ocr: dict[str, Any], output_path: Path) -> None:
    scenes = complete_focus.get("scenes", [])
    segments = matches.get("segments", [])
    unique_videos = len({scene.get("video_index") for scene in scenes})
    title_segments = sum(1 for row in title_ocr.get("segments", []) if row.get("scene_title") or row.get("operator") or row.get("team"))
    matched_segments = sum(1 for row in segments if row.get("best_match"))
    labels = ["Top videos", "Detected scenes", "Final segments", "Title OCR", "Schedule matched"]
    values = [unique_videos, len(scenes), len(segments), title_segments, matched_segments]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    bars = ax.bar(labels, values, color=PALETTE[: len(labels)], alpha=0.9)
    ax.set_title("Highlight Processing Funnel")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ymax = max(values) if values else 0
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.025, f"{value:,}", ha="center", fontsize=11, color="#374151")
    save(fig, output_path)


def draw_classification(complete_focus: dict[str, Any], output_path: Path) -> None:
    counts = Counter(scene.get("classification") or "unknown" for scene in complete_focus.get("scenes", []))
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, _texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=PALETTE[: len(labels)],
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.5},
    )
    ax.legend(wedges, [f"{label} ({value})" for label, value in zip(labels, values)], frameon=False, loc="center left", bbox_to_anchor=(0.9, 0.5))
    for text in autotexts:
        text.set_weight("bold")
    ax.set_title("Scene Complete/Focus Classification")
    save(fig, output_path)


def draw_segments_by_year(matches: dict[str, Any], output_path: Path) -> None:
    years = sorted({int(row["year"]) for row in matches.get("segments", []) if row.get("year")})
    kinds = ["complete", "focus"]
    counters = {kind: Counter() for kind in kinds}
    for row in matches.get("segments", []):
        year = row.get("year")
        kind = row.get("kind")
        if year and kind in counters:
            counters[kind][int(year)] += 1

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bottom = [0] * len(years)
    for idx, kind in enumerate(kinds):
        values = [counters[kind].get(year, 0) for year in years]
        ax.bar(years, values, bottom=bottom, label=kind, color=PALETTE[idx], alpha=0.9)
        bottom = [old + value for old, value in zip(bottom, values)]
    ax.set_title("Processed Highlight Segments by Year")
    ax.set_xlabel("Source video year")
    ax.set_ylabel("Segment count")
    ax.set_xticks(years)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, output_path)


def draw_duration_histogram(complete_focus: dict[str, Any], output_path: Path) -> None:
    durations = [value for scene in complete_focus.get("scenes", []) if (value := to_number(scene.get("duration"))) is not None]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.hist(durations, bins=10, color="#2563EB", alpha=0.78, edgecolor="white", linewidth=1.2)
    mean = sum(durations) / len(durations) if durations else 0
    if durations:
        ax.axvline(mean, color="#DC2626", linestyle="--", linewidth=2, label=f"mean {mean:.1f}s")
    ax.set_title("Detected Highlight Scene Duration Distribution")
    ax.set_xlabel("Scene duration after 5s tail trim (seconds)")
    ax.set_ylabel("Scene count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, output_path)


def draw_match_quality(matches: dict[str, Any], output_path: Path) -> None:
    rows = matches.get("segments", [])
    matched = sum(1 for row in rows if row.get("best_match"))
    unmatched = len(rows) - matched
    directions = Counter((row.get("best_match") or {}).get("match_direction", "unmatched") for row in rows)
    labels = ["matched", "unmatched", "ordered", "reverse"]
    values = [matched, unmatched, directions.get("ordered", 0), directions.get("reverse", 0)]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bars = ax.bar(labels, values, color=["#16A34A", "#DC2626", "#2563EB", "#F97316"], alpha=0.88)
    ax.set_title("Schedule Matching Quality")
    ax.set_ylabel("Segment count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ymax = max(values) if values else 0
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.025, f"{value:,}", ha="center", fontsize=11, color="#374151")
    save(fig, output_path)


def main() -> int:
    args = parse_args()
    configure_style()
    complete_focus = load_json(args.complete_focus)
    matches = load_json(args.schedule_matches)
    title_ocr = load_json(args.title_ocr)
    output_dir = Path(args.output_dir)

    draw_funnel(complete_focus, matches, title_ocr, output_dir / "highlight_processing_funnel.png")
    draw_classification(complete_focus, output_dir / "highlight_scene_classification.png")
    draw_segments_by_year(matches, output_dir / "highlight_segments_by_year_kind.png")
    draw_duration_histogram(complete_focus, output_dir / "highlight_scene_duration_distribution.png")
    draw_match_quality(matches, output_dir / "highlight_schedule_match_quality.png")

    for path in sorted(output_dir.glob("*.png")):
        print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
