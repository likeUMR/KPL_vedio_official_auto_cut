#!/usr/bin/env python3
"""Plot official KPL video metadata coverage and category distributions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager


TITLE_PREFIX_LEFT = chr(0x3010)
TITLE_PREFIX_RIGHT = chr(0x3011)
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]
PALETTE = [
    "#2563EB",
    "#16A34A",
    "#F97316",
    "#9333EA",
    "#DC2626",
    "#0891B2",
    "#CA8A04",
    "#4F46E5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create KPL official video coverage and category charts.")
    parser.add_argument("--input", default="data/kpl_programmes_enriched.json", help="Input enriched JSON.")
    parser.add_argument("--output-dir", default="data/plots", help="Directory for generated PNG files.")
    parser.add_argument("--top-n", type=int, default=8, help="Number of largest categories to emphasize.")
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


def category_from_title(title: str) -> str:
    stripped = (title or "").strip()
    if stripped.startswith(TITLE_PREFIX_LEFT):
        end = stripped.find(TITLE_PREFIX_RIGHT, len(TITLE_PREFIX_LEFT))
        if end > 0:
            return stripped[len(TITLE_PREFIX_LEFT) : end].strip()
    return "No Prefix"


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


def upload_year(video: dict[str, Any]) -> int | None:
    timestamp = to_number(video.get("create_timestamp"))
    if timestamp:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).year
    text = str(video.get("create_time_utc") or "")
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def load_videos(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError("input JSON must contain a videos list")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return videos, metadata


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, values: list[int | float], fmt: str = "{:,.0f}") -> None:
    ymax = max(values) if values else 0
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + ymax * 0.015,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#374151",
        )


def draw_category_counts(videos: list[dict[str, Any]], output_path: Path, top_n: int) -> None:
    counts = Counter(category_from_title(str(video.get("title") or "")) for video in videos)
    top = counts.most_common(top_n)
    labels = [item[0] for item in top]
    values = [item[1] for item in top]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.bar(labels, values, color=PALETTE[: len(labels)], alpha=0.9)
    ax.set_title("Official KPL Videos by Title Category")
    ax.set_ylabel("Video count")
    ax.set_xlabel("Title prefix category")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=25, labelsize=10)
    annotate_bars(ax, values)
    for bar in bars:
        bar.set_linewidth(0)
    save(fig, output_path)


def draw_yearly_totals(videos: list[dict[str, Any]], metadata: dict[str, Any], output_path: Path) -> None:
    counts: Counter[int] = Counter()
    for video in videos:
        year = upload_year(video)
        if year:
            counts[year] += 1
    years = list(range(min(counts), max(counts) + 1)) if counts else []
    values = [counts.get(year, 0) for year in years]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(years, values, color="#2563EB", linewidth=2.6, marker="o", markersize=6)
    ax.fill_between(years, values, color="#93C5FD", alpha=0.28)
    ax.set_title("Official KPL Video Coverage by Upload Year")
    subtitle = f"Fetched {len(videos):,} videos"
    expected = metadata.get("expected_total")
    fetched = metadata.get("fetched_total")
    if expected and fetched:
        subtitle += f" | API expected {int(expected):,}, fetched {int(fetched):,}"
    ax.text(0.01, 0.94, subtitle, transform=ax.transAxes, fontsize=11, color="#4B5563")
    ax.set_xlabel("Upload year")
    ax.set_ylabel("Video count")
    ax.set_xticks(years)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for year, value in zip(years, values):
        ax.text(year, value + max(values) * 0.02, f"{value:,}", ha="center", fontsize=8, color="#374151")
    save(fig, output_path)


def draw_category_share(videos: list[dict[str, Any]], output_path: Path, top_n: int) -> None:
    counts = Counter(category_from_title(str(video.get("title") or "")) for video in videos)
    top = counts.most_common(top_n)
    other = sum(counts.values()) - sum(value for _, value in top)
    labels = [label for label, _ in top] + (["Other"] if other else [])
    values = [value for _, value in top] + ([other] if other else [])

    fig, ax = plt.subplots(figsize=(9, 8))
    wedges, _texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=90,
        counterclock=False,
        colors=(PALETTE * 3)[: len(values)],
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10, "color": "#111827"},
    )
    ax.legend(wedges, [f"{label} ({value:,})" for label, value in zip(labels, values)], frameon=False, loc="center left", bbox_to_anchor=(0.92, 0.5))
    for text in autotexts:
        text.set_weight("bold")
    ax.set_title("Share of Official Videos by Category")
    save(fig, output_path)


def draw_top_category_distributions(videos: list[dict[str, Any]], output_dir: Path, top_n: int) -> None:
    counts = Counter(category_from_title(str(video.get("title") or "")) for video in videos)
    top_categories = [category for category, _ in counts.most_common(min(5, top_n))]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video in videos:
        category = category_from_title(str(video.get("title") or ""))
        if category in top_categories:
            grouped[category].append(video)

    def series(field: str, scale: float = 1.0) -> tuple[list[str], list[list[float]]]:
        labels: list[str] = []
        values: list[list[float]] = []
        for category in top_categories:
            rows = grouped[category]
            vals = [number / scale for row in rows if (number := to_number(row.get(field))) is not None]
            labels.append(f"{category}\n(n={len(vals)})")
            values.append(vals)
        return labels, values

    draw_boxplot(*series("play_count"), "Top Category Play Count Distribution", "Play count (log scale)", output_dir / "top5_categories_play_count_distribution.png", log_scale=True)
    draw_boxplot(*series("duration", 60.0), "Top Category Duration Distribution", "Duration (minutes)", output_dir / "top5_categories_duration_distribution.png")
    draw_top_category_yearly_counts(grouped, top_categories, output_dir / "top5_categories_yearly_video_counts.png")


def draw_boxplot(labels: list[str], values: list[list[float]], title: str, ylabel: str, output_path: Path, log_scale: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    box = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.55, medianprops={"color": "#111827", "linewidth": 1.7})
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(PALETTE[idx % len(PALETTE)])
        patch.set_alpha(0.32)
        patch.set_linewidth(1.1)
    for idx, vals in enumerate(values, start=1):
        step = max(1, len(vals) // 240)
        sampled = vals[::step]
        xs = [idx + (((i % 19) - 9) / 95) for i in range(len(sampled))]
        ax.scatter(xs, sampled, s=10, alpha=0.22, color=PALETTE[(idx - 1) % len(PALETTE)], linewidths=0)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    save(fig, output_path)


def draw_top_category_yearly_counts(grouped: dict[str, list[dict[str, Any]]], categories: list[str], output_path: Path) -> None:
    years = list(range(2016, 2027))
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = [0] * len(years)
    for idx, category in enumerate(categories):
        counter = Counter(upload_year(video) for video in grouped[category])
        values = [counter.get(year, 0) for year in years]
        ax.bar(years, values, bottom=bottom, label=category, color=PALETTE[idx % len(PALETTE)], alpha=0.86)
        bottom = [old + value for old, value in zip(bottom, values)]
    ax.set_title("Top Category Video Counts by Year")
    ax.set_xlabel("Upload year")
    ax.set_ylabel("Video count")
    ax.set_xticks(years)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, output_path)


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")

    configure_style()
    videos, metadata = load_videos(Path(args.input))
    output_dir = Path(args.output_dir)
    draw_category_counts(videos, output_dir / "official_video_category_counts.png", args.top_n)
    draw_yearly_totals(videos, metadata, output_dir / "official_video_yearly_coverage.png")
    draw_category_share(videos, output_dir / "official_video_category_share.png", args.top_n)
    draw_top_category_distributions(videos, output_dir, args.top_n)

    for path in sorted(output_dir.glob("*.png")):
        print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
