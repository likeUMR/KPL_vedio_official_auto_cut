#!/usr/bin/env python3
"""Plot play-count and duration distributions for the top KPL title categories."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager


PREFIX_PATTERN = re.compile(r"^\s*【([^】]+)】")
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create two distribution charts for the top title-prefix categories."
    )
    parser.add_argument(
        "--input",
        default="data/kpl_programmes_enriched.json",
        help="Input enriched JSON. Default: data/kpl_programmes_enriched.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/plots",
        help="Directory for generated PNG files. Default: data/plots",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of largest categories to plot. Default: 5",
    )
    return parser.parse_args()


def configure_font() -> None:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def category_from_title(title: str) -> str:
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


def upload_year(video: dict[str, Any]) -> int | None:
    timestamp = to_number(video.get("create_timestamp"))
    if timestamp:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).year

    create_time = str(video.get("create_time_utc") or "")
    if len(create_time) >= 4 and create_time[:4].isdigit():
        return int(create_time[:4])
    return None


def load_top_categories(input_path: Path, top_n: int) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError("input JSON must contain a videos list")

    counts = Counter(category_from_title(str(video.get("title") or "")) for video in videos)
    top_categories = [category for category, _ in counts.most_common(top_n)]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video in videos:
        category = category_from_title(str(video.get("title") or ""))
        if category in top_categories:
            grouped[category].append(video)
    return {category: grouped[category] for category in top_categories}


def extract_values(
    grouped: dict[str, list[dict[str, Any]]], field: str, scale: float = 1.0
) -> tuple[list[str], list[list[float]]]:
    labels: list[str] = []
    values: list[list[float]] = []
    for category, videos in grouped.items():
        series = [
            number / scale
            for video in videos
            if (number := to_number(video.get(field))) is not None
        ]
        labels.append(f"{category}\n(n={len(series)})")
        values.append(series)
    return labels, values


def draw_distribution(
    labels: list[str],
    values: list[list[float]],
    title: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    box = ax.boxplot(
        values,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111827", "linewidth": 1.6},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
    )
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # Overlay deterministic jittered points so dense/long-tail categories remain visible.
    for idx, series in enumerate(values, start=1):
        if not series:
            continue
        step = max(1, len(series) // 280)
        sampled = series[::step]
        xs = [idx + (((i % 17) - 8) / 90) for i in range(len(sampled))]
        ax.scatter(xs, sampled, s=9, alpha=0.24, color=colors[(idx - 1) % len(colors)], linewidths=0)

    ax.set_title(title, fontsize=17, pad=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(f"{ylabel}（log刻度）", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_yearly_counts(
    grouped: dict[str, list[dict[str, Any]]],
    years: list[int],
    output_path: Path,
) -> None:
    categories = list(grouped.keys())
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
    yearly_counts: dict[str, Counter[int]] = {}
    for category, videos in grouped.items():
        counter: Counter[int] = Counter()
        for video in videos:
            year = upload_year(video)
            if year in years:
                counter[year] += 1
        yearly_counts[category] = counter

    fig, ax = plt.subplots(figsize=(14, 7), dpi=160)
    group_width = 0.82
    bar_width = group_width / max(len(categories), 1)
    x_positions = list(range(len(years)))
    for idx, category in enumerate(categories):
        offset = (idx - (len(categories) - 1) / 2) * bar_width
        values = [yearly_counts[category].get(year, 0) for year in years]
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=bar_width * 0.92,
            label=category,
            color=colors[idx % len(colors)],
            alpha=0.82,
        )

    ax.set_title("前5种标题类别的年度视频数量（2016-2026）", fontsize=17, pad=14)
    ax.set_xlabel("年份", fontsize=12)
    ax.set_ylabel("视频数量", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(year) for year in years])
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=min(5, len(categories)), loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")

    configure_font()
    grouped = load_top_categories(Path(args.input), args.top_n)
    output_dir = Path(args.output_dir)

    play_labels, play_values = extract_values(grouped, "play_count")
    duration_labels, duration_values = extract_values(grouped, "duration", scale=60.0)

    play_path = output_dir / "top5_categories_play_count_distribution.png"
    duration_path = output_dir / "top5_categories_duration_distribution.png"
    yearly_path = output_dir / "top5_categories_yearly_video_counts.png"
    draw_distribution(
        play_labels,
        play_values,
        "前5种标题类别的播放次数分布",
        "播放次数",
        play_path,
        log_scale=True,
    )
    draw_distribution(
        duration_labels,
        duration_values,
        "前5种标题类别的视频时长分布",
        "时长（分钟）",
        duration_path,
    )
    draw_yearly_counts(grouped, list(range(2016, 2027)), yearly_path)

    print("top categories:", ", ".join(grouped.keys()))
    print(f"saved: {play_path}")
    print(f"saved: {duration_path}")
    print(f"saved: {yearly_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
