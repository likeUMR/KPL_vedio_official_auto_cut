#!/usr/bin/env python3
"""Extract the top-played 精彩集锦 videos for each upload year."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PREFIX_PATTERN = re.compile(r"^\s*【([^】]+)】")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick the top-played 精彩集锦 videos for each year."
    )
    parser.add_argument(
        "--input",
        default="data/kpl_programmes_enriched.json",
        help="Input enriched JSON. Default: data/kpl_programmes_enriched.json",
    )
    parser.add_argument(
        "--output",
        default="data/top_jingcai_jijin_by_year.json",
        help="Output JSON path. Default: data/top_jingcai_jijin_by_year.json",
    )
    parser.add_argument(
        "--category",
        default="精彩集锦",
        help="Title-prefix category to extract. Default: 精彩集锦",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of videos to keep for each year. Default: 3",
    )
    return parser.parse_args()


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


def compact_video(video: dict[str, Any], year: int) -> dict[str, Any]:
    vfid = video.get("vfid")
    return {
        "year": year,
        "vfid": vfid,
        "title": video.get("title"),
        "play_count": int(to_number(video.get("play_count")) or 0),
        "duration": video.get("duration"),
        "create_timestamp": video.get("create_timestamp"),
        "create_time_utc": video.get("create_time_utc"),
        "seasonid": video.get("seasonid"),
        "tag_id": video.get("tag_id"),
        "cover_url": video.get("cover_url"),
        "play_url": video.get("play_url"),
        "tencent_video_url": f"https://v.qq.com/x/page/{vfid}.html" if vfid else "",
        "video_description": video.get("video_description"),
        "aspect_ratio": video.get("aspect_ratio"),
        "like_count": video.get("like_count"),
        "tencent_detail_info": video.get("tencent_detail_info"),
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise SystemExit("input JSON must contain a videos list")
    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")

    videos_by_year: dict[int, dict[str, dict[str, Any]]] = {}
    scanned = 0
    for video in videos:
        if category_from_title(str(video.get("title") or "")) != args.category:
            continue
        year = upload_year(video)
        play_count = to_number(video.get("play_count"))
        if year is None or play_count is None:
            continue
        scanned += 1
        vid = str(video.get("vfid") or "")
        if not vid:
            continue
        year_videos = videos_by_year.setdefault(year, {})
        current = year_videos.get(vid)
        if current is None or play_count > (to_number(current.get("play_count")) or -1):
            year_videos[vid] = video

    results = []
    for year in sorted(videos_by_year):
        ranked = sorted(
            videos_by_year[year].values(),
            key=lambda item: to_number(item.get("play_count")) or -1,
            reverse=True,
        )[: args.top_n]
        results.append(
            {
                "year": year,
                "videos": [
                    {**compact_video(video, year), "rank": rank}
                    for rank, video in enumerate(ranked, start=1)
                ],
            }
        )
    output = {
        "metadata": {
            "input": args.input,
            "category": args.category,
            "category_rule": "Use leading full-width bracket prefix like 【精彩集锦】.",
            "top_n_per_year": args.top_n,
            "scanned_category_videos_with_year_and_play_count": scanned,
            "year_count": len(results),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "years": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"years: {len(results)}")
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
