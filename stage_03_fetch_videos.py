#!/usr/bin/env python3
"""Fetch official KPL programme/video metadata from kpl.qq.com."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


API_URL = "https://kplshop-op.timi-esports.qq.com/kplow/getProgrammeList"
PLAY_URL_PREFIX = "https://kpl.qq.com/#/PlayVideo"
DEFAULT_PAGE_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch official KPL video metadata and save it as JSON/CSV."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for generated files. Default: data",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="API page size. The official page uses 12. Default: 12",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit pages to fetch, for example 242. Default: fetch all pages from API total.",
    )
    parser.add_argument(
        "--seasonid",
        default="all",
        help='Season filter. Default: "all"',
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        default=None,
        help="Tag/category id. Default: null, same as first official page request.",
    )
    parser.add_argument(
        "--homepage-recommend",
        action="store_true",
        help="Fetch homepage-recommended videos only. Default: false.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Delay between page requests in seconds. Default: 0.15",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per page. Default: 3",
    )
    return parser.parse_args()


def post_json(payload: dict[str, Any], retries: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Content-Type": "application/json",
        "Origin": "https://kpl.qq.com",
        "Referer": "https://kpl.qq.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
        ),
    }
    request = Request(API_URL, data=body, headers=headers, method="POST")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return json.loads(response.read().decode(charset))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))

    raise RuntimeError(f"request failed after {retries} retries: {last_error}")


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_timestamp(value: Any) -> str:
    timestamp = to_int(value)
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def cover_url(vfid: str, image_url: str = "") -> str:
    if image_url:
        return image_url
    if not vfid:
        return ""
    return f"https://puui.qpic.cn/vpic_cover/{vfid}/{vfid}_hz.jpg"


def play_url(vfid: str, title: str, create_timestamp: Any) -> str:
    return (
        f"{PLAY_URL_PREFIX}?vid={quote(vfid or '')}"
        f"&title={quote(quote(title or '', safe=''), safe='')}"
        f"&time={quote(str(create_timestamp or ''))}"
    )


def normalize_video(video: dict[str, Any], page: int, index_in_page: int) -> dict[str, Any]:
    item = dict(video)
    vfid = str(item.get("vfid") or "")
    title = str(item.get("title") or "")
    create_timestamp = item.get("create_timestamp") or ""
    image_url = str(item.get("image_url") or "")

    item["vfid"] = vfid
    item["title"] = title
    item["create_timestamp"] = str(create_timestamp)
    item["duration"] = to_int(item.get("duration"))
    item["source_page"] = page
    item["source_index"] = index_in_page
    item["create_time_utc"] = format_timestamp(create_timestamp)
    item["cover_url"] = cover_url(vfid, image_url)
    item["play_url"] = play_url(vfid, title, create_timestamp)
    return item


def fetch_all(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_videos: list[dict[str, Any]] = []
    tag_list: list[dict[str, Any]] = []
    expected_total = 0
    pages_to_fetch = args.max_pages

    page = 1
    while True:
        payload = {
            "page": page,
            "page_size": args.page_size,
            "seasonid": args.seasonid,
            "tag_id": args.tag_id,
            "is_homepage_recommend": bool(args.homepage_recommend),
        }
        response = post_json(payload, args.retries)
        if response.get("result") != 0:
            raise RuntimeError(f"API returned result={response.get('result')}: {response.get('msg')}")

        data = response.get("data") or {}
        videos = data.get("video_list") or []
        if page == 1:
            expected_total = to_int(data.get("total"))
            tag_list = data.get("tag_list") or []
            if pages_to_fetch is None:
                pages_to_fetch = math.ceil(expected_total / args.page_size) if expected_total else 1

        for index, video in enumerate(videos, start=1):
            all_videos.append(normalize_video(video, page, index))

        print(f"page {page}/{pages_to_fetch}: {len(videos)} videos", flush=True)
        if page >= (pages_to_fetch or 1) or not videos:
            break

        page += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    metadata = {
        "api_url": API_URL,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_total": expected_total,
        "fetched_total": len(all_videos),
        "page_size": args.page_size,
        "pages_fetched": page,
        "seasonid": args.seasonid,
        "tag_id": args.tag_id,
        "is_homepage_recommend": bool(args.homepage_recommend),
        "tag_list": tag_list,
    }
    return all_videos, metadata


def write_outputs(output_dir: Path, videos: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "kpl_programmes.json"
    csv_path = output_dir / "kpl_programmes.csv"

    json_path.write_text(
        json.dumps({"metadata": metadata, "videos": videos}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preferred_fields = [
        "vfid",
        "title",
        "tag_id",
        "seasonid",
        "duration",
        "create_timestamp",
        "create_time_utc",
        "is_homepage_recommend",
        "image_url",
        "cover_url",
        "play_url",
        "source_page",
        "source_index",
    ]
    extra_fields = sorted({key for item in videos for key in item if key not in preferred_fields})
    fields = preferred_fields + extra_fields

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(videos)

    metadata_path = output_dir / "kpl_programmes_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {metadata_path}")


def main() -> int:
    args = parse_args()
    if args.page_size <= 0:
        print("--page-size must be positive", file=sys.stderr)
        return 2
    if args.max_pages is not None and args.max_pages <= 0:
        print("--max-pages must be positive", file=sys.stderr)
        return 2

    videos, metadata = fetch_all(args)
    write_outputs(Path(args.output_dir), videos, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
