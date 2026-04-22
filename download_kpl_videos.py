#!/usr/bin/env python3
"""Download KPL-listed videos through Tencent playback info, capped at 1080p."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GETINFO_URL = "https://vv.video.qq.com/getinfo"
DEFN_PRIORITY = [
    ("fhd", 1080),
    ("shd", 720),
    ("hd", 480),
    ("sd", 270),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download videos from a KPL-generated JSON list. The script uses KPL "
            "records/play_url/vfid as input and requests the best downloadable MP4 "
            "quality not above 1080p."
        )
    )
    parser.add_argument(
        "--input",
        default="data/top_jingcai_jijin_by_year.json",
        help="Input JSON list. Supports years[].videos[] or videos[].",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads/kpl_highlights_top1_by_year",
        help="Directory for downloaded videos.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Only download videos with this rank from years[].videos[]. Example: --rank 1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of selected videos, useful for testing.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Maximum video height. Default: 1080",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between downloads in seconds. Default: 0.2",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per network request. Default: 3",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def sanitize_filename(text: str, max_len: int = 90) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text or "")
    text = re.sub(r"\s+", " ", text).strip(" .")
    return (text[:max_len].rstrip(" .") or "untitled")


def load_video_list(path: Path, rank: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []

    if isinstance(payload.get("years"), list):
        for year_block in payload["years"]:
            videos = year_block.get("videos") or []
            for video in videos:
                if rank is None or video.get("rank") == rank:
                    selected.append(video)
    elif isinstance(payload.get("videos"), list):
        for video in payload["videos"]:
            if rank is None or video.get("rank") == rank:
                selected.append(video)
    else:
        raise ValueError("input JSON must contain years[].videos[] or videos[]")

    # Deduplicate by vfid while preserving order.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for video in selected:
        vfid = str(video.get("vfid") or "")
        if not vfid or vfid in seen:
            continue
        seen.add(vfid)
        deduped.append(video)
    return deduped


def fetch_text(url: str, headers: dict[str, str], retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"failed to fetch after {retries} retries: {last_error}")


def parse_qz_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^QZOutputJson\s*=\s*", "", text)
    text = text.rstrip(";")
    return json.loads(text)


def getinfo(vid: str, defn: str, retries: int) -> dict[str, Any]:
    url = (
        f"{GETINFO_URL}?vids={vid}&platform=101001&charge=0&otype=json"
        f"&defn={defn}&fhdswitch=0&show1080p=1"
    )
    headers = {
        "Accept": "*/*",
        "Referer": "https://kpl.qq.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
        ),
    }
    return parse_qz_json(fetch_text(url, headers, retries))


def build_download_url(video_info: dict[str, Any]) -> str:
    urls = (((video_info.get("ul") or {}).get("ui")) or [])
    filename = video_info.get("fn")
    fvkey = video_info.get("fvkey")
    if not urls or not filename or not fvkey:
        raise RuntimeError("missing url/fn/fvkey in video info")
    base = urls[0].get("url") or ""
    return f"{base}{filename}?vkey={fvkey}"


def choose_stream(vid: str, max_height: int, retries: int) -> dict[str, Any]:
    tried: list[str] = []
    for defn, nominal_height in DEFN_PRIORITY:
        if nominal_height > max_height:
            continue
        tried.append(defn)
        data = getinfo(vid, defn, retries)
        vi_list = (((data.get("vl") or {}).get("vi")) or [])
        if not vi_list:
            continue
        info = vi_list[0]
        height = int(info.get("vh") or nominal_height or 0)
        if height > max_height:
            continue
        try:
            url = build_download_url(info)
        except RuntimeError:
            continue
        return {
            "download_url": url,
            "defn": defn,
            "height": height,
            "width": int(info.get("vw") or 0),
            "filesize": int(info.get("fs") or 0),
            "filename": info.get("fn"),
            "duration_seconds": float(info.get("td") or 0),
            "title": info.get("ti") or "",
        }
    raise RuntimeError(f"no downloadable stream <= {max_height}p; tried {', '.join(tried)}")


def output_path_for(video: dict[str, Any], stream: dict[str, Any], output_dir: Path) -> Path:
    year = video.get("year") or "unknown-year"
    rank = video.get("rank")
    rank_part = f"rank{rank:02d}" if isinstance(rank, int) else "rank"
    vfid = video.get("vfid")
    title = sanitize_filename(str(video.get("title") or stream.get("title") or vfid))
    resolution = f"{stream.get('height') or 'unknown'}p"
    return output_dir / f"{year}_{rank_part}_{vfid}_{resolution}_{title}.mp4"


def download_file(url: str, path: Path, retries: int, overwrite: bool) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite and path.stat().st_size > 0:
        return {"status": "skipped", "bytes": path.stat().st_size}

    headers = {
        "Referer": "https://kpl.qq.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
        ),
    }
    temp_path = path.with_suffix(path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=60) as response, temp_path.open("wb") as file:
                total = response.headers.get("content-length")
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    downloaded += len(chunk)
                if total and downloaded != int(total):
                    raise RuntimeError(f"incomplete download: {downloaded}/{total}")
            temp_path.replace(path)
            return {"status": "downloaded", "bytes": path.stat().st_size}
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"download failed after {retries} retries: {last_error}")


def main() -> int:
    args = parse_args()
    videos = load_video_list(Path(args.input), args.rank)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        print("no videos selected", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    manifest: list[dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        vid = str(video.get("vfid") or "")
        print(f"[{index}/{len(videos)}] {vid} resolving...", flush=True)
        record = {
            "vfid": vid,
            "year": video.get("year"),
            "rank": video.get("rank"),
            "title": video.get("title"),
            "kpl_play_url": video.get("play_url"),
        }
        try:
            stream = choose_stream(vid, args.max_height, args.retries)
            path = output_path_for(video, stream, output_dir)
            result = download_file(stream["download_url"], path, args.retries, args.overwrite)
            record.update(
                {
                    "status": result["status"],
                    "output_path": str(path),
                    "bytes": result["bytes"],
                    "stream": {key: value for key, value in stream.items() if key != "download_url"},
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(
                f"  {record['status']}: {path.name} "
                f"({stream.get('width')}x{stream.get('height')}, {result['bytes']} bytes)",
                flush=True,
            )
        except Exception as exc:
            record.update({"status": "error", "error": str(exc)})
            print(f"  error: {exc}", flush=True)
        manifest.append(record)
        if index < len(videos) and args.sleep > 0:
            time.sleep(args.sleep)

    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "input": args.input,
                    "rank": args.rank,
                    "max_height": args.max_height,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                "downloads": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")
    errors = sum(1 for item in manifest if item.get("status") == "error")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
