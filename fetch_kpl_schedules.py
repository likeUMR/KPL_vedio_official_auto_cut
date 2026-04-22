#!/usr/bin/env python3
"""Fetch official KPL schedule/match records from kpl.qq.com."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE = "https://kplshop-op.timi-esports.qq.com/kplow"
SEASON_API = f"{API_BASE}/getSeasonAndStageAndTeamList"
SCHEDULE_API = f"{API_BASE}/getScheduleList"
DETAIL_API = f"{API_BASE}/getScheduleDetail"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch official KPL schedule records for all seasons."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for generated files. Default: data",
    )
    parser.add_argument(
        "--seasonid",
        action="append",
        default=None,
        help=(
            "Season id to fetch, for example KPL2026S1. "
            "Can be repeated. Default: fetch every official season."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Delay between requests in seconds. Default: 0.15",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per request. Default: 3",
    )
    parser.add_argument(
        "--max-seasons",
        type=int,
        default=None,
        help="Limit seasons for testing. Default: no limit.",
    )
    parser.add_argument(
        "--max-matches-per-season",
        type=int,
        default=None,
        help="Limit matches per season for testing. Default: no limit.",
    )
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any], retries: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Content-Type": "application/json",
            "Origin": "https://kpl.qq.com",
            "Referer": "https://kpl.qq.com/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
            ),
        },
        method="POST",
    )

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


def require_success(response: dict[str, Any], context: str) -> dict[str, Any]:
    if response.get("result") != 0:
        raise RuntimeError(
            f"{context} returned result={response.get('result')}: {response.get('msg')}"
        )
    data = response.get("data")
    return data if isinstance(data, dict) else {}


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


def fetch_seasons(retries: int) -> list[dict[str, Any]]:
    data = require_success(
        post_json(SEASON_API, {"seasonid": ""}, retries),
        "getSeasonAndStageAndTeamList",
    )
    seasons = data.get("seasons") or []
    if not isinstance(seasons, list):
        raise RuntimeError("season list is missing or malformed")
    return seasons


def fetch_season_metadata(seasonid: str, retries: int) -> dict[str, list[dict[str, Any]]]:
    data = require_success(
        post_json(SEASON_API, {"seasonid": seasonid}, retries),
        f"getSeasonAndStageAndTeamList({seasonid})",
    )
    return {
        "stages": data.get("stages") or [],
        "teams": data.get("teams") or [],
    }


def fetch_schedule_list(seasonid: str, retries: int) -> list[dict[str, Any]]:
    payload = {"seasonid": seasonid, "stageid": "", "team_id": ""}
    data = require_success(post_json(SCHEDULE_API, payload, retries), f"getScheduleList({seasonid})")
    schedules = data.get("list") or []
    if not isinstance(schedules, list):
        raise RuntimeError(f"schedule list is missing or malformed for {seasonid}")
    return schedules


def normalize_schedule(
    schedule: dict[str, Any],
    season: dict[str, Any],
    index_in_season: int,
) -> dict[str, Any]:
    item = dict(schedule)
    item["season_name"] = season.get("season_name", item.get("season_name", ""))
    item["season_year"] = season.get("season_year", "")
    item["season_time_desc"] = season.get("season_time_desc", "")
    item["index_in_season"] = index_in_season
    item["start_time_utc"] = format_timestamp(item.get("start_timestamp"))
    item["schedule_status_text"] = {
        1: "未开始",
        2: "已取消",
        3: "进行中",
        4: "已结束",
    }.get(to_int(item.get("schedule_status")), "")

    return item


def select_seasons(args: argparse.Namespace, seasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.seasonid:
        wanted = set(args.seasonid)
        selected = [season for season in seasons if season.get("seasonid") in wanted]
        missing = sorted(wanted - {str(season.get("seasonid")) for season in selected})
        if missing:
            raise RuntimeError(f"seasonid not found: {', '.join(missing)}")
    else:
        selected = list(seasons)

    if args.max_seasons is not None:
        selected = selected[: args.max_seasons]
    return selected


def fetch_all(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seasons = fetch_seasons(args.retries)
    selected = select_seasons(args, seasons)
    season_metadata: dict[str, dict[str, Any]] = {}
    all_schedules: list[dict[str, Any]] = []

    for season_index, season in enumerate(selected, start=1):
        seasonid = str(season.get("seasonid") or "")
        if not seasonid:
            continue

        metadata = fetch_season_metadata(seasonid, args.retries)
        season_metadata[seasonid] = {"season": season, **metadata}
        schedules = fetch_schedule_list(seasonid, args.retries)
        if args.max_matches_per_season is not None:
            schedules = schedules[: args.max_matches_per_season]

        print(
            f"season {season_index}/{len(selected)} {seasonid}: "
            f"{len(schedules)} matches",
            flush=True,
        )

        for match_index, schedule in enumerate(schedules, start=1):
            all_schedules.append(normalize_schedule(schedule, season, match_index))

        if args.sleep > 0:
            time.sleep(args.sleep)

    metadata = {
        "api_urls": {
            "seasons": SEASON_API,
            "schedule_list": SCHEDULE_API,
            "schedule_detail": DETAIL_API,
        },
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_details": False,
        "official_season_count": len(seasons),
        "fetched_season_count": len(selected),
        "fetched_match_count": len(all_schedules),
        "seasonids": [season.get("seasonid") for season in selected],
        "seasons": selected,
        "season_metadata": season_metadata,
    }
    return all_schedules, metadata


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_outputs(output_dir: Path, schedules: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "kpl_schedules.json"
    csv_path = output_dir / "kpl_schedules.csv"
    metadata_path = output_dir / "kpl_schedules_metadata.json"

    json_path.write_text(
        json.dumps({"metadata": metadata, "schedules": schedules}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    preferred_fields = [
        "scheduleid",
        "seasonid",
        "season_name",
        "season_year",
        "stageid",
        "stage_name",
        "competition_format",
        "start_timestamp",
        "start_time_utc",
        "schedule_status",
        "schedule_status_text",
        "team_a_id",
        "team_a_name",
        "team_a_group",
        "team_a_score",
        "team_a_logo",
        "team_b_id",
        "team_b_name",
        "team_b_group",
        "team_b_score",
        "team_b_logo",
        "bo_total",
        "round_count",
        "round_vids",
        "location_name",
        "index_in_season",
    ]
    extra_fields = sorted({key for item in schedules for key in item if key not in preferred_fields})
    fields = preferred_fields + extra_fields

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in schedules:
            writer.writerow({field: csv_value(item.get(field, "")) for field in fields})

    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {metadata_path}")


def main() -> int:
    args = parse_args()
    if args.sleep < 0:
        print("--sleep must not be negative", file=sys.stderr)
        return 2
    if args.retries <= 0:
        print("--retries must be positive", file=sys.stderr)
        return 2
    if args.max_seasons is not None and args.max_seasons <= 0:
        print("--max-seasons must be positive", file=sys.stderr)
        return 2
    if args.max_matches_per_season is not None and args.max_matches_per_season <= 0:
        print("--max-matches-per-season must be positive", file=sys.stderr)
        return 2

    schedules, metadata = fetch_all(args)
    write_outputs(Path(args.output_dir), schedules, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
