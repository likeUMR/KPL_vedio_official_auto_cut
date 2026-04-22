#!/usr/bin/env python3
"""Match segment top-UI team OCR results to official KPL schedule records."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match segment UI team OCR to official schedule records.")
    parser.add_argument(
        "--team-ocr",
        default="data/match_ui_team_roi_ocr/match_ui_team_roi_ocr.json",
        help="Team ROI OCR JSON from ocr_match_ui_team_rois.py.",
    )
    parser.add_argument(
        "--schedules",
        default="data/kpl_schedules_enriched.json",
        help="Official schedule JSON.",
    )
    parser.add_argument(
        "--top-videos",
        default="data/top_jingcai_jijin_by_year.json",
        help="Top highlight JSON containing upload timestamps by year.",
    )
    parser.add_argument("--output-dir", default="data/segment_schedule_matches", help="Output directory.")
    parser.add_argument("--min-pair-score", type=float, default=0.65, help="Minimum ordered pair score to accept.")
    parser.add_argument("--min-team-score", type=float, default=0.62, help="Minimum score required for each side.")
    parser.add_argument(
        "--max-days-before-upload",
        type=float,
        default=31.0,
        help="Drop matches older than this many days before the video upload time.",
    )
    return parser.parse_args()


def normalize_team_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip()
    text = text.replace("．", ".").replace("·", ".").replace("。", ".")
    text = text.upper()
    replacements = {
        "OYG": "DYG",
        "RI": "RW",
        "R1": "RW",
        "HERO久竞": "HERO",
        "HERO久競": "HERO",
        "南京HERO久竞": "HERO",
        "上海": "上海",
        "广州": "广州",
        "深圳": "深圳",
        "西安W": "西安WE",
        "济南RW": "济南RW侠",
    }
    for old, new in replacements.items():
        text = text.replace(old.upper(), new.upper())
    text = re.sub(r"[^A-Z0-9\u4e00-\u9fff.]", "", text)
    suffixes = ["超玩会", "久竞", "大鹅"]
    for suffix in suffixes:
        text = text.replace(suffix, "")
    return text


def team_similarity(ocr_text: str | None, team_name: str) -> float:
    left = normalize_team_text(ocr_text)
    right = normalize_team_text(team_name)
    if not left or not right:
        return 0.0
    aliases = team_aliases(team_name)
    normalized_aliases = {normalize_team_text(alias) for alias in aliases}
    if left == right:
        return 1.0
    if left in normalized_aliases:
        return 1.0

    # Short Latin team tags are high-value but risky: TES must not fuzzily match TS.
    # Require exact alias/subtoken matches and cap typo-like matches aggressively.
    if is_short_latin_code(left):
        alias_tokens = {token for alias in normalized_aliases for token in latin_tokens(alias)}
        if left in alias_tokens:
            return 1.0
        fuzzy = max((SequenceMatcher(None, left, token).ratio() for token in alias_tokens), default=0.0)
        return min(0.45, fuzzy)

    if left in right or right in left:
        return min(0.98, 0.62 + 0.04 * min(len(left), len(right)))

    scores = [SequenceMatcher(None, left, alias).ratio() for alias in normalized_aliases]
    return max(scores) if scores else 0.0


def is_short_latin_code(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9.]{1,4}", text))


def latin_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Z0-9.]{1,8}", text)


def team_aliases(team_name: str) -> list[str]:
    aliases = {team_name}
    name = normalize_team_text(team_name)
    aliases.add(name)
    city_prefixes = ["上海", "深圳", "广州", "西安", "济南", "南京", "佛山", "成都", "武汉", "北京", "杭州", "苏州", "重庆", "长沙", "厦门", "南通", "无锡", "桐乡"]
    for prefix in city_prefixes:
        if name.startswith(prefix.upper()):
            aliases.add(name[len(prefix) :])
    if "EDG" in name:
        aliases.update(["EDG", "EDG.M", "上海EDG.M"])
    if "RNG" in name:
        aliases.update(["RNG", "RNG.M", "上海RNG.M"])
    if "DYG" in name:
        aliases.update(["DYG", "深圳DYG"])
    if "TTG" in name:
        aliases.update(["TTG", "广州TTG"])
    if "WE" in name:
        aliases.update(["WE", "西安WE"])
    if "RW" in name:
        aliases.update(["RW", "RW侠", "济南RW侠"])
    if "HERO" in name:
        aliases.update(["Hero", "Hero久竞", "南京Hero", "南京Hero久竞", "南通Hero久竞"])
    if "TES" in name:
        aliases.update(["TES", "TES.A", "长沙TES", "长沙TES.A"])
    if "GK" in name or "DRG" in name:
        aliases.update(["GK", "DRG", "佛山GK", "佛山DRG", "佛山DRG.GK"])
    if "TCG" in name:
        aliases.update(["TCG", "无锡TCG"])
    if "XYG" in name:
        aliases.add("XYG")
    if "VG" in name:
        aliases.add("VG")
    if "狼" in team_name:
        aliases.update(["狼队", "重庆狼队"])
    return list(aliases)


def load_upload_times(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, int] = {}
    for year_block in data.get("years", []):
        year = int(year_block["year"])
        videos = year_block.get("videos", [])
        if not videos:
            continue
        rank1 = sorted(videos, key=lambda item: item.get("rank", 999))[0]
        timestamp = rank1.get("create_timestamp")
        if timestamp:
            result[year] = int(timestamp)
    return result


def candidate_schedules(
    schedules: list[dict[str, Any]],
    year: int,
    upload_timestamp: int | None,
    max_seconds_before_upload: float | None,
) -> list[dict[str, Any]]:
    rows = [item for item in schedules if int(item.get("season_year") or 0) == year]
    if upload_timestamp is not None:
        before = [item for item in rows if int(item.get("start_timestamp") or 0) <= upload_timestamp]
        if before:
            rows = before
        if max_seconds_before_upload is not None:
            rows = [
                item
                for item in rows
                if upload_timestamp - int(item.get("start_timestamp") or 0) <= max_seconds_before_upload
            ]
    return rows


def match_segment(
    segment: dict[str, Any],
    schedules: list[dict[str, Any]],
    upload_times: dict[int, int],
    min_pair_score: float,
    min_team_score: float,
    max_seconds_before_upload: float | None,
) -> dict[str, Any]:
    year = int(segment["year"])
    upload_timestamp = upload_times.get(year)
    left_text = segment.get("left_team_text")
    right_text = segment.get("right_team_text")
    candidates = []
    for schedule in candidate_schedules(schedules, year, upload_timestamp, max_seconds_before_upload):
        left_score = team_similarity(left_text, schedule["team_a_name"])
        right_score = team_similarity(right_text, schedule["team_b_name"])
        reverse_left = team_similarity(left_text, schedule["team_b_name"])
        reverse_right = team_similarity(right_text, schedule["team_a_name"])
        ordered_pair_score = (left_score + right_score) / 2
        reverse_pair_score = (reverse_left + reverse_right) / 2
        ordered_ok = (
            ordered_pair_score >= min_pair_score
            and left_score >= min_team_score
            and right_score >= min_team_score
        )
        reverse_ok = (
            reverse_pair_score >= min_pair_score
            and reverse_left >= min_team_score
            and reverse_right >= min_team_score
        )
        if not ordered_ok and not reverse_ok:
            continue
        match_direction = "ordered" if ordered_ok else "reverse"
        selected_pair_score = ordered_pair_score if ordered_ok else reverse_pair_score
        time_delta = None
        time_delta_days = None
        if upload_timestamp is not None:
            time_delta = upload_timestamp - int(schedule.get("start_timestamp") or 0)
            time_delta_days = time_delta / (24 * 60 * 60)
        candidates.append(
            {
                "scheduleid": schedule["scheduleid"],
                "seasonid": schedule["seasonid"],
                "season_name": schedule["season_name"],
                "stage_name": schedule["stage_name"],
                "start_timestamp": schedule["start_timestamp"],
                "start_time_utc": schedule["start_time_utc"],
                "team_a_name": schedule["team_a_name"],
                "team_a_score": schedule["team_a_score"],
                "team_b_name": schedule["team_b_name"],
                "team_b_score": schedule["team_b_score"],
                "bo_total": schedule.get("bo_total"),
                "match_playback_url": schedule.get("match_playback_url"),
                "match_direction": match_direction,
                "selected_pair_score": round(selected_pair_score, 4),
                "ordered_pair_score": round(ordered_pair_score, 4),
                "reverse_pair_score": round(reverse_pair_score, 4),
                "left_team_score": round(left_score, 4),
                "right_team_score": round(right_score, 4),
                "upload_minus_match_seconds": time_delta,
                "upload_minus_match_days": None if time_delta_days is None else round(time_delta_days, 3),
            }
        )
    candidates.sort(
        key=lambda item: (
            1 if item["match_direction"] == "ordered" else 0,
            item["selected_pair_score"],
            -(item["upload_minus_match_seconds"] or 10**12),
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else None
    return {
        "video_index": segment["video_index"],
        "scene_index": segment["scene_index"],
        "segment_index": segment["segment_index"],
        "kind": segment["kind"],
        "year": year,
        "upload_timestamp": upload_timestamp,
        "left_team_text": left_text,
        "right_team_text": right_text,
        "left_ocr_texts": [line["text"] for line in segment["left"]["ocr_lines"]],
        "right_ocr_texts": [line["text"] for line in segment["right"]["ocr_lines"]],
        "candidate_count": len(candidates),
        "best_match": best,
        "top_candidates": candidates[:5],
        "segment_path": segment["path"],
    }


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "year",
                "video_index",
                "scene_index",
                "segment_index",
                "kind",
                "left_team_text",
                "right_team_text",
                "candidate_count",
                "matched_scheduleid",
                "matched_time",
                "matched_team_a",
                "matched_score",
                "matched_team_b",
                "match_direction",
                "selected_pair_score",
                "ordered_pair_score",
                "reverse_pair_score",
                "upload_minus_match_days",
                "segment_path",
            ],
        )
        writer.writeheader()
        for record in records:
            best = record.get("best_match") or {}
            writer.writerow(
                {
                    "year": record["year"],
                    "video_index": record["video_index"],
                    "scene_index": record["scene_index"],
                    "segment_index": record["segment_index"],
                    "kind": record["kind"],
                    "left_team_text": record["left_team_text"],
                    "right_team_text": record["right_team_text"],
                    "candidate_count": record["candidate_count"],
                    "matched_scheduleid": best.get("scheduleid"),
                    "matched_time": best.get("start_time_utc"),
                    "matched_team_a": best.get("team_a_name"),
                    "matched_score": ""
                    if not best
                    else f"{best.get('team_a_score')}:{best.get('team_b_score')}",
                    "matched_team_b": best.get("team_b_name"),
                    "match_direction": best.get("match_direction"),
                    "selected_pair_score": best.get("selected_pair_score"),
                    "ordered_pair_score": best.get("ordered_pair_score"),
                    "reverse_pair_score": best.get("reverse_pair_score"),
                    "upload_minus_match_days": best.get("upload_minus_match_days"),
                    "segment_path": record["segment_path"],
                }
            )


def main() -> int:
    args = parse_args()
    team_ocr_path = Path(args.team_ocr)
    schedules_path = Path(args.schedules)
    top_videos_path = Path(args.top_videos)
    output_dir = Path(args.output_dir)

    team_ocr = json.loads(team_ocr_path.read_text(encoding="utf-8"))
    schedules_data = json.loads(schedules_path.read_text(encoding="utf-8"))
    schedules = schedules_data["schedules"]
    upload_times = load_upload_times(top_videos_path)
    max_seconds_before_upload = None
    if args.max_days_before_upload is not None and args.max_days_before_upload > 0:
        max_seconds_before_upload = args.max_days_before_upload * 24 * 60 * 60

    records = [
        match_segment(
            segment,
            schedules,
            upload_times,
            args.min_pair_score,
            args.min_team_score,
            max_seconds_before_upload,
        )
        for segment in team_ocr.get("segments", [])
    ]

    json_path = output_dir / "segment_schedule_matches.json"
    csv_path = output_dir / "segment_schedule_matches.csv"
    write_csv(records, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "team_ocr": str(team_ocr_path),
                    "schedules": str(schedules_path),
                    "top_videos": str(top_videos_path),
                    "segment_count": len(records),
                    "matched_count": sum(1 for item in records if item["best_match"]),
                    "min_pair_score": args.min_pair_score,
                    "min_team_score": args.min_team_score,
                    "max_days_before_upload": args.max_days_before_upload,
                    "summary_csv": str(csv_path),
                },
                "segments": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"matched: {sum(1 for item in records if item['best_match'])}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
