#!/usr/bin/env python3
"""Build the final scene/segment catalog from OCR, split, and schedule-match outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge final segment metadata into one scene catalog.")
    parser.add_argument(
        "--segment-manifest",
        default="downloads/pipeline/segments/complete_focus_segment_manifest.json",
        help="Manifest produced by split_complete_focus_segments.py.",
    )
    parser.add_argument(
        "--title-ocr",
        default="data/pipeline/segment_title_ocr/segment_title_ocr.json",
        help="Scene title/operator OCR JSON.",
    )
    parser.add_argument(
        "--team-ocr",
        default="data/pipeline/side_player_team_ocr/side_player_team_ocr.json",
        help="Side player team OCR JSON.",
    )
    parser.add_argument(
        "--schedule-matches",
        default="data/pipeline/side_player_schedule_matches/segment_schedule_matches.json",
        help="Schedule matching JSON.",
    )
    parser.add_argument(
        "--schedules",
        default="data/kpl_schedules_enriched.json",
        help="Enriched schedule JSON.",
    )
    parser.add_argument("--output", default="data/pipeline/final_scene_catalog.json", help="Output JSON path.")
    parser.add_argument("--csv-output", default="data/pipeline/final_scene_catalog.csv", help="Output CSV path.")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def segment_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["video_index"]), int(row["scene_index"]), int(row["segment_index"]))


def scene_key(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row["video_index"]), int(row["scene_index"]))


def index_segments(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    return {segment_key(row): row for row in rows}


def schedule_index(schedules_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["scheduleid"]: row for row in schedules_data.get("schedules", [])}


def compact_round_details(schedule: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not schedule:
        return []
    rounds = []
    for round_row in schedule.get("round_details", []) or []:
        rounds.append(
            {
                "round": round_row.get("round"),
                "win_team_name": round_row.get("win_team_name"),
                "vid": round_row.get("vid"),
                "playback_url": round_row.get("playback_url"),
                "heroes": [
                    {
                        "playerid": player.get("playerid"),
                        "hero_id": player.get("hero_id"),
                        "hero_name": player.get("hero_name"),
                        "hero_icon": player.get("hero_icon"),
                    }
                    for player in round_row.get("players", [])
                ],
            }
        )
    return rounds


def best_title_for_scene(title_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not title_rows:
        return {}
    focus_rows = [row for row in title_rows if row.get("kind") == "focus"]
    rows = focus_rows or title_rows
    rows = sorted(rows, key=lambda row: (row.get("ocr_source_count") or 0, row.get("segment_index") or 0), reverse=True)
    row = rows[0]
    return {
        "scene_title": row.get("scene_title"),
        "scene_title_lines": row.get("scene_title_lines", []),
        "operator_team_text": row.get("team"),
        "operator": row.get("operator"),
        "team_operator_candidates": row.get("team_operator_candidates", []),
    }


def build_catalog(args: argparse.Namespace) -> dict[str, Any]:
    segment_manifest = load_json(args.segment_manifest)
    title_data = load_json(args.title_ocr)
    team_data = load_json(args.team_ocr)
    matches_data = load_json(args.schedule_matches)
    schedules_by_id = schedule_index(load_json(args.schedules))

    title_by_segment = index_segments(title_data.get("segments", []))
    team_by_segment = index_segments(team_data.get("segments", []))
    match_by_segment = index_segments(matches_data.get("segments", []))

    scene_records = []
    flat_rows = []
    for scene in segment_manifest.get("scenes", []):
        scene_segments = scene.get("segments", [])
        key = scene_key(scene)
        title_rows = [title_by_segment[segment_key(segment)] for segment in scene_segments if segment_key(segment) in title_by_segment]
        title_info = best_title_for_scene(title_rows)
        segment_records = []
        match_ids = []

        for segment in scene_segments:
            skey = segment_key(
                {
                    "video_index": scene["video_index"],
                    "scene_index": scene["scene_index"],
                    "segment_index": segment["index"],
                }
            )
            title_row = title_by_segment.get(skey, {})
            team_row = team_by_segment.get(skey, {})
            match_row = match_by_segment.get(skey, {})
            best_match = match_row.get("best_match") or {}
            if best_match.get("scheduleid"):
                match_ids.append(best_match["scheduleid"])
            segment_record = {
                "segment_index": segment["index"],
                "kind": segment.get("kind"),
                "mode": segment.get("mode"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "duration": segment.get("duration"),
                "path": segment.get("output_path"),
                "scene_title": title_row.get("scene_title"),
                "operator_team_text": title_row.get("team"),
                "operator": title_row.get("operator"),
                "left_team_text": team_row.get("left_team_text"),
                "right_team_text": team_row.get("right_team_text"),
                "match": best_match or None,
            }
            segment_records.append(segment_record)

        selected_scheduleid = Counter(match_ids).most_common(1)[0][0] if match_ids else None
        selected_schedule = schedules_by_id.get(selected_scheduleid) if selected_scheduleid else None
        scene_record = {
            "video_index": scene["video_index"],
            "scene_index": scene["scene_index"],
            "classification": scene.get("classification"),
            "has_complete": any(segment.get("kind") == "complete" for segment in scene_segments),
            "has_focus": any(segment.get("kind") == "focus" for segment in scene_segments),
            "duration": scene.get("duration"),
            "input_path": scene.get("input_path"),
            **title_info,
            "scheduleid": selected_scheduleid,
            "seasonid": selected_schedule.get("seasonid") if selected_schedule else None,
            "season_name": selected_schedule.get("season_name") if selected_schedule else None,
            "stage_name": selected_schedule.get("stage_name") if selected_schedule else None,
            "match_time_utc": selected_schedule.get("start_time_utc") if selected_schedule else None,
            "team_a_name": selected_schedule.get("team_a_name") if selected_schedule else None,
            "team_a_score": selected_schedule.get("team_a_score") if selected_schedule else None,
            "team_b_name": selected_schedule.get("team_b_name") if selected_schedule else None,
            "team_b_score": selected_schedule.get("team_b_score") if selected_schedule else None,
            "match_playback_url": selected_schedule.get("match_playback_url") if selected_schedule else None,
            "round_details": compact_round_details(selected_schedule),
            "segments": segment_records,
        }
        scene_records.append(scene_record)
        flat_rows.append(
            {
                "video_index": scene_record["video_index"],
                "scene_index": scene_record["scene_index"],
                "classification": scene_record["classification"],
                "has_complete": scene_record["has_complete"],
                "has_focus": scene_record["has_focus"],
                "scene_title": scene_record.get("scene_title"),
                "operator_team_text": scene_record.get("operator_team_text"),
                "operator": scene_record.get("operator"),
                "scheduleid": scene_record.get("scheduleid"),
                "season_name": scene_record.get("season_name"),
                "stage_name": scene_record.get("stage_name"),
                "match_time_utc": scene_record.get("match_time_utc"),
                "matchup": ""
                if not scene_record.get("team_a_name")
                else f"{scene_record.get('team_a_name')} {scene_record.get('team_a_score')}:{scene_record.get('team_b_score')} {scene_record.get('team_b_name')}",
                "match_playback_url": scene_record.get("match_playback_url"),
                "segment_count": len(segment_records),
            }
        )

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "segment_manifest": args.segment_manifest,
            "title_ocr": args.title_ocr,
            "team_ocr": args.team_ocr,
            "schedule_matches": args.schedule_matches,
            "schedules": args.schedules,
            "scene_count": len(scene_records),
            "segment_count": sum(len(scene.get("segments", [])) for scene in scene_records),
        },
        "scenes": scene_records,
        "_flat_rows": flat_rows,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_index",
        "scene_index",
        "classification",
        "has_complete",
        "has_focus",
        "scene_title",
        "operator_team_text",
        "operator",
        "scheduleid",
        "season_name",
        "stage_name",
        "match_time_utc",
        "matchup",
        "match_playback_url",
        "segment_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    csv_path = Path(args.csv_output)
    catalog = build_catalog(args)
    flat_rows = catalog.pop("_flat_rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(flat_rows, csv_path)
    print(f"json: {output_path}")
    print(f"csv: {csv_path}")
    print(f"scenes: {catalog['metadata']['scene_count']}, segments: {catalog['metadata']['segment_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
