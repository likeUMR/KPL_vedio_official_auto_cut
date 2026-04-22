#!/usr/bin/env python3
"""OCR side player-list UI and infer left/right teams from repeated player labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR left/right side player UI team prefixes.")
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scene_segments_complete_focus_with_intro/complete_focus_segment_manifest.json",
        help="Segment manifest.",
    )
    parser.add_argument(
        "--schedules",
        default="data/kpl_schedules_enriched.json",
        help="Official schedule JSON used only for team-name aliases.",
    )
    parser.add_argument("--output-dir", default="data/side_player_team_ocr", help="Output directory.")
    parser.add_argument("--sample-time", type=float, default=1.5, help="Frame sample time in the opening filter.")
    parser.add_argument("--min-confidence", type=float, default=0.25, help="OCR confidence threshold.")
    parser.add_argument("--left-x2", type=float, default=0.22, help="Right edge of left side UI crop.")
    parser.add_argument("--right-x1", type=float, default=0.78, help="Left edge of right side UI crop.")
    parser.add_argument("--y1", type=float, default=0.0, help="Top edge of side UI crop.")
    parser.add_argument("--y2", type=float, default=0.96, help="Bottom edge of side UI crop.")
    return parser.parse_args()


def segment_year(path: str) -> int | None:
    match = re.search(r"(20\d{2})_", path)
    return int(match.group(1)) if match else None


def iter_segments(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scene in manifest.get("scenes", []):
        for segment in scene.get("segments", []):
            path = segment["output_path"]
            rows.append(
                {
                    "video_index": scene["video_index"],
                    "scene_index": scene["scene_index"],
                    "segment_index": segment["index"],
                    "kind": segment["kind"],
                    "mode": segment["mode"],
                    "year": segment_year(path),
                    "path": path,
                }
            )
    return rows


def read_frame(path: Path, sample_time: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count else sample_time
        time_sec = min(max(0.0, sample_time), max(0.0, duration - 0.05))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(time_sec * fps)))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read frame: {path}")
        return frame
    finally:
        cap.release()


def crop_side(frame: np.ndarray, side: str, args: argparse.Namespace) -> np.ndarray:
    height, width = frame.shape[:2]
    y1 = round(height * args.y1)
    y2 = round(height * args.y2)
    if side == "left":
        x1 = 0
        x2 = round(width * args.left_x2)
    else:
        x1 = round(width * args.right_x1)
        x2 = width
    return frame[y1:y2, x1:x2]


def preprocess(crop: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    return cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip().upper()
    text = text.replace("．", ".").replace("·", ".").replace("。", ".")
    text = re.sub(r"\s+", "", text)
    return text


def compact_text(text: str | None) -> str:
    text = normalize_text(text)
    return re.sub(r"[^A-Z0-9\u4e00-\u9fff]", "", text)


def canonical_team_name(team_name: str) -> str:
    return normalize_text(team_name)


def team_aliases(team_name: str) -> set[str]:
    name = canonical_team_name(team_name)
    aliases = {name, compact_text(name)}
    city_prefixes = [
        "上海",
        "深圳",
        "广州",
        "西安",
        "济南",
        "南京",
        "佛山",
        "成都",
        "武汉",
        "北京",
        "杭州",
        "苏州",
        "重庆",
        "长沙",
        "厦门",
        "南通",
        "无锡",
        "桐乡",
    ]
    compact = compact_text(name)
    for prefix in city_prefixes:
        if compact.startswith(prefix):
            aliases.add(compact[len(prefix) :])
    if "EDG" in compact:
        aliases.update(["EDG", "EDGM", "上海EDGM"])
    if "RNG" in compact:
        aliases.update(["RNG", "RNGM", "上海RNGM"])
    if "DYG" in compact:
        aliases.update(["DYG", "深圳DYG"])
    if "TTG" in compact:
        aliases.update(["TTG", "广州TTG"])
    if "WE" in compact:
        aliases.update(["WE", "西安WE"])
    if "RW" in compact:
        aliases.update(["RW", "RW侠", "济南RW侠"])
    if "HERO" in compact:
        aliases.update(["HERO", "HERO久竞", "南京HERO", "南京HERO久竞", "南通HERO久竞"])
    if "TES" in compact:
        aliases.update(["TES", "TESA", "长沙TES", "长沙TESA"])
    if "GK" in compact or "DRG" in compact:
        aliases.update(["GK", "DRG", "佛山GK", "佛山DRG", "佛山DRGGK"])
    if "TCG" in compact:
        aliases.update(["TCG", "无锡TCG"])
    if "XYG" in compact:
        aliases.add("XYG")
    if "VG" in compact:
        aliases.add("VG")
    if "狼" in compact:
        aliases.update(["狼队", "重庆狼队"])
    return {alias for alias in aliases if alias}


def load_team_alias_index(schedules_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(schedules_path.read_text(encoding="utf-8"))
    teams = set()
    for row in data.get("schedules", []):
        for key in ["team_a_name", "team_b_name"]:
            if row.get(key):
                teams.add(row[key])
    index: dict[str, dict[str, Any]] = {}
    for team in sorted(teams):
        for alias in team_aliases(team):
            index[compact_text(alias)] = {"team": team, "alias": alias}
    return index


def run_ocr(ocr: RapidOCR, image: np.ndarray, min_confidence: float) -> list[dict[str, Any]]:
    result, _elapsed = ocr(image)
    lines = []
    for item in result or []:
        points, text, confidence = item
        confidence = float(confidence)
        text = normalize_text(text)
        if confidence < min_confidence or not text:
            continue
        lines.append({"text": text, "confidence": round(confidence, 6), "points": points})
    return lines


def team_hits_from_lines(lines: list[dict[str, Any]], alias_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for line in lines:
        text = compact_text(line["text"])
        if not text:
            continue
        line_hits = []
        for alias_key, payload in alias_index.items():
            if len(alias_key) < 2:
                continue
            if alias_key in text:
                score = min(1.0, 0.5 + len(alias_key) / max(6, len(text)))
                line_hits.append(
                    {
                        "team": payload["team"],
                        "alias": payload["alias"],
                        "score": round(score * line["confidence"], 6),
                    }
                )
        if not line_hits:
            continue
        line_hits.sort(key=lambda item: (item["score"], len(compact_text(item["alias"]))), reverse=True)
        hits.append({"text": line["text"], "confidence": line["confidence"], **line_hits[0]})
    return hits


def consensus_team(hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not hits:
        return None
    score_by_team: dict[str, float] = defaultdict(float)
    count_by_team: Counter[str] = Counter()
    aliases_by_team: dict[str, set[str]] = defaultdict(set)
    for hit in hits:
        team = hit["team"]
        score_by_team[team] += float(hit["score"])
        count_by_team[team] += 1
        aliases_by_team[team].add(hit["alias"])
    ranked = sorted(
        score_by_team,
        key=lambda team: (count_by_team[team], score_by_team[team], len(team)),
        reverse=True,
    )
    team = ranked[0]
    return {
        "team": team,
        "vote_count": count_by_team[team],
        "score": round(score_by_team[team], 6),
        "aliases": sorted(aliases_by_team[team]),
    }


def process_segment(
    ocr: RapidOCR,
    segment: dict[str, Any],
    alias_index: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    frame = read_frame(Path(segment["path"]), args.sample_time)
    segment_id = f"v{segment['video_index']:02d}s{segment['scene_index']:02d}seg{segment['segment_index']:02d}_{segment['kind']}"
    frame_dir = output_dir / "side_frames" / segment_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    side_results = {}
    for side in ["left", "right"]:
        crop = crop_side(frame, side, args)
        prepared = preprocess(crop)
        lines = run_ocr(ocr, prepared, args.min_confidence)
        hits = team_hits_from_lines(lines, alias_index)
        crop_path = frame_dir / f"{side}_side.jpg"
        prepared_path = frame_dir / f"{side}_side_prepared.jpg"
        cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(prepared_path), prepared, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        side_results[side] = {
            "crop_path": str(crop_path),
            "prepared_path": str(prepared_path),
            "ocr_lines": lines,
            "team_hits": hits,
            "team_consensus": consensus_team(hits),
        }

    return {
        **segment,
        "sample_time": args.sample_time,
        "left_team_text": (side_results["left"]["team_consensus"] or {}).get("team"),
        "right_team_text": (side_results["right"]["team_consensus"] or {}).get("team"),
        "left": side_results["left"],
        "right": side_results["right"],
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
                "left_votes",
                "right_team_text",
                "right_votes",
                "left_ocr_texts",
                "right_ocr_texts",
                "path",
            ],
        )
        writer.writeheader()
        for record in records:
            left = record["left"]["team_consensus"] or {}
            right = record["right"]["team_consensus"] or {}
            writer.writerow(
                {
                    "year": record["year"],
                    "video_index": record["video_index"],
                    "scene_index": record["scene_index"],
                    "segment_index": record["segment_index"],
                    "kind": record["kind"],
                    "left_team_text": left.get("team"),
                    "left_votes": left.get("vote_count"),
                    "right_team_text": right.get("team"),
                    "right_votes": right.get("vote_count"),
                    "left_ocr_texts": " | ".join(line["text"] for line in record["left"]["ocr_lines"]),
                    "right_ocr_texts": " | ".join(line["text"] for line in record["right"]["ocr_lines"]),
                    "path": record["path"],
                }
            )


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    schedules_path = Path(args.schedules)
    output_dir = Path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alias_index = load_team_alias_index(schedules_path)
    segments = iter_segments(manifest)
    ocr = RapidOCR()

    records = []
    for idx, segment in enumerate(segments, start=1):
        print(
            f"[{idx}/{len(segments)}] side player OCR {segment['year']} "
            f"v{segment['video_index']:02d}s{segment['scene_index']:02d}seg{segment['segment_index']:02d}",
            flush=True,
        )
        records.append(process_segment(ocr, segment, alias_index, args, output_dir))

    json_path = output_dir / "side_player_team_ocr.json"
    csv_path = output_dir / "side_player_team_ocr_summary.csv"
    write_csv(records, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_manifest": str(manifest_path),
                    "schedules": str(schedules_path),
                    "segment_count": len(records),
                    "sample_time": args.sample_time,
                    "side_roi": {
                        "left_x2": args.left_x2,
                        "right_x1": args.right_x1,
                        "y1": args.y1,
                        "y2": args.y2,
                    },
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
