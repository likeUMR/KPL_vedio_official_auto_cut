#!/usr/bin/env python3
"""OCR left/right team-name ROIs from segment opening match UI."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR year-group left/right team-name ROIs from segment top UI.")
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scene_segments_complete_focus_with_intro/complete_focus_segment_manifest.json",
        help="Segment manifest.",
    )
    parser.add_argument(
        "--roi-config",
        default="data/match_ui_year_roi_review/match_ui_year_roi_config.json",
        help="Year-group ROI config from review_match_ui_year_rois.py.",
    )
    parser.add_argument("--output-dir", default="data/match_ui_team_roi_ocr", help="Output directory.")
    parser.add_argument("--sample-time", type=float, default=1.5, help="Frame sample time.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="OCR confidence threshold.")
    return parser.parse_args()


def segment_year(path: str) -> int | None:
    match = re.search(r"(20\d{2})_", path)
    return int(match.group(1)) if match else None


def year_group(year: int) -> str:
    if year <= 2021:
        return "2019-2021"
    if year <= 2023:
        return "2022-2023"
    return "2024-2026"


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


def crop_roi(frame: np.ndarray, top_ratio: float, roi: dict[str, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    top_height = max(1, round(height * top_ratio))
    band = frame[:top_height, :, :]
    band_h, band_w = band.shape[:2]
    x1 = round(band_w * roi["x1"])
    x2 = round(band_w * roi["x2"])
    y1 = round(band_h * roi["y1"])
    y2 = round(band_h * roi["y2"])
    return band[y1:y2, x1:x2]


def preprocess(crop: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(enlarged, (0, 0), 1.0)
    return cv2.addWeighted(enlarged, 1.7, blurred, -0.7, 0)


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("．", ".").replace("·", ".").replace("：", ":")
    return re.sub(r"\s+", "", text)


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


def best_text(lines: list[dict[str, Any]]) -> str | None:
    filtered = []
    for line in lines:
        text = line["text"]
        if re.fullmatch(r"[\d.,:%/\\|+\-]+", text):
            continue
        if re.search(r"\d+\.\d+[kK万]|vs|FPS|KDA|ms|KPL官方|CLEAR|上汽|同程|勇闯|东鹏|美团", text, re.I):
            continue
        filtered.append(line)
    if not filtered:
        return None
    filtered.sort(key=lambda item: (item["confidence"], len(item["text"])), reverse=True)
    return filtered[0]["text"]


def process_segment(
    ocr: RapidOCR,
    segment: dict[str, Any],
    roi_config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    year = segment["year"]
    if year is None:
        raise RuntimeError(f"cannot parse year from path: {segment['path']}")
    group = year_group(year)
    rois = roi_config["year_groups"][group]
    top_ratio = float(roi_config["metadata"].get("top_ratio", 0.10))
    frame = read_frame(Path(segment["path"]), args.sample_time)

    segment_id = f"v{segment['video_index']:02d}s{segment['scene_index']:02d}seg{segment['segment_index']:02d}_{segment['kind']}"
    frame_dir = output_dir / "roi_frames" / segment_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    side_results = {}
    for side in ["left", "right"]:
        crop = crop_roi(frame, top_ratio, rois[side])
        prepared = preprocess(crop)
        lines = run_ocr(ocr, prepared, args.min_confidence)
        crop_path = frame_dir / f"{side}_roi.jpg"
        prepared_path = frame_dir / f"{side}_roi_prepared.jpg"
        cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(prepared_path), prepared, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        side_results[side] = {
            "roi": rois[side],
            "crop_path": str(crop_path),
            "prepared_path": str(prepared_path),
            "ocr_lines": lines,
            "team_text_candidate": best_text(lines),
        }

    return {
        **segment,
        "year_group": group,
        "sample_time": args.sample_time,
        "left_team_text": side_results["left"]["team_text_candidate"],
        "right_team_text": side_results["right"]["team_text_candidate"],
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
                "right_team_text",
                "left_ocr_texts",
                "right_ocr_texts",
                "path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "year": record["year"],
                    "video_index": record["video_index"],
                    "scene_index": record["scene_index"],
                    "segment_index": record["segment_index"],
                    "kind": record["kind"],
                    "left_team_text": record["left_team_text"],
                    "right_team_text": record["right_team_text"],
                    "left_ocr_texts": " | ".join(line["text"] for line in record["left"]["ocr_lines"]),
                    "right_ocr_texts": " | ".join(line["text"] for line in record["right"]["ocr_lines"]),
                    "path": record["path"],
                }
            )


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    roi_config_path = Path(args.roi_config)
    output_dir = Path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roi_config = json.loads(roi_config_path.read_text(encoding="utf-8"))
    segments = iter_segments(manifest)
    ocr = RapidOCR()

    records = []
    for idx, segment in enumerate(segments, start=1):
        print(
            f"[{idx}/{len(segments)}] team ROI OCR {segment['year']} "
            f"v{segment['video_index']:02d}s{segment['scene_index']:02d}seg{segment['segment_index']:02d}",
            flush=True,
        )
        records.append(process_segment(ocr, segment, roi_config, args, output_dir))

    json_path = output_dir / "match_ui_team_roi_ocr.json"
    csv_path = output_dir / "match_ui_team_roi_ocr_summary.csv"
    write_csv(records, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_manifest": str(manifest_path),
                    "roi_config": str(roi_config_path),
                    "segment_count": len(records),
                    "sample_time": args.sample_time,
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
