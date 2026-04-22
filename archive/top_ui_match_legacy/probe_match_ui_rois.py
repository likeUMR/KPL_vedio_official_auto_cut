#!/usr/bin/env python3
"""Probe year-specific top UI ROIs for KPL team-name OCR."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


TEAM_HINTS = [
    "AG",
    "DYG",
    "EDG",
    "EDG.M",
    "eStar",
    "GK",
    "Hero",
    "KSG",
    "LGD",
    "QG",
    "RNG",
    "RNG.M",
    "RW",
    "TES",
    "TTG",
    "VG",
    "WE",
    "XYG",
    "狼队",
    "佛山",
    "广州",
    "济南",
    "南京",
    "上海",
    "深圳",
    "苏州",
    "武汉",
    "西安",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe universal/year-specific team-name ROIs in top match UI.")
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scene_segments_complete_focus_with_intro/complete_focus_segment_manifest.json",
        help="Segment manifest.",
    )
    parser.add_argument("--output-dir", default="data/match_ui_roi_probe", help="Output directory.")
    parser.add_argument("--sample-time", type=float, default=1.5, help="Frame sample time.")
    parser.add_argument("--top-ratio", type=float, default=0.10, help="Top UI band ratio.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="OCR confidence threshold.")
    parser.add_argument(
        "--max-samples-per-year",
        type=int,
        default=2,
        help="Limit representative segments per year to keep ROI probing fast. Default: 2.",
    )
    return parser.parse_args()


def segment_year(segment_path: str) -> int | None:
    match = re.search(r"(20\d{2})_", segment_path)
    return int(match.group(1)) if match else None


def iter_segments(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for scene in manifest.get("scenes", []):
        for segment in scene.get("segments", []):
            path = segment["output_path"]
            items.append(
                {
                    "video_index": scene["video_index"],
                    "scene_index": scene["scene_index"],
                    "segment_index": segment["index"],
                    "kind": segment["kind"],
                    "path": path,
                    "year": segment_year(path),
                }
            )
    return items


def read_top_band(path: Path, sample_time: float, top_ratio: float) -> np.ndarray:
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
        h = max(1, round(frame.shape[0] * top_ratio))
        return frame[:h, :, :]
    finally:
        cap.release()


def preprocess(crop: np.ndarray) -> np.ndarray:
    scale = 3
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(enlarged, (0, 0), 1.0)
    return cv2.addWeighted(enlarged, 1.6, blurred, -0.6, 0)


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("．", ".").replace("·", ".").replace("：", ":")
    return re.sub(r"\s+", "", text)


def ocr_texts(ocr: RapidOCR, image: np.ndarray, min_confidence: float) -> list[str]:
    result, _elapsed = ocr(image)
    texts = []
    for item in result or []:
        _points, text, confidence = item
        if float(confidence) < min_confidence:
            continue
        text = normalize_text(text)
        if text:
            texts.append(text)
    return texts


def teamish_score(texts: list[str]) -> float:
    score = 0.0
    joined = " ".join(texts)
    for hint in TEAM_HINTS:
        if re.search(re.escape(hint), joined, re.I):
            score += 3.0
    for text in texts:
        if re.fullmatch(r"[\d.,:%/\\|+\-]+", text):
            score -= 1.0
        if re.search(r"\d+\.\d+[kK万]|\d+vs\d+|FPS|KDA|ms|CLEAR|KPL官方合作", text, re.I):
            score -= 1.5
        if re.search(r"[A-Za-z]{2,}(\.|$)|[\u4e00-\u9fff]{2,}", text):
            score += 0.8
    return round(score, 3)


def roi_specs() -> list[dict[str, Any]]:
    specs = []
    # Compact normalized x ranges across the top band. These avoid the center kill/gold UI
    # and focus on likely left/right team-name areas.
    ranges = [
        ("left_outer", 0.05, 0.28),
        ("left_mid", 0.10, 0.36),
        ("left_inner", 0.16, 0.44),
        ("right_inner", 0.56, 0.84),
        ("right_mid", 0.64, 0.90),
        ("right_outer", 0.72, 0.95),
    ]
    y_ranges = [
        ("full", 0.00, 1.00),
        ("middle", 0.22, 0.82),
    ]
    for x_name, x1, x2 in ranges:
        for y_name, y1, y2 in y_ranges:
            side = "left" if "left" in x_name else "right" if "right" in x_name else "center"
            specs.append({"name": f"{x_name}_{y_name}", "side": side, "x1": x1, "x2": x2, "y1": y1, "y2": y2})
    return specs


def select_probe_segments(segments: list[dict[str, Any]], max_samples_per_year: int) -> list[dict[str, Any]]:
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_scene: set[tuple[int, int, int]] = set()
    for segment in segments:
        year = segment["year"]
        if year is None:
            continue
        scene_key = (year, segment["video_index"], segment["scene_index"])
        if scene_key in seen_scene:
            continue
        seen_scene.add(scene_key)
        by_year[year].append(segment)

    selected: list[dict[str, Any]] = []
    for year in sorted(by_year):
        # Use the earliest scenes first; they are enough to expose the year-specific UI layout.
        selected.extend(by_year[year][:max_samples_per_year])
    return selected


def crop_by_spec(band: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    h, w = band.shape[:2]
    x1 = max(0, min(w - 1, round(w * spec["x1"])))
    x2 = max(x1 + 1, min(w, round(w * spec["x2"])))
    y1 = max(0, min(h - 1, round(h * spec["y1"])))
    y2 = max(y1 + 1, min(h, round(h * spec["y2"])))
    return band[y1:y2, x1:x2]


def annotate_year_review(year: int, sample_band: np.ndarray, best_specs: list[dict[str, Any]], output_path: Path) -> None:
    image = cv2.resize(sample_band, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    for spec in best_specs:
        color = (0, 255, 255) if spec["side"] == "left" else (255, 0, 255)
        x1, x2 = round(w * spec["x1"]), round(w * spec["x2"])
        y1, y2 = round(h * spec["y1"]), round(h * spec["y2"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, spec["name"], (x1 + 4, max(18, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    cv2.putText(image, str(year), (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_segments = iter_segments(manifest)
    segments = select_probe_segments(all_segments, args.max_samples_per_year)
    specs = roi_specs()
    ocr = RapidOCR()

    rows = []
    sample_band_by_year: dict[int, np.ndarray] = {}
    for idx, segment in enumerate(segments, start=1):
        year = segment["year"]
        if year is None:
            continue
        print(
            f"[{idx}/{len(segments)}] probing {year} "
            f"v{segment['video_index']:02d}s{segment['scene_index']:02d}",
            flush=True,
        )
        band = read_top_band(Path(segment["path"]), args.sample_time, args.top_ratio)
        sample_band_by_year.setdefault(year, band)
        for spec in specs:
            crop = preprocess(crop_by_spec(band, spec))
            texts = ocr_texts(ocr, crop, args.min_confidence)
            rows.append(
                {
                    **segment,
                    "roi_name": spec["name"],
                    "side": spec["side"],
                    "x1": spec["x1"],
                    "x2": spec["x2"],
                    "y1": spec["y1"],
                    "y2": spec["y2"],
                    "texts": texts,
                    "teamish_score": teamish_score(texts),
                }
            )

    by_year_roi: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_year_roi[(row["year"], row["side"], row["roi_name"])].append(row["teamish_score"])

    recommendations = []
    years = sorted({row["year"] for row in rows})
    for year in years:
        year_specs = []
        for side in ["left", "right"]:
            candidates = [
                {
                    "year": year,
                    "side": side,
                    "roi_name": roi_name,
                    "mean_score": round(float(np.mean(scores)), 3),
                    "median_score": round(float(np.median(scores)), 3),
                    "sample_count": len(scores),
                }
                for (roi_year, roi_side, roi_name), scores in by_year_roi.items()
                if roi_year == year and roi_side == side
            ]
            candidates.sort(key=lambda item: (item["median_score"], item["mean_score"]), reverse=True)
            if candidates:
                best = candidates[0]
                spec = next(item for item in specs if item["name"] == best["roi_name"])
                best.update({key: spec[key] for key in ["x1", "x2", "y1", "y2"]})
                recommendations.append(best)
                year_specs.append({**spec, "side": side})
        if year in sample_band_by_year:
            annotate_year_review(year, sample_band_by_year[year], year_specs, output_dir / "year_roi_reviews" / f"{year}_roi_review.jpg")

    csv_path = output_dir / "match_ui_roi_probe_rows.csv"
    rec_path = output_dir / "match_ui_roi_recommendations.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "year",
                "video_index",
                "scene_index",
                "segment_index",
                "kind",
                "side",
                "roi_name",
                "x1",
                "x2",
                "y1",
                "y2",
                "teamish_score",
                "texts",
                "path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "texts": " | ".join(row["texts"])})

    rec_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_manifest": str(manifest_path),
                    "row_count": len(rows),
                    "source_segment_count": len(all_segments),
                    "probe_segment_count": len(segments),
                    "max_samples_per_year": args.max_samples_per_year,
                    "sample_time": args.sample_time,
                    "top_ratio": args.top_ratio,
                    "probe_csv": str(csv_path),
                },
                "recommendations": recommendations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"recommendations: {rec_path}")
    print(f"rows: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
