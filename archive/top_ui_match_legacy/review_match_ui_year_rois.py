#!/usr/bin/env python3
"""Create visual reviews for year-specific KPL match UI team-name ROIs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render top UI samples with proposed year-specific team ROIs.")
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scene_segments_complete_focus_with_intro/complete_focus_segment_manifest.json",
        help="Segment manifest.",
    )
    parser.add_argument("--output-dir", default="data/match_ui_year_roi_review", help="Output directory.")
    parser.add_argument("--sample-time", type=float, default=1.5, help="Frame sample time.")
    parser.add_argument("--top-ratio", type=float, default=0.10, help="Top UI band ratio.")
    parser.add_argument("--max-samples-per-year", type=int, default=2, help="Samples per year.")
    return parser.parse_args()


def segment_year(path: str) -> int | None:
    match = re.search(r"(20\d{2})_", path)
    return int(match.group(1)) if match else None


def iter_segments(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    seen_scene: set[tuple[int, int, int]] = set()
    for scene in manifest.get("scenes", []):
        for segment in scene.get("segments", []):
            year = segment_year(segment["output_path"])
            if year is None:
                continue
            key = (year, scene["video_index"], scene["scene_index"])
            if key in seen_scene:
                continue
            seen_scene.add(key)
            items.append(
                {
                    "year": year,
                    "video_index": scene["video_index"],
                    "scene_index": scene["scene_index"],
                    "path": segment["output_path"],
                }
            )
    return items


def proposed_rois() -> dict[str, dict[str, float]]:
    # Normalized coordinates inside the top 10% band.
    return {
        "legacy_2019_2021_left": {"x1": 0.12, "x2": 0.27, "y1": 0.12, "y2": 0.78},
        "legacy_2019_2021_right": {"x1": 0.62, "x2": 0.80, "y1": 0.12, "y2": 0.78},
        "mid_2022_2023_left": {"x1": 0.10, "x2": 0.30, "y1": 0.10, "y2": 0.82},
        "mid_2022_2023_right": {"x1": 0.62, "x2": 0.84, "y1": 0.10, "y2": 0.82},
        "modern_2024_2026_left": {"x1": 0.10, "x2": 0.28, "y1": 0.08, "y2": 0.88},
        "modern_2024_2026_right": {"x1": 0.66, "x2": 0.86, "y1": 0.08, "y2": 0.88},
    }


def rois_for_year(year: int) -> list[tuple[str, dict[str, float]]]:
    rois = proposed_rois()
    if year <= 2021:
        keys = ["legacy_2019_2021_left", "legacy_2019_2021_right"]
    elif year <= 2023:
        keys = ["mid_2022_2023_left", "mid_2022_2023_right"]
    else:
        keys = ["modern_2024_2026_left", "modern_2024_2026_right"]
    return [(key, rois[key]) for key in keys]


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
        top_h = max(1, round(frame.shape[0] * top_ratio))
        return frame[:top_h, :, :]
    finally:
        cap.release()


def draw_rois(image: np.ndarray, year: int) -> np.ndarray:
    result = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    height, width = result.shape[:2]
    for name, roi in rois_for_year(year):
        color = (0, 255, 255) if "left" in name else (255, 0, 255)
        x1, x2 = round(width * roi["x1"]), round(width * roi["x2"])
        y1, y2 = round(height * roi["y1"]), round(height * roi["y2"])
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, "left" if "left" in name else "right", (x1 + 4, max(18, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return result


def build_review(samples: list[dict[str, Any]], output_path: Path) -> None:
    item_w, item_h = 960, 160
    label_h = 34
    canvas = np.full((len(samples) * (item_h + label_h), item_w, 3), 28, dtype=np.uint8)
    for idx, sample in enumerate(samples):
        image = cv2.resize(sample["image"], (item_w, item_h), interpolation=cv2.INTER_AREA)
        y = idx * (item_h + label_h)
        canvas[y : y + item_h, :item_w] = image
        label = f"{sample['year']} v{sample['video_index']:02d}s{sample['scene_index']:02d}"
        cv2.rectangle(canvas, (0, y + item_h), (item_w, y + item_h + label_h), (16, 16, 16), -1)
        cv2.putText(canvas, label, (8, y + item_h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = iter_segments(manifest)

    counts: dict[int, int] = {}
    samples = []
    for segment in segments:
        year = segment["year"]
        if counts.get(year, 0) >= args.max_samples_per_year:
            continue
        counts[year] = counts.get(year, 0) + 1
        top_band = read_top_band(Path(segment["path"]), args.sample_time, args.top_ratio)
        annotated = draw_rois(top_band, year)
        samples.append({**segment, "image": annotated})

    review_path = output_dir / "match_ui_year_roi_review.jpg"
    config_path = output_dir / "match_ui_year_roi_config.json"
    build_review(samples, review_path)
    config = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest": str(manifest_path),
            "sample_time": args.sample_time,
            "top_ratio": args.top_ratio,
            "review_image": str(review_path),
        },
        "year_groups": {
            "2019-2021": {
                "left": proposed_rois()["legacy_2019_2021_left"],
                "right": proposed_rois()["legacy_2019_2021_right"],
            },
            "2022-2023": {
                "left": proposed_rois()["mid_2022_2023_left"],
                "right": proposed_rois()["mid_2022_2023_right"],
            },
            "2024-2026": {
                "left": proposed_rois()["modern_2024_2026_left"],
                "right": proposed_rois()["modern_2024_2026_right"],
            },
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"review: {review_path}")
    print(f"config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
