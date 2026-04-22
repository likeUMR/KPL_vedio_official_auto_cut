#!/usr/bin/env python3
"""Extract colorful title/operator regions from the opening black-white filter frame."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample each final segment at 1.5s, crop away border UI, extract saturated colorful "
            "regions from the black-white filter frame, and save review crops for OCR."
        )
    )
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scene_segments_complete_focus_with_intro/complete_focus_segment_manifest.json",
        help="Segment manifest produced by split_complete_focus_segments.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/segment_title_region_candidates",
        help="Directory for candidate crops, JSON, and review image.",
    )
    parser.add_argument("--sample-time", type=float, default=1.5, help="Representative frame time in seconds.")
    parser.add_argument("--crop-ratio", type=float, default=0.10, help="Crop this ratio from all four borders.")
    parser.add_argument("--saturation-threshold", type=int, default=55, help="HSV saturation threshold for color mask.")
    parser.add_argument("--value-threshold", type=int, default=45, help="HSV value threshold for color mask.")
    parser.add_argument("--min-area-ratio", type=float, default=0.00035, help="Minimum contour area as full-frame ratio.")
    parser.add_argument("--merge-dilate", type=int, default=19, help="Dilation kernel size used to merge nearby color text.")
    parser.add_argument("--pad", type=int, default=18, help="Padding around candidate boxes.")
    return parser.parse_args()


def read_frame(path: Path, sample_time: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count > 0 else sample_time
        time_sec = min(max(0.0, sample_time), max(0.0, duration - 0.05))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(time_sec * fps)))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read frame at {time_sec:.2f}s: {path}")
        return frame
    finally:
        cap.release()


def color_mask(frame: np.ndarray, saturation_threshold: int, value_threshold: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation >= saturation_threshold) & (value >= value_threshold)).astype("uint8") * 255
    return mask


def crop_center(frame: np.ndarray, crop_ratio: float) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = frame.shape[:2]
    left = round(width * crop_ratio)
    right = round(width * (1 - crop_ratio))
    top = round(height * crop_ratio)
    bottom = round(height * (1 - crop_ratio))
    return frame[top:bottom, left:right], (left, top)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def non_max_suppress(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    boxes = sorted(boxes, key=lambda item: item[2] * item[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if all(box_iou(box, other) < 0.45 for other in kept):
            kept.append(box)
    return kept


def extract_boxes(frame: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray]:
    cropped, (offset_x, offset_y) = crop_center(frame, args.crop_ratio)
    mask = color_mask(cropped, args.saturation_threshold, args.value_threshold)
    kernel_size = max(3, args.merge_dilate | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * args.min_area_ratio
    raw_boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(hull)
        area = w * h
        if area < min_area:
            continue
        if w < 30 or h < 16:
            continue
        x = max(0, x + offset_x - args.pad)
        y = max(0, y + offset_y - args.pad)
        x2 = min(frame.shape[1], x + w + args.pad * 2)
        y2 = min(frame.shape[0], y + h + args.pad * 2)
        raw_boxes.append((x, y, x2 - x, y2 - y))

    boxes = non_max_suppress(raw_boxes)
    records = []
    for index, (x, y, w, h) in enumerate(sorted(boxes, key=lambda item: (item[1], item[0])), start=1):
        crop = frame[y : y + h, x : x + w]
        region_mask = color_mask(crop, args.saturation_threshold, args.value_threshold)
        records.append(
            {
                "index": index,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": w * h,
                "color_pixel_ratio": round(float(np.mean(region_mask > 0)), 6),
            }
        )
    return records, mask


def iter_segments(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for scene in manifest.get("scenes", []):
        for segment in scene.get("segments", []):
            items.append(
                {
                    "video_index": scene["video_index"],
                    "scene_index": scene["scene_index"],
                    "segment_index": segment["index"],
                    "kind": segment["kind"],
                    "mode": segment["mode"],
                    "path": segment["output_path"],
                }
            )
    return items


def save_candidates(frame: np.ndarray, boxes: list[dict[str, Any]], segment_id: str, output_dir: Path) -> None:
    crop_dir = output_dir / "crops" / segment_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        crop = frame[y : y + h, x : x + w]
        path = crop_dir / f"candidate{box['index']:02d}_{x}-{y}-{w}x{h}.png"
        cv2.imwrite(str(path), crop)
        box["crop_path"] = str(path)


def build_review(records: list[dict[str, Any]], output_path: Path) -> None:
    thumb_w, thumb_h = 320, 180
    label_h = 44
    cols = 3
    rows = max(1, math.ceil(len(records) / cols))
    canvas = np.full((rows * (thumb_h + label_h), cols * thumb_w, 3), 28, dtype=np.uint8)

    for idx, record in enumerate(records):
        frame = cv2.imread(record["annotated_frame_path"])
        if frame is None:
            frame = np.full((thumb_h, thumb_w, 3), 40, dtype=np.uint8)
        frame = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(idx, cols)
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        canvas[y : y + thumb_h, x : x + thumb_w] = frame
        label1 = f"v{record['video_index']:02d}s{record['scene_index']:02d} seg{record['segment_index']:02d} {record['kind']}"
        label2 = f"boxes={len(record['boxes'])}"
        cv2.rectangle(canvas, (x, y + thumb_h), (x + thumb_w, y + thumb_h + label_h), (16, 16, 16), -1)
        cv2.putText(canvas, label1, (x + 8, y + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, label2, (x + 8, y + thumb_h + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 205, 205), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def process_segment(item: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = Path(item["path"])
    frame = read_frame(path, args.sample_time)
    boxes, _mask = extract_boxes(frame, args)
    segment_id = f"v{item['video_index']:02d}s{item['scene_index']:02d}seg{item['segment_index']:02d}_{item['kind']}"

    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    raw_frame_path = frame_dir / f"{segment_id}_frame_{args.sample_time:.2f}s.jpg"
    annotated_frame_path = frame_dir / f"{segment_id}_annotated.jpg"
    annotated = frame.copy()
    height, width = frame.shape[:2]
    cv2.rectangle(
        annotated,
        (round(width * args.crop_ratio), round(height * args.crop_ratio)),
        (round(width * (1 - args.crop_ratio)), round(height * (1 - args.crop_ratio))),
        (255, 255, 0),
        2,
    )
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(annotated, str(box["index"]), (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(raw_frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cv2.imwrite(str(annotated_frame_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    save_candidates(frame, boxes, segment_id, output_dir)

    return {
        **item,
        "sample_time": args.sample_time,
        "raw_frame_path": str(raw_frame_path),
        "annotated_frame_path": str(annotated_frame_path),
        "boxes": boxes,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = iter_segments(manifest)

    records = []
    for idx, item in enumerate(segments, start=1):
        print(f"[{idx}/{len(segments)}] extracting v{item['video_index']:02d}s{item['scene_index']:02d} {item['kind']}", flush=True)
        records.append(process_segment(item, args, output_dir))

    review_path = output_dir / "segment_title_region_review.jpg"
    json_path = output_dir / "segment_title_region_candidates.json"
    build_review(records, review_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_manifest": str(manifest_path),
                    "segment_count": len(records),
                    "sample_time": args.sample_time,
                    "crop_ratio": args.crop_ratio,
                    "saturation_threshold": args.saturation_threshold,
                    "value_threshold": args.value_threshold,
                    "review_image": str(review_path),
                },
                "segments": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"json: {json_path}")
    print(f"review: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
