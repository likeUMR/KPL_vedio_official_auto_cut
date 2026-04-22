#!/usr/bin/env python3
"""OCR the top match UI from segment opening black-white filter frames."""

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
    parser = argparse.ArgumentParser(
        description="Sample each segment at 1.5s, crop the top match UI band, and OCR teams/scores."
    )
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scene_segments_complete_focus_with_intro/complete_focus_segment_manifest.json",
        help="Segment manifest produced by split_complete_focus_segments.py.",
    )
    parser.add_argument("--output-dir", default="data/segment_match_ui_ocr", help="Output directory.")
    parser.add_argument("--sample-time", type=float, default=1.5, help="Representative frame time in seconds.")
    parser.add_argument("--top-ratio", type=float, default=0.10, help="Top UI band height ratio. Default: 0.10.")
    parser.add_argument(
        "--horizontal-crop-ratio",
        type=float,
        default=0.055,
        help="Crop this ratio from left/right edges to remove minimap/side UI. Default: 0.055.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Drop OCR lines below this confidence.")
    return parser.parse_args()


def read_frame(path: Path, sample_time: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count else sample_time
        time_sec = min(max(0.0, sample_time), max(0.0, duration - 0.05))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(time_sec * fps)))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read frame at {time_sec:.2f}s: {path}")
        return frame
    finally:
        cap.release()


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


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("：", ":").replace("｜", "|").replace("·", ".").replace("．", ".")
    text = re.sub(r"\s+", "", text)
    return text


def crop_top_ui(frame: np.ndarray, top_ratio: float, horizontal_crop_ratio: float) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = frame.shape[:2]
    top_h = max(1, round(height * top_ratio))
    left = round(width * horizontal_crop_ratio)
    right = round(width * (1 - horizontal_crop_ratio))
    return frame[:top_h, left:right], (left, 0)


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    # Upscale the thin scoreboard band. Keep color because team bars can help OCR.
    scale = 3
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    sharpen = cv2.GaussianBlur(enlarged, (0, 0), 1.0)
    return cv2.addWeighted(enlarged, 1.5, sharpen, -0.5, 0)


def ocr_lines(ocr: RapidOCR, image: np.ndarray, min_confidence: float) -> list[dict[str, Any]]:
    result, _elapsed = ocr(image)
    lines = []
    for item in result or []:
        points, text, confidence = item
        text = normalize_text(text)
        confidence = float(confidence)
        if confidence < min_confidence or not text:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        lines.append(
            {
                "text": text,
                "confidence": round(confidence, 6),
                "x_center": round(float(sum(xs) / len(xs)), 3),
                "y_center": round(float(sum(ys) / len(ys)), 3),
                "points": points,
            }
        )
    return sorted(lines, key=lambda item: (item["y_center"], item["x_center"]))


def parse_score(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"(?<!\d)([0-4])(?!\d)", text)]


def infer_ui(lines: list[dict[str, Any]], crop_width: int) -> dict[str, Any]:
    left_lines = [line for line in lines if line["x_center"] < crop_width * 0.45]
    right_lines = [line for line in lines if line["x_center"] > crop_width * 0.55]
    center_lines = [line for line in lines if crop_width * 0.35 <= line["x_center"] <= crop_width * 0.65]

    all_scores: list[tuple[int, dict[str, Any]]] = []
    for line in lines:
        for score in parse_score(line["text"]):
            all_scores.append((score, line))

    left_score = None
    right_score = None
    if len(all_scores) >= 2:
        ordered = sorted(all_scores, key=lambda item: item[1]["x_center"])
        left_score = ordered[0][0]
        right_score = ordered[-1][0]

    def best_team(side_lines: list[dict[str, Any]]) -> str | None:
        candidates = []
        for line in side_lines:
            text = line["text"]
            if re.fullmatch(r"[\d:|/\\-]+", text):
                continue
            if len(text) <= 1:
                continue
            if re.search(r"(FPS|KDA|ms|VS|/)", text, re.I):
                continue
            candidates.append(line)
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda item: (item["confidence"], len(item["text"])), reverse=True)
        return candidates[0]["text"]

    return {
        "left_team_candidate": best_team(left_lines),
        "right_team_candidate": best_team(right_lines),
        "left_score_candidate": left_score,
        "right_score_candidate": right_score,
        "left_texts": [line["text"] for line in left_lines],
        "center_texts": [line["text"] for line in center_lines],
        "right_texts": [line["text"] for line in right_lines],
    }


def annotate(crop: np.ndarray, lines: list[dict[str, Any]], output_path: Path) -> None:
    image = crop.copy()
    for index, line in enumerate(lines, start=1):
        pts = np.array(line["points"], dtype=np.int32)
        cv2.polylines(image, [pts], True, (0, 255, 255), 2)
        cv2.putText(
            image,
            str(index),
            (int(line["x_center"]), max(18, int(line["y_center"]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def process_segment(ocr: RapidOCR, item: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    frame = read_frame(Path(item["path"]), args.sample_time)
    crop, _offset = crop_top_ui(frame, args.top_ratio, args.horizontal_crop_ratio)
    prepared = preprocess_for_ocr(crop)
    lines = ocr_lines(ocr, prepared, args.min_confidence)
    inferred = infer_ui(lines, prepared.shape[1])

    segment_id = f"v{item['video_index']:02d}s{item['scene_index']:02d}seg{item['segment_index']:02d}_{item['kind']}"
    raw_path = output_dir / "frames" / f"{segment_id}_top_ui.jpg"
    prepared_path = output_dir / "frames" / f"{segment_id}_top_ui_prepared.jpg"
    annotated_path = output_dir / "frames" / f"{segment_id}_top_ui_annotated.jpg"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(raw_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cv2.imwrite(str(prepared_path), prepared, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    annotate(prepared, lines, annotated_path)

    return {
        **item,
        "sample_time": args.sample_time,
        "top_ui_frame_path": str(raw_path),
        "prepared_frame_path": str(prepared_path),
        "annotated_frame_path": str(annotated_path),
        **inferred,
        "ocr_lines": lines,
    }


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_index",
                "scene_index",
                "segment_index",
                "kind",
                "left_team_candidate",
                "right_team_candidate",
                "left_score_candidate",
                "right_score_candidate",
                "left_texts",
                "center_texts",
                "right_texts",
                "path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "video_index": record["video_index"],
                    "scene_index": record["scene_index"],
                    "segment_index": record["segment_index"],
                    "kind": record["kind"],
                    "left_team_candidate": record["left_team_candidate"],
                    "right_team_candidate": record["right_team_candidate"],
                    "left_score_candidate": record["left_score_candidate"],
                    "right_score_candidate": record["right_score_candidate"],
                    "left_texts": " | ".join(record["left_texts"]),
                    "center_texts": " | ".join(record["center_texts"]),
                    "right_texts": " | ".join(record["right_texts"]),
                    "path": record["path"],
                }
            )


def build_review(records: list[dict[str, Any]], output_path: Path) -> None:
    thumb_w, thumb_h = 420, 90
    label_h = 48
    cols = 2
    rows = (len(records) + cols - 1) // cols
    canvas = np.full((rows * (thumb_h + label_h), cols * thumb_w, 3), 28, dtype=np.uint8)
    for idx, record in enumerate(records):
        image = cv2.imread(record["annotated_frame_path"])
        if image is None:
            image = np.full((thumb_h, thumb_w, 3), 45, dtype=np.uint8)
        image = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(idx, cols)
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        canvas[y : y + thumb_h, x : x + thumb_w] = image
        label1 = f"v{record['video_index']:02d}s{record['scene_index']:02d}seg{record['segment_index']:02d} {record['kind']}"
        label2 = f"{record['left_team_candidate']} {record['left_score_candidate']}:{record['right_score_candidate']} {record['right_team_candidate']}"
        cv2.rectangle(canvas, (x, y + thumb_h), (x + thumb_w, y + thumb_h + label_h), (16, 16, 16), -1)
        cv2.putText(canvas, label1, (x + 8, y + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, label2[:54], (x + 8, y + thumb_h + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = iter_segments(manifest)
    ocr = RapidOCR()

    records = []
    for index, item in enumerate(segments, start=1):
        print(
            f"[{index}/{len(segments)}] top UI OCR v{item['video_index']:02d}s{item['scene_index']:02d}seg{item['segment_index']:02d}",
            flush=True,
        )
        records.append(process_segment(ocr, item, args, output_dir))

    json_path = output_dir / "segment_match_ui_ocr.json"
    csv_path = output_dir / "segment_match_ui_ocr_summary.csv"
    review_path = output_dir / "segment_match_ui_ocr_review.jpg"
    write_csv(records, csv_path)
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
                    "top_ratio": args.top_ratio,
                    "horizontal_crop_ratio": args.horizontal_crop_ratio,
                    "summary_csv": str(csv_path),
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
    print(f"csv: {csv_path}")
    print(f"review: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
