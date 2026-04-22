#!/usr/bin/env python3
"""Build a brightness-histogram template for valid KPL scene transition effects."""

from __future__ import annotations

import argparse
import csv
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
            "Extract the 5th frame inside every detected transition interval, compare "
            "its 0-255 luminance histogram against other candidates, drop outliers, "
            "and save a reusable transition template."
        )
    )
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scenes/scene_split_manifest.json",
        help="Scene split manifest containing input videos and detected effect intervals.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/boundary_brightness_template",
        help="Directory for template JSON, CSV details, and review image.",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=4,
        help="Zero-based offset inside the boundary effect. 4 means the human-friendly 5th frame.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=256,
        help="Number of luminance histogram bins. Default: 256.",
    )
    parser.add_argument(
        "--min-dropped",
        type=int,
        default=3,
        help="Minimum candidates on the low-similarity side when auto-picking the outlier gap.",
    )
    parser.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.5,
        help="Require at least this fraction of candidates to remain as valid transitions.",
    )
    return parser.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def luminance_histogram(frame: np.ndarray, bins: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype("float64")
    total = float(hist.sum())
    if total == 0:
        raise ValueError("empty frame histogram")
    return hist / total


def read_effect_frame(video_path: Path, effect_start: float, frame_offset: int) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_index = max(0, int(round(effect_start * fps)) + frame_offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read frame {frame_index} from {video_path}")
        return frame, frame_index / fps, frame_index
    finally:
        cap.release()


def collect_candidates(manifest: dict[str, Any], bins: int, frame_offset: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for video_index, video in enumerate(manifest.get("videos", []), start=1):
        video_path = Path(video["input_path"])
        for effect_index, effect in enumerate(video.get("effect_intervals", []), start=1):
            frame, frame_time, frame_index = read_effect_frame(video_path, float(effect["start"]), frame_offset)
            candidates.append(
                {
                    "candidate_index": len(candidates) + 1,
                    "video_index": video_index,
                    "effect_index": effect_index,
                    "input_path": str(video_path),
                    "effect_start": float(effect["start"]),
                    "effect_end": float(effect["end"]),
                    "effect_duration": round(float(effect["end"]) - float(effect["start"]), 3),
                    "frame_index": frame_index,
                    "frame_time": round(frame_time, 3),
                    "histogram": luminance_histogram(frame, bins),
                    "frame": frame,
                }
            )
    return candidates


def choose_keep_mask(histograms: np.ndarray, min_dropped: int, min_keep_ratio: float) -> tuple[np.ndarray, float, np.ndarray]:
    count = len(histograms)
    similarities = np.zeros((count, count), dtype="float64")
    for row in range(count):
        for col in range(count):
            similarities[row, col] = cosine_similarity(histograms[row], histograms[col])

    average_similarity = (similarities.sum(axis=1) - 1.0) / max(1, count - 1)
    order = np.argsort(average_similarity)
    sorted_values = average_similarity[order]
    gaps = np.diff(sorted_values)
    valid_splits = [
        index
        for index in range(len(gaps))
        if index + 1 >= min_dropped and count - (index + 1) >= math.ceil(count * min_keep_ratio)
    ]
    if not valid_splits:
        keep = np.ones(count, dtype=bool)
        return keep, float(sorted_values[0]), average_similarity

    best_gap_index = max(valid_splits, key=lambda index: gaps[index])
    threshold = float((sorted_values[best_gap_index] + sorted_values[best_gap_index + 1]) / 2)
    keep = average_similarity >= threshold
    return keep, threshold, average_similarity


def build_review_sheet(candidates: list[dict[str, Any]], output_path: Path) -> None:
    thumb_w, thumb_h = 280, 158
    label_h = 46
    cols = 4
    rows = max(1, math.ceil(len(candidates) / cols))
    canvas = np.full((rows * (thumb_h + label_h), cols * thumb_w, 3), 28, dtype=np.uint8)

    for position, item in enumerate(candidates):
        frame = cv2.resize(item["frame"], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(position, cols)
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        canvas[y : y + thumb_h, x : x + thumb_w] = frame

        status_color = (80, 220, 120) if item["is_valid_transition"] else (90, 90, 255)
        cv2.rectangle(canvas, (x, y), (x + thumb_w, y + 6), status_color, -1)
        cv2.rectangle(canvas, (x, y + thumb_h), (x + thumb_w, y + thumb_h + label_h), (16, 16, 16), -1)
        label1 = (
            f"{item['status']} #{item['candidate_index']:02d} "
            f"v{item['video_index']:02d} e{item['effect_index']:02d} {item['effect_start']:.1f}s"
        )
        label2 = f"sim={item['template_similarity']:.3f} avg={item['average_peer_similarity']:.3f}"
        cv2.putText(canvas, label1, (x + 8, y + thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, label2, (x + 8, y + thumb_h + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 205, 205), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def write_csv(candidates: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_index",
        "status",
        "is_valid_transition",
        "video_index",
        "effect_index",
        "effect_start",
        "effect_end",
        "effect_duration",
        "frame_index",
        "frame_time",
        "average_peer_similarity",
        "template_similarity",
        "input_path",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in candidates:
            writer.writerow({field: item[field] for field in fieldnames})


def main() -> int:
    args = parse_args()
    if args.frame_offset < 0:
        raise SystemExit("--frame-offset must be >= 0")
    if args.bins <= 1:
        raise SystemExit("--bins must be > 1")

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = collect_candidates(manifest, args.bins, args.frame_offset)
    if not candidates:
        raise SystemExit("no effect intervals found")

    histograms = np.stack([item["histogram"] for item in candidates])
    keep_mask, peer_threshold, average_peer_similarity = choose_keep_mask(
        histograms, args.min_dropped, args.min_keep_ratio
    )
    template = histograms[keep_mask].mean(axis=0)
    template = template / template.sum()
    template_similarity = np.array([cosine_similarity(hist, template) for hist in histograms])

    max_drop = float(template_similarity[~keep_mask].max()) if np.any(~keep_mask) else None
    min_keep = float(template_similarity[keep_mask].min())
    if max_drop is None:
        template_threshold = min_keep
    else:
        template_threshold = float((max_drop + min_keep) / 2)

    for item, is_keep, peer_sim, tmpl_sim in zip(candidates, keep_mask, average_peer_similarity, template_similarity):
        item["is_valid_transition"] = bool(is_keep)
        item["status"] = "KEEP" if is_keep else "DROP"
        item["average_peer_similarity"] = round(float(peer_sim), 6)
        item["template_similarity"] = round(float(tmpl_sim), 6)

    sorted_candidates = sorted(candidates, key=lambda item: item["template_similarity"])
    review_path = output_dir / "boundary_frame05_sorted_by_similarity.jpg"
    csv_path = output_dir / "boundary_brightness_candidates.csv"
    template_path = output_dir / "boundary_brightness_template.json"
    build_review_sheet(sorted_candidates, review_path)
    write_csv(sorted_candidates, csv_path)

    template_records = [
        {
            key: item[key]
            for key in [
                "candidate_index",
                "status",
                "is_valid_transition",
                "video_index",
                "effect_index",
                "effect_start",
                "effect_end",
                "effect_duration",
                "frame_index",
                "frame_time",
                "average_peer_similarity",
                "template_similarity",
                "input_path",
            ]
        }
        for item in sorted_candidates
    ]
    template_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_manifest": str(manifest_path),
                    "candidate_count": len(candidates),
                    "valid_transition_count": int(keep_mask.sum()),
                    "dropped_candidate_count": int((~keep_mask).sum()),
                    "frame_offset": args.frame_offset,
                    "frame_label": f"{args.frame_offset + 1}th frame inside effect interval",
                    "histogram_bins": args.bins,
                    "histogram_range": [0, 256],
                    "peer_average_similarity_threshold": round(peer_threshold, 6),
                    "template_similarity_threshold": round(template_threshold, 6),
                    "max_dropped_template_similarity": None if max_drop is None else round(max_drop, 6),
                    "min_kept_template_similarity": round(min_keep, 6),
                    "review_image": str(review_path),
                    "candidate_csv": str(csv_path),
                },
                "mean_luminance_histogram": [round(float(value), 10) for value in template],
                "candidates": template_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"candidates: {len(candidates)}")
    print(f"kept: {int(keep_mask.sum())}, dropped: {int((~keep_mask).sum())}")
    print(f"template similarity threshold: {template_threshold:.6f}")
    print(f"template: {template_path}")
    print(f"csv: {csv_path}")
    print(f"review image: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
