#!/usr/bin/env python3
"""Detect complete-to-focus split points inside already trimmed KPL scene clips."""

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
            "Use the top 7.5% fixed UI region from the opening effect as the complete-view "
            "template, then detect whether each scene is complete-only, focus-only, or "
            "complete-then-focus."
        )
    )
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scenes_brightness_filtered_trim5s/scene_tail_trim_manifest.json",
        help="Manifest produced by stage_07_trim_scene_tails.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/complete_focus_split_analysis",
        help="Directory for analysis JSON, CSV, and review image.",
    )
    parser.add_argument("--top-ratio", type=float, default=0.075, help="Top band height ratio. Default: 0.075.")
    parser.add_argument("--scan-width", type=int, default=480, help="Resize sampled top band to this width.")
    parser.add_argument("--coarse-fps", type=float, default=1.0, help="Coarse scan sample rate. Default: 1 fps.")
    parser.add_argument("--fine-step", type=float, default=0.25, help="Fine scan time step in seconds. Default: 0.25.")
    parser.add_argument(
        "--template-seconds",
        type=float,
        default=1.0,
        help="Use this many seconds from the opening effect to build the complete UI template.",
    )
    parser.add_argument(
        "--start-offset",
        type=float,
        default=0.0,
        help="Skip this many seconds before building the opening template. Default: 0.",
    )
    parser.add_argument(
        "--pixel-diff-threshold",
        type=int,
        default=18,
        help="A top-band pixel is considered unchanged when gray diff <= this value. Default: 18.",
    )
    parser.add_argument(
        "--ui-sim-threshold",
        type=float,
        default=0.5,
        help="Frame is complete-view when this ratio of top-band pixels match template. Default: 0.5.",
    )
    parser.add_argument(
        "--min-focus-duration",
        type=float,
        default=1.5,
        help="Require focus state to persist this long after the candidate split. Default: 1.5s.",
    )
    parser.add_argument(
        "--early-split-seconds",
        type=float,
        default=5.0,
        help="Split before this time means the scene is classified as focus-only. Default: 5s.",
    )
    return parser.parse_args()


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps > 0 and frames > 0:
            return float(frames / fps)
    finally:
        cap.release()
    raise RuntimeError(f"failed to read duration: {path}")


def read_frame_at(cap: cv2.VideoCapture, time_sec: float, fps: float) -> np.ndarray | None:
    frame_index = max(0, int(round(time_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None


def top_band_gray(frame: np.ndarray, top_ratio: float, scan_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    band_height = max(1, round(height * top_ratio))
    band = frame[:band_height, :, :]
    if width > scan_width:
        target_height = max(1, round(band_height * scan_width / width))
        band = cv2.resize(band, (scan_width, target_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)


def build_template(
    cap: cv2.VideoCapture,
    fps: float,
    duration: float,
    start_offset: float,
    template_seconds: float,
    top_ratio: float,
    scan_width: int,
    fine_step: float,
) -> np.ndarray:
    bands: list[np.ndarray] = []
    end = min(duration, start_offset + template_seconds)
    time_sec = start_offset
    while time_sec <= end + 1e-6:
        frame = read_frame_at(cap, time_sec, fps)
        if frame is not None:
            bands.append(top_band_gray(frame, top_ratio, scan_width).astype("float32"))
        time_sec += fine_step
    if not bands:
        raise RuntimeError("failed to build opening UI template")
    return np.median(np.stack(bands, axis=0), axis=0).astype("uint8")


def ui_similarity(frame: np.ndarray, template: np.ndarray, top_ratio: float, scan_width: int, pixel_diff_threshold: int) -> float:
    band = top_band_gray(frame, top_ratio, scan_width)
    if band.shape != template.shape:
        band = cv2.resize(band, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(band, template)
    return float(np.mean(diff <= pixel_diff_threshold))


def sample_scores(
    path: Path,
    times: list[float],
    template: np.ndarray,
    top_ratio: float,
    scan_width: int,
    pixel_diff_threshold: int,
) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        scores = []
        for time_sec in times:
            frame = read_frame_at(cap, time_sec, fps)
            if frame is None:
                continue
            similarity = ui_similarity(frame, template, top_ratio, scan_width, pixel_diff_threshold)
            scores.append(
                {
                    "time": round(float(time_sec), 3),
                    "ui_similarity": round(similarity, 6),
                    "is_complete": similarity >= 0,
                }
            )
        return scores
    finally:
        cap.release()


def classify_scores(
    scores: list[dict[str, Any]],
    threshold: float,
    min_focus_duration: float,
    early_split_seconds: float,
) -> dict[str, Any]:
    if not scores:
        return {"classification": "unknown", "split_time": None, "reason": "no scores"}

    for item in scores:
        item["is_complete"] = item["ui_similarity"] >= threshold

    last_time = scores[-1]["time"]
    candidate: float | None = None
    for index, item in enumerate(scores):
        if item["is_complete"]:
            continue
        enough_future = item["time"] + min_focus_duration <= last_time + 1e-6
        if not enough_future:
            continue
        future = [future_item for future_item in scores[index:] if future_item["time"] <= item["time"] + min_focus_duration + 1e-6]
        if future and all(not future_item["is_complete"] for future_item in future):
            candidate = float(item["time"])
            break

    if candidate is None:
        return {"classification": "complete_only", "split_time": None, "reason": "no sustained focus state"}
    if candidate < early_split_seconds:
        return {"classification": "focus_only", "split_time": candidate, "reason": "split before early threshold"}
    return {"classification": "complete_then_focus", "split_time": candidate, "reason": "sustained focus state after complete opening"}


def refine_split(
    path: Path,
    coarse_split: float,
    template: np.ndarray,
    args: argparse.Namespace,
    duration: float,
) -> tuple[float, list[dict[str, Any]]]:
    start = max(0.0, coarse_split - 1.0)
    end = min(duration, coarse_split + 1.0)
    times = []
    time_sec = start
    while time_sec <= end + 1e-6:
        times.append(round(time_sec, 3))
        time_sec += args.fine_step
    fine_scores = sample_scores(
        path,
        times,
        template,
        args.top_ratio,
        args.scan_width,
        args.pixel_diff_threshold,
    )
    fine_result = classify_scores(fine_scores, args.ui_sim_threshold, args.min_focus_duration, args.early_split_seconds)
    return float(fine_result["split_time"] if fine_result["split_time"] is not None else coarse_split), fine_scores


def analyze_scene(scene: dict[str, Any], video_index: int, args: argparse.Namespace) -> dict[str, Any]:
    path = Path(scene["output_path"])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open scene: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = video_duration(path)
        template = build_template(
            cap,
            fps,
            duration,
            args.start_offset,
            args.template_seconds,
            args.top_ratio,
            args.scan_width,
            args.fine_step,
        )
    finally:
        cap.release()

    coarse_times = [round(value, 3) for value in np.arange(0.0, duration + 1e-6, 1.0 / args.coarse_fps)]
    coarse_scores = sample_scores(
        path,
        coarse_times,
        template,
        args.top_ratio,
        args.scan_width,
        args.pixel_diff_threshold,
    )
    coarse_result = classify_scores(coarse_scores, args.ui_sim_threshold, args.min_focus_duration, args.early_split_seconds)

    fine_scores: list[dict[str, Any]] = []
    final_split = coarse_result["split_time"]
    if final_split is not None:
        final_split, fine_scores = refine_split(path, float(final_split), template, args, duration)
        refined_result = classify_scores(fine_scores, args.ui_sim_threshold, args.min_focus_duration, args.early_split_seconds)
        if refined_result["split_time"] is not None:
            final_split = refined_result["split_time"]

    if final_split is None:
        classification = "complete_only"
    elif final_split < args.early_split_seconds:
        classification = "focus_only"
    else:
        classification = "complete_then_focus"

    return {
        "video_index": video_index,
        "scene_index": scene["index"],
        "path": str(path),
        "source_video_path": scene.get("source_video_path"),
        "source_scene_start": scene.get("source_scene_start"),
        "source_scene_end": scene.get("source_scene_end"),
        "source_scene_duration": scene.get("source_scene_duration"),
        "untrimmed_source_scene_start": scene.get("untrimmed_source_scene_start"),
        "untrimmed_source_scene_end": scene.get("untrimmed_source_scene_end"),
        "duration": round(duration, 3),
        "classification": classification,
        "split_time": None if final_split is None else round(float(final_split), 3),
        "coarse_classification": coarse_result["classification"],
        "coarse_split_time": coarse_result["split_time"],
        "coarse_scores": coarse_scores,
        "fine_scores": fine_scores,
        "template": {
            "top_ratio": args.top_ratio,
            "template_seconds": args.template_seconds,
            "pixel_diff_threshold": args.pixel_diff_threshold,
            "ui_sim_threshold": args.ui_sim_threshold,
        },
    }


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_index",
                "scene_index",
                "duration",
                "classification",
                "split_time",
                "coarse_classification",
                "coarse_split_time",
                "path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})


def build_review_plot(records: list[dict[str, Any]], output_path: Path) -> None:
    row_h = 34
    left_w = 190
    chart_w = 760
    width = left_w + chart_w + 30
    height = max(row_h * len(records) + 30, 120)
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)

    for row, record in enumerate(records):
        y = 22 + row * row_h
        label = f"v{record['video_index']:02d}s{record['scene_index']:02d} {record['classification']}"
        cv2.putText(canvas, label, (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        duration = max(0.001, float(record["duration"]))
        x0 = left_w
        y0 = y - 10
        cv2.line(canvas, (x0, y0), (x0 + chart_w, y0), (80, 80, 80), 1)
        cv2.line(canvas, (x0, y0 + 12), (x0 + chart_w, y0 + 12), (80, 80, 80), 1)
        for item in record["coarse_scores"]:
            x = x0 + int((item["time"] / duration) * chart_w)
            score = float(item["ui_similarity"])
            color = (80, 220, 120) if score >= record["template"]["ui_sim_threshold"] else (90, 90, 255)
            bar_h = max(1, int(score * 22))
            cv2.line(canvas, (x, y0 + 12), (x, y0 + 12 - bar_h), color, 2)
        if record["split_time"] is not None:
            x = x0 + int((float(record["split_time"]) / duration) * chart_w)
            cv2.line(canvas, (x, y0 - 16), (x, y0 + 16), (0, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def read_thumbnail(path: Path, time_sec: float, label: str, size: tuple[int, int] = (240, 135)) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame = read_frame_at(cap, max(0.0, time_sec), fps)
        if frame is None:
            thumb = np.full((size[1], size[0], 3), 45, dtype=np.uint8)
        else:
            thumb = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (size[0], 26), (0, 0, 0), -1)
        cv2.putText(thumb, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        return thumb
    finally:
        cap.release()


def build_thumbnail_review(records: list[dict[str, Any]], output_path: Path) -> None:
    thumb_w, thumb_h = 240, 135
    label_w = 220
    cols = 4
    row_h = thumb_h + 30
    canvas = np.full((max(1, len(records)) * row_h, label_w + cols * thumb_w, 3), 28, dtype=np.uint8)

    for row, record in enumerate(records):
        y = row * row_h
        path = Path(record["path"])
        duration = float(record["duration"])
        split_time = record["split_time"]
        label = f"v{record['video_index']:02d}s{record['scene_index']:02d}"
        cls = record["classification"]
        split_label = "none" if split_time is None else f"{float(split_time):.2f}s"
        cv2.putText(canvas, label, (8, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, cls, (8, y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"split={split_label}", (8, y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if split_time is None:
            times = [(0.5, "0.5s"), (min(duration - 0.1, duration * 0.33), "1/3"), (min(duration - 0.1, duration * 0.66), "2/3"), (max(0.0, duration - 0.5), "tail")]
        else:
            split = float(split_time)
            times = [
                (0.5, "0.5s"),
                (max(0.0, split - 0.5), "split-0.5"),
                (split, "split"),
                (min(duration - 0.1, split + 0.5), "split+0.5"),
            ]

        for col, (time_sec, time_label) in enumerate(times):
            thumb = read_thumbnail(path, time_sec, time_label)
            x = label_w + col * thumb_w
            canvas[y : y + thumb_h, x : x + thumb_w] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> int:
    args = parse_args()
    if not 0 < args.top_ratio < 1:
        raise SystemExit("--top-ratio must be in (0, 1)")
    if args.coarse_fps <= 0 or args.fine_step <= 0:
        raise SystemExit("--coarse-fps and --fine-step must be positive")

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    videos = manifest.get("videos", [])
    for video_index, video in enumerate(videos, start=1):
        print(f"[{video_index}/{len(videos)}] analyzing {Path(video['input_path']).name}", flush=True)
        for scene in video.get("scenes", []):
            record = analyze_scene(scene, video_index, args)
            print(
                f"  scene{record['scene_index']:02d}: {record['classification']} split={record['split_time']}",
                flush=True,
            )
            records.append(record)

    csv_path = output_dir / "complete_focus_split_summary.csv"
    json_path = output_dir / "complete_focus_split_analysis.json"
    review_path = output_dir / "complete_focus_split_review.png"
    thumbnail_review_path = output_dir / "complete_focus_split_thumbnail_review.jpg"
    write_csv(records, csv_path)
    build_review_plot(records, review_path)
    build_thumbnail_review(records, thumbnail_review_path)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["classification"]] = counts.get(record["classification"], 0) + 1
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_manifest": str(manifest_path),
                    "scene_count": len(records),
                    "classification_counts": counts,
                    "summary_csv": str(csv_path),
                    "review_image": str(review_path),
                    "thumbnail_review_image": str(thumbnail_review_path),
                    "parameters": vars(args),
                },
                "scenes": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"counts: {counts}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")
    print(f"review: {review_path}")
    print(f"thumbnail review: {thumbnail_review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
