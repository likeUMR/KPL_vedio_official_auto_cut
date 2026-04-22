#!/usr/bin/env python3
"""Split KPL highlight videos into individual moments by low-saturation transition effects."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the obvious black/white filter transition in KPL highlight videos "
            "using low-saturation pixel ratio, then split each moment from one effect "
            "start to the next effect start."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="downloads/kpl_highlights_top1_by_year",
        help="Directory containing downloaded MP4 files.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Specific MP4 path. Can be repeated. Overrides --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads/kpl_highlights_top1_by_year_scenes",
        help="Directory for split scene clips.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=5.0,
        help="Frames per second to sample for detection. Default: 5.",
    )
    parser.add_argument(
        "--scan-width",
        type=int,
        default=320,
        help="Resize sampled frames to this width before detection. Default: 320.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=int,
        default=30,
        help="HSV saturation <= this value counts as black/white/gray. Default: 30.",
    )
    parser.add_argument(
        "--bw-ratio-threshold",
        type=float,
        default=0.5,
        help="A frame is an effect frame when this ratio of pixels is low saturation. Default: 0.5.",
    )
    parser.add_argument(
        "--min-effect-duration",
        type=float,
        default=0.35,
        help="Minimum detected effect interval duration in seconds. Default: 0.35.",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.45,
        help="Merge detected effect intervals separated by at most this many seconds. Default: 0.45.",
    )
    parser.add_argument(
        "--min-scene-duration",
        type=float,
        default=2.0,
        help="Skip split clips shorter than this many seconds. Default: 2.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Use stream copy instead of re-encoding. Faster, but cuts may be less accurate.",
    )
    parser.add_argument(
        "--brightness-template",
        default=None,
        help=(
            "Optional template JSON from stage_07_build_boundary_brightness_template.py. "
            "When set, low-saturation effect candidates are kept only if their 5th-frame "
            "luminance histogram is similar enough to the valid transition template."
        ),
    )
    parser.add_argument(
        "--brightness-sim-threshold",
        type=float,
        default=None,
        help="Override the template similarity threshold stored in --brightness-template.",
    )
    parser.add_argument(
        "--transition-frame-offset",
        type=int,
        default=None,
        help="Zero-based frame offset inside each effect for template matching. Defaults to template metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scene clips.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only detect intervals and planned scenes; do not cut files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of input videos, useful for testing.",
    )
    return parser.parse_args()


def sanitize_filename(text: str, max_len: int = 120) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text or "")
    text = re.sub(r"\s+", " ", text).strip(" .")
    return (text[:max_len].rstrip(" .") or "video")


def list_inputs(args: argparse.Namespace) -> list[Path]:
    if args.input:
        paths = [Path(item) for item in args.input]
    else:
        paths = sorted(Path(args.input_dir).glob("*.mp4"))
    if args.limit is not None:
        paths = paths[: args.limit]
    return paths


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps > 0 and frames > 0:
            return frames / fps
    finally:
        cap.release()

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip())


def low_saturation_ratio(frame: np.ndarray, scan_width: int, saturation_threshold: int) -> float:
    height, width = frame.shape[:2]
    if width > scan_width:
        scan_height = max(1, round(height * scan_width / width))
        frame = cv2.resize(frame, (scan_width, scan_height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return float(np.mean(saturation <= saturation_threshold))


def luminance_histogram(frame: np.ndarray, bins: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype("float64")
    total = float(hist.sum())
    if total == 0:
        raise ValueError("empty frame histogram")
    return hist / total


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def load_brightness_template(
    template_path: str | None,
    threshold_override: float | None,
    frame_offset_override: int | None,
) -> dict[str, Any] | None:
    if template_path is None:
        return None
    data = json.loads(Path(template_path).read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    histogram = np.array(data["mean_luminance_histogram"], dtype="float64")
    histogram = histogram / histogram.sum()
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(metadata["template_similarity_threshold"])
    )
    frame_offset = (
        int(frame_offset_override)
        if frame_offset_override is not None
        else int(metadata.get("frame_offset", 4))
    )
    return {
        "path": template_path,
        "histogram": histogram,
        "threshold": threshold,
        "frame_offset": frame_offset,
        "bins": int(metadata.get("histogram_bins", len(histogram))),
    }


def filter_effects_by_brightness_template(
    path: Path,
    effects: list[tuple[float, float]],
    source_fps: float,
    template: dict[str, Any] | None,
) -> tuple[list[tuple[float, float]], list[dict[str, Any]]]:
    if template is None or not effects:
        return effects, []

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video for brightness template filtering: {path}")

    records: list[dict[str, Any]] = []
    kept: list[tuple[float, float]] = []
    try:
        for start, end in effects:
            frame_index = max(0, int(round(start * source_fps)) + int(template["frame_offset"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                similarity = 0.0
            else:
                hist = luminance_histogram(frame, int(template["bins"]))
                similarity = cosine_similarity(hist, template["histogram"])
            keep = similarity >= float(template["threshold"])
            if keep:
                kept.append((start, end))
            records.append(
                {
                    "start": start,
                    "end": end,
                    "frame_index": frame_index,
                    "similarity": round(similarity, 6),
                    "kept": keep,
                }
            )
    finally:
        cap.release()

    return kept, records


def merge_intervals(intervals: list[tuple[float, float]], merge_gap: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= merge_gap:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_effect_intervals(
    path: Path,
    sample_fps: float,
    scan_width: int,
    saturation_threshold: int,
    bw_ratio_threshold: float,
    min_effect_duration: float,
    merge_gap: float,
) -> tuple[list[tuple[float, float]], list[dict[str, float]], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / source_fps if source_fps > 0 and frame_count > 0 else video_duration(path)
    step = max(1, int(round(source_fps / sample_fps)))
    sample_period = step / source_fps

    raw_intervals: list[tuple[float, float]] = []
    samples: list[dict[str, float]] = []
    active_start: float | None = None
    last_active_time: float | None = None

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step != 0:
            frame_index += 1
            continue

        timestamp = frame_index / source_fps
        ratio = low_saturation_ratio(frame, scan_width, saturation_threshold)
        is_effect = ratio >= bw_ratio_threshold
        samples.append({"time": round(timestamp, 3), "low_saturation_ratio": round(ratio, 5)})

        if is_effect:
            if active_start is None:
                active_start = timestamp
            last_active_time = timestamp
        elif active_start is not None and last_active_time is not None:
            raw_intervals.append((active_start, last_active_time + sample_period))
            active_start = None
            last_active_time = None

        frame_index += 1

    cap.release()
    if active_start is not None and last_active_time is not None:
        raw_intervals.append((active_start, min(duration, last_active_time + sample_period)))

    merged = merge_intervals(raw_intervals, merge_gap)
    filtered = [
        (round(start, 3), round(end, 3))
        for start, end in merged
        if end - start >= min_effect_duration
    ]
    return filtered, samples, source_fps


def planned_scenes(
    effects: list[tuple[float, float]],
    duration: float,
    min_scene_duration: float,
) -> list[tuple[float, float]]:
    starts = [start for start, _ in effects]
    scenes: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else duration
        if end - start >= min_scene_duration:
            scenes.append((round(start, 3), round(end, 3)))
    return scenes


def cut_scene(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
    copy: bool,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(input_path),
    ]
    if copy:
        command.extend(["-c", "copy"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "128k"])
    command.append(str(output_path))
    subprocess.run(command, check=True)


def process_video(path: Path, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    duration = video_duration(path)
    brightness_template = load_brightness_template(
        args.brightness_template,
        args.brightness_sim_threshold,
        args.transition_frame_offset,
    )
    effects, samples, source_fps = detect_effect_intervals(
        path,
        args.sample_fps,
        args.scan_width,
        args.saturation_threshold,
        args.bw_ratio_threshold,
        args.min_effect_duration,
        args.merge_gap,
    )
    raw_effect_count = len(effects)
    effects, brightness_records = filter_effects_by_brightness_template(
        path,
        effects,
        source_fps,
        brightness_template,
    )
    scenes = planned_scenes(effects, duration, args.min_scene_duration)

    base = sanitize_filename(path.stem)
    scene_records: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(scenes, start=1):
        scene_path = output_dir / base / f"{base}_scene{index:02d}_{start:.2f}-{end:.2f}.mp4"
        if not args.dry_run:
            cut_scene(path, scene_path, start, end, args.copy, args.overwrite)
        scene_records.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "output_path": str(scene_path),
            }
        )

    return {
        "input_path": str(path),
        "duration": round(duration, 3),
        "effect_intervals": [
            {"start": start, "end": end, "duration": round(end - start, 3)}
            for start, end in effects
        ],
        "scene_count": len(scene_records),
        "scenes": scene_records,
        "detection": {
            "sample_fps": args.sample_fps,
            "scan_width": args.scan_width,
            "saturation_threshold": args.saturation_threshold,
            "bw_ratio_threshold": args.bw_ratio_threshold,
            "min_effect_duration": args.min_effect_duration,
            "merge_gap": args.merge_gap,
            "sample_count": len(samples),
            "max_low_saturation_ratio": max((item["low_saturation_ratio"] for item in samples), default=None),
            "raw_effect_count_before_brightness_filter": raw_effect_count,
            "brightness_template": None
            if brightness_template is None
            else {
                "path": brightness_template["path"],
                "threshold": brightness_template["threshold"],
                "frame_offset": brightness_template["frame_offset"],
                "records": brightness_records,
            },
        },
    }


def main() -> int:
    args = parse_args()
    if not 0 <= args.saturation_threshold <= 255:
        raise SystemExit("--saturation-threshold must be between 0 and 255")
    if not 0 < args.bw_ratio_threshold <= 1:
        raise SystemExit("--bw-ratio-threshold must be in (0, 1]")
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be positive")

    input_paths = list_inputs(args)
    if not input_paths:
        raise SystemExit("no input videos found")

    output_dir = Path(args.output_dir)
    records: list[dict[str, Any]] = []
    for index, path in enumerate(input_paths, start=1):
        print(f"[{index}/{len(input_paths)}] detecting {path.name}", flush=True)
        record = process_video(path, args, output_dir)
        print(
            f"  effects={len(record['effect_intervals'])}, scenes={record['scene_count']}",
            flush=True,
        )
        records.append(record)

    manifest_path = output_dir / "scene_split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dry_run": args.dry_run,
                    "copy": args.copy,
                    "input_count": len(input_paths),
                },
                "videos": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
