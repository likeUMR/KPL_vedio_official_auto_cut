#!/usr/bin/env python3
"""Split KPL scene clips into complete/focus segments using detected UI split analysis."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut each scene into one or two clips according to complete/focus split analysis."
    )
    parser.add_argument(
        "--analysis",
        default="data/complete_focus_split_analysis/complete_focus_split_analysis.json",
        help="Analysis JSON produced by analyze_scene_complete_focus_split.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads/kpl_highlights_top1_by_year_scene_segments_complete_focus_with_intro",
        help="Directory for complete/focus segment clips.",
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=1.0,
        help="Skip generated segments shorter than this many seconds. Default: 1.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Use stream copy instead of re-encoding. Faster, but cuts may be keyframe-aligned.",
    )
    parser.add_argument(
        "--opening-effect-max-seconds",
        type=float,
        default=6.0,
        help="Scan at most this many seconds from the opening to find the black/white effect end. Default: 6.",
    )
    parser.add_argument(
        "--opening-effect-sample-fps",
        type=float,
        default=10.0,
        help="Sampling rate for opening black/white effect detection. Default: 10.",
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
        help="Opening frame is black/white-filter-like when this ratio of pixels is low saturation. Default: 0.5.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output clips.")
    parser.add_argument("--dry-run", action="store_true", help="Plan cuts only; do not write clips.")
    return parser.parse_args()


def sanitize_filename(text: str, max_len: int = 120) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text or "")
    text = re.sub(r"\s+", " ", text).strip(" .")
    return (text[:max_len].rstrip(" .") or "scene")


def cut_clip(input_path: Path, output_path: Path, start: float, end: float, copy: bool, overwrite: bool) -> None:
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


def low_saturation_ratio(frame: np.ndarray, saturation_threshold: int) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1] <= saturation_threshold))


def detect_opening_effect_end(
    path: Path,
    max_seconds: float,
    sample_fps: float,
    saturation_threshold: int,
    bw_ratio_threshold: float,
) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video for opening effect detection: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count > 0 else max_seconds
        step = max(1, int(round(fps / sample_fps)))
        last_effect_time: float | None = None
        seen_effect = False
        max_frame = min(frame_count - 1, int(round(min(duration, max_seconds) * fps)))
        frame_index = 0
        while frame_index <= max_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            timestamp = frame_index / fps
            is_effect = low_saturation_ratio(frame, saturation_threshold) >= bw_ratio_threshold
            if is_effect:
                seen_effect = True
                last_effect_time = timestamp
            elif seen_effect:
                return round(min(duration, timestamp), 3)
            frame_index += step
        if last_effect_time is not None:
            return round(min(duration, last_effect_time + step / fps), 3)
        return 0.0
    finally:
        cap.release()


def cut_intro_plus_tail(
    input_path: Path,
    output_path: Path,
    intro_end: float,
    tail_start: float,
    duration: float,
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
        "-i",
        str(input_path),
        "-filter_complex",
        (
            f"[0:v]trim=start=0:end={intro_end:.3f},setpts=PTS-STARTPTS[v0];"
            f"[0:a]atrim=start=0:end={intro_end:.3f},asetpts=PTS-STARTPTS[a0];"
            f"[0:v]trim=start={tail_start:.3f}:end={duration:.3f},setpts=PTS-STARTPTS[v1];"
            f"[0:a]atrim=start={tail_start:.3f}:end={duration:.3f},asetpts=PTS-STARTPTS[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        ),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def planned_segments(
    scene: dict[str, Any],
    min_segment_duration: float,
    opening_effect_end: float | None,
) -> list[dict[str, Any]]:
    duration = float(scene["duration"])
    classification = scene["classification"]
    split_time = scene["split_time"]

    if classification == "complete_then_focus" and split_time is not None:
        split = float(split_time)
        intro_end = min(float(opening_effect_end or 0.0), split, duration)
        candidates = [
            {"kind": "complete", "mode": "single", "start": 0.0, "end": split},
            {
                "kind": "focus",
                "mode": "intro_plus_tail",
                "start": 0.0,
                "end": duration,
                "components": [
                    {"start": 0.0, "end": intro_end, "role": "opening_effect"},
                    {"start": split, "end": duration, "role": "focus_body"},
                ],
            },
        ]
    elif classification == "focus_only":
        candidates = [{"kind": "focus", "mode": "single", "start": 0.0, "end": duration}]
    else:
        candidates = [{"kind": "complete", "mode": "single", "start": 0.0, "end": duration}]

    planned: list[dict[str, Any]] = []
    for segment in candidates:
        if segment["mode"] == "intro_plus_tail":
            components = [
                component
                for component in segment["components"]
                if component["end"] - component["start"] >= min_segment_duration
            ]
            duration_value = sum(component["end"] - component["start"] for component in components)
            if duration_value >= min_segment_duration:
                planned.append({**segment, "components": components, "duration": round(duration_value, 3)})
        else:
            duration_value = segment["end"] - segment["start"]
            if duration_value >= min_segment_duration:
                planned.append({**segment, "duration": round(duration_value, 3)})
    return planned


def process_scene(scene: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(scene["path"])
    scene_dir = output_dir / sanitize_filename(input_path.parent.name)
    base = sanitize_filename(input_path.stem)
    opening_effect_end = None
    if scene["classification"] == "complete_then_focus":
        opening_effect_end = detect_opening_effect_end(
            input_path,
            args.opening_effect_max_seconds,
            args.opening_effect_sample_fps,
            args.saturation_threshold,
            args.bw_ratio_threshold,
        )

    segment_records = []
    for index, segment in enumerate(planned_segments(scene, args.min_segment_duration, opening_effect_end), start=1):
        output_path = (
            scene_dir
            / f"{base}_{index:02d}_{segment['kind']}_{segment['start']:.2f}-{segment['end']:.2f}.mp4"
        )
        if not args.dry_run:
            if segment["mode"] == "intro_plus_tail":
                components = segment["components"]
                if len(components) == 1:
                    cut_clip(
                        input_path,
                        output_path,
                        float(components[0]["start"]),
                        float(components[0]["end"]),
                        args.copy,
                        args.overwrite,
                    )
                else:
                    cut_intro_plus_tail(
                        input_path,
                        output_path,
                        float(components[0]["end"]),
                        float(components[1]["start"]),
                        float(scene["duration"]),
                        args.overwrite,
                    )
            else:
                cut_clip(input_path, output_path, segment["start"], segment["end"], args.copy, args.overwrite)
        segment_records.append(
            {
                "index": index,
                "kind": segment["kind"],
                "mode": segment["mode"],
                "start": round(segment["start"], 3),
                "end": round(segment["end"], 3),
                "duration": segment["duration"],
                "components": [
                    {
                        "role": component["role"],
                        "start": round(component["start"], 3),
                        "end": round(component["end"], 3),
                        "duration": round(component["end"] - component["start"], 3),
                    }
                    for component in segment.get("components", [])
                ],
                "output_path": str(output_path),
            }
        )

    return {
        "video_index": scene["video_index"],
        "scene_index": scene["scene_index"],
        "classification": scene["classification"],
        "split_time": scene["split_time"],
        "opening_effect_end": opening_effect_end,
        "input_path": str(input_path),
        "duration": scene["duration"],
        "segment_count": len(segment_records),
        "segments": segment_records,
    }


def main() -> int:
    args = parse_args()
    if args.min_segment_duration < 0:
        raise SystemExit("--min-segment-duration must be >= 0")

    analysis_path = Path(args.analysis)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)

    records = []
    for scene in analysis.get("scenes", []):
        record = process_scene(scene, output_dir, args)
        print(
            f"v{record['video_index']:02d}s{record['scene_index']:02d}: "
            f"{record['classification']} -> {record['segment_count']} segment(s)",
            flush=True,
        )
        records.append(record)

    counts: dict[str, int] = {}
    segment_kind_counts: dict[str, int] = {}
    for record in records:
        counts[record["classification"]] = counts.get(record["classification"], 0) + 1
        for segment in record["segments"]:
            segment_kind_counts[segment["kind"]] = segment_kind_counts.get(segment["kind"], 0) + 1

    manifest = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_analysis": str(analysis_path),
            "dry_run": args.dry_run,
            "copy": args.copy,
            "min_segment_duration": args.min_segment_duration,
            "opening_effect_max_seconds": args.opening_effect_max_seconds,
            "opening_effect_sample_fps": args.opening_effect_sample_fps,
            "saturation_threshold": args.saturation_threshold,
            "bw_ratio_threshold": args.bw_ratio_threshold,
            "scene_count": len(records),
            "segment_count": sum(record["segment_count"] for record in records),
            "classification_counts": counts,
            "segment_kind_counts": segment_kind_counts,
        },
        "scenes": records,
    }
    manifest_path = output_dir / "complete_focus_segment_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
