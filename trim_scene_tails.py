#!/usr/bin/env python3
"""Trim a fixed number of seconds from the tail of each KPL scene clip."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim a fixed tail duration from scene clips in a manifest.")
    parser.add_argument(
        "--manifest",
        default="downloads/kpl_highlights_top1_by_year_scenes_brightness_filtered/scene_split_manifest.json",
        help="Manifest produced by split_highlight_scenes.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads/kpl_highlights_top1_by_year_scenes_brightness_filtered_trim5s",
        help="Directory for trimmed scene clips.",
    )
    parser.add_argument("--trim-seconds", type=float, default=5.0, help="Seconds to remove from each clip tail.")
    parser.add_argument(
        "--min-output-duration",
        type=float,
        default=2.0,
        help="Skip trimming if the output would be shorter than this. Default: 2.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Use stream copy instead of re-encoding. Faster, but cuts may be keyframe-aligned.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output clips.")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only; do not write clips.")
    return parser.parse_args()


def sanitize_filename(text: str, max_len: int = 120) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text or "")
    text = re.sub(r"\s+", " ", text).strip(" .")
    return (text[:max_len].rstrip(" .") or "scene")


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps > 0 and frames > 0:
            return float(frames / fps)
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


def cut_clip(input_path: Path, output_path: Path, end: float, copy: bool, overwrite: bool) -> None:
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
        "-to",
        f"{end:.3f}",
    ]
    if copy:
        command.extend(["-c", "copy"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "128k"])
    command.append(str(output_path))
    subprocess.run(command, check=True)


def process_scene(scene: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(scene["output_path"])
    duration = video_duration(input_path)
    trimmed_duration = max(0.0, duration - args.trim_seconds)
    should_trim = trimmed_duration >= args.min_output_duration and args.trim_seconds > 0
    if not should_trim:
        trimmed_duration = duration

    base = sanitize_filename(input_path.stem)
    output_path = output_dir / input_path.parent.name / f"{base}_trim{args.trim_seconds:g}s_0.00-{trimmed_duration:.2f}.mp4"
    if not args.dry_run:
        cut_clip(input_path, output_path, trimmed_duration, args.copy, args.overwrite)

    return {
        "index": scene["index"],
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_duration": round(duration, 3),
        "trimmed_duration": round(trimmed_duration, 3),
        "removed_tail_duration": round(duration - trimmed_duration, 3),
        "trimmed": should_trim,
    }


def main() -> int:
    args = parse_args()
    if args.trim_seconds < 0:
        raise SystemExit("--trim-seconds must be >= 0")
    if args.min_output_duration < 0:
        raise SystemExit("--min-output-duration must be >= 0")

    source_manifest_path = Path(args.manifest)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)

    total_scenes = 0
    trimmed_scenes = 0
    videos: list[dict[str, Any]] = []
    source_videos = source_manifest.get("videos", [])
    for video_index, video in enumerate(source_videos, start=1):
        print(f"[{video_index}/{len(source_videos)}] trimming {Path(video['input_path']).name}", flush=True)
        scene_records = []
        for scene in video.get("scenes", []):
            total_scenes += 1
            record = process_scene(scene, output_dir, args)
            if record["trimmed"]:
                trimmed_scenes += 1
            print(
                f"  scene{record['index']:02d}: {record['original_duration']:.2f}s -> "
                f"{record['trimmed_duration']:.2f}s",
                flush=True,
            )
            scene_records.append(record)
        videos.append({"input_path": video["input_path"], "scene_count": len(scene_records), "scenes": scene_records})

    output_manifest = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest": str(source_manifest_path),
            "trim_seconds": args.trim_seconds,
            "min_output_duration": args.min_output_duration,
            "dry_run": args.dry_run,
            "copy": args.copy,
            "total_scenes": total_scenes,
            "trimmed_scene_count": trimmed_scenes,
        },
        "videos": videos,
    }
    manifest_path = output_dir / "scene_tail_trim_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"trimmed scenes: {trimmed_scenes}/{total_scenes}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
