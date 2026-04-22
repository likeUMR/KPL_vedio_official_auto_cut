#!/usr/bin/env python3
"""End-to-end KPL highlight processing pipeline.

Default mode is dry-run: commands are printed but not executed.
Use --execute to run the pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


@dataclass
class Step:
    name: str
    description: str
    commands: list[list[str]] = field(default_factory=list)
    cleanup_dirs: list[Path] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print or execute the full KPL highlight pipeline.")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default: print only.")
    parser.add_argument("--top-n", type=int, default=3, help="Top highlight videos per year to download.")
    parser.add_argument("--max-height", type=int, default=1080, help="Maximum download height.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Default polite request sleep for network steps.")
    parser.add_argument(
        "--cleanup-video-intermediates",
        action="store_true",
        help="After successful dependent steps, delete raw split and trimmed scene working directories.",
    )
    return parser.parse_args()


def py(script: str, *args: str | int | float) -> list[str]:
    return [PYTHON, str(ROOT / script), *(str(arg) for arg in args)]


def paths() -> dict[str, Path]:
    data = ROOT / "data"
    downloads = ROOT / "downloads"
    pipeline_data = data / "pipeline"
    pipeline_downloads = downloads / "pipeline"
    return {
        "data": data,
        "downloads": downloads,
        "pipeline_data": pipeline_data,
        "pipeline_downloads": pipeline_downloads,
        "schedules": data / "kpl_schedules.json",
        "schedules_csv": data / "kpl_schedules.csv",
        "schedules_enriched": data / "kpl_schedules_enriched.json",
        "schedules_enriched_csv": data / "kpl_schedules_enriched.csv",
        "programmes": data / "kpl_programmes.json",
        "programmes_enriched": data / "kpl_programmes_enriched.json",
        "programmes_enriched_csv": data / "kpl_programmes_enriched.csv",
        "video_stats": data / "kpl_video_category_stats.json",
        "video_stats_csv": data / "kpl_video_category_stats.csv",
        "plots": data / "plots",
        "top_highlights": data / "top_jingcai_jijin_by_year.json",
        "downloaded": pipeline_downloads / "selected_highlights",
        "scenes_raw": pipeline_downloads / "scenes_raw",
        "brightness": pipeline_data / "boundary_brightness_template",
        "scenes": pipeline_downloads / "scenes",
        "complete_focus_analysis": pipeline_data / "complete_focus_split_analysis",
        "segments": pipeline_downloads / "segments",
        "title_regions": pipeline_data / "segment_title_region_candidates",
        "title_ocr": pipeline_data / "segment_title_ocr",
        "side_team_ocr": pipeline_data / "side_player_team_ocr",
        "schedule_matches": pipeline_data / "side_player_schedule_matches",
        "highlight_plots": pipeline_data / "highlight_processing_plots",
        "final_catalog": pipeline_data / "final_scene_catalog.json",
        "final_catalog_csv": pipeline_data / "final_scene_catalog.csv",
    }


def build_steps(args: argparse.Namespace) -> list[Step]:
    p = paths()
    highlight_category = "".join(chr(code) for code in [0x7CBE, 0x5F69, 0x96C6, 0x9526])
    return [
        Step(
            "01_fetch_schedules",
            "Fetch all official KPL schedules and save the raw schedule list.",
            [
                py(
                    "stage_01_fetch_schedules.py",
                    "--output-dir",
                    p["data"],
                    "--sleep",
                    args.sleep,
                )
            ],
        ),
        Step(
            "02_enrich_schedules",
            "Fetch each match detail, results, per-round hero selections, and official playback links; overwrite schedule outputs.",
            [
                py(
                    "stage_02_enrich_schedules.py",
                    "--input",
                    p["schedules"],
                    "--output",
                    p["schedules_enriched"],
                    "--csv-output",
                    p["schedules_enriched_csv"],
                    "--sleep",
                    args.sleep,
                )
            ],
        ),
        Step(
            "03_fetch_videos",
            "Fetch all KPL official programme/video records.",
            [
                py(
                    "stage_03_fetch_videos.py",
                    "--output-dir",
                    p["data"],
                    "--page-size",
                    12,
                    "--sleep",
                    args.sleep,
                )
            ],
        ),
        Step(
            "04_enrich_videos",
            "Fetch Tencent Video detail fields such as play count, likes, description, tags, duration, and cover; overwrite video outputs.",
            [
                py(
                    "stage_04_enrich_videos.py",
                    "--input",
                    p["programmes"],
                    "--output",
                    p["programmes_enriched"],
                    "--csv-output",
                    p["programmes_enriched_csv"],
                    "--sleep",
                    args.sleep,
                )
            ],
        ),
        Step(
            "05_analyze_and_plot",
            "Analyze title-prefix categories and draw plots.",
            [
                py(
                    "stage_05_analyze_video_stats.py",
                    "--input",
                    p["programmes_enriched"],
                    "--output",
                    p["video_stats"],
                    "--csv-output",
                    p["video_stats_csv"],
                ),
                py(
                    "stage_05_plot_video_stats.py",
                    "--input",
                    p["programmes_enriched"],
                    "--output-dir",
                    p["plots"],
                    "--top-n",
                    5,
                ),
            ],
        ),
        Step(
            "06_select_and_download_highlights",
            "Select each year's top-N played highlight-category videos and download best available KPL MP4 not above 1080p.",
            [
                py(
                    "stage_06_select_top_highlights.py",
                    "--input",
                    p["programmes_enriched"],
                    "--output",
                    p["top_highlights"],
                    "--category",
                    highlight_category,
                    "--top-n",
                    args.top_n,
                ),
                py(
                    "stage_06_download_highlights.py",
                    "--input",
                    p["top_highlights"],
                    "--output-dir",
                    p["downloaded"],
                    "--max-height",
                    args.max_height,
                    "--overwrite",
                    "--sleep",
                    args.sleep,
                ),
            ],
        ),
        Step(
            "07_split_scenes_and_trim",
            "Detect black/white transition effects, build brightness template, re-split with luminance filtering, then trim every scene tail by 5 seconds.",
            [
                py(
                    "stage_07_split_scenes_by_bw_filter.py",
                    "--input-dir",
                    p["downloaded"],
                    "--output-dir",
                    p["scenes_raw"],
                    "--overwrite",
                ),
                py(
                    "stage_07_build_boundary_brightness_template.py",
                    "--manifest",
                    p["scenes_raw"] / "scene_split_manifest.json",
                    "--output-dir",
                    p["brightness"],
                    "--frame-offset",
                    4,
                ),
                py(
                    "stage_07_split_scenes_by_bw_filter.py",
                    "--input-dir",
                    p["downloaded"],
                    "--output-dir",
                    p["scenes_raw"],
                    "--brightness-template",
                    p["brightness"] / "boundary_brightness_template.json",
                    "--overwrite",
                ),
                py(
                    "stage_07_trim_scene_tails.py",
                    "--manifest",
                    p["scenes_raw"] / "scene_split_manifest.json",
                    "--output-dir",
                    p["scenes"],
                    "--trim-seconds",
                    5,
                    "--overwrite",
                ),
            ],
            cleanup_dirs=[p["scenes_raw"]],
        ),
        Step(
            "08_split_complete_focus",
            "Classify each scene as complete-only, focus-only, or complete-then-focus; focus clips keep the opening black/white filter intro.",
            [
                py(
                    "stage_08_analyze_complete_focus.py",
                    "--manifest",
                    p["scenes"] / "scene_tail_trim_manifest.json",
                    "--output-dir",
                    p["complete_focus_analysis"],
                ),
                py(
                    "stage_08_split_complete_focus_segments.py",
                    "--analysis",
                    p["complete_focus_analysis"] / "complete_focus_split_analysis.json",
                    "--output-dir",
                    p["segments"],
                    "--overwrite",
                ),
            ],
            cleanup_dirs=[p["scenes"]],
        ),
        Step(
            "09_ocr_scene_titles",
            "Extract colorful title/operator regions from the black/white filter frame and OCR scene title, operator, and operator team text.",
            [
                py(
                    "stage_09_extract_title_regions.py",
                    "--manifest",
                    p["segments"] / "complete_focus_segment_manifest.json",
                    "--output-dir",
                    p["title_regions"],
                    "--sample-time",
                    1.5,
                ),
                py(
                    "stage_09_ocr_scene_titles.py",
                    "--candidates",
                    p["title_regions"] / "segment_title_region_candidates.json",
                    "--output-dir",
                    p["title_ocr"],
                ),
            ],
        ),
        Step(
            "10_match_scenes_and_catalog",
            "Use side player-list OCR to infer match teams, match to official schedules, then build the final scene catalog.",
            [
                py(
                    "stage_10_ocr_side_player_teams.py",
                    "--manifest",
                    p["segments"] / "complete_focus_segment_manifest.json",
                    "--schedules",
                    p["schedules_enriched"],
                    "--output-dir",
                    p["side_team_ocr"],
                    "--sample-time",
                    1.5,
                ),
                py(
                    "stage_10_match_segments_to_schedules.py",
                    "--team-ocr",
                    p["side_team_ocr"] / "side_player_team_ocr.json",
                    "--schedules",
                    p["schedules_enriched"],
                    "--top-videos",
                    p["top_highlights"],
                    "--output-dir",
                    p["schedule_matches"],
                    "--max-days-before-upload",
                    31,
                ),
                py(
                    "stage_10_build_scene_catalog.py",
                    "--segment-manifest",
                    p["segments"] / "complete_focus_segment_manifest.json",
                    "--title-ocr",
                    p["title_ocr"] / "segment_title_ocr.json",
                    "--team-ocr",
                    p["side_team_ocr"] / "side_player_team_ocr.json",
                    "--schedule-matches",
                    p["schedule_matches"] / "segment_schedule_matches.json",
                    "--schedules",
                    p["schedules_enriched"],
                    "--output",
                    p["final_catalog"],
                    "--csv-output",
                    p["final_catalog_csv"],
                ),
                py(
                    "stage_10_plot_highlight_processing.py",
                    "--complete-focus",
                    p["complete_focus_analysis"] / "complete_focus_split_analysis.json",
                    "--schedule-matches",
                    p["schedule_matches"] / "segment_schedule_matches.json",
                    "--title-ocr",
                    p["title_ocr"] / "segment_title_ocr.json",
                    "--output-dir",
                    p["highlight_plots"],
                ),
            ],
        ),
    ]


def command_to_text(command: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(command: list[str], execute: bool) -> None:
    print(command_to_text(command), flush=True)
    if execute:
        subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def run_steps(steps: list[Step], execute: bool, cleanup_video_intermediates: bool) -> None:
    for step in steps:
        print(f"\n# {step.name}: {step.description}", flush=True)
        for command in step.commands:
            run_command(command, execute)
        if cleanup_video_intermediates and step.cleanup_dirs:
            for directory in step.cleanup_dirs:
                print(f"# cleanup {directory}", flush=True)
                if execute and directory.exists():
                    shutil.rmtree(directory)


def write_manifest(steps: list[Step], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dry_run_by_default": True,
        "steps": [
            {
                "name": step.name,
                "description": step.description,
                "commands": [command_to_text(command) for command in step.commands],
                "cleanup_dirs": [str(path) for path in step.cleanup_dirs],
            }
            for step in steps
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    steps = build_steps(args)
    write_manifest(steps, ROOT / "data" / "pipeline" / "pipeline_manifest.json")
    run_steps(steps, args.execute, args.cleanup_video_intermediates)
    if not args.execute:
        print("\nDry run only. Add --execute to run the pipeline.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
