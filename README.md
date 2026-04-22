# KPL Official Video Auto Cut Pipeline

This project builds a reproducible pipeline for collecting official KPL match/video data, downloading top official highlight videos, cutting them into individual highlight scenes, splitting complete-view/focus-view parts, OCRing scene metadata, and matching scenes back to official match schedules.

The workflow is designed for local research and dataset preparation. Generated videos, crawled data, OCR crops, plots, and checkpoints are intentionally ignored by git.

## What It Does

1. Fetch all official KPL schedules.
2. Enrich each match with result details, per-round hero picks, and official replay links.
3. Fetch all KPL official video/programme records.
4. Enrich each video with Tencent Video detail fields such as play count, likes, description, tags, aspect ratio, and duration.
5. Analyze title-prefix categories and draw distribution plots.
6. Select the top-N played `【精彩集锦】` videos per year and download the best available KPL MP4 not above 1080p.
7. Detect black/white transition effects with luminance-histogram filtering, cut scenes, and trim each scene tail by 5 seconds.
8. Split each scene into complete/focus segments; focus segments keep the opening black/white title/filter phase.
9. OCR scene title, operator, and operator team from the stylized colored title region.
10. OCR side player-list UI to infer both match teams, match scenes to official schedules, and build a final scene catalog.

## Setup

Use Python 3.10+ on Windows. The scripts also expect `ffmpeg` to be available on `PATH`.

Install the Python dependencies you use in this workspace, for example:

```powershell
pip install opencv-python numpy matplotlib rapidocr-onnxruntime
```

Some steps call public KPL/Tencent endpoints. Use conservative worker counts if endpoints start returning rate-limit or timeout errors.

## Full Pipeline

The pipeline script is dry-run by default. It prints every command and writes a command manifest, but does not execute the pipeline unless `--execute` is passed.

```powershell
# Preview the full workflow
python .\run_full_pipeline.py --top-n 3

# Execute the full workflow
python .\run_full_pipeline.py --execute --top-n 3

# Execute and delete large intermediate scene folders after dependent steps finish
python .\run_full_pipeline.py --execute --top-n 3 --cleanup-video-intermediates
```

Primary final outputs:

- `data/pipeline/final_scene_catalog.json`
- `data/pipeline/final_scene_catalog.csv`
- `downloads/pipeline/segments/`

The pipeline stores command metadata at:

- `data/pipeline/pipeline_manifest.json`

## Individual Steps

Fetch official schedules:

```powershell
python .\fetch_kpl_schedules.py
```

Enrich schedules with match details, round hero picks, and replay links:

```powershell
python .\enrich_kpl_schedule_details.py --resume
```

Fetch official video records:

```powershell
python .\fetch_kpl_programmes.py
```

Enrich video records with Tencent detail-page fields:

```powershell
python .\enrich_kpl_video_details.py --resume
```

The schedule and video enrichment scripts both support adaptive parallelism:

```powershell
python .\enrich_kpl_schedule_details.py --max-workers 32 --min-workers 4 --resume
python .\enrich_kpl_video_details.py --max-workers 16 --min-workers 2 --resume
```

Analyze and plot video categories:

```powershell
python .\analyze_kpl_video_stats.py
python .\plot_kpl_top_categories.py
```

Select top highlight videos and download:

```powershell
python .\extract_top_highlight_by_year.py --top-n 3
python .\download_kpl_videos.py --input .\data\top_jingcai_jijin_by_year.json --output-dir .\downloads\pipeline\selected_highlights --max-height 1080 --overwrite
```

Cut scenes, trim tails, split complete/focus segments, OCR, and build the final catalog are orchestrated by `run_full_pipeline.py`.

## Important Scripts

- `run_full_pipeline.py`: end-to-end orchestration, dry-run by default.
- `build_scene_catalog.py`: merges segment cuts, title OCR, side UI OCR, and schedule matches into the final catalog.
- `fetch_kpl_schedules.py`: official KPL season/stage/team and schedule fetcher.
- `enrich_kpl_schedule_details.py`: adaptive parallel detail fetcher for match results, round hero picks, and replay links.
- `fetch_kpl_programmes.py`: official KPL programme/video metadata fetcher.
- `enrich_kpl_video_details.py`: adaptive parallel Tencent detail enrichment.
- `split_highlight_scenes.py`: black/white-filter boundary detector and scene cutter.
- `analyze_boundary_brightness_template.py`: builds luminance histogram template to reject false transition detections.
- `analyze_scene_complete_focus_split.py`: detects complete-view to focus-view splits.
- `split_complete_focus_segments.py`: writes final complete/focus segment videos.
- `extract_segment_title_regions.py` and `ocr_segment_titles.py`: title/operator OCR.
- `ocr_side_player_team_rois.py`: side player-list OCR for match team inference.
- `match_segments_to_schedules.py`: matches scene teams to official schedules with upload-time constraints.

## Data Policy

Generated artifacts are not tracked:

- downloaded source videos and cut clips
- crawled/enriched JSON/CSV outputs
- OCR crops and review images
- plots
- checkpoint files
- Python caches

This keeps the repository small and focused on reproducible code.
