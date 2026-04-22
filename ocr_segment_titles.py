#!/usr/bin/env python3
"""OCR title/operator metadata from extracted colorful segment title regions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidocr_onnxruntime import RapidOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR on colorful title candidate crops and infer scene title/team/operator metadata."
    )
    parser.add_argument(
        "--candidates",
        default="data/segment_title_region_candidates/segment_title_region_candidates.json",
        help="Candidate JSON produced by extract_segment_title_regions.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/segment_title_ocr",
        help="Directory for OCR JSON and CSV outputs.",
    )
    parser.add_argument("--top-boxes", type=int, default=5, help="OCR at most this many largest candidate boxes.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Drop OCR lines below this confidence.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("。", ".").replace("．", ".").replace("·", ".")
    text = re.sub(r"\s+", "", text)
    return text


def is_noise(text: str) -> bool:
    if not text:
        return True
    if re.fullmatch(r"[\d.,:%/\\|+\-]+", text):
        return True
    if len(text) <= 1:
        return True
    return False


def looks_like_team_or_player(text: str) -> bool:
    if "（" in text or "(" in text:
        return True
    if "." in text and re.search(r"[A-Za-z]{2,}|[A-Z]", text):
        return True
    if re.search(r"(EDG|DYG|TTG|WE|Hero|GK|XYG|VG|RNG|TES|TCG|AG|狼队|eStar)", text, re.I):
        return True
    return False


def looks_like_title(text: str) -> bool:
    if looks_like_team_or_player(text):
        return False
    if re.search(r"[A-Za-z]{3,}", text):
        return False
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    return chinese_count >= 4


def box_height(line: dict[str, Any]) -> float:
    points = line.get("points") or []
    if not points:
        return 0.0
    ys = [point[1] for point in points]
    return max(ys) - min(ys)


def run_ocr(ocr: RapidOCR, image_path: str, min_confidence: float) -> list[dict[str, Any]]:
    result, _elapsed = ocr(image_path)
    lines = []
    for item in result or []:
        points, text, confidence = item
        text = normalize_text(text)
        confidence = float(confidence)
        if confidence < min_confidence or is_noise(text):
            continue
        lines.append({"text": text, "confidence": round(confidence, 6), "points": points})
    return lines


def candidate_crop_paths(segment: dict[str, Any], top_boxes: int) -> list[str]:
    boxes = sorted(segment.get("boxes", []), key=lambda item: item.get("area", 0), reverse=True)
    paths = []
    for box in boxes[:top_boxes]:
        crop_path = box.get("crop_path")
        if crop_path:
            paths.append(crop_path)
    return paths


def dedupe_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for line in lines:
        text = line["text"]
        if text not in best or line["confidence"] > best[text]["confidence"]:
            best[text] = line
    return sorted(best.values(), key=lambda item: (-item["confidence"], item["text"]))


def infer_metadata(lines: list[dict[str, Any]]) -> dict[str, Any]:
    title_candidates = [line for line in lines if looks_like_title(line["text"])]
    title_candidates = sorted(
        title_candidates,
        key=lambda item: (box_height(item), item["confidence"], len(item["text"])),
        reverse=True,
    )
    title_lines: list[str] = []
    for line in title_candidates:
        text = line["text"]
        if text not in title_lines:
            title_lines.append(text)
        if len(title_lines) >= 2:
            break

    team_player_candidates = [line for line in lines if looks_like_team_or_player(line["text"])]
    team_player_candidates = sorted(
        team_player_candidates,
        key=lambda item: (item["confidence"], len(item["text"])),
        reverse=True,
    )

    team = None
    operator = None
    team_player_texts = [line["text"] for line in team_player_candidates]
    for text in team_player_texts:
        if "（" in text or "(" in text:
            operator = text
            break
    for text in team_player_texts:
        if "." in text or re.search(r"(EDG|DYG|TTG|WE|Hero|GK|XYG|VG|RNG|TES|TCG|AG)", text, re.I):
            team = text
            break

    return {
        "scene_title": " / ".join(title_lines) if title_lines else None,
        "scene_title_lines": title_lines,
        "team": team,
        "operator": operator,
        "team_operator_candidates": team_player_texts[:8],
    }


def process_segment(ocr: RapidOCR, segment: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ocr_sources = candidate_crop_paths(segment, args.top_boxes)
    all_lines: list[dict[str, Any]] = []
    for source in ocr_sources:
        lines = run_ocr(ocr, source, args.min_confidence)
        for line in lines:
            line["source_image"] = source
        all_lines.extend(lines)

    deduped = dedupe_lines(all_lines)
    inferred = infer_metadata(deduped)
    return {
        "video_index": segment["video_index"],
        "scene_index": segment["scene_index"],
        "segment_index": segment["segment_index"],
        "kind": segment["kind"],
        "mode": segment["mode"],
        "path": segment["path"],
        "sample_time": segment["sample_time"],
        "ocr_source_count": len(ocr_sources),
        "ocr_sources": ocr_sources,
        **inferred,
        "ocr_lines": deduped,
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
                "scene_title",
                "team",
                "operator",
                "team_operator_candidates",
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
                    "scene_title": record["scene_title"],
                    "team": record["team"],
                    "operator": record["operator"],
                    "team_operator_candidates": " | ".join(record["team_operator_candidates"]),
                    "path": record["path"],
                }
            )


def main() -> int:
    args = parse_args()
    candidates_path = Path(args.candidates)
    output_dir = Path(args.output_dir)
    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    ocr = RapidOCR()

    records = []
    for index, segment in enumerate(data.get("segments", []), start=1):
        print(
            f"[{index}/{len(data.get('segments', []))}] OCR "
            f"v{segment['video_index']:02d}s{segment['scene_index']:02d}seg{segment['segment_index']:02d} {segment['kind']}",
            flush=True,
        )
        records.append(process_segment(ocr, segment, args))

    json_path = output_dir / "segment_title_ocr.json"
    csv_path = output_dir / "segment_title_ocr_summary.csv"
    write_csv(records, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_candidates": str(candidates_path),
                    "segment_count": len(records),
                    "top_boxes": args.top_boxes,
                    "min_confidence": args.min_confidence,
                    "summary_csv": str(csv_path),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
