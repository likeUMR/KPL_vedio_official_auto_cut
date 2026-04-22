# Top UI Match Recognition Legacy Module

This folder archives the earlier top-band match UI recognition approach.

Reason for archiving:

- For 2020-2023 clips, the top match UI often shows team logos instead of readable team names.
- OCR on the top band therefore misses the most important information for schedule matching.
- Later work should prefer side-player UI recognition, where five player labels per side provide repeated team-prefix evidence.

Archived scripts:

- `ocr_segment_match_ui.py`
- `review_match_ui_year_rois.py`
- `probe_match_ui_rois.py`
- `ocr_match_ui_team_rois.py`

The original files are intentionally left in the project root for reproducibility of previous outputs.
