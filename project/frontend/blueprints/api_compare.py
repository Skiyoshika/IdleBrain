"""api_compare.py - Cross-sample brain region comparison endpoints.

Routes
------
POST /api/compare/regions   – merge cell_counts_hierarchy.csv from multiple
                              output directories and return a unified table

Body JSON::

    {
      "dirs": ["/path/to/job1/outputs", "/path/to/job2/outputs"],
      "labels": ["Sample A", "Sample B"]   // optional display names
    }

Response::

    {
      "ok": true,
      "regions": [
        {
          "region_id": 123,
          "region_name": "Isocortex",
          "acronym": "Isocortex",
          "depth": 2,
          "samples": {
            "Sample A": {"count": 450, "confidence": 0.12},
            "Sample B": {"count": 320, "confidence": 0.09}
          }
        },
        ...
      ],
      "sample_labels": ["Sample A", "Sample B"],
      "total_regions": 42
    }
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from project.frontend.api_errors import (
    ERR_INTERNAL,
    ERR_INVALID_INPUT,
)

bp = Blueprint("api_compare", __name__)

_REQUIRED_COLS = {"region_id", "region_name", "acronym", "count"}
_META_COLS = ["region_id", "region_name", "acronym", "depth", "hemisphere"]


@bp.post("/api/compare/regions")
def compare_regions():
    body = request.get_json(silent=True) or {}
    dirs = body.get("dirs", [])
    labels = body.get("labels", [])

    if not isinstance(dirs, list) or len(dirs) < 1:
        return jsonify(
            {"ok": False, "error": "dirs must be a non-empty list", "error_code": ERR_INVALID_INPUT}
        ), 400

    # Normalise labels
    if not isinstance(labels, list) or len(labels) != len(dirs):
        labels = [f"Sample {i + 1}" for i in range(len(dirs))]

    try:
        import pandas as pd
    except ImportError:
        return jsonify(
            {"ok": False, "error": "pandas not available", "error_code": ERR_INTERNAL}
        ), 500

    # Load each hierarchy CSV
    frames: list[tuple[str, pd.DataFrame]] = []
    missing: list[str] = []
    for raw_dir, label in zip(dirs, labels, strict=True):
        csv_path = Path(raw_dir) / "cell_counts_hierarchy.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        try:
            df = pd.read_csv(csv_path)
            if not _REQUIRED_COLS.issubset(df.columns):
                missing.append(f"{csv_path} (missing columns)")
                continue
            frames.append((label, df))
        except Exception as exc:
            missing.append(f"{csv_path} ({exc})")

    if not frames:
        return jsonify(
            {
                "ok": False,
                "error": "no valid hierarchy CSVs found",
                "missing": missing,
            }
        ), 404

    # Build unified region metadata (union of all region_ids, keep first occurrence)
    all_meta_rows: dict[tuple, dict] = {}
    for _label, df in frames:
        for _, row in df.iterrows():
            key = (int(row["region_id"]), str(row.get("hemisphere", "")))
            if key not in all_meta_rows:
                meta: dict = {"region_id": int(row["region_id"])}
                for col in ("region_name", "acronym", "depth", "hemisphere"):
                    if col in df.columns:
                        meta[col] = row[col]
                all_meta_rows[key] = meta

    # Build per-sample count lookup: (region_id, hemisphere) -> {count, confidence}
    sample_counts: dict[str, dict[tuple, dict]] = {}
    for label, df in frames:
        lookup: dict[tuple, dict] = {}
        for _, row in df.iterrows():
            key = (int(row["region_id"]), str(row.get("hemisphere", "")))
            lookup[key] = {
                "count": int(row["count"]),
                "confidence": round(float(row.get("confidence", 0.0)), 6),
            }
        sample_counts[label] = lookup

    sample_labels = [label for label, _ in frames]

    # Compose output rows
    regions = []
    for key, meta in sorted(all_meta_rows.items(), key=lambda x: (x[1].get("depth", 99), x[0])):
        samples_data: dict[str, dict] = {}
        for label in sample_labels:
            cell = sample_counts[label].get(key, {"count": 0, "confidence": 0.0})
            samples_data[label] = cell
        regions.append({**meta, "samples": samples_data})

    return jsonify(
        {
            "ok": True,
            "regions": regions,
            "sample_labels": sample_labels,
            "total_regions": len(regions),
            "missing": missing,
        }
    )
