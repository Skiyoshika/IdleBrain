from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu


def _norm01(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def compute_staining_stats(
    registered_path: Path,
    annotation_path: Path,
    *,
    min_signal: float = 0.02,
) -> dict[str, float]:
    registered = np.asarray(nib.load(str(registered_path)).dataobj, dtype=np.float32)
    annotation = np.asarray(nib.load(str(annotation_path)).dataobj)
    common_shape = tuple(
        min(a, b) for a, b in zip(registered.shape, annotation.shape, strict=False)
    )
    registered = registered[: common_shape[0], : common_shape[1], : common_shape[2]]
    annotation = annotation[: common_shape[0], : common_shape[1], : common_shape[2]]

    signal = _norm01(registered)
    atlas_mask = annotation > 0
    atlas_voxels = int(atlas_mask.sum())
    if atlas_voxels <= 0:
        return {
            "atlas_voxels": 0.0,
            "covered_voxels": 0.0,
            "positive_voxels": 0.0,
            "atlas_coverage": 0.0,
            "staining_rate": 0.0,
            "positive_fraction_of_atlas": 0.0,
            "signal_threshold": 0.0,
            "mean_signal": 0.0,
        }

    covered_mask = atlas_mask & (signal > float(min_signal))
    covered_voxels = int(covered_mask.sum())
    if covered_voxels <= 16:
        return {
            "atlas_voxels": float(atlas_voxels),
            "covered_voxels": float(covered_voxels),
            "positive_voxels": 0.0,
            "atlas_coverage": float(covered_voxels / atlas_voxels),
            "staining_rate": 0.0,
            "positive_fraction_of_atlas": 0.0,
            "signal_threshold": 0.0,
            "mean_signal": float(signal[covered_mask].mean()) if covered_voxels > 0 else 0.0,
        }

    signal_threshold = float(threshold_otsu(signal[covered_mask]))
    positive_mask = covered_mask & (signal >= signal_threshold)
    positive_voxels = int(positive_mask.sum())

    return {
        "atlas_voxels": float(atlas_voxels),
        "covered_voxels": float(covered_voxels),
        "positive_voxels": float(positive_voxels),
        "atlas_coverage": float(covered_voxels / atlas_voxels),
        "staining_rate": float(positive_voxels / covered_voxels),
        "positive_fraction_of_atlas": float(positive_voxels / atlas_voxels),
        "signal_threshold": signal_threshold,
        "mean_signal": float(signal[covered_mask].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute staining coverage and staining rate for a registered volume"
    )
    ap.add_argument("--registered", required=True, help="Registered brain NIfTI path")
    ap.add_argument("--annotation", required=True, help="Fixed-space annotation NIfTI path")
    ap.add_argument("--out-json", default="", help="Optional JSON output path")
    args = ap.parse_args()

    stats = compute_staining_stats(Path(args.registered), Path(args.annotation))
    text = json.dumps(stats, indent=2)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved staining stats -> {out_path}")
    print(text)


if __name__ == "__main__":
    main()
