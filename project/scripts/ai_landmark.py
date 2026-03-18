from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scripts.slice_select import select_label_slice_2d, select_real_slice_2d
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import sobel
from skimage.measure import ransac
from skimage.metrics import structural_similarity as ssim
from skimage.transform import AffineTransform
from tifffile import imread


def detect_landmarks(img: np.ndarray, max_points: int = 30, min_distance: int = 12) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., 0]
    resp = corner_harris(img.astype(np.float32))
    pts = corner_peaks(resp, min_distance=min_distance, num_peaks=max_points)
    # return x,y
    return np.array([[float(p[1]), float(p[0])] for p in pts], dtype=np.float32)


def _match_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    return a, b


def score_alignment(a: np.ndarray, b: np.ndarray) -> float:
    # Raw SSIM (can be unstable across modality differences)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a, b = _match_shape(a, b)
    return float(ssim(a, b, data_range=float(max(np.ptp(a), np.ptp(b), 1.0))))


def score_alignment_edges(a: np.ndarray, b: np.ndarray) -> float:
    # Edge-domain SSIM is usually more meaningful for atlas-vs-real matching
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a, b = _match_shape(a, b)
    ea = sobel(a)
    eb = sobel(b)
    return float(ssim(ea, eb, data_range=float(max(np.ptp(ea), np.ptp(eb), 1.0))))


def propose_landmarks(
    real_path: Path,
    atlas_path: Path,
    out_csv: Path,
    max_points: int = 30,
    min_distance: int = 12,
    ransac_residual: float = 8.0,
) -> dict:
    real = imread(str(real_path))
    atl = imread(str(atlas_path))
    real, _ = select_real_slice_2d(real, source_path=real_path)
    atl, _ = select_label_slice_2d(atl)

    rp = detect_landmarks(real, max_points=max_points, min_distance=min_distance)
    ap = detect_landmarks(atl, max_points=max_points, min_distance=min_distance)
    n = min(len(rp), len(ap))
    rp = rp[:n]
    ap = ap[:n]

    # RANSAC robust filtering for outlier landmark pairs
    inliers = np.ones(n, dtype=bool)
    if n >= 4:
        try:
            model, inliers = ransac(
                (ap, rp),
                AffineTransform,
                min_samples=3,
                residual_threshold=ransac_residual,
                max_trials=100,
            )
        except Exception:
            inliers = np.ones(n, dtype=bool)

    rp_f = rp[inliers]
    ap_f = ap[inliers]
    nf = len(rp_f)

    df = pd.DataFrame(
        {
            "real_x": rp_f[:, 0] if nf else [],
            "real_y": rp_f[:, 1] if nf else [],
            "atlas_x": ap_f[:, 0] if nf else [],
            "atlas_y": ap_f[:, 1] if nf else [],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    return {
        "landmark_pairs": int(nf),
        "raw_pairs": int(n),
        "score": score_alignment(real, atl),
        "params": {
            "max_points": max_points,
            "min_distance": min_distance,
            "ransac_residual": ransac_residual,
        },
        "csv": str(out_csv),
    }
