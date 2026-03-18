"""alignment_service.py — Business logic for landmark and nonlinear alignment operations.

Blueprints call these functions; scripts are imported here, not in route handlers.
All functions take plain Path/dict arguments and return plain dicts — no Flask objects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread, imwrite


def score_and_render_comparison(
    real_path: Path,
    label_before_path: Path,
    label_after_path: Path,
    compare_out_path: Path,
    alpha: float = 0.45,
) -> dict:
    """Score alignment before/after and save a side-by-side comparison image.

    Returns a dict with beforeScore, afterScore, beforeEdgeScore, afterEdgeScore,
    scoreWarning, and compareImage.
    """
    from scripts.ai_landmark import score_alignment, score_alignment_edges
    from scripts.compare_render import render_before_after
    from scripts.slice_select import select_real_slice_2d, select_label_slice_2d

    real = imread(str(real_path))
    atlas_before = imread(str(label_before_path))
    atlas_after = imread(str(label_after_path))
    real, _ = select_real_slice_2d(real, source_path=real_path)
    atlas_before, _ = select_label_slice_2d(atlas_before)
    atlas_after, _ = select_label_slice_2d(atlas_after)

    before = score_alignment(real, atlas_before)
    after = score_alignment(real, atlas_after)
    before_edge = score_alignment_edges(real, atlas_before)
    after_edge = score_alignment_edges(real, atlas_after)

    render_before_after(
        real_path, label_before_path, label_after_path, compare_out_path,
        alpha=alpha, before_score=before_edge, after_score=after_edge,
    )
    return {
        "beforeScore": before,
        "afterScore": after,
        "beforeEdgeScore": before_edge,
        "afterEdgeScore": after_edge,
        "scoreWarning": after_edge < before_edge,
        "compareImage": str(compare_out_path),
    }


def apply_nonlinear_alignment(
    real_path: Path,
    atlas_label_path: Path,
    pairs_csv: Path,
    out_label: Path,
    compare_out: Path,
    alpha: float = 0.45,
) -> dict:
    """Apply nonlinear landmark alignment, score result, render comparison.

    Returns merged dict of apply_landmark_nonlinear metadata + scoring results.
    """
    from scripts.align_nonlinear import apply_landmark_nonlinear

    meta = apply_landmark_nonlinear(real_path, atlas_label_path, pairs_csv, out_label)
    scores = score_and_render_comparison(real_path, atlas_label_path, out_label, compare_out, alpha)
    return {**meta, **scores}


def apply_affine_alignment(
    real_path: Path,
    atlas_label_path: Path,
    pairs_csv: Path,
    out_label: Path,
    compare_out: Path,
    alpha: float = 0.45,
) -> dict:
    """Apply affine landmark alignment, score result, render comparison.

    Returns merged dict of apply_landmark_affine metadata + scoring results.
    """
    from scripts.align_ai import apply_landmark_affine

    meta = apply_landmark_affine(real_path, atlas_label_path, pairs_csv, out_label)
    scores = score_and_render_comparison(real_path, atlas_label_path, out_label, compare_out, alpha)
    return {**meta, **scores}


def propose_landmarks(
    real_path: Path,
    atlas_path: Path,
    out_csv: Path,
    *,
    max_points: int = 30,
    min_distance: int = 12,
    ransac_residual: float = 8.0,
) -> dict:
    """Detect and propose landmark pairs between real and atlas images.

    Returns the result dict from the underlying script.
    """
    from scripts.ai_landmark import propose_landmarks as _propose_landmarks

    return _propose_landmarks(
        real_path, atlas_path, out_csv,
        max_points=max_points,
        min_distance=min_distance,
        ransac_residual=ransac_residual,
    )


def render_landmark_preview(
    real_path: Path,
    atlas_path: Path,
    pairs_csv: Path,
    out_path: Path,
) -> int:
    """Draw landmark dots on real + atlas side-by-side and save PNG.

    Returns the number of landmark pairs drawn.
    """
    from scripts.slice_select import select_real_slice_2d, select_label_slice_2d

    real = imread(str(real_path))
    atlas = imread(str(atlas_path))
    real, _ = select_real_slice_2d(real, source_path=real_path)
    atlas, _ = select_label_slice_2d(atlas)

    h = min(real.shape[0], atlas.shape[0])
    w = min(real.shape[1], atlas.shape[1])
    real = real[:h, :w]
    atlas = atlas[:h, :w]

    real_rgb = np.stack([real, real, real], axis=-1).astype(np.uint8)
    atlas_rgb = np.stack([atlas, atlas, atlas], axis=-1).astype(np.uint8)

    pairs = pd.read_csv(pairs_csv)
    for _, r in pairs.iterrows():
        rx, ry = int(r['real_x']), int(r['real_y'])
        ax, ay = int(r['atlas_x']), int(r['atlas_y'])
        if 0 <= ry < h and 0 <= rx < w:
            real_rgb[max(0, ry - 2):ry + 3, max(0, rx - 2):rx + 3] = [255, 255, 0]
        if 0 <= ay < h and 0 <= ax < w:
            atlas_rgb[max(0, ay - 2):ay + 3, max(0, ax - 2):ax + 3] = [0, 255, 255]

    pad = np.zeros((h, 8, 3), dtype=np.uint8)
    canvas = np.concatenate([real_rgb, pad, atlas_rgb], axis=1)
    imwrite(str(out_path), canvas)
    return int(len(pairs))
