"""overlay_service.py — Business logic for overlay liquify and rendering operations.

Blueprints call these functions; scripts are imported here, not in route handlers.
All functions take plain Path/dict arguments and return plain Python objects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import imread, imwrite


def apply_liquify_and_render(
    base_label_path: Path,
    drags: list[dict],
    corrected_label_path: Path,
    hover_label_path: Path,
    render_kwargs: dict,
) -> tuple[np.ndarray, dict]:
    """Apply liquify drags to an atlas label and re-render the overlay.

    Writes corrected label to both *corrected_label_path* (timestamped archive copy)
    and *hover_label_path* (live preview used by region-at queries).

    Returns ``(corrected_label_array, diagnostic_dict)`` from render_overlay.
    """
    from scripts.slice_select import select_label_slice_2d
    from scripts.overlay_render import render_overlay
    from project.frontend.server_context import _apply_liquify_drags

    lbl_raw = imread(str(base_label_path))
    lbl2d, _ = select_label_slice_2d(lbl_raw)
    tissue = lbl2d > 0
    corrected = _apply_liquify_drags(lbl2d.astype(np.int32), drags=drags, tissue_mask=tissue)

    imwrite(str(corrected_label_path), corrected.astype(np.int32))
    imwrite(str(hover_label_path), corrected.astype(np.int32))

    _, diagnostic = render_overlay(**render_kwargs)
    return corrected, diagnostic


def render_overlay_from_label(
    real_path: Path,
    label_path: Path,
    out_path: Path,
    render_kwargs: dict,
) -> dict:
    """Render an overlay PNG from an existing (pre-warped) label file.

    Used by calibration-finalize and atlas-layer routes that already have a
    ready label and just need to call render_overlay cleanly.

    Returns the *diagnostic* dict from render_overlay.
    """
    from scripts.overlay_render import render_overlay

    _, diagnostic = render_overlay(real_path, label_path, out_path, **render_kwargs)
    return diagnostic
