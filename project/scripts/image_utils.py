"""
Shared image utility functions used across scripts and the Flask server.

All functions here are pure (no IO, no side effects) so they can be unit-tested
without loading any atlas or TIFF file.
"""

from __future__ import annotations

import numpy as np


def norm_u8_robust(img: np.ndarray) -> np.ndarray:
    """Stretch image to uint8 using 1st–99th percentile clip.

    Robust to sparse fluorescence images where a few bright cells would
    otherwise dominate a simple min/max stretch.

    Parameters
    ----------
    img : np.ndarray
        Input array of any numeric dtype and shape.

    Returns
    -------
    np.ndarray
        uint8 array with same shape as ``img``.
    """
    x = img.astype(np.float32)
    p1, p99 = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if p99 <= p1:
        p1, p99 = float(np.min(x)), float(np.max(x) + 1e-6)
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def alpha_blend(base_gray: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a gray image with an RGBA/RGB color overlay at given alpha.

    Parameters
    ----------
    base_gray : np.ndarray
        2-D uint8 gray image (H, W).
    color_mask : np.ndarray
        3-D uint8 color image (H, W, 3).
    alpha : float
        Blend weight for ``color_mask`` (0 = gray only, 1 = color only).

    Returns
    -------
    np.ndarray
        uint8 RGB image (H, W, 3).
    """
    base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1).astype(np.float32)
    out = (1.0 - alpha) * base_rgb + alpha * color_mask.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Convert any image to a 2-D uint8 grayscale via robust stretch.

    If the input is multi-channel, only the first channel is used.

    Parameters
    ----------
    img : np.ndarray
        Image of shape (H, W) or (H, W, C).

    Returns
    -------
    np.ndarray
        uint8 2-D array.
    """
    if img.ndim == 3:
        img = img[..., 0]
    return norm_u8_robust(img)
