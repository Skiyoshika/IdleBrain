from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.filters import sobel


def _looks_like_rgb(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[-1] in (3, 4) and img.shape[0] > 16 and img.shape[1] > 16


def _reduce_to_3d(img: np.ndarray) -> np.ndarray:
    out = np.asarray(img)
    while out.ndim > 3:
        out = out[..., 0]
    return out


def _score_real_slice(slice_2d: np.ndarray, z_idx: int, z_dim: int) -> float:
    x = slice_2d.astype(np.float32)
    if x.size == 0:
        return -1.0

    p2, p98 = np.percentile(x, (2, 98))
    if float(p98) <= float(p2):
        return -1.0

    x = np.clip((x - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    thr = float(np.percentile(x, 70.0))
    mask = x > thr
    area = float(np.mean(mask))

    e = sobel(x)
    eth = float(np.percentile(e, 80.0))
    edge_density = float(np.mean(e > eth))

    mid = (z_dim - 1) * 0.5
    center_bias = 1.0 - abs(float(z_idx) - mid) / max(mid, 1.0)

    return float(0.62 * area + 0.28 * edge_density + 0.10 * center_bias)


def _auto_pick_real_z(stack: np.ndarray) -> int:
    z_dim = int(stack.shape[0])
    if z_dim <= 1:
        return 0

    coarse_step = max(1, z_dim // 96)
    candidates = sorted(set(list(range(0, z_dim, coarse_step)) + [z_dim // 2, z_dim - 1]))

    best_z = z_dim // 2
    best_score = -1.0
    for z in candidates:
        s = _score_real_slice(stack[z], z, z_dim)
        if s > best_score:
            best_score = s
            best_z = int(z)

    lo = max(0, best_z - coarse_step)
    hi = min(z_dim - 1, best_z + coarse_step)
    fine_step = max(1, coarse_step // 3)
    for z in range(lo, hi + 1, fine_step):
        s = _score_real_slice(stack[z], z, z_dim)
        if s > best_score:
            best_score = s
            best_z = int(z)

    return int(best_z)


def select_real_slice_2d(
    real_img: np.ndarray,
    z_index: int | None = None,
    source_path: Path | str | None = None,  # reserved for future heuristics
) -> tuple[np.ndarray, dict]:
    """Return a 2D real slice from either 2D TIFF or 3D stack."""
    arr = _reduce_to_3d(np.asarray(real_img))
    input_shape = [int(x) for x in arr.shape]

    if arr.ndim == 2:
        return arr, {
            "is_volume": False,
            "input_shape": input_shape,
            "slice_shape": [int(arr.shape[0]), int(arr.shape[1])],
            "z_axis": None,
            "z_count": 1,
            "z_index": 0,
            "selected_by": "2d_input",
        }

    if _looks_like_rgb(arr):
        sl = arr[..., 0]
        return sl, {
            "is_volume": False,
            "input_shape": input_shape,
            "slice_shape": [int(sl.shape[0]), int(sl.shape[1])],
            "z_axis": None,
            "z_count": 1,
            "z_index": 0,
            "selected_by": "rgb_channel_0",
        }

    z_axis = int(np.argmin(np.array(arr.shape, dtype=np.int64)))
    stack = np.moveaxis(arr, z_axis, 0)
    z_dim = int(stack.shape[0])

    selected_by = "manual"
    if z_index is None:
        z_index = _auto_pick_real_z(stack)
        selected_by = "auto"

    z_index = int(np.clip(int(z_index), 0, z_dim - 1))
    sl = np.asarray(stack[z_index])
    if sl.ndim > 2:
        sl = sl[..., 0]

    return sl, {
        "is_volume": True,
        "input_shape": input_shape,
        "slice_shape": [int(sl.shape[0]), int(sl.shape[1])],
        "z_axis": int(z_axis),
        "z_count": int(z_dim),
        "z_index": int(z_index),
        "selected_by": selected_by,
    }


def select_label_slice_2d(
    label_img: np.ndarray,
    z_index: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Return a 2D label slice from label image/stack."""
    arr = _reduce_to_3d(np.asarray(label_img))
    input_shape = [int(x) for x in arr.shape]

    if arr.ndim == 2:
        return arr, {
            "is_volume": False,
            "input_shape": input_shape,
            "slice_shape": [int(arr.shape[0]), int(arr.shape[1])],
            "z_axis": None,
            "z_count": 1,
            "z_index": 0,
            "selected_by": "2d_input",
        }

    if _looks_like_rgb(arr):
        sl = arr[..., 0]
        return sl, {
            "is_volume": False,
            "input_shape": input_shape,
            "slice_shape": [int(sl.shape[0]), int(sl.shape[1])],
            "z_axis": None,
            "z_count": 1,
            "z_index": 0,
            "selected_by": "rgb_channel_0",
        }

    z_axis = int(np.argmin(np.array(arr.shape, dtype=np.int64)))
    stack = np.moveaxis(arr, z_axis, 0)
    z_dim = int(stack.shape[0])
    if z_index is None:
        z_index = z_dim // 2
        selected_by = "mid"
    else:
        selected_by = "manual"

    z_index = int(np.clip(int(z_index), 0, z_dim - 1))
    sl = np.asarray(stack[z_index])
    if sl.ndim > 2:
        sl = sl[..., 0]

    return sl, {
        "is_volume": True,
        "input_shape": input_shape,
        "slice_shape": [int(sl.shape[0]), int(sl.shape[1])],
        "z_axis": int(z_axis),
        "z_count": int(z_dim),
        "z_index": int(z_index),
        "selected_by": selected_by,
    }
