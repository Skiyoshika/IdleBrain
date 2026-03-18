"""atlas_deepslice.py — AP position estimation via DeepSlice.

Wraps DeepSlice (https://github.com/PolarBean/DeepSlice) for our lightsheet
cleared-brain pipeline.

Key design choices:
  • Run on the FULL SERIES at once → DeepSlice uses propagate_angles() to
    smooth out single-slice ambiguities.
  • For each slice, compare the DeepSlice estimate with our formula estimate.
    If |deepslice - formula| > max_deviation, fall back to the formula.
  • Images are converted to 8-bit RGB with gamma correction before passing
    to DeepSlice (it expects 8-bit).

Usage
-----
  from scripts.atlas_deepslice import predict_ap_series

  results = predict_ap_series(
      slice_paths,           # list of Path objects, ordered anterior→posterior
      z_scale=0.2,
      z_offset=150,
      pixel_size_um=5.0,
      display_gamma=0.65,
      max_deviation=30,      # fall back to formula if deviation > this many voxels
  )
  # returns: dict[slice_path_stem -> atlas_z]
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Suppress TensorFlow noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

_MODEL_CACHE: dict = {}


def _get_model(species: str = "mouse"):
    if species not in _MODEL_CACHE:
        from DeepSlice import DSModel

        _MODEL_CACHE[species] = DSModel(species=species)
    return _MODEL_CACHE[species]


def _to_8bit_rgb(arr: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    """Convert 16-bit grayscale to 8-bit RGB with gamma for DeepSlice."""
    x = arr.astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    p1 = float(np.percentile(x, 1))
    p99 = float(np.percentile(x, 99))
    if p99 <= p1:
        p1, p99 = float(x.min()), float(x.max()) + 1e-6
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    if gamma != 1.0:
        x = x ** float(gamma)
    x = (x * 255.0).astype(np.uint8)
    return np.stack([x, x, x], axis=-1)


def _formula_atlas_z(
    z_filename: int, z_scale: float, z_offset: int, atlas_z_min: int = 0, atlas_z_max: int = 527
) -> int:
    az = int(round(z_filename * z_scale)) + z_offset
    return max(atlas_z_min, min(atlas_z_max, az))


def _get_ds_atlas_z(row, predictions) -> int:
    """Extract atlas_z (AP voxel index) from a DeepSlice prediction row."""
    from DeepSlice.coord_post_processing.depth_estimation import calculate_brain_center_depth

    coords = (
        predictions[["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]]
        .loc[row.name]
        .values.astype(float)
    )
    depth = float(calculate_brain_center_depth(coords))
    return int(round(depth))


def predict_ap_series(
    slice_paths: Sequence[Path],
    z_scale: float = 0.2,
    z_offset: int = 150,
    pixel_size_um: float = 5.0,
    display_gamma: float = 0.65,
    max_deviation: int = 30,
    species: str = "mouse",
    atlas_z_min: int = 0,
    atlas_z_max: int = 527,
) -> dict[str, int]:
    """Run DeepSlice on a full brain series, return per-slice atlas_z.

    Parameters
    ----------
    slice_paths : ordered list of lightsheet slice TIFFs
    z_scale / z_offset : formula parameters (atlas_z = int(z_num × scale) + offset)
    display_gamma : gamma for 8-bit conversion (< 1 brightens dim images)
    max_deviation : if |deepslice_z - formula_z| > this, fall back to formula

    Returns
    -------
    dict mapping slice stem (e.g. "z0300") to atlas_z
    """
    from tifffile import imread as tiff_imread

    try:
        from PIL import Image as PilImage
    except ImportError:
        raise RuntimeError("Pillow is required for atlas_deepslice: pip install Pillow") from None

    model = _get_model(species)

    # Parse z-numbers from filenames (handles z0300.tif → 300)
    z_nums: list[int] = []
    for p in slice_paths:
        m = re.search(r"z(\d+)", Path(p).stem)
        z_nums.append(int(m.group(1)) if m else 0)

    # Compute formula estimates
    formula_zs = [_formula_atlas_z(z, z_scale, z_offset, atlas_z_min, atlas_z_max) for z in z_nums]

    # Write images to temp directory with DeepSlice-compatible names
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        written: list[tuple[int, Path]] = []  # (sequential_idx, path)

        for seq_idx, (sp, _z_num) in enumerate(zip(slice_paths, z_nums, strict=False), start=1):
            try:
                raw = tiff_imread(str(sp))
                rgb = _to_8bit_rgb(raw, gamma=display_gamma)
                out_path = tmp / f"brain_s{seq_idx:04d}.png"
                PilImage.fromarray(rgb).save(str(out_path))
                written.append((seq_idx, out_path))
            except Exception as e:
                print(f"[deepslice] WARNING: could not prepare {sp.name}: {e}")

        if not written:
            print("[deepslice] No images prepared; returning formula estimates")
            return {Path(p).stem: fz for p, fz in zip(slice_paths, formula_zs, strict=False)}

        print(f"[deepslice] Running DeepSlice on {len(written)} slices ...")
        try:
            model.predict(image_directory=str(tmp), ensemble=True, section_numbers=True)
            model.propagate_angles()
        except Exception as e:
            print(f"[deepslice] Prediction failed: {e}; falling back to formula")
            return {Path(p).stem: fz for p, fz in zip(slice_paths, formula_zs, strict=False)}

        # Build seq_idx → deepslice atlas_z map
        ds_map: dict[int, int] = {}
        for _, row in model.predictions.iterrows():
            m2 = re.search(r"_s(\d+)", str(row.Filenames))
            if not m2:
                continue
            seq = int(m2.group(1))
            try:
                ds_z = _get_ds_atlas_z(row, model.predictions)
                ds_z = max(atlas_z_min, min(atlas_z_max, ds_z))
                ds_map[seq] = ds_z
            except Exception:
                pass

    # Combine with formula using deviation-based fallback
    results: dict[str, int] = {}
    used_ds, used_formula, used_total = 0, 0, 0
    for seq_idx, (sp, _z_num, formula_z) in enumerate(
        zip(slice_paths, z_nums, formula_zs, strict=False), start=1
    ):
        stem = Path(sp).stem
        ds_z = ds_map.get(seq_idx)
        used_total += 1
        if ds_z is not None and abs(ds_z - formula_z) <= max_deviation:
            results[stem] = ds_z
            used_ds += 1
        else:
            results[stem] = formula_z
            used_formula += 1
            if ds_z is not None:
                pass  # outlier silently discarded

    print(
        f"[deepslice] Used DeepSlice: {used_ds}/{used_total}, "
        f"formula fallback: {used_formula}/{used_total}"
    )
    return results


def predict_ap_single(
    slice_path: Path,
    z_scale: float = 0.2,
    z_offset: int = 150,
    display_gamma: float = 0.65,
    max_deviation: int = 30,
    species: str = "mouse",
) -> int:
    """Convenience wrapper for a single slice. Returns atlas_z."""
    result = predict_ap_series(
        [slice_path],
        z_scale=z_scale,
        z_offset=z_offset,
        display_gamma=display_gamma,
        max_deviation=max_deviation,
        species=species,
    )
    return next(iter(result.values()))
