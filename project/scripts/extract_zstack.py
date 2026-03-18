"""
extract_zstack.py  —  Split a 3-D Z-stack TIFF into individual 2-D slice TIFs.

Usage:
    python extract_zstack.py --input <zstack.tif> --out_dir <folder>
                             [--channel 0] [--every_n 1] [--z_min 0] [--z_max -1]

Examples
--------
# Extract every Z-plane from channel 0 into slices/
python extract_zstack.py \\
    --input "Sample/35_High...C0.tif" \\
    --out_dir project/data/35_C0_slices

# Demo: every 5th slice, covering the full brain at 5x coarser resolution
python extract_zstack.py \\
    --input "Sample/35_High...C0.tif" \\
    --out_dir project/data/35_C0_demo \\
    --every_n 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tifffile import TiffFile, imwrite


def extract_zstack(
    src: Path,
    out_dir: Path,
    channel: int = 0,
    every_n: int = 1,
    z_min: int = 0,
    z_max: int = -1,
) -> list[Path]:
    """
    Extract 2-D slices from a multi-page (Z-stack) TIFF.

    Parameters
    ----------
    src      : Path to the input Z-stack TIFF.
    out_dir  : Directory where individual slice TIFs are written.
    channel  : For (Z, H, W, C) stacks select this channel index.
               For (Z, H, W) stacks this argument is ignored.
    every_n  : Keep only every N-th slice (1 = all slices).
    z_min    : First Z index to extract (inclusive).
    z_max    : Last  Z index to extract (inclusive, -1 = last).

    Returns
    -------
    List of written file paths, sorted by Z index.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {src.name} …", flush=True)
    with TiffFile(str(src)) as tf:
        arr = tf.asarray()  # full load; typical brain ~1-4 GB

    if arr.ndim == 2:
        # Single-page TIFF — treat as one slice
        arr = arr[np.newaxis, ...]

    if arr.ndim == 4:
        # (Z, H, W, C) — pick channel
        arr = arr[:, :, :, channel]

    if arr.ndim != 3:
        raise ValueError(f"Unexpected TIFF shape after loading: {arr.shape}")

    n_z = arr.shape[0]
    if z_max < 0:
        z_max = n_z - 1
    z_max = min(z_max, n_z - 1)

    written: list[Path] = []
    for z in range(z_min, z_max + 1, every_n):
        slc = arr[z]  # (H, W) uint16
        dst = out_dir / f"z{z:04d}.tif"
        imwrite(str(dst), slc.astype(np.uint16))
        written.append(dst)

    print(
        f"Extracted {len(written)} slices (Z {z_min}–{z_max}, every {every_n}) → {out_dir}",
        flush=True,
    )
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Split Z-stack TIFF into 2-D slice files")
    ap.add_argument("--input", required=True, help="Z-stack TIFF path")
    ap.add_argument("--out_dir", required=True, help="Output directory for slice TIFs")
    ap.add_argument("--channel", type=int, default=0, help="Channel index (for ZCYX stacks)")
    ap.add_argument("--every_n", type=int, default=1, help="Keep every N-th slice")
    ap.add_argument("--z_min", type=int, default=0, help="First Z index (inclusive)")
    ap.add_argument("--z_max", type=int, default=-1, help="Last Z index (-1 = last)")
    args = ap.parse_args()

    extract_zstack(
        src=Path(args.input),
        out_dir=Path(args.out_dir),
        channel=args.channel,
        every_n=args.every_n,
        z_min=args.z_min,
        z_max=args.z_max,
    )


if __name__ == "__main__":
    main()
