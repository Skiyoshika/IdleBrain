from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scripts.slice_select import select_label_slice_2d, select_real_slice_2d
from skimage.transform import estimate_transform, warp
from tifffile import imread, imwrite


def apply_landmark_affine(
    real_path: Path, atlas_label_path: Path, pairs_csv: Path, out_path: Path
) -> dict:
    real = imread(str(real_path))
    atlas = imread(str(atlas_label_path))
    real, _ = select_real_slice_2d(real, source_path=real_path)
    atlas, _ = select_label_slice_2d(atlas)

    pairs = pd.read_csv(pairs_csv)
    if len(pairs) < 3:
        raise ValueError("Need >=3 landmark pairs for affine transform")

    src = pairs[["atlas_x", "atlas_y"]].to_numpy(dtype=np.float32)
    dst = pairs[["real_x", "real_y"]].to_numpy(dtype=np.float32)
    tform = estimate_transform("affine", src, dst)

    warped = warp(
        atlas.astype(np.float32),
        inverse_map=tform.inverse,
        output_shape=real.shape,
        order=0,
        preserve_range=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_path), warped.astype(np.uint16))

    return {
        "pairs": int(len(pairs)),
        "matrix": tform.params.tolist(),
        "warped_label": str(out_path),
    }
