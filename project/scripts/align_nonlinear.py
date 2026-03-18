from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.slice_select import select_label_slice_2d, select_real_slice_2d
from skimage.transform import PiecewiseAffineTransform, warp
from tifffile import imread, imwrite


def apply_landmark_nonlinear(
    real_path: Path, atlas_label_path: Path, pairs_csv: Path, out_path: Path
) -> dict:
    fail_dir = out_path.parent / "fail_cases"
    fail_dir.mkdir(parents=True, exist_ok=True)
    fail_log = fail_dir / "align_nonlinear_last.json"

    try:
        real = imread(str(real_path))
        atlas = imread(str(atlas_label_path))
        real, _ = select_real_slice_2d(real, source_path=real_path)
        atlas, _ = select_label_slice_2d(atlas)

        pairs = pd.read_csv(pairs_csv)
        if len(pairs) < 4:
            raise ValueError("Need >=4 landmark pairs for non-linear transform")

        src = pairs[["atlas_x", "atlas_y"]].to_numpy(dtype=np.float32)
        dst = pairs[["real_x", "real_y"]].to_numpy(dtype=np.float32)

        tform = PiecewiseAffineTransform.from_estimate(src, dst)
        if tform is None:
            raise RuntimeError("PiecewiseAffineTransform estimate returned None")

        warped = warp(
            atlas.astype(np.float32),
            inverse_map=tform.inverse,
            output_shape=real.shape,
            order=0,
            preserve_range=True,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imwrite(str(out_path), warped.astype(np.uint16))

        result = {
            "pairs": int(len(pairs)),
            "warped_label": str(out_path),
            "mode": "piecewise_affine",
            "failLog": str(fail_log),
            "status": "ok",
        }
        fail_log.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "realPath": str(real_path),
                    "atlasLabelPath": str(atlas_label_path),
                    "pairsCsv": str(pairs_csv),
                    "warpedLabel": str(out_path),
                    "pairs": int(len(pairs)),
                    "shapeReal": list(real.shape),
                    "shapeAtlas": list(atlas.shape),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
    except Exception as e:
        fail_log.write_text(
            json.dumps(
                {
                    "status": "error",
                    "realPath": str(real_path),
                    "atlasLabelPath": str(atlas_label_path),
                    "pairsCsv": str(pairs_csv),
                    "warpedLabel": str(out_path),
                    "error": str(e),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(f"nonlinear align failed; see {fail_log}: {e}") from e
