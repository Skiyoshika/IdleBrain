"""Quick single-slice overlay test.

Usage (from project/ dir):
    python scripts/test_single_slice.py --z 0300
    python scripts/test_single_slice.py --z 0200 --offset 150 --scale 0.82
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from tifffile import imwrite as tiff_imwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paths import bootstrap_sys_path

PROJECT_DIR = bootstrap_sys_path()

from scripts.overlay_render import render_overlay  # noqa: E402

ATLAS_PATH = PROJECT_DIR / "annotation_25.nii.gz"
REG_ATLAS_PATH = PROJECT_DIR / "outputs" / "registration_3d" / "annotation_registered.nii.gz"
SLICE_DIR = PROJECT_DIR / "data" / "35_C0_demo"
OUT_DIR = PROJECT_DIR / "outputs"
CONFIG = PROJECT_DIR / "configs" / "run_config_35.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", default="0300", help="z-filename stem, e.g. 0300 → z0300.tif")
    ap.add_argument("--offset", type=int, default=None, help="Override atlas_z_offset")
    ap.add_argument("--scale", type=float, default=None, help="Override tissue_shrink_factor")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--mode", type=str, default="fill", help="fill | contour | contour-major")
    ap.add_argument(
        "--use-3d", action="store_true", help="Use 3D-registered annotation if available"
    )
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    reg = cfg["registration"]

    z_scale = float(reg.get("atlas_z_z_scale", 0.2))
    offset = int(args.offset) if args.offset is not None else int(reg.get("atlas_z_offset", 150))
    shrink = (
        float(args.scale)
        if args.scale is not None
        else float(reg.get("tissue_shrink_factor", 0.88))
    )
    gamma = float(reg.get("display_gamma", 1.0))
    hemi = str(reg.get("atlas_hemisphere", "right_flipped"))
    pixel_um = float(cfg["input"]["pixel_size_um_xy"])

    z_num = int(args.z)
    atlas_z = int(round(z_num * z_scale)) + offset
    atlas_z = max(0, min(527, atlas_z))

    real_path = SLICE_DIR / f"z{z_num:04d}.tif"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"test_z{z_num:04d}_az{atlas_z}_{args.mode}.png"

    print(f"Real slice : {real_path}")
    print(f"Atlas z    : {atlas_z}  (z{z_num} × {z_scale} + {offset})")
    print(f"Shrink     : {shrink}")
    print(f"Hemisphere : {hemi}")
    print(f"Output     : {out_png}")

    # Always use raw atlas with 2D placement pipeline (best quality).
    # 3D-registered annotation (Elastix) is available via --use-3d but is
    # still in template space and needs 2D placement anyway.
    import nibabel as nib

    args.use_3d and REG_ATLAS_PATH.exists()
    atlas_source = ATLAS_PATH
    nii = nib.load(str(atlas_source))
    vol = np.asarray(nii.get_fdata(), dtype=np.int32)
    label_slice = vol[atlas_z, :, :]
    print(f"Atlas vol  : {vol.shape}  (raw atlas, AP={atlas_z})")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        tmp_label = Path(f.name)
    tiff_imwrite(str(tmp_label), label_slice.astype(np.int32))

    # For registered annotation: label is already in brain space → use prewarped mode
    # For raw atlas: use 2D placement pipeline
    warp_params = {
        "force_hemisphere": hemi,
        "tissue_shrink_factor": shrink,
        "enable_silhouette_conform": False,
        "enable_sitk_ref_refine": False,
    }

    try:
        render_overlay(
            real_slice_path=real_path,
            label_slice_path=tmp_label,
            out_png=out_png,
            alpha=args.alpha,
            mode=args.mode,
            structure_csv=PROJECT_DIR / "configs" / "allen_mouse_structure_graph.csv",
            pixel_size_um=pixel_um,
            fit_mode="cover",
            warp_params=warp_params,
            display_gamma=gamma,
            prewarped_label=False,
        )
        print(f"Done -> {out_png}")
    finally:
        tmp_label.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
