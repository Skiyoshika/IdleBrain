from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paths import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from scripts.atlas_autopick import autopick_best_z  # noqa: E402
from scripts.overlay_render import render_overlay  # noqa: E402


def _next_test_dir(outputs_root: Path) -> Path:
    outputs_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while (outputs_root / f"Test_{n}").exists():
        n += 1
    out = outputs_root / f"Test_{n}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _to_jpg_and_thumb(png_path: Path, thumb_width: int = 960) -> dict:
    jpg_path = png_path.with_suffix(".jpg")
    thumb_path = png_path.with_name(f"{png_path.stem}_thumb.png")
    with Image.open(png_path) as im:
        im.convert("RGB").save(jpg_path, quality=92)
        w, h = im.size
        tw = max(1, int(thumb_width))
        th = max(1, int(round(h * tw / max(w, 1))))
        thumb = im.resize((tw, th), Image.Resampling.BILINEAR)
        thumb.save(thumb_path)
    return {
        "jpg": str(jpg_path),
        "thumb": str(thumb_path),
    }


def _default_tag(path: Path, idx: int) -> str:
    name = path.stem.lower()
    if "_c0" in name:
        return "c0"
    if "_c1" in name:
        return "c1"
    return f"ch{idx:02d}"


def _load_tuned_params(path: Path | None) -> tuple[dict, str | None, int | None]:
    if path is None:
        return {}, None, None
    if not path.exists():
        raise FileNotFoundError(f"tuned params json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    fit_mode = None
    edge_smooth = None
    warp_params: dict = {}

    if isinstance(data, dict):
        if isinstance(data.get("warpParams"), dict):
            warp_params = dict(data["warpParams"])
        if isinstance(data.get("fitMode"), str):
            fit_mode = data["fitMode"]
        if "edgeSmoothIter" in data:
            try:
                edge_smooth = int(data["edgeSmoothIter"])
            except Exception:
                edge_smooth = None
        best = data.get("best")
        if isinstance(best, dict):
            params = best.get("params", {})
            if isinstance(params, dict):
                if isinstance(params.get("warpParams"), dict):
                    warp_params = dict(params["warpParams"])
                if isinstance(params.get("fitMode"), str):
                    fit_mode = params["fitMode"]
                if "edgeSmoothIter" in params:
                    try:
                        edge_smooth = int(params["edgeSmoothIter"])
                    except Exception:
                        edge_smooth = edge_smooth

    return warp_params, fit_mode, edge_smooth


def main() -> None:
    ap = argparse.ArgumentParser(description="Run atlas overlay test into outputs/Test_N")
    ap.add_argument(
        "--real",
        action="append",
        required=True,
        help="Path to real TIFF (can be 2D or 3D stack). Use multiple --real for multi-channel.",
    )
    ap.add_argument(
        "--annotation",
        default=str(Path(__file__).resolve().parent.parent / "annotation_25.nii.gz"),
        help="Path to annotation_25.nii.gz",
    )
    ap.add_argument(
        "--outputs-root",
        default=str(Path(__file__).resolve().parent.parent / "outputs"),
        help="Outputs root directory",
    )
    ap.add_argument("--pixel-size-um", type=float, default=0.65)
    ap.add_argument("--z-step", type=int, default=1)
    ap.add_argument("--slicing-plane", default="coronal")
    ap.add_argument("--major-top-k", type=int, default=28)
    ap.add_argument(
        "--fit-mode", default="cover", choices=["contain", "cover", "width-lock", "height-lock"]
    )
    ap.add_argument("--edge-smooth-iter", type=int, default=1)
    ap.add_argument(
        "--tuned-params-json",
        default="",
        help="Optional json with tuned params. Supports outputs/trainset_tuned_params.json format.",
    )
    args = ap.parse_args()

    annotation = Path(args.annotation)
    if not annotation.exists():
        raise FileNotFoundError(f"annotation not found: {annotation}")

    outputs_root = Path(args.outputs_root)
    out_dir = _next_test_dir(outputs_root)
    tuned_json = Path(args.tuned_params_json) if str(args.tuned_params_json).strip() else None
    warp_params, tuned_fit_mode, tuned_edge_smooth = _load_tuned_params(tuned_json)
    fit_mode = str(tuned_fit_mode or args.fit_mode)
    edge_smooth_iter = int(
        tuned_edge_smooth if tuned_edge_smooth is not None else args.edge_smooth_iter
    )

    summary: dict = {
        "output_dir": str(out_dir),
        "annotation": str(annotation),
        "pixel_size_um": float(args.pixel_size_um),
        "z_step": int(args.z_step),
        "slicing_plane": str(args.slicing_plane),
        "fit_mode": str(fit_mode),
        "edge_smooth_iter": int(edge_smooth_iter),
        "warp_params": warp_params,
        "tuned_params_json": str(tuned_json) if tuned_json is not None else "",
        "channels": {},
    }

    for i, real_s in enumerate(args.real, start=1):
        real_path = Path(real_s)
        if not real_path.exists():
            raise FileNotFoundError(f"real path not found: {real_path}")

        tag = _default_tag(real_path, i)
        prefix = f"{tag}_auto"
        label_tif = out_dir / f"{prefix}_label.tif"
        warped_tif = out_dir / f"{prefix}_warped_label.tif"
        contour_png = out_dir / f"{prefix}_contour_major.png"
        fill_png = out_dir / f"{prefix}_fill.png"

        auto_meta = autopick_best_z(
            real_path=real_path,
            annotation_nii=annotation,
            out_label_tif=label_tif,
            z_step=int(args.z_step),
            pixel_size_um=float(args.pixel_size_um),
            slicing_plane=str(args.slicing_plane),
            roi_mode="auto",
        )

        _, contour_meta = render_overlay(
            real_slice_path=real_path,
            label_slice_path=label_tif,
            out_png=contour_png,
            alpha=0.72,
            mode="contour-major",
            pixel_size_um=float(args.pixel_size_um),
            major_top_k=int(args.major_top_k),
            fit_mode=fit_mode,
            edge_smooth_iter=edge_smooth_iter,
            warp_params=warp_params,
            return_meta=True,
            warped_label_out=warped_tif,
        )
        _, fill_meta = render_overlay(
            real_slice_path=real_path,
            label_slice_path=label_tif,
            out_png=fill_png,
            alpha=0.45,
            mode="fill",
            pixel_size_um=float(args.pixel_size_um),
            major_top_k=int(args.major_top_k),
            fit_mode=fit_mode,
            edge_smooth_iter=edge_smooth_iter,
            warp_params=warp_params,
            return_meta=True,
        )

        contour_aux = _to_jpg_and_thumb(contour_png)
        fill_aux = _to_jpg_and_thumb(fill_png)

        summary["channels"][tag] = {
            "real_path": str(real_path),
            "autopick": auto_meta,
            "contour": {
                "png": str(contour_png),
                "warped_label_tif": str(warped_tif),
                "diagnostic": contour_meta,
                **contour_aux,
            },
            "fill": {
                "png": str(fill_png),
                "diagnostic": fill_meta,
                **fill_aux,
            },
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(str(out_dir))
    print(str(summary_path))


if __name__ == "__main__":
    main()
