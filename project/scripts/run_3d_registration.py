"""
3D volumetric registration pipeline for Brainfast.

Pipeline:
  1. Convert a stack TIFF or z*.tif folder into a 3D NIfTI volume
  2. Crop the Allen CCF template and annotation to the target AP range + hemisphere
  3. Register the brain volume into template space with Elastix or ANTs
  4. Save fixed-space annotation, metrics, metadata, and an overview PNG

Usage (from project/ dir):
    python scripts/run_3d_registration.py --config configs/run_config_3d_sample.json
    python scripts/run_3d_registration.py --config configs/run_config_3d_ants_sample.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_registration_report import build_outputs_index, build_run_report
from scripts.laplacian_refine import LaplacianConfig, refine_volume_with_laplacian
from scripts.paths import bootstrap_sys_path
from scripts.staining_stats import compute_staining_stats
from scripts.volume_io import inspect_volume_source, volume_source_to_nifti

PROJECT_DIR = bootstrap_sys_path()

UCI_DIR = Path("d:/UCI-ALLEN-BrainRepositoryCodeGUI-main")
ELASTIX_EXE = UCI_DIR / "elastix" / "elastix.exe"
TRANSFORMIX_EXE = UCI_DIR / "elastix" / "transformix.exe"
CCF_DATA = UCI_DIR / "CCF_DATA"
TEMPLATE_25 = CCF_DATA / "average_template_25.nii.gz"
PARAMS_RIGID = UCI_DIR / "001_parameters_Rigid.txt"
PARAMS_BSPLINE = UCI_DIR / "002_parameters_BSpline.txt"
ANNOTATION = PROJECT_DIR / "annotation_25.nii.gz"


def _compute_ap_range_from_source(
    input_source: Path | None,
    z_scale: float,
    z_offset: int,
) -> tuple[int, int]:
    if input_source is None or not input_source.is_dir():
        return 0, 528

    slices = sorted(input_source.glob("z*.tif"))
    if not slices:
        return 0, 528
    try:
        z_first = int(slices[0].stem[1:])
        z_last = int(slices[-1].stem[1:])
    except Exception:
        return 0, 528

    ap_start = max(0, int(z_first * z_scale) + z_offset - 5)
    ap_end = min(528, int(z_last * z_scale) + z_offset + 5)
    return ap_start, ap_end


def prepare_cropped_template(
    full_path,
    annot_path,
    hemisphere,
    ap_start,
    ap_end,
    out_dir,
    *,
    half_width: int | None = None,
):
    """Crop template AND annotation to AP range + hemisphere."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl_out = out_dir / "template_cropped.nii.gz"
    ann_out = out_dir / "annotation_cropped.nii.gz"

    for src, dst, is_label in [(full_path, tmpl_out, False), (annot_path, ann_out, True)]:
        if dst.exists():
            print(f"  Already exists: {dst.name}")
            continue
        img = nib.load(str(src))
        data = np.asarray(img.dataobj)
        # AP crop
        data = data[ap_start : min(ap_end, data.shape[0]), :, :]
        # ML hemisphere
        half = (
            int(half_width)
            if half_width is not None and int(half_width) > 0
            else data.shape[2] // 2
        )
        if hemisphere in ("right", "right_flipped"):
            data = data[:, :, max(0, data.shape[2] - half) :]
        else:
            data = data[:, :, : min(half, data.shape[2])]
        # Flip ML for right_flipped
        if hemisphere == "right_flipped":
            data = data[:, :, ::-1].copy()
        dtype = np.int32 if is_label else np.float32
        out_img = nib.Nifti1Image(data.astype(dtype), img.affine)
        out_img.header.set_zooms((0.025, 0.025, 0.025))
        nib.save(out_img, str(dst))
        print(f"  Saved {dst.name}  shape={data.shape}")

    return tmpl_out, ann_out


def run_elastix(fixed, moving, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ELASTIX_EXE),
        "-f",
        str(fixed),
        "-m",
        str(moving),
        "-out",
        str(out_dir),
        "-p",
        str(PARAMS_RIGID),
        "-p",
        str(PARAMS_BSPLINE),
        "-threads",
        "8",
    ]
    print(f"  Fixed:  {fixed.name}  {nib.load(str(fixed)).shape}")
    print(f"  Moving: {moving.name}  {nib.load(str(moving)).shape}")
    t0 = time.time()
    r = subprocess.run(cmd)
    print(f"  Done in {(time.time() - t0) / 60:.1f} min  (exit {r.returncode})")
    if r.returncode != 0:
        raise RuntimeError("Elastix failed - check elastix log above")
    return out_dir


def run_transformix(annotation, transform_params, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Patch to nearest-neighbor for label images
    tp = transform_params.read_text(encoding="utf-8", errors="replace")
    tp = tp.replace("(FinalBSplineInterpolationOrder 3)", "(FinalBSplineInterpolationOrder 0)")
    tp = tp.replace(
        '(ResampleInterpolator "FinalBSplineInterpolator")',
        '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
    )
    patched = out_dir / "TransformParameters_NN.txt"
    patched.write_text(tp, encoding="utf-8")
    cmd = [
        str(TRANSFORMIX_EXE),
        "-in",
        str(annotation),
        "-out",
        str(out_dir),
        "-tp",
        str(patched),
        "-threads",
        "8",
    ]
    t0 = time.time()
    r = subprocess.run(cmd)
    print(f"  Transformix done in {time.time() - t0:.1f}s  (exit {r.returncode})")
    if r.returncode != 0:
        raise RuntimeError("Transformix failed")
    for name in ["result.nii", "result.nii.gz", "result.mhd"]:
        p = out_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Transformix output not found in {out_dir}")


def run_ants_registration(
    fixed: Path,
    moving: Path,
    out_dir: Path,
    *,
    transform_type: str = "SyN",
    random_seed: int = 42,
    verbose: bool = True,
    reg_iterations: tuple[int, int, int] = (40, 20, 0),
) -> tuple[Path, dict[str, object]]:
    try:
        import ants
    except Exception as exc:
        raise RuntimeError(
            "ANTs backend requested but antspyx is not available. "
            "Install antspyx or switch backend to elastix."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    outprefix = str((out_dir / "ants_").resolve())
    fixed_img = ants.image_read(str(fixed))
    moving_img = ants.image_read(str(moving))

    result = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform=transform_type,
        outprefix=outprefix,
        reg_iterations=reg_iterations,
        random_seed=random_seed,
        verbose=verbose,
    )

    registered_path = out_dir / "ants_result.nii.gz"
    ants.image_write(result["warpedmovout"], str(registered_path))

    fwd = [str(Path(p).resolve()) for p in result.get("fwdtransforms", [])]
    inv = [str(Path(p).resolve()) for p in result.get("invtransforms", [])]
    transform_manifest = out_dir / "ants_transforms.txt"
    lines = [
        "# ANTs Registration Transform Files",
        f"# Type: {transform_type}",
        f"# Fixed: {fixed.resolve()}",
        f"# Moving: {moving.resolve()}",
        "",
        "Forward Transforms (apply in order):",
    ]
    lines.extend(f"  {p}" for p in fwd)
    lines.extend(["", "Inverse Transforms (apply in order):"])
    lines.extend(f"  {p}" for p in inv)
    transform_manifest.write_text("\n".join(lines), encoding="utf-8")

    return registered_path, {
        "transform_type": transform_type,
        "random_seed": int(random_seed),
        "reg_iterations": [int(v) for v in reg_iterations],
        "forward_transforms": fwd,
        "inverse_transforms": inv,
        "transform_manifest": str(transform_manifest.resolve()),
    }


def compute_registration_metrics(
    fixed_path: Path,
    registered_path: Path,
    *,
    sample_slices: int = 7,
) -> dict[str, float]:
    fixed = np.asarray(nib.load(str(fixed_path)).dataobj, dtype=np.float32)
    registered = np.asarray(nib.load(str(registered_path)).dataobj, dtype=np.float32)
    common_shape = tuple(min(a, b) for a, b in zip(fixed.shape, registered.shape, strict=False))
    fixed = fixed[: common_shape[0], : common_shape[1], : common_shape[2]]
    registered = registered[: common_shape[0], : common_shape[1], : common_shape[2]]

    fixed = _norm01(fixed)
    registered = _norm01(registered)

    diff = fixed - registered
    mse = float(np.mean(diff**2))
    psnr = float(peak_signal_noise_ratio(fixed, registered, data_range=1.0))
    flat_fixed = fixed.ravel()
    flat_registered = registered.ravel()
    if np.std(flat_fixed) <= 1e-8 or np.std(flat_registered) <= 1e-8:
        ncc = 0.0
    else:
        ncc = float(np.corrcoef(flat_fixed, flat_registered)[0, 1])

    z_indices = np.linspace(0, fixed.shape[0] - 1, num=max(1, sample_slices), dtype=int)
    ssim_scores = [
        float(structural_similarity(fixed[z], registered[z], data_range=1.0)) for z in z_indices
    ]

    fixed_mask = fixed > 0.10
    registered_mask = registered > 0.10
    inter = float(np.logical_and(fixed_mask, registered_mask).sum())
    denom = float(fixed_mask.sum() + registered_mask.sum())
    dice = float((2.0 * inter / denom) if denom > 0 else 0.0)

    return {
        "NCC": ncc,
        "SSIM": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        "Dice": dice,
        "MSE": mse,
        "PSNR": psnr,
    }


def write_registration_reports(
    out_dir: Path,
    *,
    metadata: dict,
    metrics: dict[str, float],
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "registration_metadata.json"
    metrics_path = out_dir / "registration_metrics.csv"
    summary_path = out_dir / "registration_summary.txt"

    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, f"{value:.6f}"])

    lines = [
        "=" * 60,
        "REGISTRATION SUMMARY",
        "=" * 60,
        "",
        f"Input source: {metadata.get('input_source', '')}",
        f"Input type:   {metadata.get('input_type', '')}",
        f"Brain NIfTI:  {metadata.get('brain_nii', '')}",
        f"Fixed image:  {metadata.get('fixed_template', '')}",
        f"Pre-result:   {metadata.get('registered_brain_pre_laplacian', '')}",
        f"Result image: {metadata.get('registered_brain', '')}",
        f"Annotation:   {metadata.get('annotation_fixed_half', '')}",
        f"Backend:      {metadata.get('backend', '')}",
        f"Laplacian:    {'enabled' if metadata.get('laplacian_enabled') else 'disabled'}",
        f"Hemisphere:   {metadata.get('hemisphere', '')}",
        f"AP range:     {metadata.get('ap_range', [])}",
        "",
        "QUALITY METRICS",
        "-" * 40,
    ]
    lines.extend(f"{key:8s} {value:0.4f}" for key, value in metrics.items())
    pre_metrics = metadata.get("metrics_before_laplacian")
    if isinstance(pre_metrics, dict) and pre_metrics:
        lines.extend(
            [
                "",
                "PRE-LAPLACIAN METRICS",
                "-" * 40,
            ]
        )
        lines.extend(f"{key:8s} {float(value):0.4f}" for key, value in pre_metrics.items())
    staining = metadata.get("staining_stats")
    if isinstance(staining, dict) and staining:
        lines.extend(
            [
                "",
                "STAINING STATS",
                "-" * 40,
                f"AtlasCov  {float(staining.get('atlas_coverage', 0.0)):0.4f}",
                f"StainRate {float(staining.get('staining_rate', 0.0)):0.4f}",
                f"PosAtlas  {float(staining.get('positive_fraction_of_atlas', 0.0)):0.4f}",
                f"MeanSig   {float(staining.get('mean_signal', 0.0)):0.4f}",
                f"Thr       {float(staining.get('signal_threshold', 0.0)):0.4f}",
            ]
        )
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return meta_path, metrics_path, summary_path


def _norm01(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _cfg_float(value: object, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _cfg_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _cfg_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _target_label(target_um: float | None) -> str:
    return "native_xy" if target_um is None else f"{target_um:g}um"


def _resolve_target_um(cli_value: float | None, volume_prep: dict[str, object]) -> float | None:
    if cli_value is not None:
        return float(cli_value)
    if _cfg_bool(volume_prep.get("native_xy"), False):
        return None
    if "target_um" in volume_prep and volume_prep["target_um"] is None:
        return None
    return _cfg_float(volume_prep.get("target_um"), 25.0)


def _resolve_cfg_path(raw_value: object) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (PROJECT_DIR / path).resolve()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_config_3d_sample.json")
    ap.add_argument("--skip-to-transformix", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--input-path", default=None, help="Override config input with stack TIFF or z*.tif folder"
    )
    ap.add_argument(
        "--input-nii", default=None, help="Use an existing brain NIfTI instead of building one"
    )
    ap.add_argument("--pixel-size-um-xy", type=float, default=None, help="Override XY pixel size")
    ap.add_argument("--z-spacing-um", type=float, default=None, help="Override Z spacing")
    ap.add_argument(
        "--target-um", type=float, default=None, help="Target isotropic XY size when stacking TIFF"
    )
    ap.add_argument(
        "--backend", default=None, help="Registration backend override: elastix or ants"
    )
    ap.add_argument("--atlas-hemisphere", default=None, help="Override atlas hemisphere")
    ap.add_argument(
        "--pad-z", type=int, default=None, help="Zero-pad input volume along Z before saving NIfTI"
    )
    ap.add_argument(
        "--pad-y", type=int, default=None, help="Zero-pad input volume height before saving NIfTI"
    )
    ap.add_argument(
        "--pad-x", type=int, default=None, help="Zero-pad input volume width before saving NIfTI"
    )
    normalize_group = ap.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize-input",
        dest="normalize_input",
        action="store_true",
        help="Normalize intensities when writing the brain NIfTI",
    )
    normalize_group.add_argument(
        "--no-normalize-input",
        dest="normalize_input",
        action="store_false",
        help="Preserve raw intensities when writing the brain NIfTI",
    )
    ap.set_defaults(normalize_input=None)
    args = ap.parse_args()

    cfg = json.loads((PROJECT_DIR / args.config).read_text())
    reg = cfg["registration"]
    inp = cfg["input"]
    volume_prep = cfg.get("volume_prep", {})

    input_source = Path(args.input_path) if args.input_path else PROJECT_DIR / inp["slice_dir"]
    input_nii = Path(args.input_nii) if args.input_nii else None
    pixel_um_xy = (
        float(args.pixel_size_um_xy)
        if args.pixel_size_um_xy is not None
        else float(inp["pixel_size_um_xy"])
    )
    z_spacing = (
        float(args.z_spacing_um)
        if args.z_spacing_um is not None
        else float(inp["slice_spacing_um"])
    )
    hemisphere = (
        str(args.atlas_hemisphere)
        if args.atlas_hemisphere
        else str(reg.get("atlas_hemisphere", "right_flipped"))
    )
    backend = (
        str(args.backend).strip().lower()
        if args.backend is not None
        else str(reg.get("backend", "elastix")).strip().lower()
    )
    if backend not in {"elastix", "ants"}:
        raise ValueError(f"unsupported backend: {backend}")
    z_scale = float(reg.get("atlas_z_z_scale", 0.2))
    z_offset = int(reg.get("atlas_z_offset", 150))
    ants_cfg = reg.get("ants", {})
    lap_cfg = reg.get("laplacian", {})
    fixed_template_override = _resolve_cfg_path(reg.get("fixed_template_path"))
    fixed_annotation_override = _resolve_cfg_path(reg.get("fixed_annotation_path"))
    template_half_width = _cfg_int(reg.get("template_half_width"), 0)
    target_um = _resolve_target_um(args.target_um, volume_prep)
    pad_z = (
        max(0, int(args.pad_z))
        if args.pad_z is not None
        else max(0, _cfg_int(volume_prep.get("pad_z"), 0))
    )
    pad_y = (
        max(0, int(args.pad_y))
        if args.pad_y is not None
        else max(0, _cfg_int(volume_prep.get("pad_y"), 0))
    )
    pad_x = (
        max(0, int(args.pad_x))
        if args.pad_x is not None
        else max(0, _cfg_int(volume_prep.get("pad_x"), 0))
    )
    normalize_input = (
        bool(args.normalize_input)
        if args.normalize_input is not None
        else _cfg_bool(volume_prep.get("normalize"), True)
    )
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_DIR / "outputs" / "registration_3d"

    print("=" * 60)
    print("Brainfast 3D Registration")
    print("=" * 60)
    print(f"Backend: {backend}")

    # 1. Brain NIfTI
    source_info = None
    if input_nii is not None:
        brain_nii = input_nii
        brain_shape = nib.load(str(brain_nii)).shape
        print(f"\n[1/4] Using existing brain NIfTI: {brain_nii}  shape={brain_shape}")
    else:
        source_info = inspect_volume_source(input_source)
        brain_nii = out_dir / f"brain_{_target_label(target_um)}.nii.gz"
        if not brain_nii.exists():
            print("\n[1/4] Building brain NIfTI from input source...")
            print(
                f"  Source: {source_info.source_type}  shape={source_info.shape}  "
                f"axes={source_info.axes or '?'}  dtype={source_info.dtype}"
            )
            print(
                "  Prep: "
                f"target_xy={_target_label(target_um)}, "
                f"pad=(z={pad_z}, y={pad_y}, x={pad_x}), "
                f"normalize={normalize_input}"
            )
            _, brain_shape = volume_source_to_nifti(
                input_source,
                brain_nii,
                pixel_um_xy,
                z_spacing,
                target_um=target_um,
                pad_z=pad_z,
                pad_y=pad_y,
                pad_x=pad_x,
                normalize=normalize_input,
            )
        else:
            brain_shape = nib.load(str(brain_nii)).shape
            print(f"\n[1/4] Brain NIfTI exists: {brain_nii}  shape={brain_shape}")

    # Compute AP range from filenames
    ap_start, ap_end = _compute_ap_range_from_source(
        input_source if input_nii is None else None, z_scale, z_offset
    )
    print(f"  AP range in atlas: [{ap_start}, {ap_end}]  ({ap_end - ap_start} slices)")

    # 2. Cropped template + annotation
    if fixed_template_override and fixed_annotation_override:
        print(f"\n[2/4] Using fixed template override: {fixed_template_override}")
        print(f"      Using fixed annotation override: {fixed_annotation_override}")
        if not fixed_template_override.exists():
            raise FileNotFoundError(f"fixed template override not found: {fixed_template_override}")
        if not fixed_annotation_override.exists():
            raise FileNotFoundError(
                f"fixed annotation override not found: {fixed_annotation_override}"
            )
        tmpl_cropped = fixed_template_override
        ann_cropped = fixed_annotation_override
    else:
        print(f"\n[2/4] Cropping template + annotation to AP range + {hemisphere}...")
        tmpl_cropped, ann_cropped = prepare_cropped_template(
            TEMPLATE_25,
            ANNOTATION,
            hemisphere,
            ap_start,
            ap_end,
            out_dir,
            half_width=(template_half_width or None),
        )

    # 3. Registration to fixed/template space
    registered_brain = out_dir / "registered_brain.nii.gz"
    registered_brain_pre_laplacian = out_dir / "registered_brain_pre_laplacian.nii.gz"
    backend_details: dict[str, object] = {"backend": backend}
    metrics_before_laplacian: dict[str, float] | None = None
    if backend == "elastix":
        elastix_out = out_dir / "elastix"
        tp_file = elastix_out / "TransformParameters.1.txt"
        if not args.skip_to_transformix:
            print("\n[3/4] Running Elastix (Rigid + BSpline) ...")
            print("  This takes ~10-15 minutes...")
            run_elastix(tmpl_cropped, brain_nii, elastix_out)
        else:
            print(f"\n[3/4] Skipping Elastix (existing: {tp_file})")
            if not tp_file.exists():
                raise FileNotFoundError(f"No transform at: {tp_file}")
        elastix_result = elastix_out / "result.1.nii"
        if not elastix_result.exists():
            raise FileNotFoundError(f"registered output not found: {elastix_result}")
        nib.save(nib.load(str(elastix_result)), str(registered_brain))
        backend_details["elastix_dir"] = str(elastix_out.resolve())
        backend_details["transform_parameters"] = str(tp_file.resolve())
    else:
        ants_out = out_dir / "ants"
        transform_manifest = ants_out / "ants_transforms.txt"
        print("\n[3/4] Running ANTs SyN registration ...")
        print("  This may take several minutes...")
        if registered_brain.exists():
            print(f"  Registered brain already exists: {registered_brain}")
            if ants_out.exists():
                backend_details["ants_dir"] = str(ants_out.resolve())
            if transform_manifest.exists():
                backend_details["transform_manifest"] = str(transform_manifest.resolve())
        else:
            reg_iterations = tuple(int(v) for v in ants_cfg.get("reg_iterations", [40, 20, 0]))
            ants_result, ants_details = run_ants_registration(
                tmpl_cropped,
                brain_nii,
                ants_out,
                transform_type=str(ants_cfg.get("transform_type", "SyN")),
                random_seed=int(ants_cfg.get("random_seed", 42)),
                verbose=_cfg_bool(ants_cfg.get("verbose"), True),
                reg_iterations=reg_iterations,
            )
            shutil.copyfile(ants_result, registered_brain)
            backend_details.update(ants_details)
            backend_details["ants_dir"] = str(ants_out.resolve())

    laplacian_enabled = _cfg_bool(lap_cfg.get("enabled"), False)
    if laplacian_enabled:
        print("\n[3.5/4] Running Laplacian refinement ...")
        if registered_brain_pre_laplacian.exists():
            print(f"  Using existing pre-Laplacian result: {registered_brain_pre_laplacian}")
        else:
            shutil.copyfile(registered_brain, registered_brain_pre_laplacian)
        metrics_before_laplacian = compute_registration_metrics(
            tmpl_cropped, registered_brain_pre_laplacian
        )
        lap_result = refine_volume_with_laplacian(
            tmpl_cropped,
            registered_brain_pre_laplacian,
            out_dir / "laplacian",
            cfg=LaplacianConfig(
                axis=_cfg_int(lap_cfg.get("axis"), 0),
                rtol=_cfg_float(lap_cfg.get("rtol"), 1e-2) or 1e-2,
                maxiter=_cfg_int(lap_cfg.get("maxiter"), 1000),
                normal_radius=_cfg_int(lap_cfg.get("normal_radius"), 3),
                degree_thresh_deg=_cfg_float(lap_cfg.get("degree_thresh_deg"), 5.0) or 5.0,
                distance_neighbours=_cfg_int(lap_cfg.get("distance_neighbours"), 30),
                template_min_component=_cfg_int(lap_cfg.get("template_min_component"), 25),
                data_min_component=_cfg_int(lap_cfg.get("data_min_component"), 100),
                print_every=_cfg_int(lap_cfg.get("print_every"), 10),
                chunk_slices=_cfg_int(lap_cfg.get("chunk_slices"), 16),
            ),
        )
        shutil.copyfile(lap_result.final_registered_path, registered_brain)
        backend_details["laplacian"] = {
            "enabled": True,
            "axis": _cfg_int(lap_cfg.get("axis"), 0),
            "rtol": _cfg_float(lap_cfg.get("rtol"), 1e-2) or 1e-2,
            "maxiter": _cfg_int(lap_cfg.get("maxiter"), 1000),
            "out_dir": str((out_dir / "laplacian").resolve()),
            "final_registered": str(lap_result.final_registered_path.resolve()),
            "deformation_field": str(lap_result.deformation_field_path.resolve()),
            "boundary_conditions_csv": str(lap_result.boundary_csv_path.resolve()),
            "boundary_conditions_npy": str(lap_result.boundary_npy_path.resolve()),
            "fpoints": str(lap_result.fpoints_path.resolve()),
            "mpoints": str(lap_result.mpoints_path.resolve()),
            "correspondences": int(lap_result.correspondences),
            "unique_boundary_voxels": int(lap_result.unique_boundary_voxels),
            "solve_seconds": float(lap_result.solve_seconds),
            "residual_history_y": [float(v) for v in lap_result.residual_history_u],
            "residual_history_z": [float(v) for v in lap_result.residual_history_v],
        }

    # 4. Fixed-space annotation + overview
    print("\n[4/4] Writing fixed-space annotation and reports...")
    fixed_annotation = out_dir / "annotation_fixed_half.nii.gz"
    nib.save(nib.load(str(ann_cropped)), str(fixed_annotation))

    metrics = compute_registration_metrics(tmpl_cropped, registered_brain)
    staining_stats = compute_staining_stats(registered_brain, fixed_annotation)
    staining_path = out_dir / "staining_stats.json"
    staining_path.write_text(json.dumps(staining_stats, indent=2), encoding="utf-8")
    overview_path = out_dir / "overview.png"
    try:
        from scripts.make_registration_overview import make_registration_overview

        make_registration_overview(
            registered_brain,
            fixed_annotation,
            overview_path,
            slices=[200, 300],
            structure_csv=PROJECT_DIR / "configs" / "allen_mouse_structure_graph.csv",
        )
    except Exception as exc:
        print(f"[warn] overview generation failed: {exc}")
        overview_path = Path("")
    metadata = {
        "input_source": str((input_nii or input_source).resolve()),
        "input_type": "input_nii"
        if input_nii is not None
        else (source_info.source_type if source_info else "unknown"),
        "brain_nii": str(brain_nii.resolve()),
        "fixed_template": str(tmpl_cropped.resolve()),
        "fixed_template_override": str(fixed_template_override.resolve())
        if fixed_template_override
        else "",
        "fixed_annotation_override": str(fixed_annotation_override.resolve())
        if fixed_annotation_override
        else "",
        "registered_brain_pre_laplacian": str(registered_brain_pre_laplacian.resolve())
        if registered_brain_pre_laplacian.exists()
        else "",
        "registered_brain": str(registered_brain.resolve()),
        "annotation_fixed_half": str(fixed_annotation.resolve()),
        "overview_png": str(overview_path.resolve()) if str(overview_path) else "",
        "staining_stats_json": str(staining_path.resolve()),
        "backend": backend,
        "laplacian_enabled": bool(laplacian_enabled),
        "hemisphere": hemisphere,
        "ap_range": [int(ap_start), int(ap_end)],
        "brain_shape": [int(v) for v in brain_shape],
        "pixel_size_um_xy": float(pixel_um_xy),
        "z_spacing_um": float(z_spacing),
        "target_um": float(target_um) if target_um is not None else None,
        "template_half_width": int(template_half_width) if template_half_width else None,
        "pad_z": int(pad_z),
        "pad_y": int(pad_y),
        "pad_x": int(pad_x),
        "normalize_input": bool(normalize_input),
        "config_path": str((PROJECT_DIR / args.config).resolve()),
        "metrics_before_laplacian": metrics_before_laplacian or {},
        "staining_stats": staining_stats,
        "backend_details": backend_details,
    }
    meta_path, metrics_path, summary_path = write_registration_reports(
        out_dir,
        metadata=metadata,
        metrics=metrics,
    )
    report_path = Path("")
    index_path = Path("")
    try:
        report_path = build_run_report(out_dir)
        index_path = build_outputs_index(PROJECT_DIR / "outputs")
    except Exception as exc:
        print(f"[warn] html report generation failed: {exc}")
    print(f"\n{'=' * 60}")
    print(f"DONE! Registered brain -> {registered_brain}")
    print(f"Fixed annotation -> {fixed_annotation}")
    if str(overview_path):
        print(f"Overview -> {overview_path}")
    if str(report_path):
        print(f"Report -> {report_path}")
    if str(index_path):
        print(f"Output Index -> {index_path}")
    print(f"Metrics -> {metrics_path}")
    print(f"Metadata -> {meta_path}")
    print(f"Summary -> {summary_path}")
    print("Run test_single_slice.py to view overlay.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
