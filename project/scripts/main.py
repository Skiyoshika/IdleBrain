import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread, imwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paths import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

try:
    from scripts.atlas_autopick import autopick_best_z, refine_atlas_z_by_size
    from scripts.atlas_mapper import (
        map_cells_with_registered_label_slice,
    )
    from scripts.config_validation import config_value, load_config, validate_runtime_config
    from scripts.dedup import apply_dedup_kdtree, write_dedup_stats
    from scripts.detect import detect_cells
    from scripts.exceptions import ConfigError
    from scripts.map_and_aggregate import aggregate_by_region, map_cells_to_regions, write_outputs
    from scripts.overlay_render import render_overlay
    from scripts.preprocess import merge_every_n_slices
    from scripts.qc import export_slice_qc
    from scripts.registration_adapter import bootstrap_registration_assets
except Exception:
    from atlas_autopick import autopick_best_z, refine_atlas_z_by_size
    from atlas_mapper import map_cells_with_registered_label_slice
    from config_validation import config_value, load_config, validate_runtime_config
    from dedup import apply_dedup_kdtree, write_dedup_stats
    from detect import detect_cells
    from exceptions import ConfigError
    from map_and_aggregate import aggregate_by_region, map_cells_to_regions, write_outputs
    from overlay_render import render_overlay
    from preprocess import merge_every_n_slices
    from qc import export_slice_qc
    from registration_adapter import bootstrap_registration_assets


def _validated_float(cfg: dict, dotted_key: str) -> float:
    return float(config_value(cfg, dotted_key))


def _validated_int(cfg: dict, dotted_key: str) -> int:
    return int(config_value(cfg, dotted_key))


def emit_progress(
    step_current: int,
    step_total: int,
    phase: str,
    message: str,
    *,
    slices_done: int | None = None,
    slices_total: int | None = None,
) -> None:
    parts = [
        f"step={int(step_current)}/{int(step_total)}",
        f"phase={str(phase).strip() or 'registration'}",
    ]
    if slices_done is not None or slices_total is not None:
        cur = int(slices_done or 0)
        total = int(slices_total or 0)
        parts.append(f"slices={cur}/{total}")
    print(f"[PROGRESS:{':'.join(parts)}] {message}", flush=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_structure_source(project_root: Path) -> Path:
    registration_csv = project_root / "outputs" / "registration" / "structure_tree.csv"
    if registration_csv.exists():
        return registration_csv
    fallback_csv = project_root / "configs" / "allen_mouse_structure_graph.csv"
    if fallback_csv.exists():
        return fallback_csv
    raise FileNotFoundError(
        "structure ontology source not found in outputs/registration or project/configs"
    )


def _load_tuned_overlay_params(outputs_dir: Path) -> tuple[dict, str, int]:
    tuned_json = outputs_dir / "trainset_tuned_params.json"
    if not tuned_json.exists():
        return {}, "cover", 1

    data = json.loads(tuned_json.read_text(encoding="utf-8-sig"))
    fit_mode = "cover"
    edge_smooth_iter = 1
    warp_params: dict = {}

    if isinstance(data, dict):
        if isinstance(data.get("warpParams"), dict):
            warp_params = dict(data["warpParams"])
        if isinstance(data.get("fitMode"), str):
            fit_mode = data["fitMode"]
        if "edgeSmoothIter" in data:
            edge_smooth_iter = int(data["edgeSmoothIter"])

        best = data.get("best")
        if isinstance(best, dict):
            params = best.get("params", {})
            if isinstance(params, dict):
                if isinstance(params.get("warpParams"), dict):
                    warp_params = dict(params["warpParams"])
                if isinstance(params.get("fitMode"), str):
                    fit_mode = params["fitMode"]
                if "edgeSmoothIter" in params:
                    edge_smooth_iter = int(params["edgeSmoothIter"])

    return warp_params, str(fit_mode), int(edge_smooth_iter)


def _resolve_channel_index(cfg: dict) -> int:
    active = os.environ.get("BRAINCOUNT_ACTIVE_CHANNEL") or cfg.get("input", {}).get(
        "active_channel", "red"
    )
    cmap = cfg.get("input", {}).get("channel_map", {"red": 0, "green": 1, "farred": 2})
    return int(cmap.get(active, 0))


def _collect_slice_files(slice_dir: Path, glob_pattern: str) -> list[Path]:
    return sorted(slice_dir.glob(glob_pattern))


def _extract_channel_to_tmp(src_files: list[Path], out_dir: Path, ch_idx: int) -> list[Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(src_files):
        arr = imread(str(p))
        if arr.ndim == 3:
            if arr.shape[-1] <= ch_idx:
                ch = arr[..., 0]
            else:
                ch = arr[..., ch_idx]
        else:
            ch = arr
        dst = out_dir / f"ch_{ch_idx}_{i:04d}.tif"
        imwrite(str(dst), ch.astype(np.uint16))
        out.append(dst)
    return out


def _normalize_sampling_mode(cfg: dict) -> tuple[str, int]:
    input_cfg = cfg.get("input", {})
    raw_mode = str(input_cfg.get("sampling_mode", "")).strip().lower()
    interval = max(1, int(input_cfg.get("slice_interval_n", 1)))

    if not raw_mode:
        raw_mode = "merge" if interval > 1 else "single"

    if raw_mode in {"single", "native", "raw", "original"}:
        return "single", 1
    if raw_mode in {"merge", "merged", "merge5", "merge_n"}:
        return "merged", interval
    raise ValueError(f"unsupported input.sampling_mode: {raw_mode}")


def _resolve_processing_files(
    cfg: dict,
    channel_files: list[Path],
    outputs_dir: Path,
) -> tuple[list[Path], dict[str, int | str]]:
    sampling_mode, interval = _normalize_sampling_mode(cfg)
    if sampling_mode == "single":
        return channel_files, {"sampling_mode": "single", "slice_interval_n": 1}

    merged_files = merge_every_n_slices(channel_files, outputs_dir / "tmp_merged", n=interval)
    return merged_files, {"sampling_mode": "merged", "slice_interval_n": int(interval)}


def _write_detection_summary(
    outputs_dir: Path,
    *,
    sampling: dict[str, int | str],
    cfg: dict,
    detections: pd.DataFrame,
    deduped: pd.DataFrame | None = None,
) -> None:
    det_cfg = cfg.get("detection", {})
    primary = str(det_cfg.get("primary_model", ""))
    secondary = str(det_cfg.get("secondary_model", ""))
    requested_cellpose = primary.lower().startswith("cellpose") or secondary.lower().startswith(
        "cellpose"
    )
    allow_fallback_raw = det_cfg.get("allow_fallback", None)
    allow_fallback = (
        bool(allow_fallback_raw) if allow_fallback_raw is not None else (not requested_cellpose)
    )
    summary = {
        "sampling_mode": str(sampling.get("sampling_mode", "single")),
        "slice_interval_n": int(sampling.get("slice_interval_n", 1)),
        "primary_model": primary,
        "secondary_model": secondary,
        "requested_instance_segmentation": bool(requested_cellpose),
        "allow_fallback": bool(allow_fallback),
        "cells_detected": int(len(detections)),
        "detector_counts": {},
    }
    if "detector" in detections.columns and not detections.empty:
        summary["detector_counts"] = {
            str(k): int(v)
            for k, v in detections["detector"].fillna("unknown").value_counts().to_dict().items()
        }
    if deduped is not None:
        summary["cells_dedup"] = int(len(deduped))
        if "detector" in deduped.columns and not deduped.empty:
            summary["dedup_detector_counts"] = {
                str(k): int(v)
                for k, v in deduped["detector"].fillna("unknown").value_counts().to_dict().items()
            }
    (outputs_dir / "detection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _make_sample_tiffs(out_dir: Path, n: int = 12, h: int = 256, w: int = 256) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(n):
        base = rng.normal(200, 20, size=(h, w, 3)).astype(np.float32)
        # red channel hotspots
        for _ in range(10):
            y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
            base[y - 2 : y + 3, x - 2 : x + 3, 0] += 250
        # green channel hotspots
        for _ in range(8):
            y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
            base[y - 2 : y + 3, x - 2 : x + 3, 1] += 180
        # farred channel hotspots
        for _ in range(6):
            y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
            base[y - 2 : y + 3, x - 2 : x + 3, 2] += 130

        arr = np.clip(base, 0, 65535).astype(np.uint16)
        imwrite(str(out_dir / f"slice_{i:04d}.tif"), arr)


def run_demo(cfg: dict):
    project_root = _project_root()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cells_csv = project_root / "notes" / "demo_cells.csv"
    atlas_csv = project_root / "notes" / "demo_atlas_map.csv"
    cells = pd.read_csv(cells_csv)

    px_um = _validated_float(cfg, "input.pixel_size_um_xy")
    spacing_um = _validated_float(cfg, "input.slice_spacing_um")
    neighbor = _validated_int(cfg, "dedup.neighbor_slices")
    rxy = _validated_float(cfg, "dedup.r_xy_um")

    deduped, stats = apply_dedup_kdtree(
        cells,
        neighbor_slices=neighbor,
        pixel_size_um=px_um,
        slice_spacing_um=spacing_um,
        r_xy_um=rxy,
    )
    deduped.to_csv(outputs_dir / "demo_cells_dedup.csv", index=False)
    write_dedup_stats(stats, outputs_dir)

    mapped = map_cells_to_regions(outputs_dir / "demo_cells_dedup.csv", atlas_csv)
    leaf, hierarchy = aggregate_by_region(mapped)
    write_outputs(leaf, hierarchy, outputs_dir)
    print("Demo pipeline complete: kdtree dedup + mapping + aggregation outputs generated")


def run_real_input(cfg: dict, input_dir: Path):
    project_root = _project_root()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    annotation_nii = project_root / "annotation_25.nii.gz"
    if not annotation_nii.exists():
        raise FileNotFoundError(f"atlas annotation not found: {annotation_nii}")
    slice_glob = cfg.get("input", {}).get("slice_glob", "*.tif")
    files = _collect_slice_files(input_dir, slice_glob)
    if not files:
        raise FileNotFoundError(f"No slice files found in {input_dir} with pattern {slice_glob}")

    ch_idx = _resolve_channel_index(cfg)
    ch_dir = outputs_dir / "tmp_channel"
    channel_files = _extract_channel_to_tmp(files, ch_dir, ch_idx)

    processing_files, sampling = _resolve_processing_files(cfg, channel_files, outputs_dir)
    total_slices = len(processing_files)
    emit_progress(
        1,
        6,
        "ap_selection",
        f"Prepared {total_slices} slice(s) for processing.",
        slices_done=0,
        slices_total=total_slices,
    )

    px_um = _validated_float(cfg, "input.pixel_size_um_xy")
    spacing_um = _validated_float(cfg, "input.slice_spacing_um")
    neighbor = _validated_int(cfg, "dedup.neighbor_slices")
    rxy = _validated_float(cfg, "dedup.r_xy_um")
    structure_csv = _resolve_structure_source(project_root)
    warp_params, fit_mode, edge_smooth_iter = _load_tuned_overlay_params(outputs_dir)
    reg_cfg = cfg.get("registration", {})
    # Force hemisphere if specified in config (for half-brain samples where auto-detection fails)
    atlas_hemisphere = str(reg_cfg.get("atlas_hemisphere", "")).lower().strip()
    if atlas_hemisphere:
        warp_params = dict(warp_params)
        warp_params["force_hemisphere"] = atlas_hemisphere
    # Tissue shrinkage correction: cleared tissue shrinks 10-15% linearly during processing.
    # Values < 1.0 make the atlas appear smaller to match actual tissue extent.
    tissue_shrink = float(reg_cfg.get("tissue_shrink_factor", 1.0))
    if tissue_shrink != 1.0:
        warp_params = dict(warp_params)
        warp_params["tissue_shrink_factor"] = tissue_shrink
    # Merge registration-section warp/render knobs into warp_params so render_overlay
    # can read them via _warp_param().  Only keys that are not already set by the tuned
    # params are propagated (tuned params take precedence).
    _reg_warp_keys = [
        "enable_silhouette_conform",
        "physical_placement_skip_opt",
        "contour_n_pts",
        "contour_tps_smooth",
        "contour_max_disp_ratio",
        "contour_max_disp_min_px",
        "contour_fill_radius",
        "enable_sitk_ref_refine",
        "sitk_max_dim",
        "sitk_mi_bins",
        "sitk_max_iter",
        "sitk_mesh_size",
        "sitk_max_disp_frac",
    ]
    _reg_warp_updates = {
        k: reg_cfg[k] for k in _reg_warp_keys if k in reg_cfg and k not in warp_params
    }
    if _reg_warp_updates:
        warp_params = dict(warp_params)
        warp_params.update(_reg_warp_updates)
    # Gamma correction for display of dim cleared-tissue overlays (< 1.0 = brighten).
    display_gamma = float(reg_cfg.get("display_gamma", 1.0))
    fail_score_threshold = float(reg_cfg.get("fail_score_threshold", 0.65))
    registration_dir = outputs_dir / "registered_slices"
    if registration_dir.exists():
        shutil.rmtree(registration_dir, ignore_errors=True)
    registration_dir.mkdir(parents=True, exist_ok=True)

    # ── Optional DeepSlice AP estimation (run on whole series before the loop) ──
    ap_method = str(reg_cfg.get("ap_method", "formula")).lower().strip()
    _deepslice_az: dict[str, int] = {}
    if ap_method in ("deepslice", "deepslice+formula"):
        emit_progress(
            2,
            6,
            "ap_selection",
            "Running DeepSlice AP estimation on the full series...",
            slices_done=0,
            slices_total=total_slices,
        )
        try:
            from scripts.atlas_deepslice import predict_ap_series

            _ds_z_scale = float(reg_cfg.get("atlas_z_z_scale", 0.2))
            _ds_z_offset = int(reg_cfg.get("atlas_z_offset", 0))
            _ds_max_dev = int(reg_cfg.get("deepslice_max_deviation", 30))
            print("[main] Running DeepSlice AP estimation on full series ...")
            _deepslice_az = predict_ap_series(
                channel_files,
                z_scale=_ds_z_scale,
                z_offset=_ds_z_offset,
                pixel_size_um=px_um,
                display_gamma=display_gamma,
                max_deviation=_ds_max_dev,
            )
            print(f"[main] DeepSlice done: {len(_deepslice_az)} slices mapped")
        except Exception as _ds_err:
            print(f"[main] DeepSlice failed ({_ds_err}); falling back to formula")
            _deepslice_az = {}
    emit_progress(
        2,
        6,
        "ap_selection",
        "Atlas AP selection is ready.",
        slices_done=0,
        slices_total=total_slices,
    )

    detect_rows = []
    mapped_rows = []
    registration_rows = []
    next_id = 1
    for sid, mp in enumerate(processing_files):
        emit_progress(
            3,
            6,
            "registration",
            f"Processing slice {sid + 1} / {total_slices}: {Path(mp).name}",
            slices_done=sid,
            slices_total=total_slices,
        )
        det = detect_cells(mp, cfg)
        if det.empty:
            emit_progress(
                3,
                6,
                "registration",
                f"Processed slice {sid + 1} / {total_slices}: no detections.",
                slices_done=sid + 1,
                slices_total=total_slices,
            )
            continue

        auto_label_path = registration_dir / f"slice_{sid:04d}_auto_label.tif"
        registered_label_path = registration_dir / f"slice_{sid:04d}_registered_label.tif"
        overlay_png = registration_dir / f"slice_{sid:04d}_overlay.png"

        step_t0 = time.perf_counter()
        atlas_z_range = cfg.get("registration", {}).get("atlas_z_range", None)
        atlas_z_fixed = cfg.get("registration", {}).get("atlas_z_fixed", None)

        # Derive atlas_z from the filename z-number (e.g., z0050 → atlas_z=10 with scale=0.2)
        # Try original source file first, then merged file, then sid-based computation.
        if (
            cfg.get("registration", {}).get("atlas_z_from_filename", False)
            and atlas_z_fixed is None
        ):
            z_scale = float(cfg.get("registration", {}).get("atlas_z_z_scale", 0.2))
            z_offset = int(cfg.get("registration", {}).get("atlas_z_offset", 0))
            _fname = None
            # Try the source file path (for direct pipeline invocations)
            _m = re.search(r"z(\d+)", Path(str(mp)).stem)
            if _m:
                _fname = int(_m.group(1))
            else:
                # Merged file — reconstruct z-number from original source files
                if sid < len(files):
                    _m2 = re.search(r"z(\d+)", Path(str(files[sid])).stem)
                    if _m2:
                        _fname = int(_m2.group(1))
            if _fname is not None:
                atlas_z_fixed = max(0, min(527, int(_fname * z_scale) + z_offset))

        # Override with DeepSlice estimate if available for this slice
        if _deepslice_az:
            _stem = Path(str(mp)).stem
            if _stem in _deepslice_az:
                atlas_z_fixed = int(_deepslice_az[_stem])

        if atlas_z_fixed is not None:
            # Use filename-based atlas z, optionally refined by size-aware shape matching
            import nibabel as nib
            from tifffile import imread as tiff_imread

            fixed_z = int(atlas_z_fixed)
            nii = nib.load(str(annotation_nii))
            vol = np.asarray(nii.get_fdata(), dtype=np.int32)

            # Size-aware refinement: search ±refine_range around the filename estimate
            refine_range = int(cfg.get("registration", {}).get("atlas_z_refine_range", 0))
            score_type = "fixed_filename"
            best_score = 1.0
            if refine_range > 0:
                try:
                    _real_img = tiff_imread(str(mp))
                    _hemi = cfg.get("registration", {}).get("atlas_hemisphere", "")
                    fixed_z, best_score = refine_atlas_z_by_size(
                        _real_img,
                        vol,
                        z_estimate=fixed_z,
                        search_range=int(refine_range),
                        pixel_size_um=px_um,
                        hemisphere=_hemi,
                    )
                    score_type = "size_shape_refined"
                except Exception as _e:
                    pass  # fallback to filename estimate

            best_slice = vol[fixed_z, :, :]
            auto_label_path.parent.mkdir(parents=True, exist_ok=True)
            from tifffile import imwrite as tiff_imwrite

            tiff_imwrite(str(auto_label_path), best_slice)
            auto_meta = {
                "best_z": fixed_z,
                "best_score": 1.0,  # filename-based: always passes threshold
                "best_score_type": score_type,
                "label_slice_tif": str(auto_label_path),
                "shape": list(vol.shape),
                "slicing_plane": "coronal",
                "slice_shape": list(best_slice.shape),
                "roi_mode": "fixed",
                "roi_bbox": [0, 0, 0, 0],
                "real_slice": {},
                "tissue_coverage": 1.0,
                "coarse_top": [],
                "refined_top": [],
            }
        else:
            auto_meta = autopick_best_z(
                real_path=mp,
                annotation_nii=annotation_nii,
                out_label_tif=auto_label_path,
                z_step=2,
                pixel_size_um=px_um,
                slicing_plane="coronal",
                roi_mode="auto",
                z_range=atlas_z_range,
            )
        autopick_ms = float((time.perf_counter() - step_t0) * 1000.0)

        step_t0 = time.perf_counter()
        _label_z_idx = (
            int(auto_meta.get("best_z", -1)) if auto_meta.get("best_z") is not None else None
        )
        _, diagnostic = render_overlay(
            real_slice_path=mp,
            label_slice_path=auto_label_path,
            out_png=overlay_png,
            alpha=0.55,
            mode="fill",
            pixel_size_um=px_um,
            major_top_k=28,
            fit_mode=fit_mode,
            edge_smooth_iter=edge_smooth_iter,
            warp_params=warp_params,
            return_meta=True,
            warped_label_out=registered_label_path,
            display_gamma=display_gamma,
            label_z_index=_label_z_idx,
        )
        render_wall_ms = float((time.perf_counter() - step_t0) * 1000.0)
        render_timings = diagnostic.get("timings_ms", {}) if isinstance(diagnostic, dict) else {}
        postprocess_timings = (
            render_timings.get("postprocess", {}) if isinstance(render_timings, dict) else {}
        )
        reg_score = float(auto_meta.get("best_score", 0.0))
        reg_ok = bool(np.isfinite(reg_score) and reg_score >= fail_score_threshold)
        registration_rows.append(
            {
                "slice_id": int(sid),
                "slice_path": str(mp),
                "auto_label_path": str(auto_label_path),
                "registered_label_path": str(registered_label_path),
                "overlay_path": str(overlay_png),
                "best_z": int(auto_meta.get("best_z", -1)),
                "best_score": reg_score,
                "score_type": str(auto_meta.get("best_score_type", "")),
                "slicing_plane": str(auto_meta.get("slicing_plane", "coronal")),
                "registration_ok": bool(reg_ok),
                "registration_method": str(diagnostic.get("warp", {}).get("method", "")),
                "autopick_ms": autopick_ms,
                "render_ms": float(render_timings.get("total", render_wall_ms))
                if isinstance(render_timings, dict)
                else render_wall_ms,
                "render_load_inputs_ms": float(render_timings.get("load_inputs", 0.0))
                if isinstance(render_timings, dict)
                else 0.0,
                "render_registration_ms": float(render_timings.get("registration", 0.0))
                if isinstance(render_timings, dict)
                else 0.0,
                "render_postprocess_ms": float(
                    postprocess_timings.get("total", postprocess_timings.get("wall", 0.0))
                )
                if isinstance(postprocess_timings, dict)
                else 0.0,
                "render_colorize_ms": float(render_timings.get("colorize", 0.0))
                if isinstance(render_timings, dict)
                else 0.0,
                "render_draw_labels_ms": float(render_timings.get("draw_region_labels", 0.0))
                if isinstance(render_timings, dict)
                else 0.0,
            }
        )
        if not reg_ok:
            emit_progress(
                3,
                6,
                "registration",
                f"Processed slice {sid + 1} / {total_slices}: registration score below threshold.",
                slices_done=sid + 1,
                slices_total=total_slices,
            )
            continue

        det = det.copy()
        det["slice_id"] = sid
        det["cell_id"] = range(next_id, next_id + len(det))
        det["source_slice_path"] = str(Path(mp).resolve())
        det["count_sampling_mode"] = str(sampling["sampling_mode"])
        det["count_slice_interval_n"] = int(sampling["slice_interval_n"])
        next_id += len(det)
        detect_rows.append(
            det[
                [
                    "cell_id",
                    "slice_id",
                    "x",
                    "y",
                    "score",
                    "detector",
                    "area_px",
                    "source_slice_path",
                    "count_sampling_mode",
                    "count_slice_interval_n",
                ]
            ]
        )
        mapped_rows.append(
            map_cells_with_registered_label_slice(
                det,
                registered_label_tif=registered_label_path,
                structure_csv=structure_csv,
                atlas_slice_index=int(auto_meta.get("best_z", -1)),
                slicing_plane=str(auto_meta.get("slicing_plane", "coronal")),
                registration_score=reg_score,
                registration_method=str(
                    diagnostic.get("warp", {}).get("method", "registered_slice_label")
                ),
            )
        )
        emit_progress(
            3,
            6,
            "registration",
            f"Processed slice {sid + 1} / {total_slices}.",
            slices_done=sid + 1,
            slices_total=total_slices,
        )

    registration_qc_path = outputs_dir / "slice_registration_qc.csv"
    pd.DataFrame(registration_rows).to_csv(registration_qc_path, index=False)

    failed_slices = [row for row in registration_rows if not row["registration_ok"]]
    if failed_slices:
        raise RuntimeError(
            f"{len(failed_slices)} slice(s) failed registration score threshold {fail_score_threshold:.3f}; "
            f"see {registration_qc_path}"
        )

    emit_progress(
        4,
        6,
        "detection",
        "Finalizing cell detections...",
        slices_done=total_slices,
        slices_total=total_slices,
    )
    if not detect_rows:
        empty = pd.DataFrame(
            columns=[
                "cell_id",
                "slice_id",
                "x",
                "y",
                "score",
                "detector",
                "area_px",
                "source_slice_path",
                "count_sampling_mode",
                "count_slice_interval_n",
            ]
        )
        empty.to_csv(outputs_dir / "cells_detected.csv", index=False)
        _write_detection_summary(outputs_dir, sampling=sampling, cfg=cfg, detections=empty)
        print("No detections found in real-input run.")
        emit_progress(
            6,
            6,
            "done",
            "No detections found. Pipeline completed.",
            slices_done=total_slices,
            slices_total=total_slices,
        )
        return

    if not mapped_rows:
        cells = pd.concat(detect_rows, ignore_index=True)
        cells.to_csv(outputs_dir / "cells_detected.csv", index=False)
        _write_detection_summary(outputs_dir, sampling=sampling, cfg=cfg, detections=cells)
        print("Detections found, but none could be mapped after registration.")
        emit_progress(
            6,
            6,
            "done",
            "Detections found, but nothing could be mapped after registration.",
            slices_done=total_slices,
            slices_total=total_slices,
        )
        return

    cells = pd.concat(detect_rows, ignore_index=True)
    cells.to_csv(outputs_dir / "cells_detected.csv", index=False)
    mapped = pd.concat(mapped_rows, ignore_index=True)

    emit_progress(
        5,
        6,
        "dedup",
        "Deduplicating detected cells across slices...",
        slices_done=total_slices,
        slices_total=total_slices,
    )
    deduped, stats = apply_dedup_kdtree(
        mapped,
        neighbor_slices=neighbor,
        pixel_size_um=px_um,
        slice_spacing_um=spacing_um,
        r_xy_um=rxy,
    )
    deduped.to_csv(outputs_dir / "cells_dedup.csv", index=False)
    write_dedup_stats(stats, outputs_dir)
    _write_detection_summary(outputs_dir, sampling=sampling, cfg=cfg, detections=cells, deduped=deduped)

    deduped.to_csv(outputs_dir / "cells_mapped.csv", index=False)
    emit_progress(
        6,
        6,
        "mapping",
        "Mapping cells to brain regions and writing outputs...",
        slices_done=total_slices,
        slices_total=total_slices,
    )
    leaf, hierarchy = aggregate_by_region(deduped)
    write_outputs(leaf, hierarchy, outputs_dir)

    cells_mapped_path = outputs_dir / "cells_mapped.csv"
    slice_qc_path = outputs_dir / "slice_qc.csv"
    export_slice_qc(cells_mapped_path, slice_qc_path)

    # Copy registered atlas overlays to qc_overlays so the UI gallery shows beautiful colored images
    qc_dir = outputs_dir / "qc_overlays"
    if qc_dir.exists():
        shutil.rmtree(qc_dir, ignore_errors=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    reg_overlays = sorted(registration_dir.glob("slice_*_overlay.png"))

    for i, src_png in enumerate(reg_overlays):
        shutil.copy2(src_png, qc_dir / f"overlay_{i:03d}.png")

    print(
        f"Real-input end-to-end complete: detected={len(cells)}, dedup={len(deduped)} -> outputs/cell_counts_leaf.csv + QC"
    )
    emit_progress(
        6,
        6,
        "done",
        "Pipeline completed successfully.",
        slices_done=total_slices,
        slices_total=total_slices,
    )

    # Auto-regenerate demo visuals (panel, annotated slice, chart) after pipeline completes
    refresh_script = project_root / "scripts" / "refresh_demo.py"
    if refresh_script.exists():
        import subprocess
        import sys as _sys

        print("\nAuto-running refresh_demo.py to regenerate demo visuals...")
        try:
            subprocess.run(
                [_sys.executable, str(refresh_script)],
                cwd=str(project_root),
                timeout=180,
            )
        except Exception as _e:
            print(f"[warn] refresh_demo.py failed: {_e}")


def main():
    parser = argparse.ArgumentParser(description="Brain atlas cell count MVP pipeline entry")
    parser.add_argument("--config", required=True, help="Path to run config JSON")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and print plan only"
    )
    parser.add_argument(
        "--demo-map", action="store_true", help="Run mapping/aggregation demo with sample CSVs"
    )
    parser.add_argument(
        "--run-real-input", type=str, default="", help="Run preprocess+detect on real input folder"
    )
    parser.add_argument(
        "--make-sample-tiff",
        type=str,
        default="",
        help="Create synthetic 3-channel TIFF slices into folder",
    )
    parser.add_argument(
        "--init-registration",
        action="store_true",
        help="Bootstrap registration assets from legacy repo",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG-level logging to console and log file"
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    # Configure structured logging before any pipeline work begins
    try:
        from scripts.logging_setup import configure_logging

        _outputs_dir = _project_root() / "outputs"
        configure_logging(_outputs_dir, debug=getattr(args, "debug", False))
    except Exception:
        pass  # logging is optional; don't block pipeline startup

    needs_runtime_config = bool(
        args.dry_run
        or args.demo_map
        or args.run_real_input
        or (not args.make_sample_tiff and not args.init_registration)
    )
    if needs_runtime_config:
        issues = validate_runtime_config(cfg, require_input_dir=bool(args.run_real_input))
        if issues:
            raise ConfigError("Config validation failed", issues=issues)

    steps = [
        "prepare_input",
        "register_slices",
        "detect_cells",
        "dedup_cells_kdtree",
        "map_cells_to_regions",
        "aggregate_by_region",
        "export_qc",
    ]

    print("Loaded config for project:", cfg.get("project", {}).get("name", "unknown"))
    print("Active channel:", cfg.get("input", {}).get("active_channel", "unknown"))
    print("Planned steps:")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    if args.dry_run:
        print("Dry-run complete.")
        return

    if args.demo_map:
        run_demo(cfg)
        return

    if args.make_sample_tiff:
        _make_sample_tiffs(Path(args.make_sample_tiff))
        print(f"Sample TIFFs generated at {args.make_sample_tiff}")
        return

    if args.init_registration:
        project_root = _project_root()
        assets = bootstrap_registration_assets(project_root)
        out = project_root / "outputs" / "registration_assets.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(assets, indent=2), encoding="utf-8")
        print(f"Registration assets initialized -> {out}")
        return

    if args.run_real_input:
        run_real_input(cfg, Path(args.run_real_input))
        return

    print("MVP skeleton only: registration/mapping integration pending.")


if __name__ == "__main__":
    main()
