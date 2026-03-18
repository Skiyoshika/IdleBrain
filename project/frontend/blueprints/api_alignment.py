"""api_alignment.py — Landmark alignment, nonlinear alignment, slice info/extract routes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, request, send_from_directory
from tifffile import imread, imwrite

import project.frontend.server_context as ctx
from project.frontend.services.alignment_service import (
    apply_affine_alignment,
    apply_nonlinear_alignment,
    propose_landmarks,
    render_landmark_preview,
)

bp = Blueprint("api_alignment", __name__, url_prefix="/api")


@bp.post("/align/nonlinear")
def align_nonlinear():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    real_path = Path(payload.get("realPath", ""))
    atlas_label_path = Path(payload.get("atlasLabelPath", ""))
    pairs_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    if not real_path.exists() or not atlas_label_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas/pairs file"}), 400

    out_label = ctx._job_file(job_id, "aligned_label_nonlinear.tif")
    compare_png = ctx._job_file(job_id, "overlay_compare_nonlinear.png")
    try:
        result = apply_nonlinear_alignment(
            real_path, atlas_label_path, pairs_csv, out_label, compare_png
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "jobId": job_id, **result})


@bp.get("/outputs/overlay-compare-nonlinear")
def outputs_overlay_compare_nonlinear():
    fp = ctx._job_file(ctx._query_job_id(), "overlay_compare_nonlinear.png")
    if not fp.exists():
        return jsonify({"ok": False, "error": "nonlinear overlay compare not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/outputs/auto-label-slice")
def outputs_auto_label_slice():
    fp = ctx._job_file(ctx._query_job_id(), "auto_label_slice.tif")
    if not fp.exists():
        return jsonify({"ok": False, "error": "auto label slice not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/outputs/landmark-preview")
def outputs_landmark_preview():
    fp = ctx._job_file(ctx._query_job_id(), "landmark_preview.png")
    if not fp.exists():
        return jsonify({"ok": False, "error": "landmark preview not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.post("/align/landmark-preview")
def align_landmark_preview():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    real_path = Path(payload.get("realPath", ""))
    atlas_path = Path(payload.get("atlasPath", ""))
    pairs_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    if not real_path.exists() or not atlas_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas or pairs file"}), 400

    fp = ctx._job_file(job_id, "landmark_preview.png")
    try:
        n_points = render_landmark_preview(real_path, atlas_path, pairs_csv, fp)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "preview": str(fp), "points": n_points, "jobId": job_id})


@bp.post("/align/landmarks")
def align_landmarks():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    real_path = Path(payload.get("realPath", ""))
    atlas_path = Path(payload.get("atlasPath", ""))
    if not real_path.exists() or not atlas_path.exists():
        return jsonify({"ok": False, "error": "real or atlas path not found"}), 400

    out_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    try:
        res = propose_landmarks(
            real_path,
            atlas_path,
            out_csv,
            max_points=int(payload.get("maxPoints", 30)),
            min_distance=int(payload.get("minDistance", 12)),
            ransac_residual=float(payload.get("ransacResidual", 8.0)),
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "jobId": job_id, **res})


@bp.post("/align/apply")
def align_apply():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    real_path = Path(payload.get("realPath", ""))
    atlas_label_path = Path(payload.get("atlasLabelPath", ""))
    pairs_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    if not real_path.exists() or not atlas_label_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas/pairs file"}), 400

    out_label = ctx._job_file(job_id, "aligned_label_ai.tif")
    compare_png = ctx._job_file(job_id, "overlay_compare.png")
    try:
        result = apply_affine_alignment(
            real_path, atlas_label_path, pairs_csv, out_label, compare_png
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "jobId": job_id, **result})


@bp.post("/align/add-manual-landmarks")
def add_manual_landmarks():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    pairs = payload.get("pairs", [])
    if not pairs:
        return jsonify({"ok": False, "error": "no pairs provided"}), 400
    pairs_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    new_rows = pd.DataFrame(pairs)
    if pairs_csv.exists():
        try:
            existing = pd.read_csv(pairs_csv)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        except Exception:
            combined = new_rows
    else:
        combined = new_rows
    combined.to_csv(pairs_csv, index=False)
    return jsonify({"ok": True, "total_pairs": int(len(combined)), "jobId": job_id})


@bp.get("/slice/info")
def slice_info():
    path = request.args.get("path", "")
    if not path or not Path(path).exists():
        return jsonify({"ok": False, "error": "file not found"}), 400
    try:
        from tifffile import TiffFile

        with TiffFile(path) as tif:
            shape = list(tif.series[0].shape)
        ndim = len(shape)
        return jsonify(
            {
                "ok": True,
                "shape": shape,
                "ndim": ndim,
                "is3d": ndim >= 3,
                "z_count": shape[0] if ndim >= 3 else 1,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@bp.post("/slice/extract-z")
def slice_extract_z():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    src = Path(payload.get("path", ""))
    z = int(payload.get("z", 0))
    if not src.exists():
        return jsonify({"ok": False, "error": "source file not found"}), 400
    try:
        img = imread(str(src))
        if img.ndim >= 3:
            z = max(0, min(z, img.shape[0] - 1))
            slc = img[z]
        else:
            slc = img
        out_path = ctx._job_file(job_id, f"extracted_z{z:04d}.tif")
        imwrite(str(out_path), slc)
        return jsonify(
            {
                "ok": True,
                "path": str(out_path),
                "z": z,
                "shape": list(slc.shape),
                "dtype": str(slc.dtype),
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
