"""api_overlay.py — Overlay preview, liquify, export, region-at routes."""

from __future__ import annotations

import datetime
import threading
import uuid as _uuid_mod
from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory
from PIL import Image

import project.frontend.server_context as ctx
from project.frontend.api_errors import (
    ERR_INVALID_INPUT,
    ERR_NOT_FOUND,
)
from project.frontend.services.overlay_service import (
    apply_liquify_and_render,
    render_overlay_from_label,
)

bp = Blueprint("api_overlay", __name__, url_prefix="/api")


@bp.post("/overlay/preview")
def overlay_preview():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    real_path = Path(payload.get("realPath", ""))
    label_path = Path(payload.get("labelPath", ""))
    raw_real_z = payload.get("realZIndex", None)
    raw_label_z = payload.get("labelZIndex", None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    label_z_index = None if raw_label_z in (None, "", "null") else int(raw_label_z)
    alpha = float(payload.get("alpha", 0.45))
    mode = payload.get("mode", "fill")
    structure_csv = Path(payload.get("structureCsv", "")) if payload.get("structureCsv") else None
    min_mean = float(payload.get("minMeanThreshold", 8.0))
    pixel_size_um = float(payload.get("pixelSizeUm", 0.65))
    rotate_deg = float(payload.get("rotateAtlas", 0.0))
    flip_mode = payload.get("flipAtlas", "none")
    fit_mode = payload.get("fitMode", "cover")
    major_top_k = int(payload.get("majorTopK", 20))
    edge_smooth_iter = int(payload.get("edgeSmoothIter", 1))
    warp_params = payload.get("warpParams", {})
    if not isinstance(warp_params, dict):
        warp_params = {}
    # Inject registration settings from active config if not already specified
    warp_params = dict(warp_params)
    reg_cfg = ctx._active_reg_cfg()
    if reg_cfg:
        if "force_hemisphere" not in warp_params and reg_cfg.get("atlas_hemisphere"):
            warp_params["force_hemisphere"] = str(reg_cfg["atlas_hemisphere"])
        if "tissue_shrink_factor" not in warp_params and reg_cfg.get("tissue_shrink_factor"):
            warp_params["tissue_shrink_factor"] = float(reg_cfg["tissue_shrink_factor"])
        if "enable_silhouette_conform" not in warp_params:
            warp_params["enable_silhouette_conform"] = bool(
                reg_cfg.get("enable_silhouette_conform", False)
            )
        if "enable_sitk_ref_refine" not in warp_params:
            warp_params["enable_sitk_ref_refine"] = bool(
                reg_cfg.get("enable_sitk_ref_refine", False)
            )

    if not real_path.exists() or not label_path.exists():
        return jsonify(
            {"ok": False, "error": "real or label path not found", "error_code": ERR_NOT_FOUND}
        ), 400

    out = ctx._job_file(job_id, "overlay_preview.png")
    hover_label_path = ctx._job_file(job_id, "overlay_label_preview.tif")

    kwargs = dict(
        real_slice_path=real_path,
        label_slice_path=label_path,
        out_png=out,
        alpha=alpha,
        mode=mode,
        structure_csv=structure_csv,
        min_mean_threshold=min_mean,
        pixel_size_um=pixel_size_um,
        rotate_deg=rotate_deg,
        flip_mode=flip_mode,
        return_meta=True,
        major_top_k=major_top_k,
        fit_mode=fit_mode,
        edge_smooth_iter=edge_smooth_iter,
        warp_params=warp_params,
        warped_label_out=hover_label_path,
        real_z_index=real_z_index,
        label_z_index=label_z_index,
    )

    token = _uuid_mod.uuid4().hex
    with ctx._task_lock:
        ctx._preview_tasks[token] = {
            "status": "pending",
            "progress": 0,
            "message": "Queued...",
            "result": None,
            "error": None,
            "jobId": job_id,
        }
    t = threading.Thread(
        target=ctx._run_preview_worker,
        args=(token, kwargs, job_id, structure_csv),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True, "token": token, "jobId": job_id, "started": True})


@bp.get("/overlay/preview/status")
def overlay_preview_status():
    token = request.args.get("token", "")
    if not token or token not in ctx._preview_tasks:
        return jsonify({"ok": False, "error": "unknown token", "error_code": ERR_NOT_FOUND}), 404
    task = ctx._preview_tasks[token]
    resp = {
        "ok": True,
        "token": token,
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
        "jobId": task.get("jobId", ""),
    }
    if task["status"] == "done":
        resp.update(task["result"])
    if task["status"] == "error":
        resp["error"] = task["error"]
        if "failCase" in task:
            resp["failCase"] = task["failCase"]
    return jsonify(resp)


@bp.post("/overlay/liquify-drag")
def overlay_liquify_drag():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    real_path = Path(payload.get("realPath", ""))
    label_path_payload = Path(payload.get("labelPath", "")) if payload.get("labelPath") else None
    raw_real_z = payload.get("realZIndex", None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    alpha = float(payload.get("alpha", 0.45))
    mode = str(payload.get("mode", "fill"))
    structure_csv = Path(payload.get("structureCsv", "")) if payload.get("structureCsv") else None
    min_mean = float(payload.get("minMeanThreshold", 8.0))
    pixel_size_um = float(payload.get("pixelSizeUm", 0.65))
    rotate_deg = float(payload.get("rotateAtlas", 0.0))
    flip_mode = payload.get("flipAtlas", "none")
    fit_mode = payload.get("fitMode", "cover")
    major_top_k = int(payload.get("majorTopK", 20))
    edge_smooth_iter = int(payload.get("edgeSmoothIter", 1))
    warp_params = payload.get("warpParams", {})
    if not isinstance(warp_params, dict):
        warp_params = {}

    if not real_path.exists():
        return jsonify(
            {"ok": False, "error": "real path not found", "error_code": ERR_NOT_FOUND}
        ), 400

    hover_label_path = ctx._job_file(job_id, "overlay_label_preview.tif")
    if hover_label_path.exists():
        base_label_path = hover_label_path
    elif label_path_payload is not None and label_path_payload.exists():
        base_label_path = label_path_payload
    else:
        return jsonify(
            {"ok": False, "error": "no aligned preview label available; run preview first"}
        ), 400

    drags = payload.get("drags", [])
    if not isinstance(drags, list):
        drags = []
    if not drags and all(k in payload for k in ("x1", "y1", "x2", "y2")):
        drags = [
            {
                "x1": payload.get("x1"),
                "y1": payload.get("y1"),
                "x2": payload.get("x2"),
                "y2": payload.get("y2"),
                "radius": payload.get("radius", 80),
                "strength": payload.get("strength", 0.72),
            }
        ]
    if not drags:
        return jsonify(
            {"ok": False, "error": "no drags provided", "error_code": ERR_INVALID_INPUT}
        ), 400

    calib_dir = ctx._job_manual_calibration_dir(job_id)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    corrected_label_path = calib_dir / f"manual_warped_label_{ts}.tif"
    out = ctx._job_file(job_id, "overlay_preview.png")

    render_kwargs = dict(
        real_slice_path=real_path,
        label_slice_path=corrected_label_path,
        out_png=out,
        alpha=alpha,
        mode=mode,
        structure_csv=structure_csv,
        min_mean_threshold=min_mean,
        pixel_size_um=pixel_size_um,
        rotate_deg=rotate_deg,
        flip_mode=flip_mode,
        return_meta=True,
        major_top_k=major_top_k,
        fit_mode=fit_mode,
        edge_smooth_iter=edge_smooth_iter,
        warp_params=warp_params,
        warped_label_out=hover_label_path,
        real_z_index=real_z_index,
        prewarped_label=True,
    )

    try:
        _, diagnostic = apply_liquify_and_render(
            base_label_path,
            drags,
            corrected_label_path,
            hover_label_path,
            render_kwargs,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "error_code": ERR_INVALID_INPUT}), 400

    return jsonify(
        {
            "ok": True,
            "preview": str(out),
            "correctedLabelPath": str(corrected_label_path),
            "diagnostic": diagnostic,
            "jobId": job_id,
        }
    )


@bp.post("/overlay/calibration/finalize")
def overlay_calibration_finalize():
    import shutil

    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    real_path = Path(payload.get("realPath", ""))
    if not real_path.exists():
        return jsonify(
            {"ok": False, "error": "real path not found", "error_code": ERR_NOT_FOUND}
        ), 400
    hover_label_path = ctx._job_file(job_id, "overlay_label_preview.tif")
    if not hover_label_path.exists():
        return jsonify(
            {
                "ok": False,
                "error": "no calibrated label to finalize",
                "error_code": ERR_INVALID_INPUT,
            }
        ), 400

    raw_real_z = payload.get("realZIndex", None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    alpha = float(payload.get("alpha", 0.45))
    mode = str(payload.get("mode", "fill"))
    structure_csv = Path(payload.get("structureCsv", "")) if payload.get("structureCsv") else None
    min_mean = float(payload.get("minMeanThreshold", 8.0))
    pixel_size_um = float(payload.get("pixelSizeUm", 0.65))
    rotate_deg = float(payload.get("rotateAtlas", 0.0))
    flip_mode = payload.get("flipAtlas", "none")
    fit_mode = payload.get("fitMode", "cover")
    major_top_k = int(payload.get("majorTopK", 20))
    edge_smooth_iter = int(payload.get("edgeSmoothIter", 1))
    warp_params = payload.get("warpParams", {})
    if not isinstance(warp_params, dict):
        warp_params = {}
    auto_learn = bool(payload.get("autoLearn", True))
    note = str(payload.get("note", "")).strip()

    calib_dir = ctx._job_manual_calibration_dir(job_id)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    final_overlay = calib_dir / f"calibrated_overlay_{ts}.png"
    final_label = calib_dir / f"calibrated_label_{ts}.tif"
    shutil.copy2(hover_label_path, final_label)

    try:
        diagnostic = render_overlay_from_label(
            real_path,
            final_label,
            final_overlay,
            render_kwargs=dict(
                alpha=alpha,
                mode=mode,
                structure_csv=structure_csv,
                min_mean_threshold=min_mean,
                pixel_size_um=pixel_size_um,
                rotate_deg=rotate_deg,
                flip_mode=flip_mode,
                return_meta=True,
                major_top_k=major_top_k,
                fit_mode=fit_mode,
                edge_smooth_iter=edge_smooth_iter,
                warp_params=warp_params,
                real_z_index=real_z_index,
                prewarped_label=True,
            ),
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "error_code": ERR_INVALID_INPUT}), 400

    manifest = {
        "timestamp": ts,
        "note": note,
        "realPath": str(real_path),
        "realZIndex": real_z_index,
        "calibratedLabelPath": str(final_label),
        "calibratedOverlayPath": str(final_overlay),
        "mode": mode,
        "alpha": alpha,
        "fitMode": fit_mode,
        "pixelSizeUm": pixel_size_um,
        "rotateAtlas": rotate_deg,
        "flipAtlas": flip_mode,
        "majorTopK": major_top_k,
        "edgeSmoothIter": edge_smooth_iter,
        "warpParams": warp_params,
        "diagnostic": diagnostic,
    }
    pair = ctx._save_calibration_pair(
        real_path=real_path,
        corrected_label_tif=final_label,
        corrected_overlay_png=final_overlay,
        manifest=manifest,
        real_z_index=real_z_index,
    )

    learning_started = ctx._learn_from_trainset_async() if auto_learn else False
    return jsonify(
        {
            "ok": True,
            "sample": pair,
            "autoLearn": auto_learn,
            "learningStarted": bool(learning_started),
            "learnStatus": ctx.learning_state,
            "jobId": job_id,
        }
    )


@bp.post("/overlay/export")
def overlay_export():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    fmt = str(payload.get("format", "png")).strip().lower()
    allowed = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}
    if fmt not in allowed:
        return jsonify(
            {"ok": False, "error": f"unsupported format: {fmt}", "error_code": ERR_INVALID_INPUT}
        ), 400

    src = ctx._job_file(job_id, "overlay_preview.png")
    if not src.exists():
        return jsonify(
            {"ok": False, "error": "overlay preview not found", "error_code": ERR_NOT_FOUND}
        ), 404

    export_dir = ctx._job_output_dir(job_id) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "jpg" if fmt == "jpeg" else "tif" if fmt == "tiff" else fmt
    out = export_dir / f"overlay_export_{ts}.{ext}"
    try:
        with Image.open(src) as im:
            if ext in ("jpg",):
                im.convert("RGB").save(out, quality=95)
            elif ext in ("tif",):
                im.save(out, compression="tiff_lzw")
            else:
                im.save(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "error_code": ERR_INVALID_INPUT}), 400

    return jsonify({"ok": True, "path": str(out), "format": ext, "jobId": job_id})


@bp.get("/overlay/region-at")
def overlay_region_at():
    job_id = ctx._query_job_id()
    label = ctx._load_hover_label(job_id)
    if label is None:
        return jsonify(
            {"ok": False, "error": "preview label not available yet", "error_code": ERR_NOT_FOUND}
        ), 404

    try:
        x = int(float(request.args.get("x", "-1")))
        y = int(float(request.args.get("y", "-1")))
    except Exception:
        return jsonify({"ok": False, "error": "invalid x/y", "error_code": ERR_INVALID_INPUT}), 400

    h, w = label.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return jsonify({"ok": True, "inside": False, "region_id": 0, "x": x, "y": y})

    rid = int(label[y, x])
    if rid <= 0:
        return jsonify({"ok": True, "inside": False, "region_id": 0, "x": x, "y": y})

    tree = ctx._load_hover_structure_tree()
    node = tree.get(str(rid), {}) if isinstance(tree, dict) else {}
    parent_map = ctx._build_parent_name_map(ctx._last_preview_structure_csv.get(job_id, ""))
    parent_name = str(node.get("parent", "")).strip() or parent_map.get(rid, "")

    return jsonify(
        {
            "ok": True,
            "inside": True,
            "x": int(x),
            "y": int(y),
            "region_id": int(rid),
            "acronym": str(node.get("acronym", "")),
            "name": str(node.get("name", "")),
            "parent": str(parent_name),
            "color": str(node.get("color", "")),
        }
    )


@bp.post("/overlay/atlas-layer")
def overlay_atlas_layer():
    """Render atlas colormap as RGBA PNG (transparent background) for client-side compositing."""
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    label_path = Path(payload.get("labelPath", ""))
    real_path = Path(payload.get("realPath", ""))
    structure_csv = Path(payload.get("structureCsv", "")) if payload.get("structureCsv") else None
    pixel_size_um = float(payload.get("pixelSizeUm", 0.65))
    rotate_deg = float(payload.get("rotateAtlas", 0.0))
    flip_mode = payload.get("flipAtlas", "none")
    fit_mode = payload.get("fitMode", "cover")
    edge_smooth_iter = int(payload.get("edgeSmoothIter", 1))
    warp_params = payload.get("warpParams", {})
    if not isinstance(warp_params, dict):
        warp_params = {}
    raw_real_z = payload.get("realZIndex", None)
    raw_label_z = payload.get("labelZIndex", None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    label_z_index = None if raw_label_z in (None, "", "null") else int(raw_label_z)

    if not label_path.exists() or not real_path.exists():
        return jsonify(
            {"ok": False, "error": "label or real path not found", "error_code": ERR_NOT_FOUND}
        ), 400
    try:
        out = ctx._job_file(job_id, "atlas_layer_rgba.png")
        diagnostic = render_overlay_from_label(
            real_path,
            label_path,
            out,
            render_kwargs=dict(
                alpha=1.0,
                mode="fill",
                structure_csv=structure_csv,
                pixel_size_um=pixel_size_um,
                rotate_deg=rotate_deg,
                flip_mode=flip_mode,
                return_meta=True,
                major_top_k=20,
                fit_mode=fit_mode,
                edge_smooth_iter=edge_smooth_iter,
                warp_params=warp_params,
                real_z_index=real_z_index,
                label_z_index=label_z_index,
            ),
        )
        return jsonify({"ok": True, "diagnostic": diagnostic, "jobId": job_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "error_code": ERR_INVALID_INPUT}), 400


@bp.get("/outputs/atlas-layer")
def get_atlas_layer():
    fp = ctx._job_file(ctx._query_job_id(), "atlas_layer_rgba.png")
    if not fp.exists():
        return jsonify(
            {"ok": False, "error": "atlas layer not rendered yet", "error_code": ERR_NOT_FOUND}
        ), 404
    return send_from_directory(str(fp.parent), fp.name)


@bp.get("/outputs/overlay-preview")
def outputs_overlay_preview():
    fp = ctx._job_file(ctx._query_job_id(), "overlay_preview.png")
    if not fp.exists():
        return jsonify(
            {"ok": False, "error": "overlay preview not found", "error_code": ERR_NOT_FOUND}
        ), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/outputs/overlay-compare")
def outputs_overlay_compare():
    fp = ctx._job_file(ctx._query_job_id(), "overlay_compare.png")
    if not fp.exists():
        return jsonify(
            {"ok": False, "error": "overlay compare not found", "error_code": ERR_NOT_FOUND}
        ), 404
    return send_from_directory(fp.parent, fp.name)
