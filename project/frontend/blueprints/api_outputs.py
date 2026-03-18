"""api_outputs.py — Read-only GET routes for CSV / file serving."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, send_from_directory

import project.frontend.server_context as ctx

bp = Blueprint("api_outputs", __name__, url_prefix="/api/outputs")


@bp.get("/leaf")
def outputs_leaf():
    fp = ctx.OUTPUT_DIR / "cell_counts_leaf.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "output not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/leaf/<channel>")
def outputs_leaf_channel(channel: str):
    fp = ctx.OUTPUT_DIR / f"cell_counts_leaf_{channel}.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": f"channel output not found: {channel}"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/hierarchy")
def outputs_hierarchy():
    fp = ctx.OUTPUT_DIR / "cell_counts_hierarchy.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "hierarchy output not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/registration-qc")
def outputs_registration_qc():
    fp = ctx.OUTPUT_DIR / "slice_registration_qc.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "registration QC not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/reg-slice-list")
def outputs_reg_slice_list():
    reg_dir = ctx.OUTPUT_DIR / "registered_slices"
    if not reg_dir.exists():
        return jsonify({"ok": True, "files": [], "count": 0})
    files = sorted(reg_dir.glob("slice_*_overlay.png"))
    return jsonify({"ok": True, "files": [f.name for f in files], "count": len(files)})


@bp.get("/reg-slice/<filename>")
def outputs_reg_slice_file(filename: str):
    reg_dir = ctx.OUTPUT_DIR / "registered_slices"
    safe = Path(filename).name
    fp = reg_dir / safe
    if not fp.exists() or not safe.endswith(".png"):
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(str(reg_dir), safe)


@bp.get("/file-list")
def outputs_file_list():
    if not ctx.OUTPUT_DIR.exists():
        return jsonify({"ok": True, "files": []})
    files = []
    for f in sorted(ctx.OUTPUT_DIR.iterdir()):
        if f.is_file():
            files.append({"name": f.name, "size": f.stat().st_size, "ext": f.suffix.lower()})
    return jsonify({"ok": True, "files": files, "dir": str(ctx.OUTPUT_DIR)})


@bp.get("/named/<filename>")
def outputs_named(filename: str):
    safe = Path(filename).name
    fp = ctx.OUTPUT_DIR / safe
    if not fp.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(str(ctx.OUTPUT_DIR), safe)


@bp.get("/qc-list")
def outputs_qc_list():
    qc_dir = ctx.OUTPUT_DIR / "qc_overlays"
    if not qc_dir.exists():
        return jsonify({"ok": True, "files": [], "count": 0})
    files = sorted(qc_dir.glob("overlay_*.png"))
    return jsonify({"ok": True, "files": [f.name for f in files], "count": len(files)})


@bp.get("/qc-file/<filename>")
def outputs_qc_file(filename: str):
    qc_dir = ctx.OUTPUT_DIR / "qc_overlays"
    safe = Path(filename).name
    return send_from_directory(str(qc_dir), safe)
