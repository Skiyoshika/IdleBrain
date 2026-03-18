"""api_demo.py — Demo/chart routes."""

from __future__ import annotations

import csv
import threading

from flask import Blueprint, jsonify, send_from_directory

import project.frontend.server_context as ctx
from project.frontend.services.demo_service import generate_cell_chart, generate_demo_comparison

bp = Blueprint("api_demo", __name__, url_prefix="/api")


@bp.get("/outputs/demo-best-slice")
def outputs_demo_best_slice():
    """Serve the pre-generated best-slice comparison image."""
    fp = ctx.OUTPUT_DIR / "demo_best_slice.jpg"
    if not fp.exists():
        return jsonify({"ok": False, "error": "Best-slice image not generated yet"}), 404
    return send_from_directory(str(ctx.OUTPUT_DIR), "demo_best_slice.jpg")


@bp.get("/outputs/demo-annotated-slice")
def outputs_demo_annotated_slice():
    """Serve the annotated single-slice with region labels."""
    fp = ctx.OUTPUT_DIR / "demo_annotated_slice.jpg"
    if not fp.exists():
        return jsonify(
            {"ok": False, "error": "Annotated slice not generated yet. Run refresh_demo.py first."}
        ), 404
    return send_from_directory(str(ctx.OUTPUT_DIR), "demo_annotated_slice.jpg")


@bp.get("/outputs/cell-chart")
def outputs_cell_chart():
    """Generate and serve the cell-count bar+pie chart."""
    chart_path = ctx.OUTPUT_DIR / "cell_count_chart.png"
    hier_path = ctx.OUTPUT_DIR / "cell_counts_hierarchy.csv"
    if not hier_path.exists():
        return jsonify({"ok": False, "error": "No hierarchy CSV yet"}), 404
    # Regenerate if stale
    if not chart_path.exists() or hier_path.stat().st_mtime > chart_path.stat().st_mtime:
        try:
            generate_cell_chart(hier_path, chart_path, ctx.PROJECT_ROOT)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Chart generation failed: {e}"}), 500
    if not chart_path.exists():
        return jsonify({"ok": False, "error": "Chart not found"}), 404
    return send_from_directory(str(ctx.OUTPUT_DIR), "cell_count_chart.png")


@bp.get("/outputs/demo-comparison/<int:slice_idx>")
def outputs_demo_comparison(slice_idx: int):
    """Generate and serve a side-by-side raw vs atlas comparison for a given slice index."""
    reg_dir = ctx.OUTPUT_DIR / "registered_slices"
    data_dir = ctx.PROJECT_ROOT / "data" / "35_C0_demo"
    ov_path = reg_dir / f"slice_{slice_idx:04d}_overlay.png"
    if not ov_path.exists():
        return jsonify({"ok": False, "error": f"slice {slice_idx} not found"}), 404

    out_path = ctx.OUTPUT_DIR / f"compare_{slice_idx:04d}.jpg"
    try:
        generate_demo_comparison(slice_idx, reg_dir, data_dir, out_path)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return send_from_directory(str(ctx.OUTPUT_DIR), out_path.name)


@bp.get("/outputs/demo-panel")
def outputs_demo_panel():
    """Serve the demo panel image, generating it on-demand if needed."""
    import subprocess
    import sys

    panel_path = ctx.OUTPUT_DIR / "demo_panel.jpg"
    reg_dir = ctx.OUTPUT_DIR / "registered_slices"
    # Auto-regenerate if stale or missing
    if not panel_path.exists() or (
        reg_dir.exists()
        and any(
            p.stat().st_mtime > panel_path.stat().st_mtime
            for p in reg_dir.glob("slice_*_overlay.png")
        )
    ):
        try:
            script = ctx.PROJECT_ROOT / "scripts" / "make_demo_panel.py"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--reg_dir",
                    str(reg_dir),
                    "--out",
                    str(panel_path),
                    "--n",
                    "12",
                    "--cols",
                    "4",
                    "--size",
                    "380",
                ],
                cwd=str(ctx.PROJECT_ROOT),
                timeout=120,
                check=True,
                capture_output=True,
            )
        except Exception as e:
            return jsonify({"ok": False, "error": f"Panel generation failed: {e}"}), 500
    if not panel_path.exists():
        return jsonify({"ok": False, "error": "Panel not found"}), 404
    return send_from_directory(str(ctx.OUTPUT_DIR), "demo_panel.jpg")


@bp.post("/outputs/refresh-demo")
def outputs_refresh_demo():
    """Run refresh_demo.py to regenerate all demo visuals (panel, annotated slice, chart)."""
    import subprocess
    import sys

    script = ctx.PROJECT_ROOT / "scripts" / "refresh_demo.py"
    if not script.exists():
        return jsonify({"ok": False, "error": "refresh_demo.py not found"}), 404

    def _run():
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(ctx.PROJECT_ROOT),
                timeout=180,
                capture_output=True,
                text=True,
            )
            ctx._append_log(f"[refresh_demo] {result.stdout.strip()}")
            if result.returncode != 0:
                ctx._append_log(f"[refresh_demo] ERROR: {result.stderr.strip()}")
        except Exception as e:
            ctx._append_log(f"[refresh_demo] exception: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "refresh_demo.py started in background"})


@bp.get("/outputs/reg-stats")
def outputs_reg_stats():
    """Return registration quality summary for display."""
    qc_path = ctx.OUTPUT_DIR / "slice_registration_qc.csv"
    if not qc_path.exists():
        return jsonify({"ok": False, "error": "No registration QC data yet"})
    try:
        with open(qc_path, newline="", encoding="utf-8") as _f:
            rows = list(csv.DictReader(_f))
        scores = [float(r["best_score"]) for r in rows if r.get("best_score")]
        ok_count = sum(1 for r in rows if r.get("registration_ok", "").lower() == "true")
        return jsonify(
            {
                "ok": True,
                "total": len(rows),
                "ok_count": ok_count,
                "mean_score": round(sum(scores) / len(scores), 3) if scores else 0,
                "min_score": round(min(scores), 3) if scores else 0,
                "max_score": round(max(scores), 3) if scores else 0,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
