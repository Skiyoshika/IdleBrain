"""Demo/chart routes."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

from flask import Blueprint, jsonify, send_from_directory

import project.frontend.server_context as ctx
from project.frontend.blueprints.api_outputs import _registration_run_dirs
from project.frontend.services.demo_service import (
    build_cell_summary,
    generate_cell_chart,
    generate_demo_comparison,
    generate_detection_confidence_samples,
    generate_registration_annotated_slice,
    generate_registration_best_slice,
    generate_registration_demo_panel,
)

bp = Blueprint("api_demo", __name__, url_prefix="/api")


def _outputs_root() -> Path:
    return ctx._job_output_dir(ctx._query_job_id())


def _latest_registration_run() -> Path | None:
    runs = _registration_run_dirs(_outputs_root())
    return runs[0] if runs else None


def _artifact_stale(target: Path, *sources: Path) -> bool:
    if not target.exists():
        return True
    target_mtime = target.stat().st_mtime
    return any(source.exists() and source.stat().st_mtime > target_mtime for source in sources)


def _run_qc_artifact(run_dir: Path, name: str) -> Path:
    return run_dir / name


def _run_sources(run_dir: Path) -> list[Path]:
    meta_path = run_dir / "registration_metadata.json"
    if not meta_path.exists():
        return [meta_path]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return [meta_path]
    sources = [meta_path]
    for key, fallback in (
        ("registered_brain", "registered_brain.nii.gz"),
        ("annotation_fixed_half", "annotation_fixed_half.nii.gz"),
    ):
        value = str(meta.get(key, "")).strip()
        path = Path(value) if value else run_dir / fallback
        if not path.is_absolute():
            path = run_dir / path
        sources.append(path)
    return sources


def _serve_generated_run_artifact(
    run_dir: Path,
    artifact_name: str,
    generator,
    *args,
    **kwargs,
):
    artifact = _run_qc_artifact(run_dir, artifact_name)
    sources = _run_sources(run_dir)
    if _artifact_stale(artifact, *sources):
        generator(run_dir, artifact, *args, **kwargs)
    return send_from_directory(str(run_dir), artifact.name)


def _detection_sample_dir() -> Path:
    return _outputs_root() / "detection_samples"


def _detection_sample_manifest_path() -> Path:
    return _detection_sample_dir() / "manifest.json"


@bp.get("/outputs/demo-best-slice")
def outputs_demo_best_slice():
    """Serve the best-slice comparison image."""

    run_dir = _latest_registration_run()
    if run_dir is not None:
        try:
            return _serve_generated_run_artifact(
                run_dir,
                "qc_best_slice.jpg",
                generate_registration_best_slice,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": f"3D best-slice generation failed: {exc}"}), 500

    outputs_root = _outputs_root()
    fp = outputs_root / "demo_best_slice.jpg"
    if not fp.exists():
        return jsonify({"ok": False, "error": "Best-slice image not generated yet"}), 404
    return send_from_directory(str(outputs_root), fp.name)


@bp.get("/outputs/demo-annotated-slice")
def outputs_demo_annotated_slice():
    """Serve the annotated single-slice with region labels."""

    run_dir = _latest_registration_run()
    if run_dir is not None:
        structure_csv = ctx.PROJECT_ROOT / "configs" / "allen_mouse_structure_graph.csv"
        try:
            return _serve_generated_run_artifact(
                run_dir,
                "qc_annotated_slice.jpg",
                generate_registration_annotated_slice,
                structure_csv=structure_csv,
                top_n=12,
            )
        except Exception as exc:
            return jsonify(
                {"ok": False, "error": f"3D annotated-slice generation failed: {exc}"}
            ), 500

    outputs_root = _outputs_root()
    fp = outputs_root / "demo_annotated_slice.jpg"
    if not fp.exists():
        return jsonify(
            {"ok": False, "error": "Annotated slice not generated yet. Run refresh_demo.py first."}
        ), 404
    return send_from_directory(str(outputs_root), fp.name)


@bp.get("/outputs/cell-chart")
def outputs_cell_chart():
    """Generate and serve the cell-count summary chart."""

    outputs_root = _outputs_root()
    chart_path = outputs_root / "cell_count_chart.png"
    hier_path = outputs_root / "cell_counts_hierarchy.csv"
    cells_path = outputs_root / "cells_mapped.csv"
    if not hier_path.exists():
        return jsonify({"ok": False, "error": "No hierarchy CSV yet"}), 404
    sources = [hier_path]
    if cells_path.exists():
        sources.append(cells_path)
    if not chart_path.exists() or any(
        src.stat().st_mtime > chart_path.stat().st_mtime for src in sources
    ):
        try:
            generate_cell_chart(hier_path, chart_path, ctx.PROJECT_ROOT)
        except Exception as exc:
            return jsonify({"ok": False, "error": f"Chart generation failed: {exc}"}), 500
    if not chart_path.exists():
        return jsonify({"ok": False, "error": "Chart not found"}), 404
    return send_from_directory(str(outputs_root), chart_path.name)


@bp.get("/outputs/cell-summary")
def outputs_cell_summary():
    """Return a product-facing summary of the current cell-count outputs."""

    hier_path = _outputs_root() / "cell_counts_hierarchy.csv"
    if not hier_path.exists():
        return jsonify({"ok": False, "error": "No hierarchy CSV yet"}), 404
    try:
        summary = build_cell_summary(hier_path)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Cell summary generation failed: {exc}"}), 500
    return jsonify({"ok": True, "summary": summary})


@bp.get("/outputs/detection-samples")
def outputs_detection_samples():
    """Return representative counted-cell overlays rendered from the final dedup table."""

    cells_path = _outputs_root() / "cells_dedup.csv"
    if not cells_path.exists():
        return jsonify({"ok": True, "samples": [], "count": 0})

    sample_dir = _detection_sample_dir()
    manifest_path = _detection_sample_manifest_path()
    regenerate = (
        not manifest_path.exists() or cells_path.stat().st_mtime > manifest_path.stat().st_mtime
    )

    if regenerate:
        try:
            manifest = generate_detection_confidence_samples(cells_path, sample_dir, sample_count=3)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            return jsonify(
                {"ok": False, "error": f"detection sample generation failed: {exc}"}
            ), 500

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = []

    samples = []
    for item in manifest:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        fp = sample_dir / name
        if not fp.exists():
            continue
        sample = dict(item)
        sample["url"] = f"/api/outputs/detection-sample/{name}"
        samples.append(sample)
    return jsonify({"ok": True, "samples": samples, "count": len(samples)})


@bp.get("/outputs/detection-sample/<filename>")
def outputs_detection_sample_file(filename: str):
    sample_dir = _detection_sample_dir()
    safe = Path(filename).name
    fp = sample_dir / safe
    if not fp.exists() or not fp.is_file():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(str(sample_dir), safe)


@bp.get("/outputs/demo-comparison/<int:slice_idx>")
def outputs_demo_comparison(slice_idx: int):
    """Generate and serve a side-by-side raw vs atlas comparison for a legacy slice."""

    outputs_root = _outputs_root()
    reg_dir = outputs_root / "registered_slices"
    data_dir = ctx.PROJECT_ROOT / "data" / "35_C0_demo"
    ov_path = reg_dir / f"slice_{slice_idx:04d}_overlay.png"
    if not ov_path.exists():
        return jsonify({"ok": False, "error": f"slice {slice_idx} not found"}), 404

    out_path = outputs_root / f"compare_{slice_idx:04d}.jpg"
    try:
        generate_demo_comparison(slice_idx, reg_dir, data_dir, out_path)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return send_from_directory(str(outputs_root), out_path.name)


@bp.get("/outputs/demo-panel")
def outputs_demo_panel():
    """Serve the demo panel image, generating it on-demand if needed."""

    run_dir = _latest_registration_run()
    if run_dir is not None:
        try:
            return _serve_generated_run_artifact(
                run_dir,
                "qc_panel.jpg",
                generate_registration_demo_panel,
                n_slices=12,
                cols=4,
                thumb_size=380,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": f"3D panel generation failed: {exc}"}), 500

    import subprocess
    import sys

    outputs_root = _outputs_root()
    panel_path = outputs_root / "demo_panel.jpg"
    reg_dir = outputs_root / "registered_slices"
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
        except Exception as exc:
            return jsonify({"ok": False, "error": f"Panel generation failed: {exc}"}), 500
    if not panel_path.exists():
        return jsonify({"ok": False, "error": "Panel not found"}), 404
    return send_from_directory(str(outputs_root), panel_path.name)


@bp.post("/outputs/refresh-demo")
def outputs_refresh_demo():
    """Regenerate the QC/demo visuals."""

    import subprocess
    import sys

    run_dir = _latest_registration_run()
    script = ctx.PROJECT_ROOT / "scripts" / "refresh_demo.py"
    if run_dir is None and not script.exists():
        return jsonify({"ok": False, "error": "refresh_demo.py not found"}), 404

    def _run():
        try:
            outputs_root = _outputs_root()
            hier_path = outputs_root / "cell_counts_hierarchy.csv"
            chart_path = outputs_root / "cell_count_chart.png"
            cells_path = outputs_root / "cells_dedup.csv"

            if run_dir is not None:
                structure_csv = ctx.PROJECT_ROOT / "configs" / "allen_mouse_structure_graph.csv"
                generate_registration_demo_panel(
                    run_dir,
                    _run_qc_artifact(run_dir, "qc_panel.jpg"),
                    n_slices=12,
                    cols=4,
                    thumb_size=380,
                )
                generate_registration_best_slice(
                    run_dir, _run_qc_artifact(run_dir, "qc_best_slice.jpg")
                )
                generate_registration_annotated_slice(
                    run_dir,
                    _run_qc_artifact(run_dir, "qc_annotated_slice.jpg"),
                    structure_csv=structure_csv,
                    top_n=12,
                )
                if hier_path.exists():
                    generate_cell_chart(hier_path, chart_path, ctx.PROJECT_ROOT)
                if cells_path.exists():
                    manifest = generate_detection_confidence_samples(
                        cells_path,
                        _detection_sample_dir(),
                        sample_count=3,
                    )
                    _detection_sample_manifest_path().write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                ctx._append_log(f"[refresh_demo] regenerated 3D QC visuals for {run_dir.name}")
                return

            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(ctx.PROJECT_ROOT),
                timeout=180,
                capture_output=True,
                text=True,
            )
            ctx._append_log(f"[refresh_demo] {result.stdout.strip()}")
            if result.returncode == 0 and cells_path.exists():
                manifest = generate_detection_confidence_samples(
                    cells_path,
                    _detection_sample_dir(),
                    sample_count=3,
                )
                _detection_sample_manifest_path().write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if result.returncode == 0 and hier_path.exists():
                generate_cell_chart(hier_path, chart_path, ctx.PROJECT_ROOT)
            if result.returncode != 0:
                ctx._append_log(f"[refresh_demo] ERROR: {result.stderr.strip()}")
        except Exception as exc:
            ctx._append_log(f"[refresh_demo] exception: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "Demo refresh started in background"})


@bp.get("/outputs/reg-stats")
def outputs_reg_stats():
    """Return registration quality summary for display."""

    run_dir = _latest_registration_run()
    if run_dir is not None:
        meta_path = run_dir / "registration_metadata.json"
        metrics_path = run_dir / "registration_metrics.csv"
        if meta_path.exists() and metrics_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                with metrics_path.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                metrics = {}
                for row in rows:
                    try:
                        metrics[str(row["metric"])] = float(row["value"])
                    except Exception:
                        continue
                staining = meta.get("staining_stats") or {}
                return jsonify(
                    {
                        "ok": True,
                        "mode": "registration_run",
                        "pipeline": f"{str(meta.get('backend', '')).upper()}{' + Laplacian' if meta.get('laplacian_enabled') else ''}",
                        "ncc": round(float(metrics.get("NCC", 0.0)), 4),
                        "ssim": round(float(metrics.get("SSIM", 0.0)), 4),
                        "dice": round(float(metrics.get("Dice", 0.0)), 4),
                        "staining_rate": round(float(staining.get("staining_rate", 0.0)), 4),
                        "atlas_coverage": round(float(staining.get("atlas_coverage", 0.0)), 4),
                    }
                )
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

    qc_path = _outputs_root() / "slice_registration_qc.csv"
    if not qc_path.exists():
        return jsonify({"ok": False, "error": "No registration QC data yet"})
    try:
        with qc_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        scores = [float(row["best_score"]) for row in rows if row.get("best_score")]
        ok_count = sum(1 for row in rows if row.get("registration_ok", "").lower() == "true")
        return jsonify(
            {
                "ok": True,
                "mode": "slice_qc",
                "total": len(rows),
                "ok_count": ok_count,
                "mean_score": round(sum(scores) / len(scores), 3) if scores else 0,
                "min_score": round(min(scores), 3) if scores else 0,
                "max_score": round(max(scores), 3) if scores else 0,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
