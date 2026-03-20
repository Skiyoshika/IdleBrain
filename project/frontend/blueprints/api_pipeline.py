"""api_pipeline.py — Pipeline run/status/cancel/logs/history routes."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request, send_from_directory

import project.frontend.server_context as ctx
from project.frontend.api_errors import (
    ERR_CONFIG_PATH_DENIED,
    ERR_INTERNAL,
    ERR_INVALID_INPUT,
    ERR_PIPELINE_RUNNING,
)
from project.scripts.config_validation import collect_runtime_config_issues, load_config

bp = Blueprint("api_pipeline", __name__)


def _resolve_job_id(payload: dict | None = None) -> str:
    if isinstance(payload, dict) and payload.get("jobId"):
        return ctx._sanitize_job_id(payload.get("jobId"))
    return ctx._query_job_id()


def _job_state(job_id: str | None = None) -> dict:
    return ctx.get_job_state(job_id)


def _job_outputs_dir(job_id: str | None = None) -> Path:
    return ctx._job_output_dir(job_id)


def _make_job_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ctx._sanitize_job_id(f"job_{stamp}_{uuid4().hex[:8]}")


def _read_version_info() -> dict[str, str]:
    version_path = ctx.PROJECT_ROOT / "version.json"
    fallback = {"version": "0.5.1", "build_date": "", "commit": ""}
    # Try version.json first (created by build_version_json.py during release)
    if version_path.exists():
        try:
            data = json.loads(version_path.read_text(encoding="utf-8"))
            return {
                "version": str(data.get("version", fallback["version"])),
                "build_date": str(data.get("build_date", "")),
                "commit": str(data.get("commit", "")),
            }
        except Exception:
            pass
    # Fall back to setuptools-scm generated _version.py
    try:
        import project._version as _v  # type: ignore[import]

        return {"version": str(_v.__version__), "build_date": "", "commit": ""}
    except Exception:
        pass
    return fallback


def _ALLOWED_CONFIG_ROOTS():
    return (ctx.PROJECT_ROOT / "configs", ctx.OUTPUT_DIR)


def _resolve_config_path(raw_path: str | None) -> Path:
    if not raw_path:
        return ctx.PROJECT_ROOT / "configs" / "run_config.template.json"
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = None
        for base in (ctx.ROOT, ctx.PROJECT_ROOT, Path.cwd()):
            r = (base / candidate).resolve()
            if r.exists():
                resolved = r
                break
        if resolved is None:
            resolved = (ctx.ROOT / candidate).resolve()
    # BUG-2 fix: containment check — only allow configs/ and outputs/ subtrees
    allowed = _ALLOWED_CONFIG_ROOTS()
    if not any(resolved.is_relative_to(r) for r in allowed):
        raise PermissionError(
            f"Config path '{resolved}' is outside allowed directories. "
            f"Allowed: {[str(r) for r in allowed]}"
        )
    return resolved


def _apply_runtime_overrides(base_cfg: dict, payload: dict) -> dict:
    cfg = deepcopy(base_cfg)
    params = payload.get("params", {}) if isinstance(payload.get("params"), dict) else {}
    input_cfg = cfg.setdefault("input", {})
    reg_cfg = cfg.setdefault("registration", {})

    input_dir = str(payload.get("inputDir", "")).strip()
    if input_dir:
        input_cfg["slice_dir"] = input_dir

    pixel_size = params.get("pixelSizeUm")
    if pixel_size not in ("", None):
        try:
            input_cfg["pixel_size_um_xy"] = float(pixel_size)
        except Exception:
            input_cfg["pixel_size_um_xy"] = pixel_size

    channels = payload.get("channels", [])
    if isinstance(channels, list) and channels:
        input_cfg["active_channel"] = str(channels[0])

    align_mode = params.get("alignMode")
    if align_mode not in ("", None):
        reg_cfg["align_mode"] = str(align_mode)

    confidence_threshold = params.get("confidenceThreshold")
    if confidence_threshold not in ("", None):
        try:
            cfg.setdefault("detection", {})["min_score"] = float(confidence_threshold)
        except Exception:
            pass

    return cfg


def _runtime_path_issues(payload: dict) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    input_dir = str(payload.get("inputDir", "")).strip()
    atlas = str(
        (payload.get("params", {}) or {}).get("atlasPath") or payload.get("atlasPath") or ""
    ).strip()
    struct = str(
        (payload.get("params", {}) or {}).get("structPath") or payload.get("structPath") or ""
    ).strip()

    if not input_dir or not Path(input_dir).exists():
        issues.append(
            {
                "field": "inputDir",
                "severity": "error",
                "message": "Input TIFF folder is missing or not found.",
            }
        )
    if not atlas or not Path(atlas).exists():
        issues.append(
            {
                "field": "atlasPath",
                "severity": "error",
                "message": "Atlas annotation file is missing or not found.",
            }
        )
    if not struct or not Path(struct).exists():
        issues.append(
            {
                "field": "structPath",
                "severity": "error",
                "message": "Structure mapping file is missing or not found.",
            }
        )
    elif Path(struct).suffix.lower() not in {".csv", ".json"}:
        issues.append(
            {
                "field": "structPath",
                "severity": "error",
                "message": "Structure mapping file must be .csv or .json.",
            }
        )
    return issues


def _collect_preflight_issues(payload: dict) -> tuple[dict, list[dict[str, str]], Path]:
    config_path = _resolve_config_path(payload.get("configPath"))
    cfg = _apply_runtime_overrides(load_config(config_path), payload)
    issues = collect_runtime_config_issues(cfg, require_input_dir=True)
    issues.extend(_runtime_path_issues(payload))
    return cfg, issues, config_path


def _materialize_runtime_config(payload: dict, *, job_id: str = ctx.DEFAULT_JOB_ID) -> Path:
    cfg, _issues, _config_path = _collect_preflight_issues(payload)
    runtime_dir = _job_outputs_dir(job_id) / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_path = runtime_dir / f"run_config_{stamp}.json"
    runtime_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return runtime_path


@bp.get("/")
def index():
    return send_from_directory(ctx.ROOT, "index.html")


@bp.get("/api/info")
def info():
    default_atlas = ctx.PROJECT_ROOT / "annotation_25.nii.gz"
    default_struct = ctx.DEFAULT_STRUCTURE_SOURCE
    version = _read_version_info()
    return jsonify(
        {
            "app": "BrainfastUI",
            "version": version["version"],
            "buildDate": version["build_date"],
            "commit": version["commit"],
            "frontend": str(ctx.ROOT),
            "project": str(ctx.PROJECT_ROOT),
            "outputs": str(ctx.OUTPUT_DIR),
            "defaults": {
                "atlasPath": str(default_atlas),
                "structPath": str(default_struct) if default_struct.exists() else "",
                "sampleLimit": int(ctx.MAX_CALIB_SAMPLES),
            },
        }
    )


@bp.get("/api/config-schema")
def config_schema():
    """Serve the run_config JSON Schema (Draft-07).

    Frontend can use this for live config validation or to render a schema viewer.
    """
    schema_path = ctx.PROJECT_ROOT / "configs" / "run_config.schema.json"
    if not schema_path.exists():
        return jsonify(
            {"ok": False, "error": "Schema file not found.", "error_code": "NOT_FOUND"}
        ), 404
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        return jsonify({"ok": True, "schema": schema})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "error_code": ERR_INTERNAL}), 500


@bp.get("/api/validate")
def validate():
    payload = {
        "inputDir": request.args.get("inputDir", ""),
        "atlasPath": request.args.get("atlasPath", ""),
        "structPath": request.args.get("structPath", ""),
    }
    field_issues = _runtime_path_issues(payload)
    return jsonify(
        {
            "ok": not any(issue["severity"] == "error" for issue in field_issues),
            "issues": [issue["message"] for issue in field_issues],
            "fieldIssues": field_issues,
        }
    )


@bp.post("/api/pipeline/preflight")
def preflight():
    payload = request.get_json(force=True) or {}
    try:
        _cfg, issues, _config_path = _collect_preflight_issues(payload)
    except Exception as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "issues": [
                        {
                            "field": "config",
                            "severity": "error",
                            "message": f"Failed to load config for preflight: {exc}",
                        }
                    ],
                }
            ),
            400,
        )
    ok = not any(issue["severity"] == "error" for issue in issues)
    return jsonify({"ok": ok, "issues": issues})


@bp.post("/api/run")
def run_pipeline():
    payload = request.get_json(force=True) or {}
    job_id = _resolve_job_id(payload)
    if not payload.get("jobId"):
        job_id = _make_job_id()
        payload["jobId"] = job_id
    job_state = _job_state(job_id)

    with ctx._run_state_lock:
        if job_state["running"]:
            return jsonify(
                {
                    "ok": False,
                    "error": "pipeline already running",
                    "error_code": ERR_PIPELINE_RUNNING,
                    "jobId": job_id,
                }
            ), 409

    try:
        config = str(_materialize_runtime_config(payload, job_id=job_id))
    except PermissionError as exc:
        return jsonify({"ok": False, "error": str(exc), "error_code": ERR_CONFIG_PATH_DENIED}), 403
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": f"failed to build runtime config: {exc}",
                "error_code": ERR_INVALID_INPUT,
            }
        ), 400
    input_dir = payload.get("inputDir", "")
    channels = payload.get("channels", ["red"])
    if isinstance(channels, str):
        channels = [channels]

    run_params = payload.get("params", {})
    # BUG-1 fix: set running=True inside the lock BEFORE starting the thread,
    # so concurrent requests see the correct state and cannot bypass the 409 check.
    with ctx._run_state_lock:
        job_state["config_path"] = config
        job_state["running"] = True
    t = threading.Thread(
        target=ctx._runner,
        args=(config, input_dir, channels, run_params),
        kwargs={"job_id": job_id},
        daemon=True,
    )
    t.start()
    return jsonify(
        {"ok": True, "started": True, "jobId": job_id, "outputsDir": str(_job_outputs_dir(job_id))}
    )


@bp.get("/api/status")
def status():
    job_id = _resolve_job_id()
    job_state = _job_state(job_id)
    progress = dict(job_state.get("progress", {}) or {})
    outputs_dir = _job_outputs_dir(job_id)
    slices_done = int(progress.get("slicesDone", 0) or 0)
    slices_total = int(progress.get("slicesTotal", 0) or 0)
    if slices_done <= 0 and slices_total <= 0:
        reg_dir = outputs_dir / "registered_slices"
        slices_done = len(list(reg_dir.glob("slice_*_overlay.png"))) if reg_dir.exists() else 0
        merged_dir = outputs_dir / "tmp_merged"
        channel_dir = outputs_dir / "tmp_channel"
        merged_total = len(list(merged_dir.glob("*.tif"))) if merged_dir.exists() else 0
        channel_total = len(list(channel_dir.glob("*.tif"))) if channel_dir.exists() else 0
        sampling_mode = ""
        config_path = job_state.get("config_path")
        if config_path:
            try:
                cfg = json.loads(Path(str(config_path)).read_text(encoding="utf-8-sig"))
                sampling_mode = str(cfg.get("input", {}).get("sampling_mode", "")).lower().strip()
            except Exception:
                sampling_mode = ""
        if sampling_mode in {"single", "native", "raw", "original"}:
            slices_total = channel_total
        elif sampling_mode in {"merge", "merged", "merge5", "merge_n"}:
            slices_total = merged_total
        else:
            slices_total = merged_total if merged_total > 0 else channel_total
    return jsonify(
        {
            "jobId": job_id,
            "outputsDir": str(outputs_dir),
            "running": job_state["running"],
            "done": job_state["done"],
            "error": job_state["error"],
            "channels": job_state["channels"],
            "currentChannel": job_state["current_channel"],
            "logCount": len(job_state["logs"]),
            "slicesDone": slices_done,
            "slicesTotal": slices_total,
            "startEpoch": job_state.get("startEpoch"),
            "progress": {
                "phase": str(progress.get("phase", "idle")),
                "stepCurrent": int(progress.get("stepCurrent", 0) or 0),
                "stepTotal": int(progress.get("stepTotal", 0) or 0),
                "message": str(progress.get("message", "")),
            },
        }
    )


@bp.post("/api/cancel")
def cancel():
    payload = request.get_json(silent=True) or {}
    job_id = _resolve_job_id(payload)
    job_state = _job_state(job_id)
    with ctx._run_state_lock:
        p = job_state.get("proc")
        if p and job_state.get("running"):
            p.terminate()
            job_state["error"] = "cancelled by user"
            job_state["running"] = False
            job_state["done"] = False
            job_state["current_channel"] = None
            job_state["channels"] = []
            job_state["startEpoch"] = None
            job_state.setdefault("progress", {})["phase"] = "cancelled"
            ctx._append_log("[cancel] user requested stop", state=job_state)
            ctx._append_error(
                "Pipeline cancelled by user.", step="cancel", recoverable=True, state=job_state
            )
            return jsonify({"ok": True, "cancelled": True, "jobId": job_id})
    return jsonify(
        {"ok": False, "cancelled": False, "error": "no running process", "jobId": job_id}
    ), 409


@bp.get("/api/logs")
def logs():
    job_id = _resolve_job_id()
    job_state = _job_state(job_id)
    return jsonify({"jobId": job_id, "logs": job_state["logs"]})


@bp.get("/api/error-log")
def error_log():
    job_id = _resolve_job_id()
    job_state = _job_state(job_id)
    errors = list(job_state.get("errors", []) or [])
    return jsonify({"ok": True, "jobId": job_id, "errors": errors, "count": len(errors)})


@bp.get("/api/history")
def history():
    job_id = _resolve_job_id()
    job_state = _job_state(job_id)
    return jsonify({"jobId": job_id, "history": job_state["history"]})


@bp.get("/api/poll")
def poll():
    """Composite poll endpoint — replaces three independent polling loops.

    Returns a single JSON object containing:
      running, done, error, progress, slicesDone, slicesTotal,
      logTail (last 20 lines), errors (structured error list).

    Frontend polls this at 500ms when active, 30s when idle.
    """
    job_id = _resolve_job_id()
    job_state = _job_state(job_id)
    progress = dict(job_state.get("progress", {}) or {})
    outputs_dir = _job_outputs_dir(job_id)

    # Slice progress (same logic as /api/status)
    slices_done = int(progress.get("slicesDone", 0) or 0)
    slices_total = int(progress.get("slicesTotal", 0) or 0)
    if slices_done <= 0 and slices_total <= 0:
        reg_dir = outputs_dir / "registered_slices"
        slices_done = len(list(reg_dir.glob("slice_*_overlay.png"))) if reg_dir.exists() else 0
        merged_dir = outputs_dir / "tmp_merged"
        channel_dir = outputs_dir / "tmp_channel"
        merged_total = len(list(merged_dir.glob("*.tif"))) if merged_dir.exists() else 0
        channel_total = len(list(channel_dir.glob("*.tif"))) if channel_dir.exists() else 0
        config_path = job_state.get("config_path")
        sampling_mode = ""
        if config_path:
            try:
                cfg = json.loads(Path(str(config_path)).read_text(encoding="utf-8-sig"))
                sampling_mode = str(cfg.get("input", {}).get("sampling_mode", "")).lower().strip()
            except Exception:
                sampling_mode = ""
        if sampling_mode in {"single", "native", "raw", "original"}:
            slices_total = channel_total
        elif sampling_mode in {"merge", "merged", "merge5", "merge_n"}:
            slices_total = merged_total
        else:
            slices_total = merged_total if merged_total > 0 else channel_total

    logs = job_state.get("logs", [])
    log_tail = logs[-20:] if len(logs) > 20 else list(logs)
    errors = list(job_state.get("errors", []) or [])

    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "running": job_state["running"],
            "done": job_state["done"],
            "error": job_state["error"],
            "slicesDone": slices_done,
            "slicesTotal": slices_total,
            "logTail": log_tail,
            "errors": errors,
            "progress": {
                "phase": str(progress.get("phase", "idle")),
                "stepCurrent": int(progress.get("stepCurrent", 0) or 0),
                "stepTotal": int(progress.get("stepTotal", 0) or 0),
                "message": str(progress.get("message", "")),
            },
        }
    )


@bp.get("/api/export/methods-text")
def export_methods_text():
    job_id = _resolve_job_id()
    outputs_dir = _job_outputs_dir(job_id)
    params_files = sorted(outputs_dir.glob("run_params_*.json"), reverse=True)
    params = {}
    if params_files:
        try:
            params = json.loads(params_files[0].read_text(encoding="utf-8"))
        except Exception:
            pass
    detection_summary = {}
    detection_summary_path = outputs_dir / "detection_summary.json"
    if detection_summary_path.exists():
        try:
            detection_summary = json.loads(detection_summary_path.read_text(encoding="utf-8"))
        except Exception:
            detection_summary = {}
    align_mode = params.get("alignMode", "affine")
    align_cn = "仿射变换 (affine)" if align_mode == "affine" else "非线性变换 (nonlinear/TPS)"
    align_en = "affine" if align_mode == "affine" else "nonlinear (thin-plate spline)"
    pixel_size = params.get("pixelSizeUm", "0.65")
    channels = params.get("channels", ["red"])
    ch_str = ", ".join(channels)
    ts = params.get("timestamp", "—")
    atlas_sha256 = detection_summary.get("atlas_sha256", "")
    detector_counts = detection_summary.get("dedup_detector_counts") or detection_summary.get(
        "detector_counts", {}
    )
    detector_names = ", ".join(detector_counts.keys()) if detector_counts else "configured detector"
    sampling_mode = str(detection_summary.get("sampling_mode", "single"))
    if sampling_mode == "merged":
        sampling_cn = "合并切片"
        sampling_en = "merged slices"
    else:
        sampling_cn = "原始单切片"
        sampling_en = "native single slices"
    if any(str(name).startswith("cellpose") for name in detector_counts):
        detection_cn = (
            f"细胞实例分割与计数采用 {detector_names}，并基于{sampling_cn}完成去重和图谱统计。"
        )
        detection_en = (
            f"Cell counting used instance segmentation via {detector_names} on {sampling_en}, "
            f"followed by deduplication and hierarchical atlas aggregation."
        )
    else:
        detection_cn = f"细胞计数采用 {detector_names}，并基于{sampling_cn}完成去重和图谱统计。"
        detection_en = (
            f"Cell counting used {detector_names} on {sampling_en}, followed by deduplication "
            f"and hierarchical atlas aggregation."
        )
    text_cn = (
        f"【方法段落参考（中文）】\n"
        f"脑图谱配准使用 Brainfast v0.3 完成（运行时间：{ts}）。"
        f"显微图像分辨率为 {pixel_size} μm/像素。"
        f"图谱配准参照 Allen 小鼠脑图谱（CCFv3，annotation_25.nii.gz，体素间距 25 μm"
        + (f"，sha256: {atlas_sha256}" if atlas_sha256 else "")
        + f"），"
        f"采用{align_cn}方法对切片进行空间配准。配准质量通过边缘 SSIM（结构相似性指标）评估。"
        f"{detection_cn}荧光通道：{ch_str}。"
    )
    text_en = (
        f"\n【Methods paragraph reference (English)】\n"
        f"Brain atlas registration was performed using Brainfast v0.3 (run: {ts}). "
        f"Microscopy images were acquired at {pixel_size} μm/pixel. "
        f"Section registration was carried out against the Allen Mouse Brain Atlas "
        f"(CCFv3, annotation_25.nii.gz, 25 μm voxel spacing"
        + (f", sha256: {atlas_sha256}" if atlas_sha256 else "")
        + f") using {align_en} transformation. "
        f"Alignment quality was evaluated by edge-SSIM. "
        f"{detection_en} Channels: {ch_str}."
    )
    return jsonify({"ok": True, "jobId": job_id, "text": text_cn + text_en, "params": params})
