"""api_projects.py - Project and sample management endpoints.

Routes
------
GET  /api/projects                 – list all projects
POST /api/projects                 – create project  {name, description?}
GET  /api/projects/<id>            – get project detail + samples
DELETE /api/projects/<id>          – delete project and all its samples

GET  /api/projects/<id>/samples    – list samples for a project
POST /api/projects/<id>/samples    – add sample  {name, configPath, inputDir, outputsDir?}
GET  /api/samples/<sid>            – get single sample
PATCH /api/samples/<sid>           – update sample fields
DELETE /api/samples/<sid>          – delete sample
GET  /api/samples/<sid>/events     – run event log
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from project.frontend.api_errors import (
    ERR_INVALID_INPUT,
    ERR_NOT_FOUND,
)
from project.frontend.services import project_manager as pm

bp = Blueprint("api_projects", __name__)


# ── Projects ─────────────────────────────────────────────────────────────────


@bp.get("/api/projects")
def projects_list():
    return jsonify({"ok": True, "projects": pm.list_projects()})


@bp.post("/api/projects")
def projects_create():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    if not name:
        return jsonify(
            {"ok": False, "error": "name is required", "error_code": ERR_INVALID_INPUT}
        ), 400
    description = str(body.get("description", ""))
    project = pm.create_project(name, description)
    return jsonify({"ok": True, "project": project}), 201


@bp.get("/api/projects/<project_id>")
def projects_get(project_id: str):
    project = pm.get_project(project_id)
    if not project:
        return jsonify(
            {"ok": False, "error": "project not found", "error_code": ERR_NOT_FOUND}
        ), 404
    samples = pm.list_samples(project_id)
    return jsonify({"ok": True, "project": project, "samples": samples})


@bp.delete("/api/projects/<project_id>")
def projects_delete(project_id: str):
    deleted = pm.delete_project(project_id)
    if not deleted:
        return jsonify(
            {"ok": False, "error": "project not found", "error_code": ERR_NOT_FOUND}
        ), 404
    return jsonify({"ok": True})


# ── Samples ───────────────────────────────────────────────────────────────────


@bp.get("/api/projects/<project_id>/samples")
def samples_list(project_id: str):
    if not pm.get_project(project_id):
        return jsonify(
            {"ok": False, "error": "project not found", "error_code": ERR_NOT_FOUND}
        ), 404
    return jsonify({"ok": True, "samples": pm.list_samples(project_id)})


@bp.post("/api/projects/<project_id>/samples")
def samples_add(project_id: str):
    if not pm.get_project(project_id):
        return jsonify(
            {"ok": False, "error": "project not found", "error_code": ERR_NOT_FOUND}
        ), 404
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    if not name:
        return jsonify(
            {"ok": False, "error": "name is required", "error_code": ERR_INVALID_INPUT}
        ), 400
    sample = pm.add_sample(
        project_id,
        name=name,
        config_path=str(body.get("configPath", "")),
        input_dir=str(body.get("inputDir", "")),
        outputs_dir=str(body.get("outputsDir", "")),
    )
    return jsonify({"ok": True, "sample": sample}), 201


@bp.get("/api/samples/<sample_id>")
def sample_get(sample_id: str):
    sample = pm.get_sample(sample_id)
    if not sample:
        return jsonify({"ok": False, "error": "sample not found", "error_code": ERR_NOT_FOUND}), 404
    return jsonify({"ok": True, "sample": sample})


@bp.delete("/api/samples/<sample_id>")
def sample_delete(sample_id: str):
    deleted = pm.delete_sample(sample_id)
    if not deleted:
        return jsonify({"ok": False, "error": "sample not found", "error_code": ERR_NOT_FOUND}), 404
    return jsonify({"ok": True})


@bp.get("/api/samples/<sample_id>/events")
def sample_events(sample_id: str):
    limit = min(500, max(1, int(request.args.get("limit", 100))))
    events = pm.get_run_events(sample_id, limit=limit)
    return jsonify({"ok": True, "events": events})
