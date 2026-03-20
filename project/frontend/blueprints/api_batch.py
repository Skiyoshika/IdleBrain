"""api_batch.py - Batch processing queue endpoints.

Routes
------
POST /api/batch/enqueue    – add sample to batch queue
GET  /api/batch/status     – current queue status (active + pending items)
POST /api/batch/cancel     – remove a queued (not yet running) sample
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from project.frontend.api_errors import (
    ERR_INVALID_INPUT,
    ERR_NOT_FOUND,
)
from project.frontend.services import batch_queue as bq
from project.frontend.services import project_manager as pm

bp = Blueprint("api_batch", __name__)


@bp.post("/api/batch/enqueue")
def batch_enqueue():
    """Add a sample run to the batch queue.

    Body JSON::

        {
          "sampleId":   "uuid-...",
          "configPath": "/path/to/config.json",
          "inputDir":   "/path/to/slices",
          "channels":   ["red"],       // optional, default ["red"]
          "runParams":  {}             // optional extra params
        }
    """
    body = request.get_json(silent=True) or {}
    sample_id = str(body.get("sampleId", "")).strip()
    config_path = str(body.get("configPath", "")).strip()
    input_dir = str(body.get("inputDir", "")).strip()

    if not sample_id:
        return jsonify(
            {"ok": False, "error": "sampleId is required", "error_code": ERR_INVALID_INPUT}
        ), 400
    if not config_path:
        return jsonify(
            {"ok": False, "error": "configPath is required", "error_code": ERR_INVALID_INPUT}
        ), 400
    if not input_dir:
        return jsonify(
            {"ok": False, "error": "inputDir is required", "error_code": ERR_INVALID_INPUT}
        ), 400

    if not pm.get_sample(sample_id):
        return jsonify({"ok": False, "error": "sample not found", "error_code": ERR_NOT_FOUND}), 404

    channels = body.get("channels", ["red"])
    if not isinstance(channels, list) or not channels:
        channels = ["red"]

    job_id = bq.enqueue_sample(
        sample_id,
        config_path=config_path,
        input_dir=input_dir,
        channels=[str(c) for c in channels],
        run_params=body.get("runParams"),
    )
    return jsonify({"ok": True, "jobId": job_id, "sampleId": sample_id})


@bp.get("/api/batch/status")
def batch_status():
    """Return current queue status."""
    return jsonify({"ok": True, **bq.get_queue_status()})


@bp.post("/api/batch/cancel")
def batch_cancel():
    """Remove a queued (pending, not yet running) sample from the queue.

    Body JSON::

        {"sampleId": "uuid-..."}
    """
    body = request.get_json(silent=True) or {}
    sample_id = str(body.get("sampleId", "")).strip()
    if not sample_id:
        return jsonify(
            {"ok": False, "error": "sampleId is required", "error_code": ERR_INVALID_INPUT}
        ), 400
    removed = bq.cancel_queued(sample_id)
    return jsonify({"ok": True, "removed": removed})
