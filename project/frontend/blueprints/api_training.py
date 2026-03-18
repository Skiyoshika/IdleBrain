"""api_training.py — Calibration learn-status route."""

from __future__ import annotations

from flask import Blueprint, jsonify

import project.frontend.server_context as ctx

bp = Blueprint("api_training", __name__, url_prefix="/api/calibration")


@bp.get("/learn-status")
def calibration_learn_status():
    return jsonify({"ok": True, "state": ctx.learning_state})
