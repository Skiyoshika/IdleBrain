from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import project.frontend.server_context as ctx
from project.frontend.server import app
from project.scripts.paths import RunPaths


def _minimal_cfg(input_dir: Path) -> dict:
    return {
        "project": {"name": "test_project"},
        "input": {
            "slice_dir": str(input_dir),
            "slice_glob": "z*.tif",
            "pixel_size_um_xy": 5.0,
            "slice_spacing_um": 25.0,
            "channel_map": {"red": 0},
            "active_channel": "red",
        },
        "registration": {"atlas_z_refine_range": 0},
        "detection": {"primary_model": "fallback"},
        "dedup": {"neighbor_slices": 1, "r_xy_um": 8.0},
        "outputs": {
            "leaf_csv": "outputs/leaf.csv",
            "hierarchy_csv": "outputs/hierarchy.csv",
            "qc_dir": "outputs/qc",
        },
    }


def test_preflight_returns_structured_warning(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    atlas_path = tmp_path / "annotation_25.nii.gz"
    atlas_path.write_text("atlas", encoding="utf-8")
    struct_path = tmp_path / "allen.csv"
    struct_path.write_text("id,name\n1,root\n", encoding="utf-8")
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(_minimal_cfg(input_dir)), encoding="utf-8")
    # OUTPUT_DIR must contain the config_path so _resolve_config_path() containment check passes
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    payload = {
        "configPath": str(config_path),
        "inputDir": str(input_dir),
        "atlasPath": str(atlas_path),
        "structPath": str(struct_path),
        "channels": ["red"],
        "params": {
            "pixelSizeUm": "5.0",
            "alignMode": "affine",
            "atlasPath": str(atlas_path),
            "structPath": str(struct_path),
        },
    }

    with app.test_client() as client:
        resp = client.post("/api/pipeline/preflight", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert any(
            issue["field"] == "registration.atlas_z_refine_range"
            and issue["severity"] == "warning"
            for issue in data["issues"]
        )


def test_preflight_returns_structured_error_for_invalid_runtime_config(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    atlas_path = tmp_path / "annotation_25.nii.gz"
    atlas_path.write_text("atlas", encoding="utf-8")
    struct_path = tmp_path / "allen.csv"
    struct_path.write_text("id,name\n1,root\n", encoding="utf-8")
    cfg = _minimal_cfg(input_dir)
    cfg["input"]["pixel_size_um_xy"] = 0
    config_path = tmp_path / "bad_run_config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    # OUTPUT_DIR must contain the config_path so containment check passes
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    payload = {
        "configPath": str(config_path),
        "inputDir": str(input_dir),
        "atlasPath": str(atlas_path),
        "structPath": str(struct_path),
        "channels": ["red"],
        "params": {
            "pixelSizeUm": "",
            "alignMode": "affine",
            "atlasPath": str(atlas_path),
            "structPath": str(struct_path),
        },
    }

    with app.test_client() as client:
        resp = client.post("/api/pipeline/preflight", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is False
        assert any(
            issue["field"] == "input.pixel_size_um_xy" and issue["severity"] == "error"
            for issue in data["issues"]
        )


def test_error_log_and_status_expose_structured_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    job_id = "job_test123"
    monkeypatch.setattr(
        ctx,
        "_job_states",
        {
            ctx.DEFAULT_JOB_ID: ctx.run_state,
            job_id: {
            "running": True,
            "done": False,
            "error": None,
            "logs": ["[PROGRESS:step=3/6:phase=registration:slices=4/12] Processing"],
            "errors": [
                {
                    "timestamp": "2026-03-19T10:00:00",
                    "message": "Registration score dropped below threshold.",
                    "step": "registration",
                    "recoverable": False,
                    "source": "backend",
                }
            ],
            "channels": ["red"],
            "proc": None,
            "current_channel": "red",
            "history": [],
            "config_path": None,
            "startEpoch": 1234567890,
            "job_id": job_id,
            "outputs_dir": str(tmp_path / "jobs" / job_id),
            "progress": {
                "phase": "registration",
                "stepCurrent": 3,
                "stepTotal": 6,
                "slicesDone": 4,
                "slicesTotal": 12,
                "message": "Processing slice 4 / 12",
            },
            },
        },
    )

    with app.test_client() as client:
        error_resp = client.get(f"/api/error-log?job={job_id}")
        assert error_resp.status_code == 200
        error_data = error_resp.get_json()
        assert error_data["count"] == 1
        assert error_data["errors"][0]["step"] == "registration"

        status_resp = client.get(f"/api/status?job={job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.get_json()
        assert status_data["jobId"] == job_id
        assert status_data["progress"]["phase"] == "registration"
        assert status_data["progress"]["stepCurrent"] == 3
        assert status_data["slicesDone"] == 4
        assert status_data["slicesTotal"] == 12


def test_job_output_dirs_do_not_overlap() -> None:
    left = ctx._job_output_dir("job_alpha")
    right = ctx._job_output_dir("job_beta")
    assert left != right
    assert left.name == "job_alpha"
    assert right.name == "job_beta"


def test_runpaths_accepts_custom_outputs_dir(tmp_path: Path) -> None:
    cfg = _minimal_cfg(tmp_path / "input")
    custom_root = tmp_path / "jobs" / "job_alpha"
    paths = RunPaths.from_project_root(tmp_path, cfg, outputs_dir=custom_root)
    assert paths.outputs == custom_root
    assert paths.cells_detected == custom_root / "cells_detected.csv"
    assert paths.registered_slices == custom_root / "registered_slices"


def test_info_reads_version_json() -> None:
    with app.test_client() as client:
        resp = client.get("/api/info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["version"] == "0.5.1"
