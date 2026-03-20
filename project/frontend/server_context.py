"""server_context.py — Shared globals, helpers, and background worker functions.

All blueprint modules must access state via `import project.frontend.server_context as ctx`
and reference `ctx.run_state`, `ctx.OUTPUT_DIR`, etc. — never import names directly —
so that mutations made here propagate correctly to all blueprints.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import request
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from tifffile import imread

# ---------------------------------------------------------------------------
# Paths – populated by server.py (thin orchestrator) before any request
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent  # frontend dir
PROJECT_ROOT: Path = ROOT.parent  # project dir
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CALIB_SAMPLES = int(os.environ.get("IDLEBRAIN_MAX_CALIB_SAMPLES", "180"))
DEFAULT_STRUCTURE_SOURCE = PROJECT_ROOT / "configs" / "allen_mouse_structure_graph.csv"
DEFAULT_JOB_ID = "default"

# ---------------------------------------------------------------------------
# Shared mutable state
# ---------------------------------------------------------------------------
_autopick_tasks: dict = {}  # token -> {status, progress, step, total, message, result, error}
_preview_tasks: dict = {}  # token -> {status, progress, message, result, error}
_task_lock = threading.Lock()
_run_state_lock = threading.Lock()

run_state: dict = {
    "running": False,
    "done": False,
    "error": None,
    "logs": [],
    "errors": [],
    "channels": [],
    "proc": None,
    "current_channel": None,
    "history": [],
    "config_path": None,
    "startEpoch": None,
    "progress": {
        "phase": "idle",
        "stepCurrent": 0,
        "stepTotal": 0,
        "slicesDone": 0,
        "slicesTotal": 0,
        "message": "",
    },
}

_learning_lock = threading.Lock()
learning_state: dict = {
    "running": False,
    "started_at": "",
    "finished_at": "",
    "ok": None,
    "error": "",
    "out_json": "",
    "log_tail": [],
}

_hover_label_cache: dict = {"path": "", "mtime": 0.0, "img": None}
_hover_tree_cache: dict | None = None
_hover_parent_cache: dict = {"csv": "", "mtime": 0.0, "map": {}}
_last_preview_structure_csv: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _sanitize_job_id(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return DEFAULT_JOB_ID
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "-" for ch in text)
    safe = safe.strip("-_")
    return safe[:64] or DEFAULT_JOB_ID


def _job_output_dir(job_id: str | None) -> Path:
    safe_job_id = _sanitize_job_id(job_id)
    if safe_job_id == DEFAULT_JOB_ID:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR
    out = OUTPUT_DIR / "jobs" / safe_job_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _job_file(job_id: str | None, filename: str) -> Path:
    return _job_output_dir(job_id) / filename


def _job_manual_calibration_dir(job_id: str | None) -> Path:
    out = _job_output_dir(job_id) / "manual_calibration"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _payload_job_id(payload: dict | None = None) -> str:
    if isinstance(payload, dict):
        return _sanitize_job_id(payload.get("jobId"))
    return DEFAULT_JOB_ID


def _query_job_id() -> str:
    return _sanitize_job_id(request.args.get("jobId", ""))


def _append_log(line: str):
    run_state["logs"].append(line.rstrip())
    if len(run_state["logs"]) > 500:
        run_state["logs"] = run_state["logs"][-500:]


def _append_error(
    message: str,
    *,
    step: str = "general",
    recoverable: bool = True,
    source: str = "backend",
) -> None:
    item = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "message": str(message).strip(),
        "step": str(step or "general"),
        "recoverable": bool(recoverable),
        "source": str(source or "backend"),
    }
    run_state["errors"].append(item)
    if len(run_state["errors"]) > 200:
        run_state["errors"] = run_state["errors"][-200:]


_PROGRESS_RE = re.compile(r"^\[PROGRESS:(?P<meta>[^\]]+)\]\s*(?P<message>.*)$")


def _parse_progress_line(line: str) -> dict[str, object] | None:
    match = _PROGRESS_RE.match(str(line).strip())
    if not match:
        return None
    meta = match.group("meta")
    payload: dict[str, object] = {"message": match.group("message").strip()}
    for chunk in meta.split(":"):
        if "=" not in chunk:
            continue
        key, raw = chunk.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if key in {"step", "slices"} and "/" in raw:
            cur, total = raw.split("/", 1)
            try:
                payload[f"{key}Current"] = int(cur)
                payload[f"{key}Total"] = int(total)
            except Exception:
                continue
        else:
            payload[key] = raw
    return payload


def _apply_progress_update(parsed: dict[str, object]) -> None:
    progress = run_state.setdefault(
        "progress",
        {
            "phase": "idle",
            "stepCurrent": 0,
            "stepTotal": 0,
            "slicesDone": 0,
            "slicesTotal": 0,
            "message": "",
        },
    )
    phase = str(parsed.get("phase", progress.get("phase", "idle"))).strip() or "idle"
    progress["phase"] = phase
    progress["message"] = str(parsed.get("message", progress.get("message", "")))
    if "stepCurrent" in parsed:
        progress["stepCurrent"] = int(parsed["stepCurrent"])
    if "stepTotal" in parsed:
        progress["stepTotal"] = int(parsed["stepTotal"])
    if "slicesCurrent" in parsed:
        progress["slicesDone"] = int(parsed["slicesCurrent"])
    if "slicesTotal" in parsed:
        progress["slicesTotal"] = int(parsed["slicesTotal"])


# ---------------------------------------------------------------------------
# Training-set helpers
# ---------------------------------------------------------------------------


def _next_train_pair_id(train_dir: Path) -> int:
    n = 1
    seen = set()
    for p in train_dir.glob("*_Ori.png"):
        try:
            seen.add(int(p.stem.replace("_Ori", "")))
        except Exception:
            continue
    while n in seen:
        n += 1
    return n


def _trainset_pair_ids(train_dir: Path) -> list[int]:
    ori_ids = set()
    target_ids = set()
    for p in train_dir.glob("*_Ori.png"):
        try:
            ori_ids.add(int(p.stem.replace("_Ori", "")))
        except Exception:
            continue
    for p in train_dir.glob("*_Show.png"):
        try:
            target_ids.add(int(p.stem.replace("_Show", "")))
        except Exception:
            continue
    for p in train_dir.glob("*_Label.tif"):
        try:
            target_ids.add(int(p.stem.replace("_Label", "")))
        except Exception:
            continue
    return sorted(ori_ids & target_ids)


def _prune_trainset_if_needed(train_dir: Path, max_samples: int) -> dict:
    max_n = max(10, int(max_samples))
    ids = _trainset_pair_ids(train_dir)
    if len(ids) <= max_n:
        return {"pruned": 0, "kept": len(ids), "max": max_n}

    remove_ids = ids[: len(ids) - max_n]
    removed = 0
    calib_dir = OUTPUT_DIR / "manual_calibration"
    for sid in remove_ids:
        for suffix in ("_Ori.png", "_Show.png", "_Label.tif"):
            p = train_dir / f"{sid}{suffix}"
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        m = calib_dir / f"calibration_sample_{sid}.json"
        if m.exists():
            try:
                m.unlink()
            except Exception:
                pass
        removed += 1

    ids_after = _trainset_pair_ids(train_dir)
    return {"pruned": int(removed), "kept": len(ids_after), "max": max_n}


# ---------------------------------------------------------------------------
# Liquify / warp helpers
# ---------------------------------------------------------------------------


def _warp_label_with_flow(label: np.ndarray, flow_v: np.ndarray, flow_u: np.ndarray) -> np.ndarray:
    h, w = label.shape[:2]
    rr, cc = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    src_r = np.clip(rr + flow_v.astype(np.float32), 0.0, float(h - 1))
    src_c = np.clip(cc + flow_u.astype(np.float32), 0.0, float(w - 1))
    warped = map_coordinates(
        label.astype(np.float32),
        [src_r, src_c],
        order=0,
        mode="nearest",
    )
    return warped.astype(np.int32)


def _apply_liquify_drags(
    label: np.ndarray,
    drags: list[dict],
    tissue_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = label.astype(np.int32).copy()
    h, w = out.shape[:2]
    if h < 2 or w < 2 or not drags:
        return out

    flow_v = np.zeros((h, w), dtype=np.float32)
    flow_u = np.zeros((h, w), dtype=np.float32)

    for d in drags:
        try:
            x1 = float(d.get("x1"))
            y1 = float(d.get("y1"))
            x2 = float(d.get("x2"))
            y2 = float(d.get("y2"))
        except Exception:
            continue
        radius = float(d.get("radius", 80.0))
        strength = float(d.get("strength", 0.72))
        radius = float(np.clip(radius, 8.0, 260.0))
        strength = float(np.clip(strength, 0.05, 1.5))
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        drag_len = float(np.hypot(dx, dy))
        if drag_len < 0.75:
            continue

        sigma = max(4.0, radius * 0.46)
        influence = radius * 1.65
        y0 = max(0, int(np.floor(y1 - influence)))
        y1i = min(h, int(np.ceil(y1 + influence + 1)))
        x0 = max(0, int(np.floor(x1 - influence)))
        x1i = min(w, int(np.ceil(x1 + influence + 1)))
        if y1i <= y0 or x1i <= x0:
            continue

        yy, xx = np.meshgrid(
            np.arange(y0, y1i, dtype=np.float32),
            np.arange(x0, x1i, dtype=np.float32),
            indexing="ij",
        )
        dist2 = (yy - y1) ** 2 + (xx - x1) ** 2
        wt = np.exp(-0.5 * dist2 / (sigma * sigma)).astype(np.float32)
        wt *= float(strength)

        if tissue_mask is not None and tissue_mask.shape == out.shape:
            local_tm = tissue_mask[y0:y1i, x0:x1i].astype(np.float32)
            wt *= local_tm

        flow_u[y0:y1i, x0:x1i] += (dx * wt).astype(np.float32)
        flow_v[y0:y1i, x0:x1i] += (dy * wt).astype(np.float32)

    if not np.any(flow_u) and not np.any(flow_v):
        return out

    flow_u = gaussian_filter(flow_u, sigma=1.0).astype(np.float32)
    flow_v = gaussian_filter(flow_v, sigma=1.0).astype(np.float32)
    mag = np.hypot(flow_u, flow_v)
    max_disp = float(np.percentile(mag, 99.5))
    max_allowed = float(max(6.0, min(h, w) * 0.11))
    if max_disp > max_allowed:
        s = max_allowed / max(max_disp, 1e-6)
        flow_u *= float(s)
        flow_v *= float(s)

    warped = _warp_label_with_flow(out, flow_v=flow_v, flow_u=flow_u)
    return warped.astype(np.int32)


# ---------------------------------------------------------------------------
# Calibration save / learn
# ---------------------------------------------------------------------------


def _save_calibration_pair(
    real_path: Path,
    corrected_label_tif: Path,
    corrected_overlay_png: Path,
    manifest: dict,
    real_z_index: int | None = None,
) -> dict:
    from scripts.image_utils import norm_u8_robust
    from scripts.slice_select import select_real_slice_2d

    train_dir = PROJECT_ROOT / "train_data_set"
    train_dir.mkdir(parents=True, exist_ok=True)
    sid = _next_train_pair_id(train_dir)
    ori_png = train_dir / f"{sid}_Ori.png"
    show_png = train_dir / f"{sid}_Show.png"
    label_tif = train_dir / f"{sid}_Label.tif"

    real_raw = imread(str(real_path))
    real2d, _meta = select_real_slice_2d(real_raw, z_index=real_z_index, source_path=real_path)
    real_u8 = norm_u8_robust(real2d)
    real_rgb = np.stack([real_u8, real_u8, real_u8], axis=-1).astype(np.uint8)
    Image.fromarray(real_rgb).save(ori_png)
    shutil.copy2(corrected_overlay_png, show_png)
    shutil.copy2(corrected_label_tif, label_tif)

    calib_dir = OUTPUT_DIR / "manual_calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = calib_dir / f"calibration_sample_{sid}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    prune = _prune_trainset_if_needed(train_dir, max_samples=MAX_CALIB_SAMPLES)

    return {
        "sample_id": int(sid),
        "ori_png": str(ori_png),
        "show_png": str(show_png),
        "label_tif": str(label_tif),
        "manifest_json": str(manifest_path),
        "sample_limit": int(MAX_CALIB_SAMPLES),
        "prune": prune,
    }


def _learn_from_trainset_async():
    with _learning_lock:
        if learning_state.get("running"):
            return False
        learning_state["running"] = True  # reserve under lock before spawning thread

    def _worker():
        learning_state.update(
            {
                "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "finished_at": "",
                "ok": None,
                "error": "",
                "out_json": str(OUTPUT_DIR / "trainset_tuned_params.json"),
                "log_tail": [],
            }
        )
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "learn_from_trainset.py"),
            "--train-dir",
            str(PROJECT_ROOT / "train_data_set"),
            "--annotation",
            str(PROJECT_ROOT / "annotation_25.nii.gz"),
            "--out-json",
            str(OUTPUT_DIR / "trainset_tuned_params.json"),
            "--fit-modes",
            "contain,cover",
            "--smooth-values",
            "0,1",
            "--profiles",
            "balanced,internal_strong",
            "--sample-limit",
            str(MAX_CALIB_SAMPLES),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            lines: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    lines.append(line)
                    if len(lines) > 40:
                        lines = lines[-40:]
            code = proc.wait()
            learning_state["ok"] = bool(code == 0)
            if code != 0:
                learning_state["error"] = f"learn_from_trainset exited with code {code}"
            learning_state["log_tail"] = lines
        except Exception as e:
            learning_state["ok"] = False
            learning_state["error"] = str(e)
        finally:
            learning_state["running"] = False
            learning_state["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return True


# ---------------------------------------------------------------------------
# Hover / label cache helpers
# ---------------------------------------------------------------------------


def _load_hover_label(job_id: str | None = None) -> np.ndarray | None:
    global _hover_label_cache
    hover_label_path = _job_file(job_id, "overlay_label_preview.tif")
    if not hover_label_path.exists():
        return None
    mtime = float(hover_label_path.stat().st_mtime)
    if (
        _hover_label_cache.get("img") is None
        or _hover_label_cache.get("path") != str(hover_label_path)
        or float(_hover_label_cache.get("mtime", 0.0)) != mtime
    ):
        arr = imread(str(hover_label_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        _hover_label_cache = {"path": str(hover_label_path), "mtime": mtime, "img": arr}
    return _hover_label_cache.get("img")


def _load_hover_structure_tree() -> dict:
    global _hover_tree_cache
    if _hover_tree_cache is not None:
        return _hover_tree_cache
    candidates = [
        PROJECT_ROOT / "configs" / "allen_structure_tree.json",
        PROJECT_ROOT / "scripts" / "allen_structure_tree.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                _hover_tree_cache = json.loads(p.read_text(encoding="utf-8"))
                return _hover_tree_cache
            except Exception:
                continue
    _hover_tree_cache = {}
    return _hover_tree_cache


def _build_parent_name_map(structure_csv_path: str) -> dict[int, str]:
    global _hover_parent_cache
    if not structure_csv_path:
        return {}
    p = Path(structure_csv_path)
    if not p.exists():
        return {}

    mtime = float(p.stat().st_mtime)
    if (
        _hover_parent_cache.get("csv") == str(p)
        and float(_hover_parent_cache.get("mtime", 0.0)) == mtime
    ):
        return _hover_parent_cache.get("map", {})

    if p.suffix.lower() == ".json":
        try:
            tree = json.loads(p.read_text(encoding="utf-8"))
            out: dict[int, str] = {}
            if isinstance(tree, dict):
                for raw_key, info in tree.items():
                    if not isinstance(info, dict):
                        continue
                    try:
                        rid = int(info.get("id", raw_key))
                    except Exception:
                        continue
                    parent_name = str(info.get("parent_name", "")).strip()
                    if not parent_name:
                        parent_id = info.get("parent_structure_id", info.get("parent_id"))
                        if parent_id is not None:
                            parent_info = tree.get(str(parent_id), {})
                            if isinstance(parent_info, dict):
                                parent_name = str(parent_info.get("name", "")).strip()
                    if parent_name:
                        out[rid] = parent_name
            _hover_parent_cache = {"csv": str(p), "mtime": mtime, "map": out}
            return out
        except Exception:
            _hover_parent_cache = {"csv": str(p), "mtime": mtime, "map": {}}
            return {}

    try:
        df = pd.read_csv(p)
    except Exception:
        _hover_parent_cache = {"csv": str(p), "mtime": mtime, "map": {}}
        return {}

    col_map = {str(c).strip().lower(): str(c) for c in df.columns}
    id_col = next(
        (col_map[k] for k in ["id", "structure_id", "region_id", "atlas_id"] if k in col_map), None
    )
    if id_col is None:
        _hover_parent_cache = {"csv": str(p), "mtime": mtime, "map": {}}
        return {}

    parent_name_col = next(
        (
            col_map[k]
            for k in ["parent_name", "parent_region_name", "parent_structure_name"]
            if k in col_map
        ),
        None,
    )
    parent_id_col = next(
        (col_map[k] for k in ["parent_structure_id", "parent_id", "parent"] if k in col_map), None
    )
    name_col = next(
        (
            col_map[k]
            for k in ["name", "region_name", "structure_name", "safe_name"]
            if k in col_map
        ),
        None,
    )

    out: dict[int, str] = {}
    try:
        if parent_name_col is not None:
            tmp = df[[id_col, parent_name_col]].dropna()
            for _, r in tmp.iterrows():
                rid = int(r[id_col])
                pnm = str(r[parent_name_col]).strip()
                if pnm:
                    out[rid] = pnm
        elif parent_id_col is not None and name_col is not None:
            name_map = {}
            for _, r in df[[id_col, name_col]].dropna().iterrows():
                name_map[int(r[id_col])] = str(r[name_col]).strip()
            for _, r in df[[id_col, parent_id_col]].dropna().iterrows():
                rid = int(r[id_col])
                pid = int(r[parent_id_col])
                pnm = name_map.get(pid, "")
                if pnm:
                    out[rid] = pnm
    except Exception:
        out = {}

    _hover_parent_cache = {"csv": str(p), "mtime": mtime, "map": out}
    return out


# ---------------------------------------------------------------------------
# Pipeline runner thread
# ---------------------------------------------------------------------------


def _runner(config_path: str, input_dir: str, channels: list[str], run_params=None):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_state.update(
        {
            "running": True,
            "done": False,
            "error": None,
            "errors": [],
            "channels": channels,
            "logs": [],
            "startTime": ts,
            "startEpoch": time.time(),
            "progress": {
                "phase": "queued",
                "stepCurrent": 0,
                "stepTotal": 0,
                "slicesDone": 0,
                "slicesTotal": 0,
                "message": "Queued...",
            },
        }
    )

    # Save run params for reproducibility / paper methods section
    if run_params:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        params_to_save = dict(run_params)
        params_to_save["timestamp"] = ts
        params_to_save["channels"] = channels
        try:
            (OUTPUT_DIR / f"run_params_{ts}.json").write_text(
                json.dumps(params_to_save, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    try:
        for ch in channels:
            run_state["current_channel"] = ch
            _append_log(f"[run] channel={ch}")
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "main.py"),
                "--config",
                config_path,
                "--run-real-input",
                input_dir,
            ]
            env = os.environ.copy()
            env["BRAINCOUNT_ACTIVE_CHANNEL"] = ch

            p = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            run_state["proc"] = p
            assert p.stdout is not None
            for line in p.stdout:
                parsed = _parse_progress_line(line)
                if parsed is not None:
                    _apply_progress_update(parsed)
                _append_log(line)
            code = p.wait()
            _append_log(f"[exit] channel={ch} code={code}")
            if code == 0:
                leaf = OUTPUT_DIR / "cell_counts_leaf.csv"
                if leaf.exists():
                    shutil.copy2(leaf, OUTPUT_DIR / f"cell_counts_leaf_{ch}.csv")
            else:
                run_state["error"] = f"channel {ch} failed with code {code}"
                _append_error(
                    run_state["error"],
                    step=str(run_state.get("progress", {}).get("phase", "pipeline")),
                    recoverable=False,
                )
                break

        run_state["done"] = run_state["error"] is None
        if run_state["done"]:
            _apply_progress_update(
                {
                    "phase": "done",
                    "message": "Pipeline completed",
                    "stepCurrent": run_state.get("progress", {}).get("stepTotal", 0),
                    "stepTotal": run_state.get("progress", {}).get("stepTotal", 0),
                    "slicesCurrent": run_state.get("progress", {}).get("slicesTotal", 0),
                    "slicesTotal": run_state.get("progress", {}).get("slicesTotal", 0),
                }
            )
        run_state["history"].append(
            {
                "channels": channels,
                "ok": run_state["error"] is None,
                "error": run_state["error"],
                "logCount": len(run_state["logs"]),
                "timestamp": ts,
            }
        )
        if len(run_state["history"]) > 20:
            run_state["history"] = run_state["history"][-20:]
    except Exception as exc:
        run_state["error"] = f"pipeline crashed: {exc}"
        run_state["done"] = False
        _append_log(f"[crash] {exc}")
        _append_error(
            run_state["error"],
            step=str(run_state.get("progress", {}).get("phase", "pipeline")),
            recoverable=False,
        )
    finally:
        run_state["running"] = False
        run_state["proc"] = None
        run_state["current_channel"] = None
        run_state["startEpoch"] = None
        if run_state.get("error"):
            run_state.setdefault("progress", {})["phase"] = "error"


# ---------------------------------------------------------------------------
# Task cleanup
# ---------------------------------------------------------------------------

_MAX_PREVIEW_CSV_CACHE = 50


def _cleanup_stale_tasks(max_age_s: float = 3600.0) -> None:
    """Remove finished tasks older than *max_age_s* seconds (default 1 hour)."""
    cutoff = time.time() - max_age_s
    with _task_lock:
        for tasks_dict in (_autopick_tasks, _preview_tasks):
            for token in list(tasks_dict.keys()):
                t = tasks_dict[token]
                if t.get("status") in ("done", "error") and t.get("_finished_at", 0.0) < cutoff:
                    del tasks_dict[token]
        # Cap _last_preview_structure_csv to prevent unbounded growth
        if len(_last_preview_structure_csv) > _MAX_PREVIEW_CSV_CACHE:
            overflow = len(_last_preview_structure_csv) - _MAX_PREVIEW_CSV_CACHE
            for old_key in list(_last_preview_structure_csv.keys())[:overflow]:
                del _last_preview_structure_csv[old_key]


# ---------------------------------------------------------------------------
# Atlas autopick worker thread
# ---------------------------------------------------------------------------


def _run_autopick_worker(token: str, real_path, annotation_path, out_label, kwargs: dict):
    from scripts.atlas_autopick import autopick_best_z

    task = _autopick_tasks[token]

    def cb(step: int, total: int, msg: str):
        if task.get("cancel_requested"):
            raise RuntimeError("cancelled by user")
        task["step"] = step
        task["total"] = total
        task["message"] = msg
        task["progress"] = int(step * 100 // max(total, 1))

    cb(1, 100, "Loading images...")
    try:
        task["status"] = "running"
        res = autopick_best_z(real_path, annotation_path, out_label, progress_cb=cb, **kwargs)
        if task.get("cancel_requested"):
            task.update(
                {
                    "status": "cancelled",
                    "message": "Cancelled by user.",
                    "_finished_at": time.time(),
                }
            )
            return
        task.update(
            {
                "status": "done",
                "progress": 100,
                "message": "Done!",
                "result": res,
                "_finished_at": time.time(),
            }
        )
    except Exception as e:
        if "cancelled by user" in str(e).lower():
            task.update(
                {
                    "status": "cancelled",
                    "message": "Cancelled by user.",
                    "_finished_at": time.time(),
                }
            )
            return
        task.update(
            {
                "status": "error",
                "error": str(e),
                "message": "Failed: " + str(e),
                "_finished_at": time.time(),
            }
        )
    finally:
        _cleanup_stale_tasks()


# ---------------------------------------------------------------------------
# Overlay preview worker thread
# ---------------------------------------------------------------------------


def _run_preview_worker(token: str, kwargs: dict, job_id: str, structure_csv):
    from scripts.overlay_render import render_overlay

    task = _preview_tasks[token]
    try:
        task["status"] = "running"
        task["progress"] = 10
        task["message"] = "Rendering overlay..."
        _, diagnostic = render_overlay(**kwargs)
        task["progress"] = 90
        task["message"] = "Finalizing..."
        # store structure_csv path for hover queries
        if structure_csv is not None:
            _last_preview_structure_csv[job_id] = str(structure_csv)
        out_png = kwargs.get("out_png") or kwargs.get("out")
        task.update(
            {
                "status": "done",
                "progress": 100,
                "message": "Done!",
                "_finished_at": time.time(),
                "result": {
                    "ok": True,
                    "preview": str(out_png) if out_png else "",
                    "diagnostic": diagnostic,
                    "jobId": job_id,
                },
            }
        )
    except Exception as e:
        task.update(
            {
                "status": "error",
                "error": str(e),
                "message": "Failed: " + str(e),
                "failCase": str(e),
                "_finished_at": time.time(),
            }
        )
    finally:
        _cleanup_stale_tasks()


def _active_reg_cfg() -> dict:
    """Return the registration section of the active config, or {} if unavailable.

    Priority order:
      1. Last config used by run_pipeline (stored in run_state["config_path"])
      2. run_config.template.json fallback
    """
    try:
        import json as _json

        cfg_candidates = []
        active = run_state.get("config_path")
        if active:
            cfg_candidates.append(Path(active))
        cfg_candidates.append(PROJECT_ROOT / "configs" / "run_config.template.json")
        for p in cfg_candidates:
            if p.exists():
                data = _json.loads(p.read_text(encoding="utf-8-sig"))
                return data.get("registration", {})
    except Exception:
        pass
    return {}
