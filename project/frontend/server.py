from __future__ import annotations

import os
import sys
import threading
import subprocess
import shutil
import json
import datetime
import time
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ROOT = Path(sys._MEIPASS)
    PROJECT_ROOT = Path(sys.executable).resolve().parent
    if str(PROJECT_ROOT).endswith("frontend"):
        PROJECT_ROOT = PROJECT_ROOT.parent
else:
    ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.overlay_render import render_overlay
from scripts.ai_landmark import propose_landmarks, score_alignment, score_alignment_edges
from scripts.align_ai import apply_landmark_affine
from scripts.align_nonlinear import apply_landmark_nonlinear
from scripts.compare_render import render_before_after
from scripts.atlas_autopick import autopick_best_z
from scripts.slice_select import select_real_slice_2d, select_label_slice_2d
from tifffile import imread, imwrite
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from PIL import Image

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAX_CALIB_SAMPLES = int(os.environ.get("IDLEBRAIN_MAX_CALIB_SAMPLES", "180"))

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")

run_state = {
    "running": False,
    "done": False,
    "error": None,
    "logs": [],
    "channels": [],
    "proc": None,
    "current_channel": None,
    "history": [],
}

learning_state = {
    "running": False,
    "started_at": "",
    "finished_at": "",
    "ok": None,
    "error": "",
    "out_json": "",
    "log_tail": [],
}

_HOVER_LABEL_PATH = OUTPUT_DIR / "overlay_label_preview.tif"
_hover_label_cache: dict = {"path": "", "mtime": 0.0, "img": None}
_hover_tree_cache: dict | None = None
_hover_parent_cache: dict = {"csv": "", "mtime": 0.0, "map": {}}
_last_preview_structure_csv: str = ""


def _append_log(line: str):
    run_state["logs"].append(line.rstrip())
    if len(run_state["logs"]) > 500:
        run_state["logs"] = run_state["logs"][-500:]


def _norm_u8_robust(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    if p99 <= p1:
        p1, p99 = float(np.min(x)), float(np.max(x) + 1e-6)
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (x * 255.0).astype(np.uint8)


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
    show_ids = set()
    for p in train_dir.glob("*_Ori.png"):
        try:
            ori_ids.add(int(p.stem.replace("_Ori", "")))
        except Exception:
            continue
    for p in train_dir.glob("*_Show.png"):
        try:
            show_ids.add(int(p.stem.replace("_Show", "")))
        except Exception:
            continue
    return sorted(ori_ids & show_ids)


def _prune_trainset_if_needed(train_dir: Path, max_samples: int) -> dict:
    max_n = max(10, int(max_samples))
    ids = _trainset_pair_ids(train_dir)
    if len(ids) <= max_n:
        return {"pruned": 0, "kept": len(ids), "max": max_n}

    remove_ids = ids[: len(ids) - max_n]
    removed = 0
    calib_dir = OUTPUT_DIR / "manual_calibration"
    for sid in remove_ids:
        for suffix in ("_Ori.png", "_Show.png"):
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


def _save_calibration_pair(
    real_path: Path,
    corrected_overlay_png: Path,
    manifest: dict,
    real_z_index: int | None = None,
) -> dict:
    train_dir = PROJECT_ROOT / "train_data_set"
    train_dir.mkdir(parents=True, exist_ok=True)
    sid = _next_train_pair_id(train_dir)
    ori_png = train_dir / f"{sid}_Ori.png"
    show_png = train_dir / f"{sid}_Show.png"

    real_raw = imread(str(real_path))
    real2d, _meta = select_real_slice_2d(real_raw, z_index=real_z_index, source_path=real_path)
    real_u8 = _norm_u8_robust(real2d)
    real_rgb = np.stack([real_u8, real_u8, real_u8], axis=-1).astype(np.uint8)
    Image.fromarray(real_rgb).save(ori_png)
    shutil.copy2(corrected_overlay_png, show_png)

    calib_dir = OUTPUT_DIR / "manual_calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = calib_dir / f"calibration_sample_{sid}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    prune = _prune_trainset_if_needed(train_dir, max_samples=MAX_CALIB_SAMPLES)

    return {
        "sample_id": int(sid),
        "ori_png": str(ori_png),
        "show_png": str(show_png),
        "manifest_json": str(manifest_path),
        "sample_limit": int(MAX_CALIB_SAMPLES),
        "prune": prune,
    }


def _learn_from_trainset_async():
    if learning_state.get("running"):
        return False

    def _worker():
        learning_state.update(
            {
                "running": True,
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


def _load_hover_label() -> np.ndarray | None:
    global _hover_label_cache
    if not _HOVER_LABEL_PATH.exists():
        return None
    mtime = float(_HOVER_LABEL_PATH.stat().st_mtime)
    if (
        _hover_label_cache.get("img") is None
        or _hover_label_cache.get("path") != str(_HOVER_LABEL_PATH)
        or float(_hover_label_cache.get("mtime", 0.0)) != mtime
    ):
        arr = imread(str(_HOVER_LABEL_PATH))
        if arr.ndim == 3:
            arr = arr[..., 0]
        _hover_label_cache = {"path": str(_HOVER_LABEL_PATH), "mtime": mtime, "img": arr}
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
    if _hover_parent_cache.get("csv") == str(p) and float(_hover_parent_cache.get("mtime", 0.0)) == mtime:
        return _hover_parent_cache.get("map", {})

    try:
        df = pd.read_csv(p)
    except Exception:
        _hover_parent_cache = {"csv": str(p), "mtime": mtime, "map": {}}
        return {}

    col_map = {str(c).strip().lower(): str(c) for c in df.columns}
    id_col = next((col_map[k] for k in ["id", "structure_id", "region_id", "atlas_id"] if k in col_map), None)
    if id_col is None:
        _hover_parent_cache = {"csv": str(p), "mtime": mtime, "map": {}}
        return {}

    parent_name_col = next(
        (col_map[k] for k in ["parent_name", "parent_region_name", "parent_structure_name"] if k in col_map),
        None,
    )
    parent_id_col = next((col_map[k] for k in ["parent_structure_id", "parent_id", "parent"] if k in col_map), None)
    name_col = next((col_map[k] for k in ["name", "region_name", "structure_name", "safe_name"] if k in col_map), None)

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


def _runner(config_path: str, input_dir: str, channels: list[str], run_params=None):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_state.update({"running": True, "done": False, "error": None, "channels": channels, "logs": [], "startTime": ts})

    # Save run params for reproducibility / paper methods section
    if run_params:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        params_to_save = dict(run_params)
        params_to_save['timestamp'] = ts
        params_to_save['channels'] = channels
        try:
            (OUTPUT_DIR / f'run_params_{ts}.json').write_text(
                json.dumps(params_to_save, indent=2, ensure_ascii=False), encoding='utf-8'
            )
        except Exception:
            pass

    for ch in channels:
        run_state["current_channel"] = ch
        _append_log(f"[run] channel={ch}")
        cmd = [
            "python",
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
            _append_log(line)
        code = p.wait()
        _append_log(f"[exit] channel={ch} code={code}")
        if code == 0:
            leaf = OUTPUT_DIR / "cell_counts_leaf.csv"
            if leaf.exists():
                shutil.copy2(leaf, OUTPUT_DIR / f"cell_counts_leaf_{ch}.csv")
        else:
            run_state["error"] = f"channel {ch} failed with code {code}"
            break

    run_state["running"] = False
    run_state["done"] = run_state["error"] is None
    run_state["proc"] = None
    run_state["current_channel"] = None
    run_state["history"].append({
        "channels": channels,
        "ok": run_state["error"] is None,
        "error": run_state["error"],
        "logCount": len(run_state["logs"]),
        "timestamp": ts,
    })
    if len(run_state["history"]) > 20:
        run_state["history"] = run_state["history"][-20:]


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get('/api/info')
def info():
    default_atlas = PROJECT_ROOT / "annotation_25.nii.gz"
    default_struct = PROJECT_ROOT / "configs" / "1_adult_mouse_brain_graph_mapping.csv"
    return jsonify({
        "app": "IdleBrainUI",
        "version": "0.3.0-desktop",
        "frontend": str(ROOT),
        "project": str(PROJECT_ROOT),
        "outputs": str(OUTPUT_DIR),
        "defaults": {
            "atlasPath": str(default_atlas),
            "structPath": str(default_struct) if default_struct.exists() else "",
            "sampleLimit": int(MAX_CALIB_SAMPLES),
        },
    })


@app.get("/api/validate")
def validate():
    input_dir = request.args.get("inputDir", "")
    atlas = request.args.get("atlasPath", "")
    struct = request.args.get("structPath", "")

    issues = []
    if not input_dir or not Path(input_dir).exists():
        issues.append("Input TIFF folder missing or not found")
    if not atlas or not Path(atlas).exists():
        issues.append("Atlas annotation file missing or not found")
    if not struct or not Path(struct).exists():
        issues.append("Structure mapping CSV missing or not found")

    return jsonify({"ok": len(issues) == 0, "issues": issues})


@app.post("/api/run")
def run_pipeline():
    if run_state["running"]:
        return jsonify({"ok": False, "error": "pipeline already running"}), 409

    payload = request.get_json(force=True)
    config = payload.get("configPath") or str(PROJECT_ROOT / "configs" / "run_config.template.json")
    input_dir = payload.get("inputDir", "")
    channels = payload.get("channels", ["red"])
    if isinstance(channels, str):
        channels = [channels]

    run_params = payload.get("params", {})
    t = threading.Thread(target=_runner, args=(config, input_dir, channels, run_params), daemon=True)
    t.start()
    return jsonify({"ok": True, "started": True})


@app.get("/api/status")
def status():
    return jsonify(
        {
            "running": run_state["running"],
            "done": run_state["done"],
            "error": run_state["error"],
            "channels": run_state["channels"],
            "currentChannel": run_state["current_channel"],
            "logCount": len(run_state["logs"]),
        }
    )


@app.post("/api/cancel")
def cancel():
    p = run_state.get("proc")
    if p and run_state.get("running"):
        p.terminate()
        run_state["error"] = "cancelled by user"
        run_state["running"] = False
        run_state["done"] = False
        _append_log("[cancel] user requested stop")
        return jsonify({"ok": True, "cancelled": True})
    return jsonify({"ok": False, "cancelled": False, "error": "no running process"}), 409


@app.get("/api/logs")
def logs():
    return jsonify({"logs": run_state["logs"]})


@app.get('/api/history')
def history():
    return jsonify({"history": run_state["history"]})


@app.post('/api/atlas/autopick-z')
def atlas_autopick_z():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    annotation_path = Path(payload.get('annotationPath', ''))
    z_step = int(payload.get('zStep', 1))
    raw_real_z = payload.get('realZIndex', None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    pixel_size_um = float(payload.get('pixelSizeUm', 0.65))
    slicing_plane = str(payload.get('slicingPlane', 'coronal'))
    roi_mode = str(payload.get('roiMode', 'auto'))
    if not real_path.exists() or not annotation_path.exists():
        return jsonify({"ok": False, "error": "missing real or annotation path"}), 400

    out_label = OUTPUT_DIR / 'auto_label_slice.tif'
    try:
        res = autopick_best_z(
            real_path,
            annotation_path,
            out_label,
            z_step=z_step,
            pixel_size_um=pixel_size_um,
            slicing_plane=slicing_plane,
            roi_mode=roi_mode,
            real_z_index=real_z_index,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, **res})


@app.post('/api/overlay/preview')
def overlay_preview():
    global _last_preview_structure_csv
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    label_path = Path(payload.get('labelPath', ''))
    raw_real_z = payload.get('realZIndex', None)
    raw_label_z = payload.get('labelZIndex', None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    label_z_index = None if raw_label_z in (None, "", "null") else int(raw_label_z)
    alpha = float(payload.get('alpha', 0.45))
    mode = payload.get('mode', 'fill')
    structure_csv = Path(payload.get('structureCsv', '')) if payload.get('structureCsv') else None
    min_mean = float(payload.get('minMeanThreshold', 8.0))
    pixel_size_um = float(payload.get('pixelSizeUm', 0.65))
    rotate_deg = float(payload.get('rotateAtlas', 0.0))
    flip_mode = payload.get('flipAtlas', 'none')
    fit_mode = payload.get('fitMode', 'cover')
    major_top_k = int(payload.get('majorTopK', 20))
    edge_smooth_iter = int(payload.get('edgeSmoothIter', 1))
    warp_params = payload.get('warpParams', {})
    if not isinstance(warp_params, dict):
        warp_params = {}

    if not real_path.exists() or not label_path.exists():
        return jsonify({"ok": False, "error": "real or label path not found"}), 400

    out = OUTPUT_DIR / 'overlay_preview.png'
    try:
        _, diagnostic = render_overlay(
            real_path,
            label_path,
            out,
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
            warped_label_out=_HOVER_LABEL_PATH,
            real_z_index=real_z_index,
            label_z_index=label_z_index,
        )
        _last_preview_structure_csv = str(structure_csv) if structure_csv and structure_csv.exists() else ""
    except Exception as e:
        fail_dir = OUTPUT_DIR / 'fail_cases'
        fail_dir.mkdir(parents=True, exist_ok=True)
        fail_json = fail_dir / 'overlay_fail_last.json'
        fail_json.write_text(json.dumps({
            'realPath': str(real_path),
            'labelPath': str(label_path),
            'alpha': alpha,
            'mode': mode,
            'structureCsv': str(structure_csv) if structure_csv else '',
            'minMeanThreshold': min_mean,
            'pixelSizeUm': pixel_size_um,
            'rotateAtlas': rotate_deg,
            'flipAtlas': flip_mode,
            'fitMode': fit_mode,
            'edgeSmoothIter': edge_smooth_iter,
            'warpParams': warp_params,
            'realZIndex': real_z_index,
            'labelZIndex': label_z_index,
            'error': str(e)
        }, indent=2), encoding='utf-8')
        return jsonify({"ok": False, "error": str(e), "failCase": str(fail_json)}), 400
    return jsonify({"ok": True, "preview": str(out), "minMeanThreshold": min_mean, "diagnostic": diagnostic})


@app.post('/api/overlay/liquify-drag')
def overlay_liquify_drag():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    label_path_payload = Path(payload.get('labelPath', '')) if payload.get('labelPath') else None
    raw_real_z = payload.get('realZIndex', None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    alpha = float(payload.get('alpha', 0.45))
    mode = str(payload.get('mode', 'fill'))
    structure_csv = Path(payload.get('structureCsv', '')) if payload.get('structureCsv') else None
    min_mean = float(payload.get('minMeanThreshold', 8.0))
    pixel_size_um = float(payload.get('pixelSizeUm', 0.65))
    rotate_deg = float(payload.get('rotateAtlas', 0.0))
    flip_mode = payload.get('flipAtlas', 'none')
    fit_mode = payload.get('fitMode', 'cover')
    major_top_k = int(payload.get('majorTopK', 20))
    edge_smooth_iter = int(payload.get('edgeSmoothIter', 1))
    warp_params = payload.get('warpParams', {})
    if not isinstance(warp_params, dict):
        warp_params = {}

    if not real_path.exists():
        return jsonify({"ok": False, "error": "real path not found"}), 400

    if _HOVER_LABEL_PATH.exists():
        base_label_path = _HOVER_LABEL_PATH
    elif label_path_payload is not None and label_path_payload.exists():
        base_label_path = label_path_payload
    else:
        return jsonify({"ok": False, "error": "no aligned preview label available; run preview first"}), 400

    drags = payload.get('drags', [])
    if not isinstance(drags, list):
        drags = []
    if not drags and all(k in payload for k in ('x1', 'y1', 'x2', 'y2')):
        drags = [{
            "x1": payload.get("x1"),
            "y1": payload.get("y1"),
            "x2": payload.get("x2"),
            "y2": payload.get("y2"),
            "radius": payload.get("radius", 80),
            "strength": payload.get("strength", 0.72),
        }]
    if not drags:
        return jsonify({"ok": False, "error": "no drags provided"}), 400

    try:
        lbl_raw = imread(str(base_label_path))
        lbl2d, _ = select_label_slice_2d(lbl_raw)
        tissue = lbl2d > 0
        corrected = _apply_liquify_drags(lbl2d.astype(np.int32), drags=drags, tissue_mask=tissue)

        calib_dir = OUTPUT_DIR / "manual_calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        corrected_label_path = calib_dir / f'manual_warped_label_{ts}.tif'
        imwrite(str(corrected_label_path), corrected.astype(np.int32))
        imwrite(str(_HOVER_LABEL_PATH), corrected.astype(np.int32))

        out = OUTPUT_DIR / 'overlay_preview.png'
        _, diagnostic = render_overlay(
            real_path,
            corrected_label_path,
            out,
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
            warped_label_out=_HOVER_LABEL_PATH,
            real_z_index=real_z_index,
            prewarped_label=True,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({
        "ok": True,
        "preview": str(out),
        "correctedLabelPath": str(corrected_label_path),
        "diagnostic": diagnostic,
    })


@app.post('/api/overlay/calibration/finalize')
def overlay_calibration_finalize():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    if not real_path.exists():
        return jsonify({"ok": False, "error": "real path not found"}), 400
    if not _HOVER_LABEL_PATH.exists():
        return jsonify({"ok": False, "error": "no calibrated label to finalize"}), 400

    raw_real_z = payload.get('realZIndex', None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    alpha = float(payload.get('alpha', 0.45))
    mode = str(payload.get('mode', 'fill'))
    structure_csv = Path(payload.get('structureCsv', '')) if payload.get('structureCsv') else None
    min_mean = float(payload.get('minMeanThreshold', 8.0))
    pixel_size_um = float(payload.get('pixelSizeUm', 0.65))
    rotate_deg = float(payload.get('rotateAtlas', 0.0))
    flip_mode = payload.get('flipAtlas', 'none')
    fit_mode = payload.get('fitMode', 'cover')
    major_top_k = int(payload.get('majorTopK', 20))
    edge_smooth_iter = int(payload.get('edgeSmoothIter', 1))
    warp_params = payload.get('warpParams', {})
    if not isinstance(warp_params, dict):
        warp_params = {}
    auto_learn = bool(payload.get('autoLearn', True))
    note = str(payload.get('note', '')).strip()

    calib_dir = OUTPUT_DIR / "manual_calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    final_overlay = calib_dir / f'calibrated_overlay_{ts}.png'
    final_label = calib_dir / f'calibrated_label_{ts}.tif'
    shutil.copy2(_HOVER_LABEL_PATH, final_label)

    try:
        _, diagnostic = render_overlay(
            real_path,
            final_label,
            final_overlay,
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
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

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
    pair = _save_calibration_pair(
        real_path=real_path,
        corrected_overlay_png=final_overlay,
        manifest=manifest,
        real_z_index=real_z_index,
    )

    learning_started = False
    if auto_learn:
        learning_started = _learn_from_trainset_async()

    return jsonify({
        "ok": True,
        "sample": pair,
        "autoLearn": auto_learn,
        "learningStarted": bool(learning_started),
        "learnStatus": learning_state,
    })


@app.get('/api/calibration/learn-status')
def calibration_learn_status():
    return jsonify({"ok": True, "state": learning_state})


@app.get("/api/outputs/leaf")
def outputs_leaf():
    fp = OUTPUT_DIR / "cell_counts_leaf.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "output not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.get('/api/outputs/leaf/<channel>')
def outputs_leaf_channel(channel: str):
    fp = OUTPUT_DIR / f"cell_counts_leaf_{channel}.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": f"channel output not found: {channel}"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.get('/api/outputs/overlay-preview')
def outputs_overlay_preview():
    fp = OUTPUT_DIR / 'overlay_preview.png'
    if not fp.exists():
        return jsonify({"ok": False, "error": "overlay preview not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.post('/api/overlay/export')
def overlay_export():
    payload = request.get_json(force=True)
    fmt = str(payload.get("format", "png")).strip().lower()
    allowed = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}
    if fmt not in allowed:
        return jsonify({"ok": False, "error": f"unsupported format: {fmt}"}), 400

    src = OUTPUT_DIR / "overlay_preview.png"
    if not src.exists():
        return jsonify({"ok": False, "error": "overlay preview not found"}), 404

    export_dir = OUTPUT_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "path": str(out), "format": ext})


@app.get('/api/overlay/region-at')
def overlay_region_at():
    label = _load_hover_label()
    if label is None:
        return jsonify({"ok": False, "error": "preview label not available yet"}), 404

    try:
        x = int(float(request.args.get("x", "-1")))
        y = int(float(request.args.get("y", "-1")))
    except Exception:
        return jsonify({"ok": False, "error": "invalid x/y"}), 400

    h, w = label.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return jsonify({"ok": True, "inside": False, "region_id": 0, "x": x, "y": y})

    rid = int(label[y, x])
    if rid <= 0:
        return jsonify({"ok": True, "inside": False, "region_id": 0, "x": x, "y": y})

    tree = _load_hover_structure_tree()
    node = tree.get(str(rid), {}) if isinstance(tree, dict) else {}
    parent_map = _build_parent_name_map(_last_preview_structure_csv)
    parent_name = str(node.get("parent", "")).strip() or parent_map.get(rid, "")

    return jsonify({
        "ok": True,
        "inside": True,
        "x": int(x),
        "y": int(y),
        "region_id": int(rid),
        "acronym": str(node.get("acronym", "")),
        "name": str(node.get("name", "")),
        "parent": str(parent_name),
        "color": str(node.get("color", "")),
    })


@app.get('/api/outputs/overlay-compare')
def outputs_overlay_compare():
    fp = OUTPUT_DIR / 'overlay_compare.png'
    if not fp.exists():
        return jsonify({"ok": False, "error": "overlay compare not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.post('/api/align/nonlinear')
def align_nonlinear():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    atlas_label_path = Path(payload.get('atlasLabelPath', ''))
    pairs_csv = OUTPUT_DIR / 'landmark_pairs.csv'
    if not real_path.exists() or not atlas_label_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas/pairs file"}), 400

    out_label = OUTPUT_DIR / 'aligned_label_nonlinear.tif'
    fail_log = OUTPUT_DIR / 'fail_cases' / 'align_nonlinear_last.json'
    try:
        meta = apply_landmark_nonlinear(real_path, atlas_label_path, pairs_csv, out_label)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "failLog": str(fail_log)}), 400

    real = imread(str(real_path)); atlas_before = imread(str(atlas_label_path)); atlas_after = imread(str(out_label))
    real, _ = select_real_slice_2d(real, source_path=real_path)
    atlas_before, _ = select_label_slice_2d(atlas_before)
    atlas_after, _ = select_label_slice_2d(atlas_after)

    before = score_alignment(real, atlas_before)
    after = score_alignment(real, atlas_after)
    before_edge = score_alignment_edges(real, atlas_before)
    after_edge = score_alignment_edges(real, atlas_after)

    compare_png = OUTPUT_DIR / 'overlay_compare_nonlinear.png'
    render_before_after(real_path, atlas_label_path, out_label, compare_png, alpha=0.45, before_score=before_edge, after_score=after_edge)
    return jsonify({
        "ok": True,
        "beforeScore": before,
        "afterScore": after,
        "beforeEdgeScore": before_edge,
        "afterEdgeScore": after_edge,
        "scoreWarning": after_edge < before_edge,
        "compareImage": str(compare_png),
        **meta,
    })


@app.get('/api/outputs/overlay-compare-nonlinear')
def outputs_overlay_compare_nonlinear():
    fp = OUTPUT_DIR / 'overlay_compare_nonlinear.png'
    if not fp.exists():
        return jsonify({"ok": False, "error": "nonlinear overlay compare not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.get('/api/outputs/auto-label-slice')
def outputs_auto_label_slice():
    fp = OUTPUT_DIR / 'auto_label_slice.tif'
    if not fp.exists():
        return jsonify({"ok": False, "error": "auto label slice not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.get('/api/outputs/landmark-preview')
def outputs_landmark_preview():
    fp = OUTPUT_DIR / 'landmark_preview.png'
    if not fp.exists():
        return jsonify({"ok": False, "error": "landmark preview not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.post('/api/align/landmark-preview')
def align_landmark_preview():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    atlas_path = Path(payload.get('atlasPath', ''))
    pairs_csv = OUTPUT_DIR / 'landmark_pairs.csv'
    if not real_path.exists() or not atlas_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas or pairs file"}), 400

    real = imread(str(real_path)); atlas = imread(str(atlas_path))
    real, _ = select_real_slice_2d(real, source_path=real_path)
    atlas, _ = select_label_slice_2d(atlas)

    h = min(real.shape[0], atlas.shape[0]); w = min(real.shape[1], atlas.shape[1])
    real = real[:h,:w]; atlas = atlas[:h,:w]

    real_rgb = np.stack([real, real, real], axis=-1).astype(np.uint8)
    atlas_rgb = np.stack([atlas, atlas, atlas], axis=-1).astype(np.uint8)

    pairs = pd.read_csv(pairs_csv)
    for _, r in pairs.iterrows():
        rx, ry = int(r['real_x']), int(r['real_y'])
        ax, ay = int(r['atlas_x']), int(r['atlas_y'])
        if 0 <= ry < h and 0 <= rx < w:
            real_rgb[max(0,ry-2):ry+3, max(0,rx-2):rx+3] = [255, 255, 0]
        if 0 <= ay < h and 0 <= ax < w:
            atlas_rgb[max(0,ay-2):ay+3, max(0,ax-2):ax+3] = [0, 255, 255]

    pad = np.zeros((h, 8, 3), dtype=np.uint8)
    canvas = np.concatenate([real_rgb, pad, atlas_rgb], axis=1)
    fp = OUTPUT_DIR / 'landmark_preview.png'
    imwrite(str(fp), canvas)
    return jsonify({"ok": True, "preview": str(fp), "points": int(len(pairs))})


@app.post('/api/align/landmarks')
def align_landmarks():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    atlas_path = Path(payload.get('atlasPath', ''))
    if not real_path.exists() or not atlas_path.exists():
        return jsonify({"ok": False, "error": "real or atlas path not found"}), 400

    max_points = int(payload.get('maxPoints', 30))
    min_distance = int(payload.get('minDistance', 12))
    ransac_residual = float(payload.get('ransacResidual', 8.0))

    out_csv = OUTPUT_DIR / 'landmark_pairs.csv'
    res = propose_landmarks(
        real_path,
        atlas_path,
        out_csv,
        max_points=max_points,
        min_distance=min_distance,
        ransac_residual=ransac_residual,
    )
    return jsonify({"ok": True, **res})


@app.post('/api/align/apply')
def align_apply():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    atlas_label_path = Path(payload.get('atlasLabelPath', ''))
    pairs_csv = OUTPUT_DIR / 'landmark_pairs.csv'
    if not real_path.exists() or not atlas_label_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas/pairs file"}), 400

    out_label = OUTPUT_DIR / 'aligned_label_ai.tif'
    meta = apply_landmark_affine(real_path, atlas_label_path, pairs_csv, out_label)

    real = imread(str(real_path))
    atlas_before = imread(str(atlas_label_path))
    atlas_after = imread(str(out_label))
    real, _ = select_real_slice_2d(real, source_path=real_path)
    atlas_before, _ = select_label_slice_2d(atlas_before)
    atlas_after, _ = select_label_slice_2d(atlas_after)

    before = score_alignment(real, atlas_before)
    after = score_alignment(real, atlas_after)
    before_edge = score_alignment_edges(real, atlas_before)
    after_edge = score_alignment_edges(real, atlas_after)
    compare_png = OUTPUT_DIR / 'overlay_compare.png'
    render_before_after(real_path, atlas_label_path, out_label, compare_png, alpha=0.45, before_score=before_edge, after_score=after_edge)

    return jsonify({
        "ok": True,
        "beforeScore": before,
        "afterScore": after,
        "beforeEdgeScore": before_edge,
        "afterEdgeScore": after_edge,
        "scoreWarning": after_edge < before_edge,
        "compareImage": str(compare_png),
        **meta,
    })


@app.get('/api/slice/info')
def slice_info():
    path = request.args.get('path', '')
    if not path or not Path(path).exists():
        return jsonify({"ok": False, "error": "file not found"}), 400
    try:
        from tifffile import TiffFile
        with TiffFile(path) as tif:
            shape = list(tif.series[0].shape)
        ndim = len(shape)
        return jsonify({"ok": True, "shape": shape, "ndim": ndim, "is3d": ndim >= 3, "z_count": shape[0] if ndim >= 3 else 1})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post('/api/slice/extract-z')
def slice_extract_z():
    payload = request.get_json(force=True)
    src = Path(payload.get('path', ''))
    z = int(payload.get('z', 0))
    if not src.exists():
        return jsonify({"ok": False, "error": "source file not found"}), 400
    try:
        img = imread(str(src))
        if img.ndim >= 3:
            z = max(0, min(z, img.shape[0] - 1))
            slc = img[z]
        else:
            slc = img
        out_path = OUTPUT_DIR / f'extracted_z{z:04d}.tif'
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        imwrite(str(out_path), slc)
        return jsonify({"ok": True, "path": str(out_path), "z": z, "shape": list(slc.shape), "dtype": str(slc.dtype)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post('/api/overlay/atlas-layer')
def overlay_atlas_layer():
    """Render atlas colormap as RGBA PNG (transparent background) for client-side compositing."""
    payload = request.get_json(force=True)
    label_path = Path(payload.get('labelPath', ''))
    structure_csv = Path(payload.get('structureCsv', '')) if payload.get('structureCsv') else None
    pixel_size_um = float(payload.get('pixelSizeUm', 0.65))
    rotate_deg = float(payload.get('rotateAtlas', 0.0))
    flip_mode = payload.get('flipAtlas', 'none')
    fit_mode = payload.get('fitMode', 'cover')
    edge_smooth_iter = int(payload.get('edgeSmoothIter', 1))
    warp_params = payload.get('warpParams', {})
    if not isinstance(warp_params, dict):
        warp_params = {}
    raw_real_z = payload.get('realZIndex', None)
    raw_label_z = payload.get('labelZIndex', None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    label_z_index = None if raw_label_z in (None, "", "null") else int(raw_label_z)
    real_path = Path(payload.get('realPath', ''))
    if not label_path.exists() or not real_path.exists():
        return jsonify({"ok": False, "error": "label or real path not found"}), 400
    try:
        real_img = imread(str(real_path))
        real_img, _ = select_real_slice_2d(real_img, z_index=real_z_index, source_path=real_path)
        target_shape = (real_img.shape[0], real_img.shape[1])
        out = OUTPUT_DIR / 'atlas_layer_rgba.png'
        _, diagnostic = render_overlay(
            real_path, label_path, out, alpha=1.0, mode='fill',
            structure_csv=structure_csv, pixel_size_um=pixel_size_um,
            rotate_deg=rotate_deg, flip_mode=flip_mode, return_meta=True,
            major_top_k=20, fit_mode=fit_mode,
            edge_smooth_iter=edge_smooth_iter,
            warp_params=warp_params,
            real_z_index=real_z_index, label_z_index=label_z_index,
        )
        return jsonify({"ok": True, "diagnostic": diagnostic})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.get('/api/outputs/atlas-layer')
def get_atlas_layer():
    fp = OUTPUT_DIR / 'atlas_layer_rgba.png'
    if not fp.exists():
        return jsonify({"ok": False, "error": "atlas layer not rendered yet"}), 404
    return send_from_directory(str(fp.parent), fp.name)


@app.post('/api/align/add-manual-landmarks')
def add_manual_landmarks():
    payload = request.get_json(force=True)
    pairs = payload.get('pairs', [])
    if not pairs:
        return jsonify({"ok": False, "error": "no pairs provided"}), 400
    pairs_csv = OUTPUT_DIR / 'landmark_pairs.csv'
    new_rows = pd.DataFrame(pairs)
    if pairs_csv.exists():
        try:
            existing = pd.read_csv(pairs_csv)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        except Exception:
            combined = new_rows
    else:
        combined = new_rows
    combined.to_csv(pairs_csv, index=False)
    return jsonify({"ok": True, "total_pairs": int(len(combined))})


@app.get('/api/outputs/file-list')
def outputs_file_list():
    if not OUTPUT_DIR.exists():
        return jsonify({"ok": True, "files": []})
    files = []
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            files.append({"name": f.name, "size": f.stat().st_size, "ext": f.suffix.lower()})
    return jsonify({"ok": True, "files": files, "dir": str(OUTPUT_DIR)})


@app.get('/api/outputs/named/<filename>')
def outputs_named(filename: str):
    safe = Path(filename).name
    fp = OUTPUT_DIR / safe
    if not fp.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(str(OUTPUT_DIR), safe)


@app.post('/api/browse/folder')
def browse_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        path = filedialog.askdirectory(title='选择文件夹')
        root.destroy()
        return jsonify({"ok": True, "path": path or ""})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post('/api/browse/file')
def browse_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        payload = request.get_json(force=True) or {}
        filetypes_raw = payload.get('filetypes', '')
        if filetypes_raw:
            exts = [e.strip() for e in filetypes_raw.split(',')]
            filetypes = [(f'.{e} 文件', f'*.{e}') for e in exts] + [('所有文件', '*.*')]
        else:
            filetypes = [('所有文件', '*.*')]
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        path = filedialog.askopenfilename(title='选择文件', filetypes=filetypes)
        root.destroy()
        return jsonify({"ok": True, "path": path or ""})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get('/api/outputs/qc-list')
def outputs_qc_list():
    qc_dir = OUTPUT_DIR / 'qc_overlays'
    if not qc_dir.exists():
        return jsonify({"ok": True, "files": [], "count": 0})
    files = sorted(qc_dir.glob('overlay_*.png'))
    return jsonify({"ok": True, "files": [f.name for f in files], "count": len(files)})


@app.get('/api/outputs/qc-file/<filename>')
def outputs_qc_file(filename: str):
    qc_dir = OUTPUT_DIR / 'qc_overlays'
    safe = Path(filename).name
    return send_from_directory(str(qc_dir), safe)


@app.get('/api/export/methods-text')
def export_methods_text():
    params_files = sorted(OUTPUT_DIR.glob('run_params_*.json'), reverse=True)
    params = {}
    if params_files:
        try:
            params = json.loads(params_files[0].read_text(encoding='utf-8'))
        except Exception:
            pass
    align_mode = params.get('alignMode', 'affine')
    align_cn = '仿射变换 (affine)' if align_mode == 'affine' else '非线性变换 (nonlinear/TPS)'
    align_en = 'affine' if align_mode == 'affine' else 'nonlinear (thin-plate spline)'
    pixel_size = params.get('pixelSizeUm', '0.65')
    channels = params.get('channels', ['red'])
    ch_str = ', '.join(channels)
    ts = params.get('timestamp', '—')
    text_cn = (
        f"【方法段落参考（中文）】\n"
        f"脑图谱配准使用 IdleBrain v0.3 完成（运行时间：{ts}）。"
        f"显微图像分辨率为 {pixel_size} μm/像素。"
        f"图谱配准参照 Allen 小鼠脑图谱（CCFv3，annotation_25.nii.gz，体素间距 25 μm），"
        f"采用{align_cn}方法对切片进行空间配准。配准质量通过边缘 SSIM（结构相似性指标）评估。"
        f"细胞检测采用 Cellpose 算法；去重后按图谱分级脑区统计细胞数量。荧光通道：{ch_str}。"
    )
    text_en = (
        f"\n【Methods paragraph reference (English)】\n"
        f"Brain atlas registration was performed using IdleBrain v0.3 (run: {ts}). "
        f"Microscopy images were acquired at {pixel_size} μm/pixel. "
        f"Section registration was carried out against the Allen Mouse Brain Atlas "
        f"(CCFv3, annotation_25.nii.gz, 25 μm voxel spacing) using {align_en} transformation. "
        f"Alignment quality was evaluated by edge-SSIM. "
        f"Cell detection used the Cellpose algorithm; deduplicated cells were assigned to "
        f"atlas regions and counts were aggregated hierarchically. Channels: {ch_str}."
    )
    return jsonify({"ok": True, "text": text_cn + text_en, "params": params})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)

