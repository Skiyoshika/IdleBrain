from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from tifffile import imread
from skimage import measure
from skimage.feature import blob_log, peak_local_max

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency fallback
    cKDTree = None


_CELLPOSE_MODEL_CACHE: dict[Tuple[str, bool], Any] = {}


def _read_gray(slice_path: Path) -> np.ndarray:
    img = imread(str(slice_path))
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32, copy=False)


def _norm_for_cellpose(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    p1, p99 = np.percentile(x, [1.0, 99.0])
    if float(p99) <= float(p1):
        p1, p99 = float(np.min(x)), float(np.max(x) + 1e-6)
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    return x.astype(np.float32, copy=False)


def _resolve_model_type(name: str) -> str:
    s = str(name or "").strip().lower()
    if "nuclei" in s:
        return "nuclei"
    if "cyto3" in s:
        return "cyto3"
    if "cyto2" in s:
        return "cyto2"
    if "cyto" in s:
        return "cyto"
    return "cyto3"


def _pixel_size_um_from_cfg(cfg: Dict[str, Any]) -> float | None:
    raw = cfg.get("input", {}).get("pixel_size_um_xy", None)
    if raw in (None, "", "TODO"):
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except Exception:
        return None


def _diameter_px(det_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> float | None:
    raw_px = det_cfg.get("cellpose_diameter_px", None)
    if raw_px not in (None, "", 0):
        try:
            v = float(raw_px)
            if v > 0:
                return v
        except Exception:
            pass

    raw_um = det_cfg.get("cellpose_diameter_um", None)
    if raw_um in (None, "", 0):
        return None
    try:
        d_um = float(raw_um)
    except Exception:
        return None
    px_um = _pixel_size_um_from_cfg(cfg)
    if px_um is None or px_um <= 0:
        return None
    d_px = d_um / px_um
    return float(d_px) if d_px > 0 else None


def _use_gpu(cfg: Dict[str, Any], det_cfg: Dict[str, Any]) -> bool:
    forced = det_cfg.get("cellpose_gpu", None)
    if forced is not None:
        return bool(forced)
    dev = str(cfg.get("compute", {}).get("device", "cpu")).lower()
    return dev in ("cuda", "gpu")


def _load_cellpose_model(model_type: str, use_gpu: bool):
    key = (str(model_type), bool(use_gpu))
    if key in _CELLPOSE_MODEL_CACHE:
        return _CELLPOSE_MODEL_CACHE[key]
    from cellpose import models

    model = models.Cellpose(gpu=bool(use_gpu), model_type=str(model_type))
    _CELLPOSE_MODEL_CACHE[key] = model
    return model


def _masks_to_centroids(masks: np.ndarray, detector: str) -> pd.DataFrame:
    if masks is None or masks.size == 0 or int(np.max(masks)) <= 0:
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])

    props = measure.regionprops_table(
        masks.astype(np.int32, copy=False),
        properties=("label", "centroid", "area"),
    )
    if not props or len(props.get("label", [])) == 0:
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])

    df = pd.DataFrame(
        {
            "x": np.asarray(props["centroid-1"], dtype=np.float32),
            "y": np.asarray(props["centroid-0"], dtype=np.float32),
            "area_px": np.asarray(props["area"], dtype=np.float32),
        }
    )
    df["score"] = np.clip(np.sqrt(df["area_px"].astype(np.float32)), 0.0, None)
    df["cell_id"] = np.arange(1, len(df) + 1, dtype=np.int32)
    df["detector"] = str(detector)
    return df[["cell_id", "x", "y", "score", "detector", "area_px"]]


def _dedup_xy(df: pd.DataFrame, radius_px: float = 4.0) -> pd.DataFrame:
    if len(df) <= 1 or radius_px <= 0:
        return df
    if cKDTree is None:
        return df

    arr = df[["x", "y"]].to_numpy(dtype=np.float32, copy=False)
    scores = df["score"].to_numpy(dtype=np.float32, copy=False)
    order = np.argsort(-scores)
    tree = cKDTree(arr)
    keep = np.ones(len(df), dtype=bool)
    for idx in order:
        if not keep[idx]:
            continue
        neigh = tree.query_ball_point(arr[idx], r=float(radius_px))
        for j in neigh:
            if j == idx:
                continue
            keep[j] = False
    out = df.loc[keep].copy().reset_index(drop=True)
    out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def detect_cells_fallback(
    slice_path: Path,
    min_distance: int = 8,
    threshold_abs: float = 200.0,
) -> pd.DataFrame:
    img = _read_gray(slice_path)
    coords = peak_local_max(
        img,
        min_distance=max(1, int(min_distance)),
        threshold_abs=float(threshold_abs),
    )
    rows = []
    for i, (y, x) in enumerate(coords, 1):
        rows.append(
            {
                "cell_id": i,
                "x": float(x),
                "y": float(y),
                "score": float(img[y, x]),
                "detector": "fallback_peak",
                "area_px": 1.0,
            }
        )
    return pd.DataFrame(rows, columns=["cell_id", "x", "y", "score", "detector", "area_px"])


def detect_cells_log_fallback(
    slice_path: Path,
    min_sigma: float = 1.2,
    max_sigma: float = 5.0,
    num_sigma: int = 8,
    threshold_rel: float = 0.03,
) -> pd.DataFrame:
    img = _read_gray(slice_path)
    x = _norm_for_cellpose(img)
    blobs = blob_log(
        x,
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
        num_sigma=max(1, int(num_sigma)),
        threshold=float(threshold_rel),
    )
    if blobs is None or len(blobs) == 0:
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])

    rows = []
    for i, b in enumerate(blobs, 1):
        y, x0, sigma = float(b[0]), float(b[1]), float(b[2])
        r = np.sqrt(2.0) * sigma
        y0 = int(np.clip(round(y), 0, img.shape[0] - 1))
        x1 = int(np.clip(round(x0), 0, img.shape[1] - 1))
        rows.append(
            {
                "cell_id": i,
                "x": x0,
                "y": y,
                "score": float(img[y0, x1]),
                "detector": "fallback_log",
                "area_px": float(np.pi * r * r),
            }
        )
    return pd.DataFrame(rows, columns=["cell_id", "x", "y", "score", "detector", "area_px"])


def detect_cells_cellpose(
    slice_path: Path,
    model_type: str = "cyto3",
    diameter_px: float | None = None,
    *,
    use_gpu: bool = False,
    channels: list[int] | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 8,
) -> pd.DataFrame:
    try:
        model = _load_cellpose_model(model_type=model_type, use_gpu=use_gpu)
    except Exception:
        return pd.DataFrame()

    img = _read_gray(slice_path)
    imgf = _norm_for_cellpose(img)
    ch = channels if isinstance(channels, list) and len(channels) == 2 else [0, 0]

    # Keep eval kwargs conservative for compatibility across Cellpose versions.
    kwargs = dict(
        diameter=diameter_px,
        channels=ch,
        flow_threshold=float(flow_threshold),
        cellprob_threshold=float(cellprob_threshold),
        min_size=max(0, int(min_size)),
    )

    try:
        masks, flows, styles, diams = model.eval(imgf, **kwargs)
    except TypeError:
        # Older versions may not accept all kwargs.
        kwargs2 = dict(diameter=diameter_px, channels=ch)
        masks, flows, styles, diams = model.eval(imgf, **kwargs2)
    except Exception:
        return pd.DataFrame()

    return _masks_to_centroids(masks, detector=f"cellpose_{model_type}")


def _run_cellpose_by_name(slice_path: Path, model_name: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    det_cfg = cfg.get("detection", {})
    model_type = _resolve_model_type(model_name)
    d_px = _diameter_px(det_cfg, cfg)
    use_gpu = _use_gpu(cfg, det_cfg)
    channels = det_cfg.get("cellpose_channels", [0, 0])
    flow_thr = float(det_cfg.get("cellpose_flow_threshold", 0.4))
    prob_thr = float(det_cfg.get("cellpose_cellprob_threshold", 0.0))
    min_sz = int(det_cfg.get("cellpose_min_size_px", 8))

    return detect_cells_cellpose(
        slice_path=slice_path,
        model_type=model_type,
        diameter_px=d_px,
        use_gpu=use_gpu,
        channels=channels if isinstance(channels, list) else [0, 0],
        flow_threshold=flow_thr,
        cellprob_threshold=prob_thr,
        min_size=min_sz,
    )


def detect_cells(slice_path: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    det_cfg = cfg.get("detection", {})
    primary = str(det_cfg.get("primary_model", "cellpose_cyto2"))
    secondary = str(det_cfg.get("secondary_model", "cellpose_nuclei"))
    merge_secondary = bool(det_cfg.get("merge_primary_secondary", False))
    within_slice_dedup_px = float(det_cfg.get("within_slice_dedup_px", 4.0))

    primary_df = pd.DataFrame()
    if primary.startswith("cellpose"):
        primary_df = _run_cellpose_by_name(slice_path, primary, cfg)
        if not primary_df.empty and not merge_secondary:
            out = _dedup_xy(primary_df, radius_px=within_slice_dedup_px)
            out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
            return out

    if secondary.startswith("cellpose"):
        secondary_df = _run_cellpose_by_name(slice_path, secondary, cfg)
        if not secondary_df.empty:
            if primary_df.empty:
                out = _dedup_xy(secondary_df, radius_px=within_slice_dedup_px)
                out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
                return out
            if merge_secondary:
                combined = pd.concat([primary_df, secondary_df], ignore_index=True)
                out = _dedup_xy(combined, radius_px=within_slice_dedup_px)
                out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
                return out

    fallback_model = str(det_cfg.get("fallback_model", "log")).lower()
    if "log" in fallback_model:
        log_df = detect_cells_log_fallback(
            slice_path,
            min_sigma=float(det_cfg.get("fallback_log_min_sigma", 1.2)),
            max_sigma=float(det_cfg.get("fallback_log_max_sigma", 5.0)),
            num_sigma=int(det_cfg.get("fallback_log_num_sigma", 8)),
            threshold_rel=float(det_cfg.get("fallback_log_threshold_rel", 0.03)),
        )
        if not log_df.empty:
            log_df["cell_id"] = np.arange(1, len(log_df) + 1, dtype=np.int32)
            return log_df

    thr = float(det_cfg.get("fallback_threshold", 200.0))
    md = int(det_cfg.get("fallback_min_distance", 8))
    out = detect_cells_fallback(slice_path, min_distance=md, threshold_abs=thr)
    out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out
