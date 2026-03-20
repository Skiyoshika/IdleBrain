from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage import measure
from skimage.feature import blob_log, peak_local_max
from tifffile import imread

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency fallback
    cKDTree = None


_CELLPOSE_MODEL_CACHE: dict[tuple[str, bool], Any] = {}


class CellposeDetectionError(RuntimeError):
    """Raised when Cellpose was requested but could not produce a valid run."""


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


def _is_cellpose_model(name: str) -> bool:
    return str(name or "").strip().lower().startswith("cellpose")


def _pixel_size_um_from_cfg(cfg: dict[str, Any]) -> float | None:
    raw = cfg.get("input", {}).get("pixel_size_um_xy", None)
    if raw in (None, "", "TODO"):
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except Exception:
        return None


def _diameter_px(det_cfg: dict[str, Any], cfg: dict[str, Any]) -> float | None:
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


def _use_gpu(cfg: dict[str, Any], det_cfg: dict[str, Any]) -> bool:
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


def _masks_to_centroids(
    masks: np.ndarray,
    detector: str,
    intensity_image: np.ndarray | None = None,
) -> pd.DataFrame:
    _empty_cols = [
        "cell_id",
        "x",
        "y",
        "score",
        "detector",
        "area_px",
        "elongation",
        "mean_intensity",
    ]
    if masks is None or masks.size == 0 or int(np.max(masks)) <= 0:
        return pd.DataFrame(columns=_empty_cols)

    base_props = ["label", "centroid", "area"]
    extra_props: list[str] = []
    if intensity_image is not None:
        extra_props.append("mean_intensity")
    # minor/major axis needs ≥3px objects; safe to request always
    extra_props += ["minor_axis_length", "major_axis_length"]

    props = measure.regionprops_table(
        masks.astype(np.int32, copy=False),
        intensity_image=intensity_image,
        properties=base_props + extra_props,
    )
    if not props or len(props.get("label", [])) == 0:
        return pd.DataFrame(columns=_empty_cols)

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

    # elongation: minor/major axis ratio (1.0 = circle, 0.0 = line)
    minor = np.asarray(props.get("minor_axis_length", []), dtype=np.float32)
    major = np.asarray(props.get("major_axis_length", []), dtype=np.float32)
    if len(minor) == len(df) and len(major) == len(df):
        with np.errstate(invalid="ignore", divide="ignore"):
            elong = np.where(major > 0, minor / major, 1.0)
        df["elongation"] = np.clip(elong, 0.0, 1.0).astype(np.float32)
    else:
        df["elongation"] = np.float32(1.0)

    # mean_intensity from intensity image
    if intensity_image is not None and "mean_intensity" in props:
        mi = np.asarray(props["mean_intensity"], dtype=np.float32)
        df["mean_intensity"] = mi if len(mi) == len(df) else np.float32(0.0)
    else:
        df["mean_intensity"] = np.float32(0.0)

    return df[_empty_cols]


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
    _cols = ["cell_id", "x", "y", "score", "detector", "area_px", "elongation", "mean_intensity"]
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
                "elongation": 1.0,
                "mean_intensity": float(img[y, x]),
            }
        )
    return pd.DataFrame(rows, columns=_cols)


def detect_cells_log_fallback(
    slice_path: Path,
    min_sigma: float = 1.2,
    max_sigma: float = 5.0,
    num_sigma: int = 8,
    threshold_rel: float = 0.03,
) -> pd.DataFrame:
    _cols = ["cell_id", "x", "y", "score", "detector", "area_px", "elongation", "mean_intensity"]
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
        return pd.DataFrame(columns=_cols)

    h, w = img.shape[:2]
    rows = []
    for i, b in enumerate(blobs, 1):
        y, x0, sigma = float(b[0]), float(b[1]), float(b[2])
        r = np.sqrt(2.0) * sigma
        y0 = int(np.clip(round(y), 0, h - 1))
        x1 = int(np.clip(round(x0), 0, w - 1))
        # Disk mean intensity: sample pixels within radius r
        iy0 = int(max(0, y0 - r))
        iy1 = int(min(h, y0 + r + 1))
        ix0 = int(max(0, x1 - r))
        ix1 = int(min(w, x1 + r + 1))
        patch = img[iy0:iy1, ix0:ix1]
        mean_int = float(np.mean(patch)) if patch.size > 0 else float(img[y0, x1])
        rows.append(
            {
                "cell_id": i,
                "x": x0,
                "y": y,
                "score": float(img[y0, x1]),
                "detector": "fallback_log",
                "area_px": float(np.pi * r * r),
                "elongation": 1.0,  # circular blob approximation
                "mean_intensity": mean_int,
            }
        )
    return pd.DataFrame(rows, columns=_cols)


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or length <= tile_size:
        return [0]
    step = max(1, int(tile_size) - int(overlap))
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = max(0, int(length) - int(tile_size))
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(int(v) for v in starts))


def _eval_cellpose_masks(
    model: Any,
    imgf: np.ndarray,
    *,
    slice_label: str,
    model_type: str,
    diameter_px: float | None,
    channels: list[int],
    flow_threshold: float,
    cellprob_threshold: float,
    min_size: int,
    batch_size: int,
    tile_overlap: float,
    resample: bool,
    raise_on_error: bool,
):
    kwargs = dict(
        diameter=diameter_px,
        channels=channels,
        flow_threshold=float(flow_threshold),
        cellprob_threshold=float(cellprob_threshold),
        min_size=max(0, int(min_size)),
        batch_size=max(1, int(batch_size)),
        tile=True,
        tile_overlap=float(tile_overlap),
        resample=bool(resample),
        normalize=False,
    )

    try:
        return model.eval(imgf, **kwargs)
    except TypeError:
        kwargs2 = dict(diameter=diameter_px, channels=channels)
        try:
            return model.eval(imgf, **kwargs2)
        except Exception as exc:
            if raise_on_error:
                raise CellposeDetectionError(
                    f"Cellpose eval failed for model '{model_type}' on {slice_label}: {exc}"
                ) from exc
            return None
    except Exception as exc:
        if raise_on_error:
            raise CellposeDetectionError(
                f"Cellpose eval failed for model '{model_type}' on {slice_label}: {exc}"
            ) from exc
        return None


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
    batch_size: int = 1,
    tile_overlap: float = 0.05,
    resample: bool = False,
    external_tile_size_px: int | None = None,
    external_tile_overlap_px: int = 64,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    try:
        model = _load_cellpose_model(model_type=model_type, use_gpu=use_gpu)
    except Exception as exc:
        if raise_on_error:
            raise CellposeDetectionError(
                f"failed to load Cellpose model '{model_type}' (gpu={bool(use_gpu)}): {exc}"
            ) from exc
        return pd.DataFrame()

    img = _read_gray(slice_path)
    imgf = _norm_for_cellpose(img)
    ch = channels if isinstance(channels, list) and len(channels) == 2 else [0, 0]
    tile_size = int(external_tile_size_px or 0)
    if tile_size <= 0 and not use_gpu and max(imgf.shape[:2]) > 384:
        tile_size = 384
    overlap_px = max(0, int(external_tile_overlap_px))

    if tile_size > 0 and max(imgf.shape[:2]) > tile_size:
        tile_rows: list[pd.DataFrame] = []
        y_starts = _tile_starts(int(imgf.shape[0]), tile_size, overlap_px)
        x_starts = _tile_starts(int(imgf.shape[1]), tile_size, overlap_px)
        for y0 in y_starts:
            y1 = min(int(imgf.shape[0]), int(y0) + tile_size)
            for x0 in x_starts:
                x1 = min(int(imgf.shape[1]), int(x0) + tile_size)
                tile_img = imgf[y0:y1, x0:x1]
                result = _eval_cellpose_masks(
                    model,
                    tile_img,
                    slice_label=f"{slice_path.name}@y{y0}:{y1},x{x0}:{x1}",
                    model_type=model_type,
                    diameter_px=diameter_px,
                    channels=ch,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    min_size=min_size,
                    batch_size=batch_size,
                    tile_overlap=tile_overlap,
                    resample=resample,
                    raise_on_error=raise_on_error,
                )
                if result is None:
                    continue
                masks, flows, styles, diams = result
                tile_intensity = img[y0:y1, x0:x1].astype(np.float32)
                tile_df = _masks_to_centroids(
                    masks,
                    detector=f"cellpose_{model_type}",
                    intensity_image=tile_intensity,
                )
                if tile_df.empty:
                    continue
                tile_df["x"] = tile_df["x"].astype(np.float32) + float(x0)
                tile_df["y"] = tile_df["y"].astype(np.float32) + float(y0)
                tile_rows.append(tile_df)
        _ecols = [
            "cell_id",
            "x",
            "y",
            "score",
            "detector",
            "area_px",
            "elongation",
            "mean_intensity",
        ]
        if not tile_rows:
            return pd.DataFrame(columns=_ecols)
        out = pd.concat(tile_rows, ignore_index=True)
        out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
        return out[_ecols]

    result = _eval_cellpose_masks(
        model,
        imgf,
        slice_label=slice_path.name,
        model_type=model_type,
        diameter_px=diameter_px,
        channels=ch,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        batch_size=batch_size,
        tile_overlap=tile_overlap,
        resample=resample,
        raise_on_error=raise_on_error,
    )
    if result is None:
        return pd.DataFrame()
    masks, flows, styles, diams = result
    return _masks_to_centroids(
        masks,
        detector=f"cellpose_{model_type}",
        intensity_image=img.astype(np.float32),
    )


def _run_cellpose_by_name(
    slice_path: Path,
    model_name: str,
    cfg: dict[str, Any],
    *,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    det_cfg = cfg.get("detection", {})
    model_type = _resolve_model_type(model_name)
    d_px = _diameter_px(det_cfg, cfg)
    use_gpu = _use_gpu(cfg, det_cfg)
    channels = det_cfg.get("cellpose_channels", [0, 0])
    flow_thr = float(det_cfg.get("cellpose_flow_threshold", 0.4))
    prob_thr = float(det_cfg.get("cellpose_cellprob_threshold", 0.0))
    min_sz = int(det_cfg.get("cellpose_min_size_px", 8))
    batch_size = int(det_cfg.get("cellpose_batch_size", 1))
    tile_overlap = float(det_cfg.get("cellpose_tile_overlap", 0.05))
    resample = bool(det_cfg.get("cellpose_resample", False))
    tile_size_px = det_cfg.get("cellpose_external_tile_size_px", None)
    external_tile_size_px = int(tile_size_px) if tile_size_px not in (None, "", 0) else None
    external_tile_overlap_px = int(det_cfg.get("cellpose_external_tile_overlap_px", 64))

    return detect_cells_cellpose(
        slice_path=slice_path,
        model_type=model_type,
        diameter_px=d_px,
        use_gpu=use_gpu,
        channels=channels if isinstance(channels, list) else [0, 0],
        flow_threshold=flow_thr,
        cellprob_threshold=prob_thr,
        min_size=min_sz,
        batch_size=batch_size,
        tile_overlap=tile_overlap,
        resample=resample,
        external_tile_size_px=external_tile_size_px,
        external_tile_overlap_px=external_tile_overlap_px,
        raise_on_error=raise_on_error,
    )


def detect_cells(slice_path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    det_cfg = cfg.get("detection", {})
    primary = str(det_cfg.get("primary_model", "cellpose_cyto2"))
    secondary = str(det_cfg.get("secondary_model", "cellpose_nuclei"))
    merge_secondary = bool(det_cfg.get("merge_primary_secondary", False))
    within_slice_dedup_px = float(det_cfg.get("within_slice_dedup_px", 4.0))
    requested_cellpose = _is_cellpose_model(primary) or _is_cellpose_model(secondary)
    allow_fallback_raw = det_cfg.get("allow_fallback", None)
    allow_fallback = (
        bool(allow_fallback_raw) if allow_fallback_raw is not None else (not requested_cellpose)
    )
    cellpose_errors: list[Exception] = []

    primary_df = pd.DataFrame()
    if _is_cellpose_model(primary):
        try:
            primary_df = _run_cellpose_by_name(
                slice_path,
                primary,
                cfg,
                raise_on_error=not allow_fallback,
            )
        except CellposeDetectionError as exc:
            cellpose_errors.append(exc)
            primary_df = pd.DataFrame()
        if not primary_df.empty and not merge_secondary:
            out = _dedup_xy(primary_df, radius_px=within_slice_dedup_px)
            out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
            return out

    secondary_df = pd.DataFrame()
    if _is_cellpose_model(secondary):
        try:
            secondary_df = _run_cellpose_by_name(
                slice_path,
                secondary,
                cfg,
                raise_on_error=not allow_fallback,
            )
        except CellposeDetectionError as exc:
            cellpose_errors.append(exc)
            secondary_df = pd.DataFrame()
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

    if requested_cellpose and not allow_fallback:
        if cellpose_errors:
            raise cellpose_errors[0]
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])

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
