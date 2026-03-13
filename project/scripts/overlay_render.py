from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import rescale, rotate, resize, SimilarityTransform, warp as skwarp
from skimage import measure, morphology
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.registration import optical_flow_tvl1
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt, laplace
from scipy.interpolate import Rbf
from scripts.allen_colors import load_allen_color_map
from scripts.slice_select import select_real_slice_2d, select_label_slice_2d

# 鈹€鈹€ Allen structure tree (region names + official colors) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_STRUCTURE_TREE: dict | None = None

def _load_structure_tree() -> dict:
    global _STRUCTURE_TREE
    if _STRUCTURE_TREE is not None:
        return _STRUCTURE_TREE
    # Try project configs, then same dir as this file
    candidates = [
        Path(__file__).resolve().parent.parent / "configs" / "allen_structure_tree.json",
        Path(__file__).resolve().parent / "allen_structure_tree.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _STRUCTURE_TREE = json.load(f)
            return _STRUCTURE_TREE
    _STRUCTURE_TREE = {}
    return _STRUCTURE_TREE


# 鈹€鈹€ Utilities 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def alpha_blend(base_gray: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1).astype(np.float32)
    out = (1.0 - alpha) * base_rgb + alpha * color_mask.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _norm_u8_robust(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    if p99 <= p1:
        p1, p99 = float(np.min(x)), float(np.max(x) + 1e-6)
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (x * 255.0).astype(np.uint8)


def _warp_param(params: dict | None, key: str, default):
    if not isinstance(params, dict):
        return default
    val = params.get(key, default)
    if isinstance(default, bool):
        return bool(val)
    if isinstance(default, int):
        try:
            return int(val)
        except Exception:
            return int(default)
    if isinstance(default, float):
        try:
            return float(val)
        except Exception:
            return float(default)
    return val


def _detect_tissue(real: np.ndarray) -> dict:
    """Find the dominant tissue region in a real-space image."""
    rf = real.astype(np.float32)
    if rf.ndim == 3:
        rf = rf[..., 0]
    rh, rw = rf.shape

    p2, p98 = np.percentile(rf, 2), np.percentile(rf, 98)
    if p98 - p2 < 1:
        return {"ok": False}

    thr = p2 + (p98 - p2) * 0.12
    raw_mask = rf > thr

    try:
        closed = morphology.closing(raw_mask, morphology.disk(max(3, rw // 300)))
        lbl_tmp = measure.label(closed)
        min_area = max(500, rh * rw // 2000)
        areas = {r.label: r.area for r in measure.regionprops(lbl_tmp)}
        cleaned = np.isin(lbl_tmp, [k for k, v in areas.items() if v >= min_area])
    except Exception:
        cleaned = raw_mask

    labeled = measure.label(cleaned, connectivity=2)
    props = sorted(measure.regionprops(labeled), key=lambda r: r.area, reverse=True)
    if not props:
        return {"ok": False}

    tissue = props[0]
    ty0, tx0, ty1, tx1 = tissue.bbox
    t_cy, t_cx = tissue.centroid
    t_h, t_w = ty1 - ty0, tx1 - tx0

    # Build tight tissue mask (largest region only)
    tight_mask = labeled == tissue.label
    major_axis = getattr(tissue, "axis_major_length", None)
    if major_axis is None:
        major_axis = getattr(tissue, "major_axis_length", 1.0)
    minor_axis = getattr(tissue, "axis_minor_length", None)
    if minor_axis is None:
        minor_axis = getattr(tissue, "minor_axis_length", 1.0)

    return {
        "ok": True,
        "bbox": (int(ty0), int(tx0), int(ty1), int(tx1)),
        "centroid": (float(t_cy), float(t_cx)),
        "hw": (int(t_h), int(t_w)),
        "orientation": float(tissue.orientation),      # radians, skimage convention
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "mask": tight_mask,                            # bool array, real image space
    }


def _atlas_bbox(label: np.ndarray) -> tuple[int, int, int, int]:
    """Return (y0, x0, y1, x1) of the non-zero region of an atlas label."""
    nnz = np.argwhere(label > 0)
    if len(nnz) == 0:
        return (0, 0, label.shape[0], label.shape[1])
    ay0, ax0 = nnz.min(axis=0)
    ay1, ax1 = nnz.max(axis=0)
    return int(ay0), int(ax0), int(ay1), int(ax1)


def _similarity_warp(
    label: np.ndarray,
    a_cy: float, a_cx: float, a_h: float, a_w: float,
    t_cy: float, t_cx: float, t_h: float, t_w: float,
    phys_scale: float,
    fit_mode: str,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Apply similarity transform to warp atlas label into real image space."""
    # Fit scale: stretch atlas non-zero region to match tissue region
    fit_h = t_h / max(a_h * phys_scale, 1.0)
    fit_w = t_w / max(a_w * phys_scale, 1.0)

    mode = str(fit_mode or "contain").lower()
    if mode == "contain":
        fit_scale = min(fit_h, fit_w)
    elif mode == "cover":
        fit_scale = max(fit_h, fit_w)
    elif mode == "width-lock":
        fit_scale = fit_w
    elif mode == "height-lock":
        fit_scale = fit_h
    else:
        fit_scale = (fit_h * fit_w) ** 0.5

    total_scale = phys_scale * fit_scale

    # Translation: align atlas centroid 鈫?tissue centroid
    t_dy = t_cy - a_cy * total_scale
    t_dx = t_cx - a_cx * total_scale

    # skwarp expects inverse_map: output_coords 鈫?input_coords
    # forward: atlas 鈫?real  (scale up by total_scale)
    # inverse: real  鈫?atlas (scale down by 1/total_scale)
    inv_scale = 1.0 / total_scale
    inverse = SimilarityTransform(
        scale=inv_scale,
        translation=(-t_dx * inv_scale, -t_dy * inv_scale),
    )
    warped = skwarp(
        label.astype(np.float32),
        inverse,
        output_shape=out_shape,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    return warped.astype(np.int32), total_scale, t_dx, t_dy


def _inner_boundaries_fast(label: np.ndarray) -> np.ndarray:
    """Fast inner-boundary approximation for integer label images."""
    li = label.astype(np.int32, copy=False)
    nz = li > 0
    bd = np.zeros(li.shape, dtype=bool)

    d_h = (li[:, 1:] != li[:, :-1]) & nz[:, 1:] & nz[:, :-1]
    bd[:, 1:] |= d_h
    bd[:, :-1] |= d_h

    d_v = (li[1:, :] != li[:-1, :]) & nz[1:, :] & nz[:-1, :]
    bd[1:, :] |= d_v
    bd[:-1, :] |= d_v

    d_d1 = (li[1:, 1:] != li[:-1, :-1]) & nz[1:, 1:] & nz[:-1, :-1]
    bd[1:, 1:] |= d_d1
    bd[:-1, :-1] |= d_d1

    d_d2 = (li[1:, :-1] != li[:-1, 1:]) & nz[1:, :-1] & nz[:-1, 1:]
    bd[1:, :-1] |= d_d2
    bd[:-1, 1:] |= d_d2

    return bd.astype(np.float32)


def _atlas_edge_feature_fast(
    label: np.ndarray,
    inner_weight: float = 1.00,
    outer_weight: float = 0.55,
) -> np.ndarray:
    """Combined outer+inner atlas boundary feature."""
    atlas_mask = (label > 0).astype(np.float32)
    outer = sobel(atlas_mask)
    inner = _inner_boundaries_fast(label)
    feat = float(outer_weight) * outer + float(inner_weight) * inner
    mx = float(np.max(feat))
    if mx > 0:
        feat /= mx
    return feat.astype(np.float32)


def _edge_overlap_score(
    real_u8: np.ndarray,
    warped_label: np.ndarray,
    inner_weight: float = 1.00,
    outer_weight: float = 0.55,
) -> float:
    """Quick quality score: edge correlation with outer+inner atlas boundaries."""
    real_e = sobel(real_u8.astype(np.float32) / 255.0)
    atlas_e = _atlas_edge_feature_fast(
        warped_label.astype(np.int32),
        inner_weight=float(inner_weight),
        outer_weight=float(outer_weight),
    )
    dr = float(max(np.ptp(real_e), np.ptp(atlas_e), 1e-6))
    # Downsample for speed
    s = 4
    re = real_e[::s, ::s]
    ae = atlas_e[::s, ::s]
    try:
        return float(ssim(re, ae, data_range=dr))
    except Exception:
        return float(np.corrcoef(re.ravel(), ae.ravel())[0, 1])


def _optimize_warp(
    real_u8: np.ndarray,
    atlas_label: np.ndarray,
    init_scale: float,
    init_dx: float,
    init_dy: float,
    out_shape: tuple[int, int],
    ds: int = 4,
    maxiter: int = 150,
    init_angle_deg: float = 0.0,
    edge_inner_weight: float = 1.00,
    edge_outer_weight: float = 0.55,
    outside_penalty: float = 2.2,
) -> tuple[float, float, float, float]:
    """
    Refine (scale, angle, dx, dy) by maximizing atlas-boundary 鈫?real-edge overlap.
    Works on downsampled images (ds脳) for speed.
    Returns (refined_scale, refined_angle_rad, refined_dx, refined_dy).
    """
    from scipy.optimize import minimize

    rh, rw = out_shape
    real_ds = real_u8[::ds, ::ds].astype(np.float32) / 255.0
    atlas_ds = atlas_label[::ds, ::ds].astype(np.float32)  # pre-slice once
    ds_h, ds_w = real_ds.shape
    real_e = sobel(real_ds)
    p = float(np.percentile(real_e, 66))
    real_e = np.clip((real_e - p) / (float(np.max(real_e)) - p + 1e-6), 0.0, 1.0)

    # Build tissue mask from real image (downsampled)
    tissue_mask = real_ds > 0.05
    # Use tight tissue mask from full-res detection if available; else derive from downsampled
    from skimage.morphology import disk as sk_disk
    # Small dilation (1px at ds=4 鈮?4px real) to allow for boundary noise
    dilated = morphology.dilation(tissue_mask.astype(np.uint8), sk_disk(1)).astype(bool)

    def _warp_score(params):
        sf, angle_deg, dx_adj, dy_adj = params
        # Clamp to valid ranges (Nelder-Mead ignores bounds otherwise)
        sf = float(np.clip(sf, 0.7, 1.4))
        angle_deg = float(np.clip(angle_deg, -18.0, 18.0))
        dx_adj = float(np.clip(dx_adj, -300.0, 300.0))
        dy_adj = float(np.clip(dy_adj, -300.0, 300.0))
        s = init_scale * sf
        angle_rad = np.deg2rad(angle_deg)
        dx = init_dx + dx_adj
        dy = init_dy + dy_adj
        # Both atlas and real are downsampled by ds, so scale is unchanged;
        # only translation (in real-image pixels) is divided by ds
        fwd = SimilarityTransform(scale=s, rotation=angle_rad,
                                  translation=(dx / ds, dy / ds))
        try:
            w = skwarp(
                atlas_ds,
                fwd.inverse,
                output_shape=(ds_h, ds_w),
                order=0, preserve_range=True, mode="constant", cval=0.0,
            )
        except Exception:
            return 0.0
        wi = np.rint(w).astype(np.int32)
        atlas_mask = (wi > 0).astype(np.float32)
        inner = _inner_boundaries_fast(wi)
        outer = sobel(atlas_mask)
        inner_ov = float(np.sum(inner * real_e))
        outer_ov = float(np.sum(outer * real_e))
        overlap = float(edge_inner_weight) * inner_ov + float(edge_outer_weight) * outer_ov
        # Strong penalty: atlas pixels outside tissue (prevents atlas from growing past tissue edge)
        outside = float(np.sum(atlas_mask * (~dilated).astype(float)))
        atlas_total = float(np.sum(atlas_mask)) + 1e-6
        outside_frac = outside / atlas_total  # fraction of atlas outside tissue [0..1]
        return -(overlap - float(outside_penalty) * outside_frac * overlap)

    x0 = [1.0, float(init_angle_deg), 0.0, 0.0]  # sf=1, angle=init, dx_adj=0, dy_adj=0
    result = minimize(
        _warp_score, x0, method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 0.3, "fatol": 1e-6, "adaptive": True},
    )
    sf, angle_deg, dx_adj, dy_adj = result.x
    sf = float(np.clip(sf, 0.7, 1.4))
    angle_deg = float(np.clip(angle_deg, -18.0, 18.0))
    dx_adj = float(np.clip(dx_adj, -300.0, 300.0))
    dy_adj = float(np.clip(dy_adj, -300.0, 300.0))
    return (
        float(init_scale * sf),
        float(np.deg2rad(angle_deg)),
        float(init_dx + dx_adj),
        float(init_dy + dy_adj),
    )


def _apply_warp(
    label: np.ndarray,
    scale: float, angle_rad: float, dx: float, dy: float,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Apply similarity transform (scale, rotation, translation) to atlas label."""
    fwd = SimilarityTransform(scale=scale, rotation=angle_rad, translation=(dx, dy))
    warped = skwarp(
        label.astype(np.float32),
        fwd.inverse,
        output_shape=out_shape,
        order=0, preserve_range=True, mode="constant", cval=0.0,
    )
    return warped.astype(np.int32)


def _atlas_boundary_feature(label: np.ndarray) -> np.ndarray:
    """Build a dense feature map that emphasizes both outer + inner atlas boundaries."""
    outer = find_boundaries(label > 0, mode="outer", connectivity=2).astype(np.float32)
    inner = find_boundaries(label.astype(np.int32), mode="inner", connectivity=2).astype(np.float32)
    feat = 0.55 * outer + 1.00 * inner
    feat = gaussian_filter(feat, sigma=0.9)
    mx = float(np.max(feat))
    if mx > 0:
        feat /= mx
    return feat.astype(np.float32)


def _real_edge_feature(real_u8: np.ndarray, tissue_mask: np.ndarray | None = None) -> np.ndarray:
    """Real-image edge feature map used for non-linear refinement."""
    x = real_u8.astype(np.float32) / 255.0
    e = sobel(x)
    p = float(np.percentile(e, 68))
    e = np.clip((e - p) / (float(np.max(e)) - p + 1e-6), 0.0, 1.0)
    if tissue_mask is not None:
        support = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(4)).astype(np.float32)
        e *= support
    return e.astype(np.float32)


def _warp_label_with_flow(label: np.ndarray, flow_v: np.ndarray, flow_u: np.ndarray) -> np.ndarray:
    """Warp label image with dense flow using nearest-neighbor sampling."""
    h, w = label.shape
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
        mode="constant",
        cval=0.0,
    )
    return warped.astype(np.int32)


def _mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0)
    bb = (b > 0)
    den = float(np.sum(aa) + np.sum(bb)) + 1e-6
    inter = float(np.sum(aa & bb))
    return 2.0 * inter / den


def _alignment_quality(
    real_u8: np.ndarray,
    label: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    edge_inner_weight: float = 1.00,
    edge_outer_weight: float = 0.55,
    mask_dice_weight: float = 0.35,
) -> float:
    """Composite quality score: edge overlap + outer mask agreement."""
    edge_score = _edge_overlap_score(
        real_u8,
        label,
        inner_weight=float(edge_inner_weight),
        outer_weight=float(edge_outer_weight),
    )
    if tissue_mask is None:
        return float(edge_score)
    dice = _mask_dice(label > 0, tissue_mask)
    return float(edge_score + float(mask_dice_weight) * dice)


def _coverage_stats(label: np.ndarray, tissue_mask: np.ndarray | None) -> dict:
    atlas = (label > 0)
    if tissue_mask is None or tissue_mask.shape != atlas.shape or not np.any(tissue_mask):
        cov = float(np.mean(atlas))
        return {
            "coverage": cov,
            "left_coverage": cov,
            "right_coverage": cov,
            "lr_gap": 0.0,
            "min_lr": cov,
            "missing_tissue_px": 0,
        }

    tm = tissue_mask.astype(bool)
    cov = float(np.mean(atlas[tm]))
    yy, xx = np.nonzero(tm)
    if len(xx) == 0:
        return {
            "coverage": cov,
            "left_coverage": cov,
            "right_coverage": cov,
            "lr_gap": 0.0,
            "min_lr": cov,
            "missing_tissue_px": int(np.sum(tm & (~atlas))),
        }

    xmin, xmax = int(np.min(xx)), int(np.max(xx))
    xmid = (xmin + xmax) // 2
    cols = np.arange(atlas.shape[1])[None, :]
    left_tm = tm & (cols <= xmid)
    right_tm = tm & (cols > xmid)
    left_cov = float(np.mean(atlas[left_tm])) if np.any(left_tm) else cov
    right_cov = float(np.mean(atlas[right_tm])) if np.any(right_tm) else cov
    gap = float(abs(left_cov - right_cov))
    return {
        "coverage": float(cov),
        "left_coverage": float(left_cov),
        "right_coverage": float(right_cov),
        "lr_gap": float(gap),
        "min_lr": float(min(left_cov, right_cov)),
        "missing_tissue_px": int(np.sum(tm & (~atlas))),
    }


def _candidate_objective_score(
    q: float,
    cov: dict,
    tissue_aspect: float,
    full_aspect: float,
    half_aspect: float,
    candidate_name: str,
    warp_params: dict | None = None,
) -> float:
    cov_w = float(_warp_param(warp_params, "linear_cov_weight", 0.34))
    minlr_w = float(_warp_param(warp_params, "linear_minlr_weight", 0.30))
    gap_pen = float(_warp_param(warp_params, "linear_gap_penalty", 1.08))
    gap_allow = float(_warp_param(warp_params, "linear_gap_allow", 0.14))
    min_cov_target = float(_warp_param(warp_params, "linear_min_coverage_target", 0.93))
    min_cov_pen = float(_warp_param(warp_params, "linear_min_cov_penalty", 2.15))
    minlr_floor = float(_warp_param(warp_params, "linear_minlr_floor", 0.78))
    minlr_floor_pen = float(_warp_param(warp_params, "linear_minlr_floor_penalty", 1.30))
    aspect_w = float(_warp_param(warp_params, "linear_aspect_prior_weight", 0.06))

    base = float(q)
    c = float(cov.get("coverage", 0.0))
    min_lr = float(cov.get("min_lr", c))
    lr_gap = float(cov.get("lr_gap", 0.0))
    obj = base + cov_w * c + minlr_w * min_lr
    obj -= gap_pen * max(0.0, lr_gap - gap_allow)
    obj -= min_cov_pen * max(0.0, min_cov_target - c)
    obj -= minlr_floor_pen * max(0.0, minlr_floor - min_lr)

    full_err = abs(float(tissue_aspect) - float(full_aspect))
    half_err = abs(float(tissue_aspect) - float(half_aspect))
    if candidate_name == "full":
        obj += aspect_w * (half_err - full_err)
    else:
        obj += aspect_w * (full_err - half_err)
    return float(obj)


def _clip_label_to_tissue(label: np.ndarray, tissue_mask: np.ndarray | None, pad: int = 5) -> np.ndarray:
    if tissue_mask is None:
        return label.astype(np.int32)
    clip = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(max(1, int(pad)))).astype(bool)
    return np.where(clip, label, 0).astype(np.int32)


def _remove_small_label_islands(
    label: np.ndarray,
    min_area_px: int,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = label.astype(np.int32).copy()
    min_area = max(1, int(min_area_px))
    ids = np.unique(out.astype(np.int32))
    ids = ids[ids > 0]
    for rid in ids.tolist():
        rid_mask = (out == int(rid))
        if not np.any(rid_mask):
            continue
        cc = measure.label(rid_mask, connectivity=2)
        if int(np.max(cc)) <= 1:
            continue
        props = measure.regionprops(cc)
        if not props:
            continue
        keep = [int(p.label) for p in props if int(p.area) >= min_area]
        if not keep:
            keep = [int(max(props, key=lambda x: x.area).label)]
        remove = rid_mask & (~np.isin(cc, keep))
        if support_mask is not None:
            remove &= support_mask.astype(bool)
        out[remove] = 0
    return out.astype(np.int32)


def _fill_small_voids_nearest(
    label: np.ndarray,
    support_mask: np.ndarray,
    max_area_px: int,
) -> np.ndarray:
    out = label.astype(np.int32).copy()
    support = support_mask.astype(bool)
    holes = support & (out <= 0)
    if not np.any(holes):
        return out

    cc = measure.label(holes, connectivity=2)
    props = measure.regionprops(cc)
    if not props:
        return out
    small_ids = [int(p.label) for p in props if int(p.area) <= max(1, int(max_area_px))]
    if not small_ids:
        return out

    fill_mask = np.isin(cc, small_ids)
    valid = (out > 0) & support
    if not np.any(valid):
        return out

    _, indices = distance_transform_edt((~valid).astype(np.uint8), return_indices=True)
    iy = indices[0]
    ix = indices[1]
    vals = out[iy[fill_mask], ix[fill_mask]]
    out[fill_mask] = vals.astype(np.int32)
    return out.astype(np.int32)


def _fill_outer_missing_tissue_nearest(
    label: np.ndarray,
    support_mask: np.ndarray,
    max_dist_px: float = 220.0,
) -> np.ndarray:
    out = label.astype(np.int32).copy()
    support = support_mask.astype(bool)
    missing = support & (out <= 0)
    if not np.any(missing):
        return out

    ring = find_boundaries(support.astype(np.uint8), mode="inner", connectivity=2)
    cc = measure.label(missing, connectivity=2)
    if int(np.max(cc)) <= 0:
        return out
    touch_ids = np.unique(cc[ring & (cc > 0)]).astype(np.int32)
    if touch_ids.size == 0:
        return out

    fill_mask = np.isin(cc, touch_ids)
    valid = (out > 0) & support
    if not np.any(valid):
        return out

    dist, indices = distance_transform_edt((~valid).astype(np.uint8), return_indices=True)
    if float(max_dist_px) > 0:
        fill_mask &= (dist <= float(max_dist_px))
    if not np.any(fill_mask):
        return out

    iy = indices[0]
    ix = indices[1]
    vals = out[iy[fill_mask], ix[fill_mask]]
    out[fill_mask] = vals.astype(np.int32)
    return out.astype(np.int32)


def _cleanup_label_topology(
    label: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    warp_params: dict | None = None,
) -> np.ndarray:
    if not bool(_warp_param(warp_params, "cleanup_enable", True)):
        return label.astype(np.int32)

    out = label.astype(np.int32).copy()
    support_pad = int(_warp_param(warp_params, "cleanup_support_pad", 4))
    if tissue_mask is not None and tissue_mask.shape == out.shape and np.any(tissue_mask):
        support = morphology.dilation(
            tissue_mask.astype(np.uint8),
            morphology.disk(max(1, int(support_pad))),
        ).astype(bool)
        out = np.where(support, out, 0).astype(np.int32)
        tissue_area = float(np.sum(tissue_mask.astype(bool)))
    else:
        support = (out > 0)
        tissue_area = float(np.sum(support))
    if tissue_area <= 1.0:
        return out.astype(np.int32)

    island_ratio = float(_warp_param(warp_params, "cleanup_min_region_area_ratio", 2.2e-4))
    hole_ratio = float(_warp_param(warp_params, "cleanup_max_hole_area_ratio", 8.5e-4))
    island_min_px = int(max(24, round(tissue_area * island_ratio)))
    hole_max_px = int(max(30, round(tissue_area * hole_ratio)))

    out = _remove_small_label_islands(
        out,
        min_area_px=island_min_px,
        support_mask=support,
    )
    out = _fill_small_voids_nearest(
        out,
        support_mask=support,
        max_area_px=hole_max_px,
    )
    if bool(_warp_param(warp_params, "cleanup_fill_outer_missing", True)):
        outer_fill_max_dist = float(_warp_param(warp_params, "cleanup_outer_fill_max_dist_px", 240.0))
        out = _fill_outer_missing_tissue_nearest(
            out,
            support_mask=support,
            max_dist_px=outer_fill_max_dist,
        )
    return out.astype(np.int32)


def _row_mode_int(values: np.ndarray) -> np.ndarray:
    """Vectorized row-wise mode for small integer neighborhoods."""
    if values.size == 0:
        return np.zeros((0,), dtype=np.int32)
    sv = np.sort(values.astype(np.int32), axis=1)
    n, m = sv.shape
    best = sv[:, 0].copy()
    best_cnt = np.ones(n, dtype=np.int16)
    cur = sv[:, 0].copy()
    cur_cnt = np.ones(n, dtype=np.int16)
    for k in range(1, m):
        same = sv[:, k] == cur
        cur_cnt = np.where(same, cur_cnt + 1, 1).astype(np.int16)
        cur = np.where(same, cur, sv[:, k])
        better = cur_cnt > best_cnt
        best_cnt = np.where(better, cur_cnt, best_cnt)
        best = np.where(better, cur, best)
    return best.astype(np.int32)


def _smooth_label_edges(
    label: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    iterations: int = 1,
) -> np.ndarray:
    out = label.astype(np.int32).copy()
    iters = max(0, int(iterations))
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    for _ in range(iters):
        bd = find_boundaries(out.astype(np.int32), mode="thick", connectivity=2)
        if tissue_mask is not None:
            support = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(2)).astype(bool)
            bd &= support
        if not np.any(bd):
            break
        coords = np.argwhere(bd)
        if coords.size == 0:
            break
        yy = coords[:, 0]
        xx = coords[:, 1]
        pad = np.pad(out, 1, mode="edge")
        yp = yy + 1
        xp = xx + 1
        neigh = np.stack([pad[yp + dy, xp + dx] for dy, dx in offsets], axis=1)
        mode_vals = _row_mode_int(neigh)
        cand = out.copy()
        cand[yy, xx] = mode_vals
        if tissue_mask is not None:
            cand = _clip_label_to_tissue(cand, tissue_mask=tissue_mask, pad=2)
        out = cand
    return out.astype(np.int32)


def _gaussian_vote_boundary_smooth(
    label: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    warp_params: dict | None = None,
) -> np.ndarray:
    """
    Smooth staircase artifacts on label boundaries using Gaussian vote interpolation.
    Inspired by ITK Generic/Label Gaussian interpolator for multilabel images.
    Applied only on a boundary band to preserve interior topology.
    """
    if not bool(_warp_param(warp_params, "gaussian_label_interp_enable", False)):
        return label.astype(np.int32)

    out = label.astype(np.int32).copy()
    sigma = float(_warp_param(warp_params, "gaussian_label_interp_sigma", 1.0))
    band_radius = int(_warp_param(warp_params, "gaussian_label_interp_band_radius", 2))
    min_area = int(_warp_param(warp_params, "gaussian_label_interp_min_area_px", 24))
    max_labels = int(_warp_param(warp_params, "gaussian_label_interp_max_labels", 160))

    sigma = float(np.clip(sigma, 0.5, 2.6))
    band_radius = int(np.clip(band_radius, 1, 6))
    min_area = int(max(4, min_area))
    max_labels = int(max(8, max_labels))

    bd = find_boundaries(out, mode="thick", connectivity=2)
    band = morphology.dilation(
        bd.astype(np.uint8),
        morphology.disk(band_radius),
    ).astype(bool)
    if tissue_mask is not None and tissue_mask.shape == out.shape and np.any(tissue_mask):
        support = morphology.dilation(
            tissue_mask.astype(np.uint8),
            morphology.disk(2),
        ).astype(bool)
        band &= support
    if not np.any(band):
        return out.astype(np.int32)

    ids, counts = np.unique(out[out > 0], return_counts=True)
    if len(ids) == 0:
        return out.astype(np.int32)

    order = np.argsort(counts)[::-1]
    ids = ids[order]
    counts = counts[order]
    if len(ids) > max_labels:
        ids = ids[:max_labels]
        counts = counts[:max_labels]

    best_score = np.zeros(out.shape, dtype=np.float32)
    best_label = np.zeros(out.shape, dtype=np.int32)

    for rid, cnt in zip(ids.tolist(), counts.tolist()):
        if int(cnt) < min_area:
            continue
        m = (out == int(rid)).astype(np.float32)
        s = gaussian_filter(m, sigma=sigma).astype(np.float32)
        upd = band & (s > best_score)
        if np.any(upd):
            best_score[upd] = s[upd]
            best_label[upd] = int(rid)

    upd_final = band & (best_label > 0)
    if np.any(upd_final):
        out[upd_final] = best_label[upd_final]
    if tissue_mask is not None and tissue_mask.shape == out.shape and np.any(tissue_mask):
        out = _clip_label_to_tissue(out, tissue_mask=tissue_mask, pad=2)
    return out.astype(np.int32)


def _sample_mask_points(mask: np.ndarray, max_points: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if n > max_points:
        idx = np.linspace(0, n - 1, int(max_points), dtype=np.int64)
        ys = ys[idx]
        xs = xs[idx]
    return np.stack([ys, xs], axis=1).astype(np.int32)


def _refine_warp_flow_candidate(
    real_u8: np.ndarray,
    label: np.ndarray,
    tissue_mask: np.ndarray | None,
    q_before: float,
    warp_params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    h, w = label.shape
    if h < 40 or w < 40:
        return label, {"ok": False, "reason": "small_image"}

    edge_inner_weight = float(_warp_param(warp_params, "edge_inner_weight", 1.00))
    edge_outer_weight = float(_warp_param(warp_params, "edge_outer_weight", 0.55))
    mask_dice_weight = float(_warp_param(warp_params, "mask_dice_weight", 0.35))
    flow_max_disp_ratio = float(_warp_param(warp_params, "flow_max_disp_ratio", 0.10))
    flow_max_disp_min = float(_warp_param(warp_params, "flow_max_disp_min_px", 10.0))
    flow_support_dilate = int(_warp_param(warp_params, "flow_support_dilate", 7))
    lap_iter = int(_warp_param(warp_params, "laplacian_smooth_iter", 1))
    lap_lambda = float(_warp_param(warp_params, "laplacian_lambda", 0.18))

    if max(h, w) >= 1800:
        ds = 4
    elif max(h, w) >= 1100:
        ds = 3
    else:
        ds = 2

    out_h = max(64, h // ds)
    out_w = max(64, w // ds)
    real_feat = _real_edge_feature(real_u8, tissue_mask=tissue_mask)
    atlas_feat = _atlas_boundary_feature(label)

    real_ds = resize(
        real_feat,
        (out_h, out_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    atlas_ds = resize(
        atlas_feat,
        (out_h, out_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    def _build_candidate(
        flow_v_ds: np.ndarray,
        flow_u_ds: np.ndarray,
        sign: float,
        tag: str,
    ) -> tuple[np.ndarray, float, dict]:
        sy = float(h) / float(out_h)
        sx = float(w) / float(out_w)
        flow_v = resize(
            flow_v_ds,
            (h, w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32) * sy * float(sign)
        flow_u = resize(
            flow_u_ds,
            (h, w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32) * sx * float(sign)

        flow_v = gaussian_filter(flow_v, sigma=1.6).astype(np.float32)
        flow_u = gaussian_filter(flow_u, sigma=1.6).astype(np.float32)

        support = None
        if tissue_mask is not None:
            support = morphology.dilation(
                tissue_mask.astype(np.uint8),
                morphology.disk(max(1, int(flow_support_dilate))),
            ).astype(np.float32)
            flow_v *= support
            flow_u *= support
        flow_v, flow_u = _laplacian_smooth_flow(
            flow_v,
            flow_u,
            iterations=lap_iter,
            lam=lap_lambda,
            support_mask=support,
        )

        max_disp = float(max(flow_max_disp_min, min(h, w) * flow_max_disp_ratio))
        flow_v, flow_u = _clip_flow_magnitude(flow_v, flow_u, max_disp=max_disp)

        refined = _warp_label_with_flow(label, flow_v, flow_u)
        refined = _clip_label_to_tissue(refined, tissue_mask, pad=5)
        q_after = _alignment_quality(
            real_u8,
            refined,
            tissue_mask=tissue_mask,
            edge_inner_weight=edge_inner_weight,
            edge_outer_weight=edge_outer_weight,
            mask_dice_weight=mask_dice_weight,
        )
        return refined, float(q_after), {
            "tag": str(tag),
            "downsample": int(ds),
            "max_displacement_px": float(max_disp),
            "score_after": float(q_after),
        }

    candidates: list[tuple[np.ndarray, float, dict]] = []
    try:
        fv_ra, fu_ra = optical_flow_tvl1(
            real_ds,
            atlas_ds,
            attachment=8.0,
            tightness=0.42,
            num_warp=8,
            num_iter=50,
            tol=1e-4,
            prefilter=True,
            dtype=np.float32,
        )
        candidates.append(_build_candidate(fv_ra, fu_ra, 1.0, "flow_real_atlas"))
    except Exception as e:
        pass

    try:
        fv_ar, fu_ar = optical_flow_tvl1(
            atlas_ds,
            real_ds,
            attachment=8.0,
            tightness=0.42,
            num_warp=8,
            num_iter=50,
            tol=1e-4,
            prefilter=True,
            dtype=np.float32,
        )
        candidates.append(_build_candidate(fv_ar, fu_ar, -1.0, "flow_atlas_real_neg"))
        candidates.append(_build_candidate(fv_ar, fu_ar, 1.0, "flow_atlas_real_pos"))
    except Exception:
        pass

    if not candidates:
        return label, {"ok": False, "reason": "flow_failed"}

    best_label, best_q, best_meta = max(candidates, key=lambda x: x[1])
    best_meta = {
        "ok": True,
        "method": "flow",
        "score_before": float(q_before),
        "score_after": float(best_q),
        "score_delta": float(best_q - q_before),
        **best_meta,
    }
    return best_label, best_meta


def _filter_liquify_pairs(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    max_disp: float,
    min_points: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    if len(src_xy) == 0 or len(dst_xy) == 0:
        return src_xy, dst_xy

    du = (dst_xy[:, 0] - src_xy[:, 0]).astype(np.float32)
    dv = (dst_xy[:, 1] - src_xy[:, 1]).astype(np.float32)
    mag = np.sqrt(du * du + dv * dv).astype(np.float32)
    keep = np.isfinite(mag) & (mag <= float(max_disp))
    if int(np.sum(keep)) >= int(min_points):
        src_xy = src_xy[keep]
        dst_xy = dst_xy[keep]
        mag = mag[keep]

    if len(src_xy) < int(min_points):
        return src_xy, dst_xy

    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med))) + 1e-6
    limit = max(med + 3.8 * mad, med + 2.5)
    keep2 = mag <= float(limit)
    if int(np.sum(keep2)) >= int(min_points):
        src_xy = src_xy[keep2]
        dst_xy = dst_xy[keep2]

    return src_xy, dst_xy


def _dense_field_from_matches(
    h: int,
    w: int,
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    sigma_px: float,
    max_disp: float,
    tissue_mask: np.ndarray | None = None,
    support_dilate: int = 10,
    lap_iterations: int = 0,
    lap_lambda: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    disp_u = np.zeros((h, w), dtype=np.float32)
    disp_v = np.zeros((h, w), dtype=np.float32)
    wt = np.zeros((h, w), dtype=np.float32)

    sx_i = np.clip(np.rint(src_xy[:, 0]).astype(np.int32), 0, w - 1)
    sy_i = np.clip(np.rint(src_xy[:, 1]).astype(np.int32), 0, h - 1)
    du = (dst_xy[:, 0] - src_xy[:, 0]).astype(np.float32)
    dv = (dst_xy[:, 1] - src_xy[:, 1]).astype(np.float32)
    np.add.at(disp_u, (sy_i, sx_i), du)
    np.add.at(disp_v, (sy_i, sx_i), dv)
    np.add.at(wt, (sy_i, sx_i), 1.0)

    sigma = float(max(1.0, sigma_px))
    den = gaussian_filter(wt, sigma=sigma) + 1e-4
    disp_u = gaussian_filter(disp_u, sigma=sigma) / den
    disp_v = gaussian_filter(disp_v, sigma=sigma) / den

    support = None
    if tissue_mask is not None:
        support = morphology.dilation(
            tissue_mask.astype(np.uint8),
            morphology.disk(max(1, int(support_dilate))),
        ).astype(np.float32)
        disp_u *= support
        disp_v *= support

    disp_v, disp_u = _laplacian_smooth_flow(
        disp_v,
        disp_u,
        iterations=max(0, int(lap_iterations)),
        lam=float(lap_lambda),
        support_mask=support,
    )
    disp_v, disp_u = _clip_flow_magnitude(disp_v, disp_u, max_disp=float(max_disp))
    return disp_v.astype(np.float32), disp_u.astype(np.float32)


def _clip_flow_magnitude(
    flow_v: np.ndarray,
    flow_u: np.ndarray,
    max_disp: float,
) -> tuple[np.ndarray, np.ndarray]:
    fv = flow_v.astype(np.float32, copy=True)
    fu = flow_u.astype(np.float32, copy=True)
    md = float(max(0.5, max_disp))
    mag = np.sqrt(fv * fv + fu * fu).astype(np.float32) + 1e-6
    scale = np.minimum(1.0, md / mag).astype(np.float32)
    fv *= scale
    fu *= scale
    return fv, fu


def _laplacian_smooth_flow(
    flow_v: np.ndarray,
    flow_u: np.ndarray,
    iterations: int = 1,
    lam: float = 0.18,
    support_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Laplacian diffusion smoothing for dense displacement fields."""
    iters = max(0, int(iterations))
    if iters <= 0:
        return flow_v.astype(np.float32), flow_u.astype(np.float32)
    alpha = float(np.clip(lam, 0.0, 0.40))
    fv = flow_v.astype(np.float32, copy=True)
    fu = flow_u.astype(np.float32, copy=True)
    sup = None
    if support_mask is not None:
        sup = (support_mask > 0).astype(np.float32)
    for _ in range(iters):
        fv = fv + alpha * laplace(fv, mode="nearest").astype(np.float32)
        fu = fu + alpha * laplace(fu, mode="nearest").astype(np.float32)
        if sup is not None:
            fv *= sup
            fu *= sup
    return fv.astype(np.float32), fu.astype(np.float32)


def _tps_field_from_matches(
    h: int,
    w: int,
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    max_ctrl: int,
    smooth: float,
    grid_step: int,
    max_disp: float,
    tissue_mask: np.ndarray | None = None,
    support_dilate: int = 10,
    lap_iterations: int = 0,
    lap_lambda: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    if len(src_xy) < 20:
        raise ValueError("insufficient_matches_for_tps")

    n = len(src_xy)
    if n > int(max_ctrl):
        idx = np.linspace(0, n - 1, int(max_ctrl), dtype=np.int64)
        src = src_xy[idx]
        dst = dst_xy[idx]
    else:
        src = src_xy
        dst = dst_xy

    du = (dst[:, 0] - src[:, 0]).astype(np.float32)
    dv = (dst[:, 1] - src[:, 1]).astype(np.float32)

    step = max(2, int(grid_step))
    gy = np.arange(0, h, step, dtype=np.float32)
    gx = np.arange(0, w, step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)

    rbf_u = Rbf(
        src[:, 0].astype(np.float64),
        src[:, 1].astype(np.float64),
        du.astype(np.float64),
        function="thin_plate",
        smooth=float(max(0.0, smooth)),
    )
    rbf_v = Rbf(
        src[:, 0].astype(np.float64),
        src[:, 1].astype(np.float64),
        dv.astype(np.float64),
        function="thin_plate",
        smooth=float(max(0.0, smooth)),
    )

    du_coarse = rbf_u(grid_x, grid_y).astype(np.float32)
    dv_coarse = rbf_v(grid_x, grid_y).astype(np.float32)
    disp_u = resize(
        du_coarse,
        (h, w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    disp_v = resize(
        dv_coarse,
        (h, w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    disp_u = gaussian_filter(disp_u, sigma=1.0).astype(np.float32)
    disp_v = gaussian_filter(disp_v, sigma=1.0).astype(np.float32)

    support = None
    if tissue_mask is not None:
        support = morphology.dilation(
            tissue_mask.astype(np.uint8),
            morphology.disk(max(1, int(support_dilate))),
        ).astype(np.float32)
        disp_u *= support
        disp_v *= support

    disp_v, disp_u = _laplacian_smooth_flow(
        disp_v,
        disp_u,
        iterations=max(0, int(lap_iterations)),
        lam=float(lap_lambda),
        support_mask=support,
    )
    disp_v, disp_u = _clip_flow_magnitude(disp_v, disp_u, max_disp=float(max_disp))
    return disp_v.astype(np.float32), disp_u.astype(np.float32)


def _refine_warp_ants_candidate(
    real_u8: np.ndarray,
    label: np.ndarray,
    tissue_mask: np.ndarray | None,
    q_before: float,
    warp_params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    ants_enabled = bool(_warp_param(warp_params, "enable_ants_refine", False))
    if not ants_enabled:
        return label, {"ok": False, "reason": "ants_disabled"}

    try:
        import ants  # type: ignore
    except Exception as e:
        return label, {"ok": False, "reason": f"ants_unavailable: {e}"}

    h, w = label.shape
    if h < 60 or w < 60 or float(np.mean(label > 0)) < 0.01:
        return label, {"ok": False, "reason": "insufficient_label_coverage"}

    edge_inner_weight = float(_warp_param(warp_params, "edge_inner_weight", 1.00))
    edge_outer_weight = float(_warp_param(warp_params, "edge_outer_weight", 0.55))
    mask_dice_weight = float(_warp_param(warp_params, "mask_dice_weight", 0.35))
    ants_transform = str(_warp_param(warp_params, "ants_transform", "SyNOnly"))
    ants_max_dim = int(_warp_param(warp_params, "ants_max_dim", 880))
    ants_support_dilate = int(_warp_param(warp_params, "ants_support_dilate", 10))
    ants_max_disp_ratio = float(_warp_param(warp_params, "ants_max_disp_ratio", 0.085))
    ants_max_disp_min = float(_warp_param(warp_params, "ants_max_disp_min_px", 10.0))
    ants_max_mask_dice_drop = float(_warp_param(warp_params, "ants_max_mask_dice_drop", 0.006))
    ants_max_interior_loss_frac = float(_warp_param(warp_params, "ants_max_interior_loss_frac", 0.012))
    lap_iter = int(_warp_param(warp_params, "ants_laplacian_iter", _warp_param(warp_params, "laplacian_smooth_iter", 1)))
    lap_lambda = float(_warp_param(warp_params, "ants_laplacian_lambda", _warp_param(warp_params, "laplacian_lambda", 0.18)))
    keep_temp = bool(_warp_param(warp_params, "ants_keep_temp", False))

    real_feat = _real_edge_feature(real_u8, tissue_mask=tissue_mask).astype(np.float32)
    atlas_feat = _atlas_boundary_feature(label).astype(np.float32)
    if float(np.mean(real_feat > 0.05)) < 0.001:
        return label, {"ok": False, "reason": "weak_real_edges"}

    mx_r = float(np.max(real_feat))
    mx_a = float(np.max(atlas_feat))
    if mx_r > 0:
        real_feat = real_feat / mx_r
    if mx_a > 0:
        atlas_feat = atlas_feat / mx_a

    max_dim = max(h, w)
    if max_dim > max(64, ants_max_dim):
        scale = float(max(64, ants_max_dim)) / float(max_dim)
        out_h = max(64, int(round(h * scale)))
        out_w = max(64, int(round(w * scale)))
    else:
        out_h, out_w = h, w

    real_ds = resize(
        real_feat,
        (out_h, out_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    atlas_ds = resize(
        atlas_feat,
        (out_h, out_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    tx_used: list[str] = []
    try:
        fixed = ants.from_numpy(real_ds)
        moving = ants.from_numpy(atlas_ds)
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=str(ants_transform),
            verbose=False,
        )
        tx_used = [str(p) for p in reg.get("fwdtransforms", [])]
        if not tx_used:
            return label, {"ok": False, "reason": "ants_no_transform"}

        yy, xx = np.meshgrid(
            np.arange(out_h, dtype=np.float32),
            np.arange(out_w, dtype=np.float32),
            indexing="ij",
        )
        y_img = ants.from_numpy(yy)
        x_img = ants.from_numpy(xx)
        wy = ants.apply_transforms(
            fixed=fixed,
            moving=y_img,
            transformlist=tx_used,
            interpolator="linear",
        ).numpy().astype(np.float32)
        wx = ants.apply_transforms(
            fixed=fixed,
            moving=x_img,
            transformlist=tx_used,
            interpolator="linear",
        ).numpy().astype(np.float32)
        flow_v_ds = wy - yy
        flow_u_ds = wx - xx
    except Exception as e:
        return label, {"ok": False, "reason": f"ants_registration_failed: {e}"}
    finally:
        if not keep_temp:
            try:
                for arr in ("fwdtransforms", "invtransforms"):
                    for p in reg.get(arr, []) if "reg" in locals() and isinstance(reg, dict) else []:
                        pp = Path(str(p))
                        if pp.exists():
                            pp.unlink(missing_ok=True)
            except Exception:
                pass

    if (out_h, out_w) != (h, w):
        sy = float(h) / float(out_h)
        sx = float(w) / float(out_w)
        flow_v = resize(
            flow_v_ds,
            (h, w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32) * sy
        flow_u = resize(
            flow_u_ds,
            (h, w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32) * sx
    else:
        flow_v = flow_v_ds.astype(np.float32)
        flow_u = flow_u_ds.astype(np.float32)

    support = None
    if tissue_mask is not None:
        support = morphology.dilation(
            tissue_mask.astype(np.uint8),
            morphology.disk(max(1, int(ants_support_dilate))),
        ).astype(np.float32)
        flow_v *= support
        flow_u *= support

    flow_v, flow_u = _laplacian_smooth_flow(
        flow_v,
        flow_u,
        iterations=max(0, int(lap_iter)),
        lam=float(lap_lambda),
        support_mask=support,
    )
    max_disp = float(max(ants_max_disp_min, min(h, w) * ants_max_disp_ratio))
    flow_v, flow_u = _clip_flow_magnitude(flow_v, flow_u, max_disp=max_disp)

    refined = _warp_label_with_flow(label, flow_v, flow_u)
    refined = _clip_label_to_tissue(refined, tissue_mask=tissue_mask, pad=5)
    baseline_dice = None
    refined_dice = None
    interior_loss_frac = None
    if tissue_mask is not None:
        baseline_dice = float(_mask_dice(label > 0, tissue_mask))
        refined_dice = float(_mask_dice(refined > 0, tissue_mask))
        if refined_dice < baseline_dice - float(ants_max_mask_dice_drop):
            return label, {
                "ok": False,
                "reason": "ants_outer_mask_drop",
                "baseline_mask_dice": float(baseline_dice),
                "refined_mask_dice": float(refined_dice),
                "max_mask_dice_drop": float(ants_max_mask_dice_drop),
            }
        interior = morphology.erosion(tissue_mask.astype(np.uint8), morphology.disk(6)).astype(bool)
        base_inside = (label > 0) & interior
        den = float(np.sum(base_inside)) + 1e-6
        lost_inside = base_inside & (refined <= 0)
        interior_loss_frac = float(np.sum(lost_inside) / den)
        if interior_loss_frac > float(ants_max_interior_loss_frac):
            return label, {
                "ok": False,
                "reason": "ants_interior_label_loss",
                "interior_loss_frac": float(interior_loss_frac),
                "max_interior_loss_frac": float(ants_max_interior_loss_frac),
            }
    q_after = _alignment_quality(
        real_u8,
        refined,
        tissue_mask=tissue_mask,
        edge_inner_weight=edge_inner_weight,
        edge_outer_weight=edge_outer_weight,
        mask_dice_weight=mask_dice_weight,
    )
    return refined.astype(np.int32), {
        "ok": True,
        "method": "ants_laplacian",
        "score_before": float(q_before),
        "score_after": float(q_after),
        "score_delta": float(q_after - q_before),
        "ants_transform": str(ants_transform),
        "ants_shape_ds": [int(out_h), int(out_w)],
        "laplacian_iter": int(lap_iter),
        "laplacian_lambda": float(lap_lambda),
        "max_disp_px": float(max_disp),
        "baseline_mask_dice": float(baseline_dice) if baseline_dice is not None else None,
        "refined_mask_dice": float(refined_dice) if refined_dice is not None else None,
        "interior_loss_frac": float(interior_loss_frac) if interior_loss_frac is not None else None,
    }


def _refine_warp_liquify_candidate(
    real_u8: np.ndarray,
    label: np.ndarray,
    tissue_mask: np.ndarray | None,
    q_before: float,
    warp_params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    h, w = label.shape
    if h < 60 or w < 60 or float(np.mean(label > 0)) < 0.01:
        return label, {"ok": False, "reason": "insufficient_label_coverage"}

    edge_inner_weight = float(_warp_param(warp_params, "edge_inner_weight", 1.00))
    edge_outer_weight = float(_warp_param(warp_params, "edge_outer_weight", 0.55))
    mask_dice_weight = float(_warp_param(warp_params, "mask_dice_weight", 0.35))
    edge_percentile = float(_warp_param(warp_params, "liquify_edge_percentile", 72.0))
    edge_floor = float(_warp_param(warp_params, "liquify_edge_floor", 0.10))
    edge_support_dilate = int(_warp_param(warp_params, "liquify_edge_support_dilate", 8))
    match_support_dilate = int(_warp_param(warp_params, "liquify_match_support_dilate", 10))
    match_dist_ratio = float(_warp_param(warp_params, "liquify_max_match_dist_ratio", 0.045))
    match_dist_min = float(_warp_param(warp_params, "liquify_max_match_dist_min_px", 10.0))
    inner_points = int(_warp_param(warp_params, "liquify_inner_points", 2400))
    outer_points = int(_warp_param(warp_params, "liquify_outer_points", 1200))
    max_ctrl = int(_warp_param(warp_params, "liquify_max_ctrl", 2600))
    max_disp_ratio = float(_warp_param(warp_params, "liquify_max_disp_ratio", 0.06))
    max_disp_min = float(_warp_param(warp_params, "liquify_max_disp_min_px", 8.0))
    dense_sigma_ratio = float(_warp_param(warp_params, "liquify_sigma_ratio", 0.020))
    dense_sigma_min = float(_warp_param(warp_params, "liquify_sigma_min_px", 6.0))
    use_tps = bool(_warp_param(warp_params, "liquify_use_tps", True))
    tps_ctrl = int(_warp_param(warp_params, "liquify_tps_ctrl", 260))
    tps_smooth = float(_warp_param(warp_params, "liquify_tps_smooth", 2.5))
    tps_grid_step = int(_warp_param(warp_params, "liquify_tps_grid_step", 4))
    lap_iter = int(_warp_param(warp_params, "laplacian_smooth_iter", 1))
    lap_lambda = float(_warp_param(warp_params, "laplacian_lambda", 0.18))

    real_feat = _real_edge_feature(real_u8, tissue_mask=tissue_mask)
    nz = real_feat > 0
    if float(np.mean(nz)) < 0.0005:
        return label, {"ok": False, "reason": "weak_real_edges"}
    thr = float(np.percentile(real_feat[nz], edge_percentile)) if np.any(nz) else 0.2
    real_edge = real_feat >= max(edge_floor, thr)
    if tissue_mask is not None:
        support = morphology.dilation(
            tissue_mask.astype(np.uint8),
            morphology.disk(max(1, int(edge_support_dilate))),
        ).astype(bool)
        real_edge &= support
    if float(np.mean(real_edge)) < 0.001:
        return label, {"ok": False, "reason": "too_few_real_edges"}

    outer = find_boundaries(label > 0, mode="outer", connectivity=2)
    inner = find_boundaries(label.astype(np.int32), mode="inner", connectivity=2)
    src_inner = _sample_mask_points(inner, max_points=max(64, inner_points))
    src_outer = _sample_mask_points(outer, max_points=max(32, outer_points))
    if len(src_inner) == 0 and len(src_outer) == 0:
        return label, {"ok": False, "reason": "no_atlas_boundaries"}
    src_rc = np.vstack([src_inner, src_outer]) if len(src_inner) and len(src_outer) else (
        src_inner if len(src_inner) else src_outer
    )

    dist, inds = distance_transform_edt(~real_edge, return_indices=True)
    yy = src_rc[:, 0]
    xx = src_rc[:, 1]
    ny = inds[0, yy, xx]
    nx = inds[1, yy, xx]
    d = dist[yy, xx]

    max_match_dist = float(max(match_dist_min, min(h, w) * match_dist_ratio))
    keep = d <= max_match_dist
    if tissue_mask is not None:
        support2 = morphology.dilation(
            tissue_mask.astype(np.uint8),
            morphology.disk(max(1, int(match_support_dilate))),
        ).astype(bool)
        keep &= support2[ny, nx]
    if int(np.sum(keep)) < 24:
        return label, {"ok": False, "reason": "too_few_matches"}

    src_xy = np.stack([xx[keep], yy[keep]], axis=1).astype(np.float32)
    dst_xy = np.stack([nx[keep], ny[keep]], axis=1).astype(np.float32)

    if len(src_xy) > int(max_ctrl):
        idx = np.linspace(0, len(src_xy) - 1, max_ctrl, dtype=np.int64)
        src_xy = src_xy[idx]
        dst_xy = dst_xy[idx]
    pair_key = np.round(np.concatenate([src_xy, dst_xy], axis=1), 1)
    _, uniq_idx = np.unique(pair_key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    src_xy = src_xy[uniq_idx]
    dst_xy = dst_xy[uniq_idx]
    if len(src_xy) < 20:
        return label, {"ok": False, "reason": "too_few_unique_matches"}

    max_disp = float(max(max_disp_min, min(h, w) * max_disp_ratio))
    src_xy, dst_xy = _filter_liquify_pairs(
        src_xy,
        dst_xy,
        max_disp=max_disp,
        min_points=20,
    )
    if len(src_xy) < 20:
        return label, {"ok": False, "reason": "too_few_inlier_matches"}

    candidate_results: list[tuple[str, np.ndarray, float, dict]] = []

    if use_tps:
        try:
            flow_v_tps, flow_u_tps = _tps_field_from_matches(
                h=h,
                w=w,
                src_xy=src_xy,
                dst_xy=dst_xy,
                max_ctrl=max(32, int(tps_ctrl)),
                smooth=float(tps_smooth),
                grid_step=max(2, int(tps_grid_step)),
                max_disp=max_disp,
                tissue_mask=tissue_mask,
                support_dilate=max(1, int(match_support_dilate)),
                lap_iterations=lap_iter,
                lap_lambda=lap_lambda,
            )
            refined_tps = _warp_label_with_flow(label, flow_v_tps, flow_u_tps)
            refined_tps = _clip_label_to_tissue(refined_tps, tissue_mask, pad=5)
            q_tps = _alignment_quality(
                real_u8,
                refined_tps,
                tissue_mask=tissue_mask,
                edge_inner_weight=edge_inner_weight,
                edge_outer_weight=edge_outer_weight,
                mask_dice_weight=mask_dice_weight,
            )
            candidate_results.append((
                "liquify_tps",
                refined_tps.astype(np.int32),
                float(q_tps),
                {
                    "tps_ctrl": int(min(len(src_xy), max(32, int(tps_ctrl)))),
                    "tps_smooth": float(tps_smooth),
                    "tps_grid_step": int(max(2, int(tps_grid_step))),
                },
            ))
        except Exception:
            pass

    try:
        sigma = float(max(dense_sigma_min, min(h, w) * dense_sigma_ratio))
        flow_v_dense, flow_u_dense = _dense_field_from_matches(
            h=h,
            w=w,
            src_xy=src_xy,
            dst_xy=dst_xy,
            sigma_px=sigma,
            max_disp=max_disp,
            tissue_mask=tissue_mask,
            support_dilate=max(1, int(match_support_dilate)),
            lap_iterations=lap_iter,
            lap_lambda=lap_lambda,
        )
        refined_dense = _warp_label_with_flow(label, flow_v_dense, flow_u_dense)
        refined_dense = _clip_label_to_tissue(refined_dense, tissue_mask, pad=5)
        q_dense = _alignment_quality(
            real_u8,
            refined_dense,
            tissue_mask=tissue_mask,
            edge_inner_weight=edge_inner_weight,
            edge_outer_weight=edge_outer_weight,
            mask_dice_weight=mask_dice_weight,
        )
        candidate_results.append((
            "liquify_dense_field",
            refined_dense.astype(np.int32),
            float(q_dense),
            {
                "sigma_px": float(sigma),
            },
        ))
    except Exception:
        pass

    if not candidate_results:
        return label, {"ok": False, "reason": "liquify_failed"}
    method, refined, q_after, extra_meta = max(candidate_results, key=lambda x: x[2])
    candidates_meta = {m: float(q) for m, _, q, _ in candidate_results}
    return refined.astype(np.int32), {
        "ok": True,
        "method": str(method),
        "score_before": float(q_before),
        "score_after": float(q_after),
        "score_delta": float(q_after - q_before),
        "matches": int(len(src_xy)),
        "max_disp_px": float(max_disp),
        "max_match_dist_px": float(max_match_dist),
        "candidates": candidates_meta,
        **extra_meta,
    }


def _refine_warp_nonlinear(
    real_u8: np.ndarray,
    label: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    warp_params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Non-linear refinement: evaluate multiple deformers and keep the best."""
    edge_inner_weight = float(_warp_param(warp_params, "edge_inner_weight", 1.00))
    edge_outer_weight = float(_warp_param(warp_params, "edge_outer_weight", 0.55))
    mask_dice_weight = float(_warp_param(warp_params, "mask_dice_weight", 0.35))
    prefer_liquify_margin = float(_warp_param(warp_params, "liquify_prefer_margin", 0.003))
    prefer_liquify_gain = float(_warp_param(warp_params, "liquify_prefer_min_gain", 0.006))
    prefer_ants_margin = float(_warp_param(warp_params, "ants_prefer_margin", 0.0025))
    prefer_ants_gain = float(_warp_param(warp_params, "ants_prefer_min_gain", 0.004))
    iter_gain_threshold = float(_warp_param(warp_params, "liquify_iter_gain_threshold", 0.0015))
    max_quality_drop = float(_warp_param(warp_params, "max_quality_drop", 0.006))
    use_liquify_pref = bool(_warp_param(warp_params, "prefer_liquify_when_close", True))
    use_ants_pref = bool(_warp_param(warp_params, "prefer_ants_when_close", True))

    q_before = _alignment_quality(
        real_u8,
        label,
        tissue_mask=tissue_mask,
        edge_inner_weight=edge_inner_weight,
        edge_outer_weight=edge_outer_weight,
        mask_dice_weight=mask_dice_weight,
    )

    flow_label, flow_meta = _refine_warp_flow_candidate(
        real_u8,
        label,
        tissue_mask=tissue_mask,
        q_before=q_before,
        warp_params=warp_params,
    )
    liquify_label, liquify_meta = _refine_warp_liquify_candidate(
        real_u8,
        label,
        tissue_mask=tissue_mask,
        q_before=q_before,
        warp_params=warp_params,
    )
    ants_label, ants_meta = _refine_warp_ants_candidate(
        real_u8,
        label,
        tissue_mask=tissue_mask,
        q_before=q_before,
        warp_params=warp_params,
    )
    hybrid_label = label
    hybrid_meta: dict = {"ok": False, "reason": "flow_or_liquify_unavailable"}
    if flow_meta.get("ok"):
        q_flow = float(flow_meta.get("score_after", q_before))
        h_label, h_meta = _refine_warp_liquify_candidate(
            real_u8,
            flow_label,
            tissue_mask=tissue_mask,
            q_before=q_flow,
            warp_params=warp_params,
        )
        if h_meta.get("ok"):
            hybrid_label = h_label
            hybrid_meta = {
                "ok": True,
                "method": "liquify_after_flow",
                "score_before": float(q_before),
                "score_after": float(h_meta.get("score_after", q_flow)),
                "score_delta": float(h_meta.get("score_after", q_flow) - q_before),
                "inner": h_meta,
            }

    ants_hybrid_label = label
    ants_hybrid_meta: dict = {"ok": False, "reason": "ants_or_liquify_unavailable"}
    if ants_meta.get("ok"):
        q_ants = float(ants_meta.get("score_after", q_before))
        ah_label, ah_meta = _refine_warp_liquify_candidate(
            real_u8,
            ants_label,
            tissue_mask=tissue_mask,
            q_before=q_ants,
            warp_params=warp_params,
        )
        if ah_meta.get("ok"):
            ants_hybrid_label = ah_label
            ants_hybrid_meta = {
                "ok": True,
                "method": "ants_plus_liquify",
                "score_before": float(q_before),
                "score_after": float(ah_meta.get("score_after", q_ants)),
                "score_delta": float(ah_meta.get("score_after", q_ants) - q_before),
                "inner": ah_meta,
            }

    candidates: list[tuple[str, np.ndarray, float]] = [("baseline", label, float(q_before))]
    if flow_meta.get("ok"):
        candidates.append(("flow", flow_label, float(flow_meta.get("score_after", -1e9))))
    if liquify_meta.get("ok"):
        candidates.append(("liquify", liquify_label, float(liquify_meta.get("score_after", -1e9))))
    if hybrid_meta.get("ok"):
        candidates.append(("liquify_after_flow", hybrid_label, float(hybrid_meta.get("score_after", -1e9))))
    if ants_meta.get("ok"):
        candidates.append(("ants_laplacian", ants_label, float(ants_meta.get("score_after", -1e9))))
    if ants_hybrid_meta.get("ok"):
        candidates.append(("ants_plus_liquify", ants_hybrid_label, float(ants_hybrid_meta.get("score_after", -1e9))))

    best_method, best_label, best_q = max(candidates, key=lambda x: x[2])
    best_q = float(best_q)

    # Prefer true liquify deformation when quality is close to the best candidate.
    liq_like = [c for c in candidates if c[0] in ("liquify", "liquify_after_flow")]
    if liq_like and use_liquify_pref:
        l_method, l_label, l_q = max(liq_like, key=lambda x: x[2])
        l_q = float(l_q)
        if l_q >= best_q - prefer_liquify_margin and l_q >= q_before + prefer_liquify_gain:
            best_method, best_label, best_q = l_method, l_label, l_q

    ants_like = [c for c in candidates if c[0] in ("ants_laplacian", "ants_plus_liquify")]
    if ants_like and use_ants_pref:
        a_method, a_label, a_q = max(ants_like, key=lambda x: x[2])
        a_q = float(a_q)
        if a_q >= best_q - prefer_ants_margin and a_q >= q_before + prefer_ants_gain:
            best_method, best_label, best_q = a_method, a_label, a_q

    if best_method == "baseline":
        return label, {
            "applied": False,
            "reason": "no_quality_gain",
            "score_before": float(q_before),
            "score_after": float(q_before),
            "score_delta": 0.0,
            "flow": flow_meta,
            "liquify": liquify_meta,
            "hybrid": hybrid_meta,
            "ants": ants_meta,
            "ants_hybrid": ants_hybrid_meta,
        }

    # One more local liquify pass can improve internal boundaries after coarse non-linear warp.
    iter_label, iter_meta = _refine_warp_liquify_candidate(
        real_u8,
        best_label,
        tissue_mask=tissue_mask,
        q_before=best_q,
        warp_params=warp_params,
    )
    if iter_meta.get("ok"):
        iter_q = float(iter_meta.get("score_after", best_q))
        if iter_q >= best_q + iter_gain_threshold:
            best_label = iter_label
            best_q = iter_q
            best_method = f"{best_method}_plus_liquify_iter"

    # Allow slight local tradeoff for internal fit, but avoid visibly worse global alignment.
    if best_q < q_before - max_quality_drop:
        return label, {
            "applied": False,
            "reason": "quality_drop_too_large",
            "score_before": float(q_before),
            "score_after": float(best_q),
            "score_delta": float(best_q - q_before),
            "flow": flow_meta,
            "liquify": liquify_meta,
            "hybrid": hybrid_meta,
            "ants": ants_meta,
            "ants_hybrid": ants_hybrid_meta,
            "iter_liquify": iter_meta,
        }

    return best_label.astype(np.int32), {
        "applied": True,
        "method": str(best_method),
        "score_before": float(q_before),
        "score_after": float(best_q),
        "score_delta": float(best_q - q_before),
        "flow": flow_meta,
        "liquify": liquify_meta,
        "hybrid": hybrid_meta,
        "ants": ants_meta,
        "ants_hybrid": ants_hybrid_meta,
        "iter_liquify": iter_meta,
    }


# 鈹€鈹€ Region label drawing 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def draw_region_labels(
    overlay: np.ndarray,
    warped_label: np.ndarray,
    min_area_px: int = 5000,
    max_labels: int = 25,
    font_scale: float = 0.75,
    text_color: tuple = (255, 255, 255),
    shadow_color: tuple = (0, 0, 0),
) -> np.ndarray:
    """Draw Allen region acronyms on the overlay at each region's centroid."""
    try:
        import cv2 as _cv2
        _HAS_CV2 = True
    except ImportError:
        _HAS_CV2 = False

    tree = _load_structure_tree()

    ids, counts = np.unique(warped_label, return_counts=True)
    # Sort by area descending, skip background (0)
    id_count = [(int(i), int(c)) for i, c in zip(ids, counts) if i != 0]
    id_count.sort(key=lambda x: -x[1])
    id_count = id_count[:max_labels]

    props_map: dict[int, measure.RegionProperties] = {}
    for region_id in np.unique(warped_label):
        if region_id == 0:
            continue
        mask = (warped_label == region_id)
        if mask.sum() < min_area_px:
            continue
        region_lbl = measure.label(mask, connectivity=2)
        rprops = measure.regionprops(region_lbl)
        if rprops:
            biggest = max(rprops, key=lambda r: r.area)
            props_map[int(region_id)] = biggest

    out = overlay.copy()

    # Set up PIL once before the loop (avoid per-region conversion overhead)
    pil_draw = None
    pil_img = None
    if not _HAS_CV2:
        from PIL import Image, ImageDraw, ImageFont
        font_size = max(14, int(font_scale * 28))
        fnt = None
        for fname in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                      "C:/Windows/Fonts/arial.ttf"]:
            try:
                fnt = ImageFont.truetype(fname, font_size)
                break
            except Exception:
                pass
        if fnt is None:
            fnt = ImageFont.load_default(size=font_size) if hasattr(ImageFont, 'load_default') else ImageFont.load_default()
        pil_img = Image.fromarray(out)
        pil_draw = ImageDraw.Draw(pil_img)

    for rid, count in id_count:
        if rid not in props_map:
            continue
        cy, cx = props_map[rid].centroid
        cy, cx = int(cy), int(cx)

        info = tree.get(str(rid), {})
        acronym = info.get("acronym", "")
        if not acronym:
            continue  # skip unknown IDs
        if len(acronym) > 8:
            acronym = acronym[:8]

        if _HAS_CV2:
            font = _cv2.FONT_HERSHEY_SIMPLEX
            lw = max(1, int(font_scale * 1.5))
            _cv2.putText(out, acronym, (cx + 1, cy + 1), font, font_scale,
                         shadow_color, lw + 1, _cv2.LINE_AA)
            _cv2.putText(out, acronym, (cx, cy), font, font_scale,
                         text_color, lw, _cv2.LINE_AA)
        else:
            pil_draw.text((cx + 1, cy + 1), acronym, fill=shadow_color, font=fnt)
            pil_draw.text((cx, cy), acronym, fill=text_color, font=fnt)

    if pil_img is not None:
        out = np.array(pil_img)
    return out


# 鈹€鈹€ Core registration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _tissue_guided_warp(
    real: np.ndarray,
    label: np.ndarray,
    atlas_res_um: float = 25.0,
    real_res_um: float = 0.65,
    fit_mode: str = "contain",
    enable_nonlinear: bool = True,
    opt_maxiter: int = 150,
    warp_params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Tissue-guided registration using full/half candidate competition."""
    rh, rw = real.shape[:2]
    real_u8 = _norm_u8_robust(real)
    phys_scale = float(atlas_res_um) / float(real_res_um)
    edge_inner_weight = float(_warp_param(warp_params, "edge_inner_weight", 1.00))
    edge_outer_weight = float(_warp_param(warp_params, "edge_outer_weight", 0.55))
    mask_dice_weight = float(_warp_param(warp_params, "mask_dice_weight", 0.35))
    outside_penalty = float(_warp_param(warp_params, "outside_penalty", 2.2))
    default_refine_topn = 2 if enable_nonlinear else 1
    candidate_top_n = int(_warp_param(warp_params, "linear_candidate_top_n", default_refine_topn))
    candidate_top_n = int(np.clip(candidate_top_n, 1, 3))
    opt_iter_primary = int(_warp_param(warp_params, "linear_opt_maxiter_primary", int(opt_maxiter)))
    opt_iter_secondary = int(_warp_param(warp_params, "linear_opt_maxiter_secondary", max(45, int(opt_maxiter * 0.72))))
    keep_margin = float(_warp_param(warp_params, "linear_opt_keep_margin", -0.003))

    info = _detect_tissue(real)
    if not info["ok"]:
        out, _fitted = _align_shape_physical(label, (rh, rw), atlas_res_um, real_res_um, fit_mode)
        return out.astype(np.int32), {"method": "fallback_center"}

    t_cy, t_cx = info["centroid"]
    t_h, t_w = info["hw"]
    tissue_aspect = t_h / max(t_w, 1)
    tissue_orientation = info.get("orientation", 0.0)
    tissue_mask = info.get("mask", None)

    ay0, ax0, ay1, ax1 = _atlas_bbox(label)
    a_h = float(ay1 - ay0)
    a_w = float(ax1 - ax0)
    a_mid_x = (ax0 + ax1) // 2
    full_aspect = a_h / max(a_w, 1)
    half_aspect = a_h / max(a_w / 2, 1)

    def _atlas_orientation(lbl: np.ndarray) -> float:
        props = measure.regionprops(measure.label(lbl > 0))
        if props:
            return float(max(props, key=lambda p: p.area).orientation)
        return 0.0

    def _clamp_angle(a: float) -> float:
        aa = float(a)
        if abs(aa) > np.deg2rad(20):
            alt = aa - np.pi * np.sign(aa)
            if abs(alt) < abs(aa):
                aa = float(alt)
        return float(np.clip(aa, np.deg2rad(-18), np.deg2rad(18)))

    full_label = label.astype(np.int32).copy()
    left_label = label.astype(np.int32).copy()
    left_label[:, a_mid_x:] = 0
    right_label = np.fliplr(label.astype(np.int32).copy())
    right_label[:, a_mid_x:] = 0
    raw_candidates = [
        ("full", full_label, False, "full"),
        ("left", left_label, True, "left"),
        ("right_mirrored", right_label, True, "right_mirrored"),
    ]

    candidates: list[dict] = []
    for name, cand_label, is_half, hemi in raw_candidates:
        by0, bx0, by1, bx1 = _atlas_bbox(cand_label)
        bh = float(max(1, by1 - by0))
        bw = float(max(1, bx1 - bx0))
        bcy = (by0 + by1) * 0.5
        bcx = (bx0 + bx1) * 0.5
        _, init_s, init_dx, init_dy = _similarity_warp(
            cand_label,
            bcy,
            bcx,
            bh,
            bw,
            t_cy,
            t_cx,
            t_h,
            t_w,
            phys_scale,
            fit_mode,
            (rh, rw),
        )
        if name == "full":
            init_angle = 0.0
        else:
            init_angle = _clamp_angle(float(tissue_orientation - _atlas_orientation(cand_label)))

        warped_init = _apply_warp(cand_label, float(init_s), float(init_angle), float(init_dx), float(init_dy), (rh, rw))
        q_init = _alignment_quality(
            real_u8,
            warped_init,
            tissue_mask=tissue_mask,
            edge_inner_weight=edge_inner_weight,
            edge_outer_weight=edge_outer_weight,
            mask_dice_weight=mask_dice_weight,
        )
        cov_init = _coverage_stats(warped_init, tissue_mask=tissue_mask)
        obj_init = _candidate_objective_score(
            q=float(q_init),
            cov=cov_init,
            tissue_aspect=float(tissue_aspect),
            full_aspect=float(full_aspect),
            half_aspect=float(half_aspect),
            candidate_name=str(name if name == "full" else "half"),
            warp_params=warp_params,
        )
        candidates.append(
            {
                "name": str(name),
                "label": cand_label.astype(np.int32),
                "is_half": bool(is_half),
                "hemisphere": str(hemi),
                "init": {
                    "scale": float(init_s),
                    "angle_rad": float(init_angle),
                    "dx": float(init_dx),
                    "dy": float(init_dy),
                    "quality": float(q_init),
                    "coverage": cov_init,
                    "objective": float(obj_init),
                    "warped": warped_init.astype(np.int32),
                },
            }
        )

    ranked_idx = sorted(range(len(candidates)), key=lambda i: candidates[i]["init"]["objective"], reverse=True)
    for rank, idx in enumerate(ranked_idx):
        c = candidates[idx]
        c["best"] = dict(c["init"])
        c["best"]["state"] = "initial"
        if rank >= int(candidate_top_n):
            continue
        try:
            maxiter_local = int(opt_iter_primary if rank == 0 else opt_iter_secondary)
            opt_scale, opt_angle, opt_dx, opt_dy = _optimize_warp(
                real_u8,
                c["label"],
                float(c["init"]["scale"]),
                float(c["init"]["dx"]),
                float(c["init"]["dy"]),
                (rh, rw),
                init_angle_deg=float(np.rad2deg(float(c["init"]["angle_rad"]))),
                maxiter=maxiter_local,
                edge_inner_weight=edge_inner_weight,
                edge_outer_weight=edge_outer_weight,
                outside_penalty=outside_penalty,
            )
            warped_opt = _apply_warp(c["label"], opt_scale, opt_angle, opt_dx, opt_dy, (rh, rw))
            q_opt = _alignment_quality(
                real_u8,
                warped_opt,
                tissue_mask=tissue_mask,
                edge_inner_weight=edge_inner_weight,
                edge_outer_weight=edge_outer_weight,
                mask_dice_weight=mask_dice_weight,
            )
            cov_opt = _coverage_stats(warped_opt, tissue_mask=tissue_mask)
            obj_opt = _candidate_objective_score(
                q=float(q_opt),
                cov=cov_opt,
                tissue_aspect=float(tissue_aspect),
                full_aspect=float(full_aspect),
                half_aspect=float(half_aspect),
                candidate_name=str(c["name"] if c["name"] == "full" else "half"),
                warp_params=warp_params,
            )
            c["optimized"] = {
                "scale": float(opt_scale),
                "angle_rad": float(opt_angle),
                "dx": float(opt_dx),
                "dy": float(opt_dy),
                "quality": float(q_opt),
                "coverage": cov_opt,
                "objective": float(obj_opt),
                "warped": warped_opt.astype(np.int32),
            }
            if float(obj_opt) >= float(c["init"]["objective"]) + float(keep_margin):
                c["best"] = dict(c["optimized"])
                c["best"]["state"] = "optimized"
        except Exception as e:
            c["opt_error"] = str(e)

    chosen = max(candidates, key=lambda x: float(x["best"]["objective"]))
    full_candidate = next((c for c in candidates if c["name"] == "full"), None)
    selection_override = "none"

    if chosen["is_half"] and full_candidate is not None:
        half_min_lr = float(chosen["best"]["coverage"].get("min_lr", 0.0))
        half_cov = float(chosen["best"]["coverage"].get("coverage", 0.0))
        full_min_lr = float(full_candidate["best"]["coverage"].get("min_lr", 0.0))
        full_cov = float(full_candidate["best"]["coverage"].get("coverage", 0.0))
        half_obj = float(chosen["best"]["objective"])
        full_obj = float(full_candidate["best"]["objective"])

        half_lr_hard = float(_warp_param(warp_params, "linear_half_min_lr_hard", 0.72))
        full_obj_margin = float(_warp_param(warp_params, "linear_half_to_full_obj_margin", 0.05))
        full_cov_gain = float(_warp_param(warp_params, "linear_half_to_full_cov_gain", 0.06))
        full_minlr_gain = float(_warp_param(warp_params, "linear_half_to_full_minlr_gain", 0.11))
        if half_min_lr < half_lr_hard and (
            full_min_lr >= half_min_lr + full_minlr_gain
            or full_cov >= half_cov + full_cov_gain
            or full_obj >= half_obj - full_obj_margin
        ):
            chosen = full_candidate
            selection_override = "forced_full_for_balance"

    chosen_cov = float(chosen["best"]["coverage"].get("coverage", 0.0))
    min_cov_hard = float(_warp_param(warp_params, "linear_force_coverage_min", 0.88))
    if tissue_mask is not None and chosen_cov < min_cov_hard:
        if full_candidate is not None:
            full_cov = float(full_candidate["best"]["coverage"].get("coverage", 0.0))
            if full_cov > chosen_cov + 0.02:
                chosen = full_candidate
                selection_override = "forced_full_for_coverage"
                chosen_cov = full_cov
        if chosen_cov < min_cov_hard:
            alt = max(candidates, key=lambda x: float(x["best"]["coverage"].get("coverage", 0.0)))
            alt_cov = float(alt["best"]["coverage"].get("coverage", 0.0))
            if alt_cov > chosen_cov + 0.04:
                chosen = alt
                selection_override = "forced_max_coverage"

    warped = chosen["best"]["warped"].astype(np.int32)
    chosen_scale = float(chosen["best"]["scale"])
    chosen_angle = float(chosen["best"]["angle_rad"])
    chosen_dx = float(chosen["best"]["dx"])
    chosen_dy = float(chosen["best"]["dy"])

    if enable_nonlinear:
        warped, nl_meta = _refine_warp_nonlinear(
            real_u8,
            warped,
            tissue_mask=tissue_mask,
            warp_params=warp_params,
        )
    else:
        nl_meta = {"applied": False, "reason": "disabled"}

    score_left = next((float(c["best"]["quality"]) for c in candidates if c["name"] == "left"), None)
    score_right = next((float(c["best"]["quality"]) for c in candidates if c["name"] == "right_mirrored"), None)
    cand_summaries = []
    for c in sorted(candidates, key=lambda x: float(x["best"]["objective"]), reverse=True):
        cand_summaries.append(
            {
                "name": str(c["name"]),
                "is_half": bool(c["is_half"]),
                "state": str(c["best"].get("state", "initial")),
                "init_objective": float(c["init"]["objective"]),
                "best_objective": float(c["best"]["objective"]),
                "best_quality": float(c["best"]["quality"]),
                "best_coverage": dict(c["best"]["coverage"]),
            }
        )

    return warped, {
        "method": "tissue_guided_auto_candidates",
        "is_half_brain": bool(chosen["is_half"]),
        "hemisphere_chosen": str(chosen["hemisphere"]),
        "candidate_selected": str(chosen["name"]),
        "candidate_state": str(chosen["best"].get("state", "initial")),
        "selection_override": str(selection_override),
        "candidate_refined_top_n": int(candidate_top_n),
        "score_left": float(score_left) if score_left is not None else None,
        "score_right": float(score_right) if score_right is not None else None,
        "total_scale": float(chosen_scale),
        "angle_deg": float(np.rad2deg(chosen_angle)),
        "translation": [float(chosen_dx), float(chosen_dy)],
        "linear_candidates": cand_summaries,
        "nonlinear_refine": nl_meta,
        "tissue_center": [float(t_cx), float(t_cy)],
        "tissue_hw": [int(t_h), int(t_w)],
        "tissue_aspect": float(tissue_aspect),
        "full_atlas_aspect": float(full_aspect),
        "half_atlas_aspect": float(half_aspect),
        "tissue_mask": tissue_mask,
    }


def _align_shape_physical(label_slice: np.ndarray, target_shape: tuple[int, int],
                           atlas_res_um: float = 25.0, real_res_um: float = 0.65,
                           fit_mode: str = "contain"):
    """Fallback: physical scale + fit mode + center crop/pad."""
    scale = atlas_res_um / real_res_um
    scaled_label = rescale(label_slice.astype(np.float32), scale, order=0,
                           preserve_range=True, anti_aliasing=False).astype(np.int32)
    th, tw = target_shape
    sh, sw = scaled_label.shape
    fh, fw = th / max(sh, 1), tw / max(sw, 1)
    mode = str(fit_mode or "contain").lower()
    fit = min(fh, fw) if mode in ("contain",) else max(fh, fw) if mode == "cover" \
        else fw if mode == "width-lock" else fh if mode == "height-lock" else min(fh, fw)
    if abs(float(fit) - 1.0) > 1e-6:
        scaled_label = rescale(scaled_label.astype(np.float32), float(fit), order=0,
                               preserve_range=True, anti_aliasing=False).astype(np.int32)
    fitted_shape = (int(scaled_label.shape[0]), int(scaled_label.shape[1]))
    sh, sw = scaled_label.shape
    out = np.zeros((th, tw), dtype=scaled_label.dtype)
    min_h, min_w = min(th, sh), min(tw, sw)
    oy, ox = (th - min_h) // 2, (tw - min_w) // 2
    iy, ix = (sh - min_h) // 2, (sw - min_w) // 2
    out[oy:oy + min_h, ox:ox + min_w] = scaled_label[iy:iy + min_h, ix:ix + min_w]
    return out, fitted_shape


def _roi_bbox_from_real(real_img: np.ndarray, pad: int = 4) -> tuple[int, int, int, int]:
    x = real_img.astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    thr = float(np.percentile(x, 88))
    mask = x > thr
    lbl = measure.label(mask, connectivity=2)
    props = measure.regionprops(lbl)
    if not props:
        h, w = x.shape
        return (0, 0, int(w), int(h))
    props = sorted(props, key=lambda r: r.area, reverse=True)
    y0, x0, y1, x1 = props[0].bbox
    h, w = x.shape
    return (max(0, int(x0) - pad), max(0, int(y0) - pad),
            int(min(w, x1 + pad) - max(0, x0 - pad)),
            int(min(h, y1 + pad) - max(0, y0 - pad)))


def _paint_boundaries(canvas: np.ndarray, mask: np.ndarray, color) -> None:
    """Paint region boundaries on canvas in-place using find_boundaries."""
    bd = find_boundaries(mask, mode="outer", connectivity=2)
    canvas[bd] = color


def _paint_contours_smooth(canvas: np.ndarray, mask: np.ndarray, color) -> None:
    """Draw anti-aliased contours if cv2 is available; fallback to boundary raster."""
    try:
        import cv2 as _cv2
        arr = (mask > 0).astype(np.uint8)
        contours, _ = _cv2.findContours(arr, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_NONE)
        if contours:
            _cv2.drawContours(
                canvas,
                contours,
                contourIdx=-1,
                color=tuple(int(c) for c in color),
                thickness=1,
                lineType=_cv2.LINE_AA,
            )
        return
    except Exception:
        pass
    _paint_boundaries(canvas, mask, color)


def draw_contours(label_img: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    h, w = label_img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    _paint_contours_smooth(canvas, label_img, color)
    return canvas


def draw_contours_major(
    label_img: np.ndarray,
    top_k: int = 12,
    tissue_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Outer brain boundary (cyan) + major region boundaries (white).

    If tissue_mask is provided it is used for the outer boundary, which
    guarantees the cyan line perfectly traces the real tissue edge.
    """
    h, w = label_img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # Outer boundary: prefer real tissue mask so it perfectly follows tissue edge
    outer = tissue_mask if tissue_mask is not None else (label_img > 0).astype(np.uint8)
    _paint_contours_smooth(canvas, outer, np.array([0, 255, 255], dtype=np.uint8))
    # Major region boundaries (from warped atlas)
    ids, counts = np.unique(label_img[label_img > 0], return_counts=True)
    if len(ids) > 0:
        for rid in ids[np.argsort(counts)[::-1][:max(1, int(top_k))]]:
            _paint_contours_smooth(
                canvas,
                (label_img == rid).astype(np.uint8),
                np.array([255, 255, 255], dtype=np.uint8),
            )
    return canvas


# 鈹€鈹€ Public entry point 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def render_overlay(
    real_slice_path: Path,
    label_slice_path: Path,
    out_png: Path,
    alpha: float = 0.45,
    mode: str = "fill",
    structure_csv: Path | None = None,
    min_mean_threshold: float = 8.0,
    pixel_size_um: float = 0.65,
    rotate_deg: float = 0.0,
    flip_mode: str = "none",
    return_meta: bool = False,
    major_top_k: int = 20,
    fit_mode: str = "cover",
    warped_label_out: Path | None = None,
    real_z_index: int | None = None,
    label_z_index: int | None = None,
    edge_smooth_iter: int = 1,
    warp_params: dict | None = None,
    prewarped_label: bool = False,
):
    real_raw = imread(str(real_slice_path))
    label_raw = imread(str(label_slice_path))
    real, real_slice_meta = select_real_slice_2d(
        real_raw,
        z_index=real_z_index,
        source_path=real_slice_path,
    )
    label, label_slice_meta = select_label_slice_2d(
        label_raw,
        z_index=label_z_index,
    )

    label_shape_before = tuple(int(x) for x in label.shape)
    real_shape = tuple(int(x) for x in real.shape)

    # User pre-transform: rotate / flip atlas
    if rotate_deg != 0.0:
        label = rotate(label.astype(np.float32), rotate_deg, order=0,
                       preserve_range=True, resize=True).astype(np.int32)
    if flip_mode == "horizontal":
        label = np.fliplr(label)
    elif flip_mode == "vertical":
        label = np.flipud(label)

    # 鈹€鈹€ Tissue-guided registration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if bool(prewarped_label):
        # Manual-calibration path: label is already in real-image coordinates.
        if label.shape != real.shape:
            label = resize(
                label.astype(np.float32),
                real.shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.int32)
        tm = _detect_tissue(real).get("mask", None)
        warp_meta = {
            "method": "manual_prewarped_label",
            "is_half_brain": None,
            "hemisphere_chosen": "manual",
            "nonlinear_refine": {"applied": False, "reason": "manual_prewarped"},
            "tissue_mask": tm,
        }
    else:
        label, warp_meta = _tissue_guided_warp(
            real, label,
            atlas_res_um=25.0,
            real_res_um=pixel_size_um,
            fit_mode=fit_mode,
            warp_params=warp_params,
        )

    # 鈹€鈹€ Clip atlas to actual tissue footprint 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # Reuse tissue mask from registration (avoids second _detect_tissue call).
    try:
        tight = warp_meta.get("tissue_mask")
        if tight is not None:
            # 3px dilation: enough to keep atlas at tissue boundary, won't let it far outside
            tissue_clip = morphology.dilation(tight.astype(np.uint8), morphology.disk(3)).astype(bool)
            label = np.where(tissue_clip, label, 0).astype(np.int32)
    except Exception:
        pass
    try:
        cleanup_params = dict(warp_params or {})
        if str(mode or "fill").lower() != "fill":
            cleanup_params["cleanup_fill_outer_missing"] = False
        label = _cleanup_label_topology(
            label.astype(np.int32),
            tissue_mask=warp_meta.get("tissue_mask"),
            warp_params=cleanup_params,
        )
    except Exception:
        pass
    try:
        label = _smooth_label_edges(
            label.astype(np.int32),
            tissue_mask=warp_meta.get("tissue_mask"),
            iterations=max(0, int(edge_smooth_iter)),
        )
    except Exception:
        pass
    try:
        if str(mode or "fill").lower() in ("fill",):
            label = _gaussian_vote_boundary_smooth(
                label.astype(np.int32),
                tissue_mask=warp_meta.get("tissue_mask"),
                warp_params=warp_params,
            )
    except Exception:
        pass
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    label_shape_after = tuple(int(x) for x in label.shape)
    roi_bbox = _roi_bbox_from_real(real)
    rx, ry, rw2, rh2 = roi_bbox
    cx_full = float(rx + rw2 / 2.0)
    cy_full = float(ry + rh2 / 2.0)

    real_u8 = _norm_u8_robust(real)

    # 鈹€鈹€ Coloring: Allen official colors (hex) 鈫?prefer over CSV / LUT 鈹€鈹€鈹€鈹€鈹€鈹€
    tree = _load_structure_tree()
    colored = None

    if tree:
        colored = np.zeros((*label.shape, 3), dtype=np.uint8)
        for rid in np.unique(label.astype(np.int32)):
            if rid == 0:
                continue
            info = tree.get(str(rid))
            if info and info.get("color") and len(info["color"]) == 6:
                hx = info["color"]
                r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
                colored[label == rid] = (r, g, b)
            else:
                # Fallback deterministic color
                rng = (int(rid) * 2654435761) & 0xFFFFFF
                colored[label == rid] = ((rng >> 16) & 0xFF, (rng >> 8) & 0xFF, rng & 0xFF)
    elif structure_csv is not None:
        cmap = load_allen_color_map(structure_csv)
        if cmap:
            colored = np.zeros((*label.shape, 3), dtype=np.uint8)
            for rid in np.unique(label.astype(np.int32)):
                if rid == 0:
                    continue
                color = cmap.get(int(rid))
                if color is not None:
                    colored[label == rid] = np.array(color, dtype=np.uint8)

    if colored is None:
        lut = np.array([
            [20, 20, 20], [0, 200, 255], [0, 255, 120], [255, 120, 180],
            [255, 70, 70], [220, 220, 80], [160, 120, 255], [255, 180, 70],
        ], dtype=np.uint8)
        colored = lut[(label.astype(np.int32) % len(lut))]

    # Retrieve tissue mask for outer boundary (cached from registration)
    _tissue_mask_for_contour = warp_meta.get("tissue_mask")

    if mode == "contour":
        colored = draw_contours(label)
    elif mode == "contour-major":
        colored = draw_contours_major(label, top_k=major_top_k,
                                      tissue_mask=_tissue_mask_for_contour)

    overlay = alpha_blend(real_u8, colored, alpha)

    # Quality guard (lower threshold for contour modes which are mostly black)
    effective_thr = min_mean_threshold if mode == "fill" else min(float(min_mean_threshold) * 0.25, 2.0)
    if float(np.mean(overlay)) < effective_thr:
        raise ValueError(
            f"overlay quality check failed: near-black output "
            f"(mean={float(np.mean(overlay)):.2f}, threshold={effective_thr:.2f})"
        )

    # 鈹€鈹€ Region labels 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if mode in ("fill",):
        overlay = draw_region_labels(overlay, label)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_png), overlay)

    if warped_label_out is not None:
        warped_label_out.parent.mkdir(parents=True, exist_ok=True)
        imwrite(str(warped_label_out), label.astype(np.int32))

    # Strip non-serializable internals from warp_meta before returning
    warp_meta.pop("tissue_mask", None)

    if return_meta:
        return out_png, {
            "real_shape": real_shape,
            "label_shape_before": label_shape_before,
            "label_shape_after": label_shape_after,
            "real_slice": real_slice_meta,
            "label_slice": label_slice_meta,
            "scale": float(25.0 / float(pixel_size_um)),
            "pixelSizeUm": float(pixel_size_um),
            "rotateAtlas": float(rotate_deg),
            "flipAtlas": str(flip_mode),
            "fitMode": str(fit_mode),
            "edgeSmoothIter": int(edge_smooth_iter),
            "warpParams": dict(warp_params or {}),
            "real_aspect": float(real_shape[1] / max(real_shape[0], 1)),
            "atlas_aspect_before": float(label_shape_before[1] / max(label_shape_before[0], 1)),
            "atlas_aspect": float(label_shape_after[1] / max(label_shape_after[0], 1)),
            "roi_bbox": [int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])],
            "roi_center_full": [cx_full, cy_full],
            "roi_roundtrip_error": 0.0,
            "warped_label_path": str(warped_label_out) if warped_label_out is not None else "",
            "warp": warp_meta,
        }

    return out_png

