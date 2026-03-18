from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.interpolate import Rbf
from scipy.ndimage import distance_transform_edt, gaussian_filter, laplace, map_coordinates
from scripts.image_utils import alpha_blend
from scripts.image_utils import norm_u8_robust as _norm_u8_robust
from scripts.overlay_assets import colorize_label, load_structure_tree
from scripts.overlay_postprocess import finalize_registered_label
from scripts.slice_select import select_label_slice_2d, select_real_slice_2d
from skimage import measure, morphology
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.registration import optical_flow_tvl1
from skimage.segmentation import find_boundaries
from skimage.transform import SimilarityTransform, rescale, resize, rotate
from skimage.transform import warp as skwarp
from tifffile import imread, imwrite


def _load_structure_tree() -> dict:
    return load_structure_tree()


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
        "orientation": float(tissue.orientation),  # radians, skimage convention
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "mask": tight_mask,  # bool array, real image space
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
    a_cy: float,
    a_cx: float,
    a_h: float,
    a_w: float,
    t_cy: float,
    t_cx: float,
    t_h: float,
    t_w: float,
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
        fwd = SimilarityTransform(scale=s, rotation=angle_rad, translation=(dx / ds, dy / ds))
        try:
            w = skwarp(
                atlas_ds,
                fwd.inverse,
                output_shape=(ds_h, ds_w),
                order=0,
                preserve_range=True,
                mode="constant",
                cval=0.0,
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
        _warp_score,
        x0,
        method="Nelder-Mead",
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
    scale: float,
    angle_rad: float,
    dx: float,
    dy: float,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Apply similarity transform (scale, rotation, translation) to atlas label."""
    fwd = SimilarityTransform(scale=scale, rotation=angle_rad, translation=(dx, dy))
    warped = skwarp(
        label.astype(np.float32),
        fwd.inverse,
        output_shape=out_shape,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    return warped.astype(np.int32)


_ALLEN_THUMB_UM_PX = 67.008  # Allen thumbnail resolution (µm per pixel at downsample=6)
_ALLEN_REF_CACHE = Path(__file__).resolve().parent.parent / "configs" / "allen_ref_cache"


_ALLEN_NISSL_VOL_CACHE: dict[str, np.ndarray] = {}  # path → loaded volume
_ALLEN_NISSL_25_PATH = Path(__file__).resolve().parent.parent / "configs" / "ara_nissl_25.nrrd"
_ALLEN_NISSL_10_PATH = Path(__file__).resolve().parent.parent / "configs" / "ara_nissl_10.nrrd"


def _load_nissl_slice(atlas_z: int) -> np.ndarray | None:
    """Load 2D Nissl reference slice from ara_nissl volume (25 or 10 µm).

    The volume is in CCFv3 space (same as annotation), so atlas_z directly
    indexes the correct coronal section.  Volume is cached in memory after
    first load.  Returns float32 [0,1] gray slice, or None if unavailable.
    """
    nissl_path = (
        _ALLEN_NISSL_25_PATH
        if _ALLEN_NISSL_25_PATH.exists()
        else (_ALLEN_NISSL_10_PATH if _ALLEN_NISSL_10_PATH.exists() else None)
    )
    if nissl_path is None:
        return None
    key = str(nissl_path)
    if key not in _ALLEN_NISSL_VOL_CACHE:
        try:
            import nrrd

            data, _ = nrrd.read(str(nissl_path))
            # NRRD axis order for Allen CCF: (AP, DV, ML) — same as annotation NIfTI
            _ALLEN_NISSL_VOL_CACHE[key] = data.astype(np.float32)
        except Exception:
            return None
    vol = _ALLEN_NISSL_VOL_CACHE[key]
    az = int(np.clip(atlas_z, 0, vol.shape[0] - 1))
    slc = vol[az].astype(np.float32)
    lo, hi = float(np.percentile(slc, 1)), float(np.percentile(slc, 99))
    if hi > lo:
        slc = np.clip((slc - lo) / (hi - lo), 0.0, 1.0)
    else:
        slc = np.zeros_like(slc)
    return slc


def _load_placed_allen_ref(
    atlas_z: int,
    total_scale: float,
    angle_rad: float,
    dx: float,
    dy: float,
    out_shape: tuple[int, int],
    hemisphere: str = "right_flipped",
    cache_dir: Path | None = None,
) -> np.ndarray | None:
    """Load Allen Nissl reference for atlas_z and place it in image coordinates.

    Preferred source: ara_nissl_25.nrrd / ara_nissl_10.nrrd volume slice (same
    CCFv3 coordinate space as annotation, so atlas_z indexes directly).
    Fallback: 67µm/px thumbnail from cache.

    Returns float32 [0,1] image (dark background, bright tissue).
    Nissl is inverted so white-matter fibers (dark in Nissl) become BRIGHT,
    matching how fiber tracts appear in cleared-brain fluorescence images.
    """
    nissl_slc = _load_nissl_slice(atlas_z)
    if nissl_slc is not None:
        # Nissl volume: background is dark (0), tissue/cells are bright.
        # White-matter tracts are DARKER than cell-dense regions in Nissl.
        # Cleared-brain fluorescence: background dark, labeled cells bright,
        # fiber tracts relatively dark.  So we use Nissl directly (no inversion)
        # and then extract "fiber-like" features via local contrast enhancement.
        gray = nissl_slc  # already [0,1], dark background
        # Hemisphere crop (volume ML axis = axis 1)
        vw = gray.shape[1]
        if hemisphere in ("right", "right_flipped"):
            gray = gray[:, vw // 2 :]  # right half of coronal section
        elif hemisphere == "left":
            gray = gray[:, : vw // 2]
        if hemisphere == "right_flipped":
            gray = np.fliplr(gray)  # lateral at LEFT, medial at RIGHT
        # Place in image space at atlas voxel scale (same scale as label)
        nissl_um = 25.0 if _ALLEN_NISSL_25_PATH.exists() else 10.0
        ref_scale = total_scale * (nissl_um / 25.0)  # = total_scale for 25µm vol
        fwd = SimilarityTransform(scale=ref_scale, rotation=angle_rad, translation=(dx, dy))
        placed = skwarp(
            gray,
            fwd.inverse,
            output_shape=out_shape,
            order=1,
            preserve_range=True,
            mode="constant",
            cval=0.0,
        ).astype(np.float32)
        return placed

    # ── Fallback: low-res thumbnail ───────────────────────────────────────────
    cache_dir = cache_dir or _ALLEN_REF_CACHE
    thumb_path = cache_dir / f"thumb_z{atlas_z:04d}.jpg"
    if not thumb_path.exists():
        return None
    try:
        from PIL import Image as _PILImage

        img_pil = _PILImage.open(str(thumb_path)).convert("RGB")
        img = np.array(img_pil, dtype=np.float32)
        gray = img[..., :3].mean(axis=-1)
        th, tw = gray.shape
        if hemisphere in ("right", "right_flipped"):
            gray = gray[:, tw // 2 :]
        elif hemisphere == "left":
            gray = gray[:, : tw // 2]
        if hemisphere == "right_flipped":
            gray = np.fliplr(gray)
        gray = 255.0 - gray  # invert: thumbnail has bright background
        lo, hi = float(np.percentile(gray, 2)), float(np.percentile(gray, 98))
        if hi > lo:
            gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
        else:
            gray = np.zeros_like(gray)
        thumb_scale = total_scale * (_ALLEN_THUMB_UM_PX / 25.0)
        fwd = SimilarityTransform(scale=thumb_scale, rotation=angle_rad, translation=(dx, dy))
        placed = skwarp(
            gray,
            fwd.inverse,
            output_shape=out_shape,
            order=1,
            preserve_range=True,
            mode="constant",
            cval=0.0,
        ).astype(np.float32)
        return placed
    except Exception:
        return None


def _structure_feature_map(img: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to a structure-feature map for cross-modal registration.

    Highlights boundaries and dark structures (fiber tracts, ventricles) that are
    visible in BOTH Nissl and cleared-brain fluorescence images:
      - Local contrast (Sobel edges)
      - Dark-region map (inverted, so fiber tracts become bright peaks)
    Returns float32 [0,1].
    """
    from scipy.ndimage import gaussian_filter as _gf

    f = img.astype(np.float32)
    # Normalise
    lo, hi = (
        float(np.percentile(f[f > 0], 5)) if f.max() > 0 else (0.0, 1.0),
        float(np.percentile(f, 95)),
    )
    if isinstance(lo, tuple):
        lo = lo[0]
    if hi > lo:
        f = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    # Edge component (Sobel magnitude)
    from skimage.filters import sobel as _sobel

    edges = _sobel(f)
    emx = float(edges.max())
    if emx > 0:
        edges /= emx
    # Dark-region component: local minimum contrast (captures fiber tracts & ventricles)
    smooth = _gf(f, sigma=3.0)
    dark = 1.0 - smooth  # bright where tissue is dark
    # Combine: edges + dark regions, both normalised
    feat = 0.6 * edges + 0.4 * dark
    mx = float(feat.max())
    return (feat / mx).astype(np.float32) if mx > 0 else feat.astype(np.float32)


def _sitk_nonlinear_register(
    placed_ref: np.ndarray,
    real_u8: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    max_dim: int = 512,
    mi_bins: int = 32,
    max_iter: int = 80,
    mesh_size: int = 6,
    max_disp_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Register placed Allen reference image to lightsheet image using SimpleITK BSpline.

    Both images are converted to structure-feature maps (edges + dark regions)
    before registration so that cross-modal intensity differences don't interfere.
    Returns (flow_v, flow_u) displacement in full-resolution pixels, or None on failure.
    """
    try:
        import SimpleITK as sitk

        h, w = real_u8.shape[:2]
        ds = max(1, max(h, w) // max_dim)
        h_ds, w_ds = h // ds, w // ds

        # Convert both to structure-feature maps before registration
        real_f32 = real_u8.astype(np.float32) / 255.0
        fixed_feat = _structure_feature_map(real_f32)
        moving_feat = _structure_feature_map(placed_ref)

        fixed_arr = fixed_feat[: h_ds * ds : ds, : w_ds * ds : ds]
        moving_arr = moving_feat[: h_ds * ds : ds, : w_ds * ds : ds]

        # Mask: zero out moving image pixels with no tissue (prevents background MI pollution)
        if tissue_mask is not None:
            tmask_ds = tissue_mask[: h_ds * ds : ds, : w_ds * ds : ds].astype(bool)
            moving_arr = moving_arr * tmask_ds.astype(np.float32)

        fixed_sitk = sitk.GetImageFromArray(fixed_arr)
        moving_sitk = sitk.GetImageFromArray(moving_arr)
        spacing_mm = (25.0 * ds / 1000.0, 25.0 * ds / 1000.0)  # coarse enough to be scale-agnostic
        fixed_sitk.SetSpacing(spacing_mm)
        moving_sitk.SetSpacing(spacing_mm)

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=mi_bins)
        reg.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
        reg.SetMetricSamplingPercentage(0.20, seed=42)
        reg.SetInterpolator(sitk.sitkLinear)

        tx = sitk.BSplineTransformInitializer(
            fixed_sitk,
            transformDomainMeshSize=[mesh_size, mesh_size],
            order=3,
        )
        reg.SetInitialTransform(tx, inPlace=True)
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=max_iter,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7,
        )
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

        tx_final = reg.Execute(fixed_sitk, moving_sitk)

        # Convert BSpline transform → dense displacement field
        disp_img = sitk.TransformToDisplacementField(
            tx_final,
            sitk.sitkVectorFloat32,
            size=[w_ds, h_ds],
            outputSpacing=spacing_mm,
        )
        disp = sitk.GetArrayFromImage(disp_img)  # (h_ds, w_ds, 2): [y-disp, x-disp] in mm

        # Convert from mm → full-resolution pixels
        ds / (25.0 * ds / 1000.0) / 1000.0  # simplifies to 1/25 * 1000 … keep explicit
        # Simpler: disp in mm → ds-pixels → full pixels
        # 1 ds-pixel corresponds to ds real pixels; spacing = 25*ds/1000 mm/px_ds
        px_per_mm_ds = 1000.0 / (25.0 * ds)
        flow_v_ds = disp[..., 0] * px_per_mm_ds * ds  # → full-res pixels (y)
        flow_u_ds = disp[..., 1] * px_per_mm_ds * ds  # → full-res pixels (x)

        # Upsample to full resolution
        from skimage.transform import resize as sk_resize

        flow_v = sk_resize(
            flow_v_ds, (h, w), order=1, anti_aliasing=False, preserve_range=True
        ).astype(np.float32)
        flow_u = sk_resize(
            flow_u_ds, (h, w), order=1, anti_aliasing=False, preserve_range=True
        ).astype(np.float32)

        # Clamp excessive displacement
        max_disp_px = max_disp_frac * min(h, w)
        mag = np.sqrt(flow_v**2 + flow_u**2)
        scale_clamp = np.where(mag > max_disp_px, max_disp_px / (mag + 1e-6), 1.0)
        flow_v *= scale_clamp
        flow_u *= scale_clamp

        return flow_v, flow_u
    except Exception as _e:
        return None


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
        support = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(4)).astype(
            np.float32
        )
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
    aa = a > 0
    bb = b > 0
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
    atlas = label > 0
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


def _clip_label_to_tissue(
    label: np.ndarray, tissue_mask: np.ndarray | None, pad: int = 5
) -> np.ndarray:
    if tissue_mask is None:
        return label.astype(np.int32)
    clip = morphology.dilation(
        tissue_mask.astype(np.uint8), morphology.disk(max(1, int(pad)))
    ).astype(bool)
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
        rid_mask = out == int(rid)
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
        fill_mask &= dist <= float(max_dist_px)
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
        support = out > 0
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
        outer_fill_max_dist = float(
            _warp_param(warp_params, "cleanup_outer_fill_max_dist_px", 240.0)
        )
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
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for _ in range(iters):
        bd = find_boundaries(out.astype(np.int32), mode="thick", connectivity=2)
        if tissue_mask is not None:
            support = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(2)).astype(
                bool
            )
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

    for rid, cnt in zip(ids.tolist(), counts.tolist(), strict=False):
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
        flow_v = (
            resize(
                flow_v_ds,
                (h, w),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            * sy
            * float(sign)
        )
        flow_u = (
            resize(
                flow_u_ds,
                (h, w),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            * sx
            * float(sign)
        )

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
        return (
            refined,
            float(q_after),
            {
                "tag": str(tag),
                "downsample": int(ds),
                "max_displacement_px": float(max_disp),
                "score_after": float(q_after),
            },
        )

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
    except Exception:
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
    ants_max_interior_loss_frac = float(
        _warp_param(warp_params, "ants_max_interior_loss_frac", 0.012)
    )
    lap_iter = int(
        _warp_param(
            warp_params, "ants_laplacian_iter", _warp_param(warp_params, "laplacian_smooth_iter", 1)
        )
    )
    lap_lambda = float(
        _warp_param(
            warp_params, "ants_laplacian_lambda", _warp_param(warp_params, "laplacian_lambda", 0.18)
        )
    )
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
        wy = (
            ants.apply_transforms(
                fixed=fixed,
                moving=y_img,
                transformlist=tx_used,
                interpolator="linear",
            )
            .numpy()
            .astype(np.float32)
        )
        wx = (
            ants.apply_transforms(
                fixed=fixed,
                moving=x_img,
                transformlist=tx_used,
                interpolator="linear",
            )
            .numpy()
            .astype(np.float32)
        )
        flow_v_ds = wy - yy
        flow_u_ds = wx - xx
    except Exception as e:
        return label, {"ok": False, "reason": f"ants_registration_failed: {e}"}
    finally:
        if not keep_temp:
            try:
                for arr in ("fwdtransforms", "invtransforms"):
                    for p in (
                        reg.get(arr, []) if "reg" in locals() and isinstance(reg, dict) else []
                    ):
                        pp = Path(str(p))
                        if pp.exists():
                            pp.unlink(missing_ok=True)
            except Exception:
                pass

    if (out_h, out_w) != (h, w):
        sy = float(h) / float(out_h)
        sx = float(w) / float(out_w)
        flow_v = (
            resize(
                flow_v_ds,
                (h, w),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            * sy
        )
        flow_u = (
            resize(
                flow_u_ds,
                (h, w),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            * sx
        )
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
    src_rc = (
        np.vstack([src_inner, src_outer])
        if len(src_inner) and len(src_outer)
        else (src_inner if len(src_inner) else src_outer)
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
            candidate_results.append(
                (
                    "liquify_tps",
                    refined_tps.astype(np.int32),
                    float(q_tps),
                    {
                        "tps_ctrl": int(min(len(src_xy), max(32, int(tps_ctrl)))),
                        "tps_smooth": float(tps_smooth),
                        "tps_grid_step": int(max(2, int(tps_grid_step))),
                    },
                )
            )
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
        candidate_results.append(
            (
                "liquify_dense_field",
                refined_dense.astype(np.int32),
                float(q_dense),
                {
                    "sigma_px": float(sigma),
                },
            )
        )
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


def _silhouette_conform_warp(
    label: np.ndarray,
    real_u8: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    ds_size: int = 512,
    fill_radius: int = 30,
    flow_attachment: float = 12.0,
    max_disp_ratio: float = 0.10,
    max_disp_min_px: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """Warp atlas to match brain silhouette using TV-L1 optical flow on distance transforms.

    Registration target: signed distance transform of filled brain silhouette vs
    atlas silhouette. Shape-based (not intensity-based) → works for sparse
    cleared-brain data where fluorescent signal is sparse cells, not solid tissue.

    TV-L1 flow has built-in Total-Variation regularisation that prevents tearing.
    max_disp_ratio=0.10 limits displacement to 10% of image → conservative, safe.
    """
    from scipy.ndimage import distance_transform_edt

    h, w = label.shape
    if int(np.sum(label > 0)) < 100:
        return label, {"ok": False, "reason": "empty_label"}

    ds = float(ds_size) / float(max(h, w))
    dh = max(64, int(round(h * ds)))
    dw = max(64, int(round(w * ds)))
    fill_r = max(4, int(round(fill_radius * ds)))

    # ── Brain silhouette (low-threshold fill → solid outline) ────────────
    real_small = resize(
        real_u8.astype(np.float32), (dh, dw), order=1, preserve_range=True, anti_aliasing=True
    )
    nz = real_small[real_small > 3]
    thr_low = float(np.percentile(nz, 10)) if nz.size > 50 else 10.0
    brain_rough = real_small > thr_low
    brain_filled = morphology.closing(brain_rough, morphology.disk(fill_r))
    lbl_t = measure.label(brain_filled)
    if lbl_t.max() > 1:
        areas = {r.label: r.area for r in measure.regionprops(lbl_t)}
        brain_filled = lbl_t == max(areas, key=areas.__getitem__)
    # Restrict to hemisphere region if tissue hint provided
    if tissue_mask is not None and int(np.sum(tissue_mask)) > 100:
        hemi_s = resize(tissue_mask.astype(np.float32), (dh, dw), order=0) > 0.5
        brain_filled = brain_filled & morphology.dilation(hemi_s, morphology.disk(max(3, fill_r)))

    if int(np.sum(brain_filled)) < 50:
        return label, {"ok": False, "reason": "brain_silhouette_empty"}

    # ── Atlas silhouette ──────────────────────────────────────────────────
    atlas_small = resize((label > 0).astype(np.float32), (dh, dw), order=0) > 0.5
    lbl_a = measure.label(atlas_small)
    if lbl_a.max() > 1:
        areas_a = {r.label: r.area for r in measure.regionprops(lbl_a)}
        atlas_small = lbl_a == max(areas_a, key=areas_a.__getitem__)

    # ── Distance transforms → smooth registration target ─────────────────
    brain_dt = distance_transform_edt(brain_filled).astype(np.float32)
    atlas_dt = distance_transform_edt(atlas_small).astype(np.float32)
    brain_dt /= max(float(brain_dt.max()), 1e-6)
    atlas_dt /= max(float(atlas_dt.max()), 1e-6)

    # ── TV-L1 optical flow: atlas_dt → brain_dt ──────────────────────────
    try:
        fv, fu = optical_flow_tvl1(
            brain_dt,
            atlas_dt,
            attachment=float(flow_attachment),
            tightness=0.3,
            num_warp=6,
            num_iter=40,
            tol=1e-3,
            prefilter=False,
            dtype=np.float32,
        )
    except Exception as e:
        return label, {"ok": False, "reason": f"flow_failed: {e}"}

    # Scale flow back to full resolution
    max_disp = float(max(max_disp_min_px, min(h, w) * max_disp_ratio))
    flow_v = resize(
        fv * (float(h) / dh), (h, w), order=1, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    flow_u = resize(
        fu * (float(w) / dw), (h, w), order=1, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    flow_v = gaussian_filter(flow_v, sigma=3.0).astype(np.float32)
    flow_u = gaussian_filter(flow_u, sigma=3.0).astype(np.float32)

    flow_mag = np.sqrt(flow_v**2 + flow_u**2)
    over = flow_mag > max_disp
    if np.any(over):
        sc = max_disp / (flow_mag[over] + 1e-6)
        flow_v[over] *= sc
        flow_u[over] *= sc

    # ── Apply warp ────────────────────────────────────────────────────────
    warped = _warp_label_with_flow(label, flow_v, flow_u)

    # Clip to dilated brain region (gentle — large dilation to not over-clip)
    clip_small = morphology.dilation(brain_filled, morphology.disk(max(3, fill_r))).astype(bool)
    clip_full = resize(clip_small.astype(np.float32), (h, w), order=0) > 0.5
    warped = np.where(clip_full, warped, 0).astype(np.int32)

    return warped, {
        "ok": True,
        "method": "silhouette_tvl1_flow",
        "max_disp_px": float(max_disp),
        "ds_size": int(ds_size),
    }


def _contour_conform_warp(
    label: np.ndarray,
    real_u8: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    n_contour_pts: int = 72,
    tps_smooth: float = 4.0,
    max_disp_ratio: float = 0.22,
    max_disp_min_px: float = 30.0,
    fill_radius: int = 30,
    ds_size: int = 320,
) -> tuple[np.ndarray, dict]:
    """Boundary-conforming warp: fit atlas outer boundary to tissue contour via TPS.

    Designed for sparse cleared-brain data where intensity-based registration fails.
    Derives the brain outline from real_u8 using a low global threshold + morphological
    closing, which captures the full brain boundary even when fluorescent signal is sparse
    or when the tissue_mask only covers a narrow bright strip.

    All morphological operations run on a downsampled image (ds_size px longest side)
    to keep runtime fast; contour points are scaled back to full resolution before TPS.
    """
    h, w = label.shape
    if int(np.sum(label > 0)) < 100:
        return label, {"ok": False, "reason": "empty_label"}

    # ── Downsample scale for fast morphology ─────────────────────────────
    ds = float(ds_size) / float(max(h, w))
    dh = max(32, int(round(h * ds)))
    dw = max(32, int(round(w * ds)))

    def _ds_bin(arr: np.ndarray) -> np.ndarray:
        return (
            resize(
                arr.astype(np.float32), (dh, dw), order=0, preserve_range=True, anti_aliasing=False
            )
            > 0.5
        )

    # ── Brain silhouette: prefer pre-computed solid mask; fall back to raw image ──
    # When a solid mask is provided (e.g. 4-sigma corners threshold + disk-closing from
    # the linear placement block), use it directly — far more reliable than raw-image
    # re-detection for sparse cleared-brain fluorescence where signal is punctate.
    fill_r_ds = max(4, int(round(fill_radius * ds)))
    _mask_area = int(np.sum(tissue_mask)) if tissue_mask is not None else 0
    _use_mask_direct = _mask_area > int(h * w * 0.02)  # solid mask: covers >2% of image

    from scipy.ndimage import binary_fill_holes as _bfh

    if _use_mask_direct:
        # Downsample, re-close at downsampled scale, fill interior holes.
        hemi_small = _ds_bin(tissue_mask)
        tissue_filled_d = morphology.closing(hemi_small, morphology.disk(max(fill_r_ds, 3)))
        tissue_filled_d = _bfh(tissue_filled_d)
        lbl_t = measure.label(tissue_filled_d)
        if lbl_t.max() > 1:
            areas = {r.label: r.area for r in measure.regionprops(lbl_t)}
            tissue_filled_d = lbl_t == max(areas, key=areas.__getitem__)
        tissue_small = tissue_filled_d.astype(bool)
    else:
        # Derive silhouette from raw image: large closing disk to bridge sparse signals.
        real_small = resize(
            real_u8.astype(np.float32), (dh, dw), order=1, preserve_range=True, anti_aliasing=True
        )
        nz = real_small[real_small > 3]
        thr_low = (
            float(np.percentile(nz, 10)) if nz.size > 50 else float(np.percentile(real_small, 10))
        )
        brain_rough = real_small > thr_low
        # Closing disk: at least 6% of shorter dim to bridge sparse-signal gaps
        fill_r_big = max(fill_r_ds, int(min(dh, dw) * 0.06))
        tissue_filled = morphology.closing(brain_rough, morphology.disk(fill_r_big))
        tissue_filled = _bfh(tissue_filled)
        lbl_t = measure.label(tissue_filled)
        if lbl_t.max() > 1:
            areas = {r.label: r.area for r in measure.regionprops(lbl_t)}
            tissue_filled = lbl_t == max(areas, key=areas.__getitem__)
        if tissue_mask is not None and _mask_area > 100:
            hemi_small = _ds_bin(tissue_mask)
            hemi_dilated = morphology.dilation(hemi_small, morphology.disk(max(3, fill_r_ds)))
            tissue_filled = tissue_filled & hemi_dilated
        tissue_small = tissue_filled.astype(bool)

    label_small = _ds_bin(label > 0)

    # ── 1. Tissue outer contour ───────────────────────────────────────────
    t_contour = find_boundaries(tissue_small, mode="outer", connectivity=2)
    t_pts_ds = np.column_stack(np.where(t_contour))  # [row_ds, col_ds]
    if len(t_pts_ds) < 20:
        return label, {"ok": False, "reason": "too_few_tissue_contour_pts"}

    # ── 2. Atlas outer boundary (on small image) ──────────────────────────
    atlas_filled = morphology.dilation(label_small, morphology.disk(max(1, int(round(3 * ds)))))
    lbl_a = measure.label(atlas_filled)
    if lbl_a.max() > 1:
        areas_a = {r.label: r.area for r in measure.regionprops(lbl_a)}
        atlas_filled = lbl_a == max(areas_a, key=areas_a.__getitem__)
    a_contour = find_boundaries(atlas_filled, mode="outer", connectivity=2)
    a_pts_ds = np.column_stack(np.where(a_contour))  # [row_ds, col_ds]
    if len(a_pts_ds) < 20:
        return label, {"ok": False, "reason": "too_few_atlas_contour_pts"}

    # Scale contour points back to full resolution
    scale_r = float(h) / float(dh)
    scale_c = float(w) / float(dw)
    t_pts = t_pts_ds.astype(np.float32) * np.array([scale_r, scale_c], dtype=np.float32)
    a_pts = a_pts_ds.astype(np.float32) * np.array([scale_r, scale_c], dtype=np.float32)

    # ── 3. Angular parameterisation → matched contour pairs ──────────────
    t_cy = float(t_pts[:, 0].mean())
    t_cx = float(t_pts[:, 1].mean())
    a_cy = float(a_pts[:, 0].mean())
    a_cx = float(a_pts[:, 1].mean())
    t_angles = np.arctan2(t_pts[:, 0] - t_cy, t_pts[:, 1] - t_cx)
    a_angles = np.arctan2(a_pts[:, 0] - a_cy, a_pts[:, 1] - a_cx)

    sample_angles = np.linspace(-np.pi, np.pi, n_contour_pts, endpoint=False)
    src_rc: list[np.ndarray] = []
    dst_rc: list[np.ndarray] = []
    for ang in sample_angles:
        a_diff = np.abs(((a_angles - ang + np.pi) % (2 * np.pi)) - np.pi)
        t_diff = np.abs(((t_angles - ang + np.pi) % (2 * np.pi)) - np.pi)
        src_rc.append(a_pts[int(np.argmin(a_diff))])
        dst_rc.append(t_pts[int(np.argmin(t_diff))])

    src_rc_arr = np.array(src_rc, dtype=np.float32)
    dst_rc_arr = np.array(dst_rc, dtype=np.float32)

    # Remove near-duplicates (at downsampled precision)
    pair_key = np.round(np.concatenate([src_rc_arr, dst_rc_arr], axis=1) * ds, 0)
    _, uniq_idx = np.unique(pair_key, axis=0, return_index=True)
    src_rc_arr = src_rc_arr[np.sort(uniq_idx)]
    dst_rc_arr = dst_rc_arr[np.sort(uniq_idx)]
    if len(src_rc_arr) < 8:
        return label, {"ok": False, "reason": "too_few_contour_pairs"}

    # ── 4. TPS displacement field (RBF thin-plate) ───────────────────────
    du = (dst_rc_arr[:, 1] - src_rc_arr[:, 1]).astype(np.float64)
    dv = (dst_rc_arr[:, 0] - src_rc_arr[:, 0]).astype(np.float64)

    max_disp = float(max(max_disp_min_px, min(h, w) * max_disp_ratio))
    mag = np.sqrt(du**2 + dv**2)
    clip_mask_arr = mag > max_disp
    if np.any(clip_mask_arr):
        sc0 = max_disp / (mag[clip_mask_arr] + 1e-6)
        du[clip_mask_arr] *= sc0
        dv[clip_mask_arr] *= sc0

    # RBF is evaluated on a coarse grid (speed), then upsampled
    grid_step = max(8, min(h, w) // 60)
    gy = np.arange(0, h, grid_step, dtype=np.float32)
    gx = np.arange(0, w, grid_step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)
    try:
        rbf_u = Rbf(
            src_rc_arr[:, 1], src_rc_arr[:, 0], du, function="thin_plate", smooth=float(tps_smooth)
        )
        rbf_v = Rbf(
            src_rc_arr[:, 1], src_rc_arr[:, 0], dv, function="thin_plate", smooth=float(tps_smooth)
        )
    except Exception as e:
        return label, {"ok": False, "reason": f"rbf_failed: {e}"}

    du_coarse = rbf_u(grid_x, grid_y).astype(np.float32)
    dv_coarse = rbf_v(grid_x, grid_y).astype(np.float32)
    flow_u = resize(du_coarse, (h, w), order=1, preserve_range=True, anti_aliasing=True).astype(
        np.float32
    )
    flow_v = resize(dv_coarse, (h, w), order=1, preserve_range=True, anti_aliasing=True).astype(
        np.float32
    )
    flow_u = gaussian_filter(flow_u, sigma=3.0).astype(np.float32)
    flow_v = gaussian_filter(flow_v, sigma=3.0).astype(np.float32)

    flow_mag = np.sqrt(flow_u**2 + flow_v**2)
    over = flow_mag > max_disp
    if np.any(over):
        sc2 = max_disp / (flow_mag[over] + 1e-6)
        flow_u[over] *= sc2
        flow_v[over] *= sc2

    # ── 5. Apply warp → clip to tissue ───────────────────────────────────
    warped = _warp_label_with_flow(label, flow_v, flow_u)
    # Dilate tissue mask slightly so atlas isn't over-clipped at boundaries
    clip_disk = morphology.disk(max(2, int(round(20 * ds))))
    clip_region = morphology.dilation(tissue_small.astype(np.uint8), clip_disk).astype(bool)
    clip_region_full = (
        resize(clip_region.astype(np.float32), (h, w), order=0, preserve_range=True) > 0.5
    )
    warped = np.where(clip_region_full, warped, 0).astype(np.int32)

    return warped, {
        "ok": True,
        "method": "contour_conform_tps",
        "n_ctrl_pts": int(len(src_rc_arr)),
        "max_disp_px": float(max_disp),
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
        candidates.append(
            ("liquify_after_flow", hybrid_label, float(hybrid_meta.get("score_after", -1e9)))
        )
    if ants_meta.get("ok"):
        candidates.append(("ants_laplacian", ants_label, float(ants_meta.get("score_after", -1e9))))
    if ants_hybrid_meta.get("ok"):
        candidates.append(
            (
                "ants_plus_liquify",
                ants_hybrid_label,
                float(ants_hybrid_meta.get("score_after", -1e9)),
            )
        )

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
    id_count = [(int(i), int(c)) for i, c in zip(ids, counts, strict=False) if i != 0]
    id_count.sort(key=lambda x: -x[1])
    id_count = id_count[:max_labels]

    props_map: dict[int, measure.RegionProperties] = {}
    for region_id in np.unique(warped_label):
        if region_id == 0:
            continue
        mask = warped_label == region_id
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
        for fname in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]:
            try:
                fnt = ImageFont.truetype(fname, font_size)
                break
            except Exception:
                pass
        if fnt is None:
            fnt = (
                ImageFont.load_default(size=font_size)
                if hasattr(ImageFont, "load_default")
                else ImageFont.load_default()
            )
        pil_img = Image.fromarray(out)
        pil_draw = ImageDraw.Draw(pil_img)

    for rid, _count in id_count:
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
            _cv2.putText(
                out, acronym, (cx + 1, cy + 1), font, font_scale, shadow_color, lw + 1, _cv2.LINE_AA
            )
            _cv2.putText(out, acronym, (cx, cy), font, font_scale, text_color, lw, _cv2.LINE_AA)
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
    # tissue_shrink_factor < 1.0 makes the atlas smaller to compensate for
    # tissue shrinkage during clearing (e.g. iDISCO+: ~10-15% linear shrinkage → 0.85-0.90)
    _shrink = float(_warp_param(warp_params, "tissue_shrink_factor", 1.0))
    if _shrink != 1.0:
        phys_scale = phys_scale * float(np.clip(_shrink, 0.5, 1.5))
    edge_inner_weight = float(_warp_param(warp_params, "edge_inner_weight", 1.00))
    edge_outer_weight = float(_warp_param(warp_params, "edge_outer_weight", 0.55))
    mask_dice_weight = float(_warp_param(warp_params, "mask_dice_weight", 0.35))
    outside_penalty = float(_warp_param(warp_params, "outside_penalty", 2.2))
    default_refine_topn = 2 if enable_nonlinear else 1
    candidate_top_n = int(_warp_param(warp_params, "linear_candidate_top_n", default_refine_topn))
    candidate_top_n = int(np.clip(candidate_top_n, 1, 3))
    opt_iter_primary = int(_warp_param(warp_params, "linear_opt_maxiter_primary", int(opt_maxiter)))
    opt_iter_secondary = int(
        _warp_param(warp_params, "linear_opt_maxiter_secondary", max(45, int(opt_maxiter * 0.72)))
    )
    keep_margin = float(_warp_param(warp_params, "linear_opt_keep_margin", -0.003))

    # If force_hemisphere is set, compute tissue centroid from the correct image half
    # using a robust high-percentile threshold (ignores background noise).
    # This fixes sparse cleared-brain data where _detect_tissue picks up the whole image.
    # NOTE: "right" hemisphere atlas is placed in original (non-mirrored) orientation with
    # medial face at image center and lateral face to the RIGHT of center.
    # "left" hemisphere: medial at center, lateral to the LEFT.
    force_hemi_for_tissue = str(_warp_param(warp_params, "force_hemisphere", "")).lower().strip()
    if force_hemi_for_tissue in ("right", "right_flipped"):
        # Right hemisphere → look at RIGHT half of image (brain is on right side)
        _half = real[:, rw // 2 :].astype(np.float32)
        _thr = float(np.percentile(_half, 85))
        _mask_half = _half > _thr
        _ys, _xs = np.where(_mask_half)
        if len(_ys) > 50:
            t_cy = float(np.mean(_ys))
            t_cx = float(np.mean(_xs)) + rw // 2  # shift back to full image coords
            t_h = int(_ys.max() - _ys.min() + 1)
            t_w = int(_xs.max() - _xs.min() + 1)
        else:
            t_cy, t_cx = rh / 2, rw * 3 / 4
            t_h, t_w = rh, rw // 2
        tissue_aspect = t_h / max(t_w, 1)
        tissue_orientation = 0.0
        _full_mask = np.zeros((rh, rw), dtype=bool)
        _full_mask[:, rw // 2 :] = real[:, rw // 2 :].astype(np.float32) > _thr
        tissue_mask = _full_mask
    elif force_hemi_for_tissue == "right_mirrored":
        # right_mirrored: brain on LEFT (legacy/radiological convention)
        _half = real[:, : rw // 2].astype(np.float32)
        _thr = float(np.percentile(_half, 85))
        _mask_half = _half > _thr
        _ys, _xs = np.where(_mask_half)
        if len(_ys) > 50:
            t_cy = float(np.mean(_ys))
            t_cx = float(np.mean(_xs))
            t_h = int(_ys.max() - _ys.min() + 1)
            t_w = int(_xs.max() - _xs.min() + 1)
        else:
            t_cy, t_cx = rh / 2, rw / 4
            t_h, t_w = rh, rw // 2
        tissue_aspect = t_h / max(t_w, 1)
        tissue_orientation = 0.0
        _full_mask = np.zeros((rh, rw), dtype=bool)
        _full_mask[:, : rw // 2] = real[:, : rw // 2].astype(np.float32) > _thr
        tissue_mask = _full_mask
    elif force_hemi_for_tissue == "left":
        # Left hemisphere → look at RIGHT half of image
        _half = real[:, rw // 2 :].astype(np.float32)
        _thr = float(np.percentile(_half, 85))
        _mask_half = _half > _thr
        _ys, _xs = np.where(_mask_half)
        if len(_ys) > 50:
            t_cy = float(np.mean(_ys))
            t_cx = float(np.mean(_xs)) + rw // 2  # shift back to full image coords
            t_h = int(_ys.max() - _ys.min() + 1)
            t_w = int(_xs.max() - _xs.min() + 1)
        else:
            t_cy, t_cx = rh / 2, rw * 3 / 4
            t_h, t_w = rh, rw // 2
        tissue_aspect = t_h / max(t_w, 1)
        tissue_orientation = 0.0
        _full_mask = np.zeros((rh, rw), dtype=bool)
        _full_mask[:, rw // 2 :] = real[:, rw // 2 :].astype(np.float32) > _thr
        tissue_mask = _full_mask
    else:
        info = _detect_tissue(real)
        if not info["ok"]:
            out, _fitted = _align_shape_physical(
                label, (rh, rw), atlas_res_um, real_res_um, fit_mode
            )
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
    # right_orig: right hemisphere in ORIGINAL (non-flipped) orientation
    # x increases from medial (a_mid_x) to lateral (a_w-1).
    # This places medial at image center and lateral to the RIGHT — correct for
    # a right hemisphere that appears on the right side of the light-sheet image.
    right_orig_label = label.astype(np.int32).copy()
    right_orig_label[:, :a_mid_x] = 0
    # right_flipped: right hemisphere LR-flipped so lateral is at LEFT, medial at RIGHT.
    # Use when the brain is on the RIGHT of the image but with lateral surface facing
    # inward (left) and medial cut surface facing outward (right).
    right_flipped_label = np.fliplr(right_orig_label.copy())
    all_candidates = [
        ("full", full_label, False, "full"),
        ("left", left_label, True, "left"),
        ("right_mirrored", right_label, True, "right_mirrored"),
        ("right_orig", right_orig_label, True, "right_orig"),
        ("right_flipped", right_flipped_label, True, "right_flipped"),
    ]
    # If caller specifies hemisphere, skip the irrelevant candidates entirely
    # (avoids the scoring competition that always favours full-brain on sparse tissue)
    force_hemi = str(_warp_param(warp_params, "force_hemisphere", "")).lower().strip()
    if force_hemi == "left":
        raw_candidates = [c for c in all_candidates if c[0] == "left"]
    elif force_hemi == "right":
        raw_candidates = [c for c in all_candidates if c[0] == "right_orig"]
    elif force_hemi == "right_mirrored":
        raw_candidates = [c for c in all_candidates if c[0] == "right_mirrored"]
    elif force_hemi == "right_flipped":
        raw_candidates = [c for c in all_candidates if c[0] == "right_flipped"]
    else:
        raw_candidates = all_candidates

    # Save tissue mask from detection block for boundary-conforming warp later.
    # The physical placement block below sets tissue_mask=None for candidate scoring,
    # but we want the detected mask for _contour_conform_warp.
    _detected_tissue_mask = tissue_mask

    # For forced hemisphere, set atlas physical size (t_h/t_w) from atlas voxel scale.
    # For "right": t_cx comes from the tissue-detection block above (right-half centroid).
    # For "right_mirrored"/"left": t_cx is formula-based (legacy).
    # t_cy = image vertical center; tissue_mask = None (use all atlas voxels for scoring).
    if force_hemi in ("right", "right_flipped", "right_mirrored", "left"):
        # Atlas bbox in voxels → image pixels at physical scale
        cand_label_for_size = raw_candidates[0][1]
        cb0y, cb0x, cb1y, cb1x = _atlas_bbox(cand_label_for_size)
        atlas_h_px = float(cb1y - cb0y) * phys_scale
        atlas_w_px = float(cb1x - cb0x) * phys_scale
        t_cy = float(rh) / 2.0
        t_h = atlas_h_px
        t_w = atlas_w_px
        tissue_aspect = atlas_h_px / max(atlas_w_px, 1.0)
        tissue_mask = None

        if force_hemi == "right":
            # right_orig: align atlas MEDIAL edge (cb0x) with brain's LEFT edge (medial cut).
            # Brain's left edge = robust leftmost extent of brain tissue in full image.
            # t_cx = x_brain_left + atlas_w_px/2  (places cb0x at x_brain_left exactly)
            _img_f = real.astype(np.float32)
            _nz_f = _img_f[_img_f > 5]
            _thr_full = float(np.percentile(_nz_f, 15)) if len(_nz_f) > 500 else 50.0
            _bmask_full = _img_f > _thr_full
            try:
                _bmask_full = morphology.closing(_bmask_full, morphology.disk(8))
            except Exception:
                pass
            _bcols = np.where(_bmask_full.any(axis=0))[0]
            if len(_bcols) > 100:
                x_brain_left = float(np.percentile(_bcols, 3))
                t_cx = x_brain_left + atlas_w_px / 2.0
        elif force_hemi == "right_flipped":
            # right_flipped: single right hemisphere, lateral at LEFT, medial at RIGHT.
            # Detect the actual brain bounding box and scale atlas to cover it directly.
            # This handles the case where the tissue size differs from the atlas size
            # (e.g. due to clearing-induced expansion/shrinkage or pixel-size uncertainty).
            _img_f = real.astype(np.float32)
            _nz_f = _img_f[_img_f > 5]
            _thr_full = float(np.percentile(_nz_f, 30)) if len(_nz_f) > 500 else 80.0
            _bmask_full = _img_f > _thr_full
            try:
                # Large closing disk to fill interior holes; then keep only the largest blob
                _bmask_full = morphology.closing(_bmask_full, morphology.disk(40))
                from skimage import measure as _skm

                _labeled = _skm.label(_bmask_full)
                if _labeled.max() > 0:
                    _sizes = np.bincount(_labeled.ravel())
                    _sizes[0] = 0
                    _bmask_full = _labeled == _sizes.argmax()
            except Exception:
                pass
            _bcols = np.where(_bmask_full.any(axis=0))[0]
            _brows = np.where(_bmask_full.any(axis=1))[0]
            if len(_bcols) > 100 and len(_brows) > 100:
                x_brain_left = float(np.percentile(_bcols, 2))
                x_brain_right = float(np.percentile(_bcols, 98))
                y_brain_top = float(np.percentile(_brows, 2))
                y_brain_bot = float(np.percentile(_brows, 98))
                t_cy = (y_brain_bot + y_brain_top) / 2.0
                tissue_aspect = (y_brain_bot - y_brain_top) / max(x_brain_right - x_brain_left, 1.0)

                # Compute accurate tissue mask using 4-sigma corner-background threshold.
                # This is far more precise than _bmask_full (30th percentile) and is used for:
                #   (a) area-adaptive atlas scaling
                #   (b) tissue_mask passed to finalize_registered_label and _contour_conform_warp
                # For right_flipped: brain occupies the RIGHT side of the image, so the
                # right corners contain brain signal — only use LEFT corners as background.
                _b = 80
                _corners_flat = np.concatenate(
                    [_img_f[:_b, :_b].ravel(), _img_f[-_b:, :_b].ravel()]
                )
                _bg_thr4s = float(np.mean(_corners_flat) + 4.0 * np.std(_corners_flat))
                _tissue_accurate_4sig = _img_f > _bg_thr4s
                try:
                    # disk(50): 250µm closing radius, bridges sparse cleared-brain signal gaps
                    _tissue_accurate_4sig = morphology.closing(
                        _tissue_accurate_4sig, morphology.disk(50)
                    )
                    from scipy.ndimage import binary_fill_holes as _bfh2

                    _tissue_accurate_4sig = _bfh2(_tissue_accurate_4sig)
                    from skimage import measure as _skm2

                    _lbl2 = _skm2.label(_tissue_accurate_4sig)
                    if _lbl2.max() > 0:
                        _sz2 = np.bincount(_lbl2.ravel())
                        _sz2[0] = 0
                        _tissue_accurate_4sig = _lbl2 == _sz2.argmax()
                except Exception:
                    pass
                _bmask_area = float(_tissue_accurate_4sig.sum())
                # Area-adaptive scaling: scale atlas so its footprint area matches
                # the detected tissue area. This prevents the atlas from overflowing
                # tissue boundaries when the atlas cross-section at this AP is larger
                # than the actual cleared tissue (common for anterior/posterior slices).
                _atlas_count = float((cand_label_for_size > 0).sum())
                _cb_bh = float(max(1, cb1y - cb0y))
                _cb_bw = float(max(1, cb1x - cb0x))
                if _atlas_count > 100 and _bmask_area > 100:
                    _area_scale = float(
                        np.sqrt(_bmask_area / max(_atlas_count * phys_scale**2, 1.0))
                    )
                    _area_scale = float(np.clip(_area_scale, 0.5, 1.5))
                else:
                    _area_scale = 1.0
                # Override t_h/t_w to encode the area scale: fit_h = fit_w = _area_scale.
                # Anchor atlas medial edge (right side of flipped atlas) to x_brain_right.
                t_w = _cb_bw * phys_scale * _area_scale
                t_h = _cb_bh * phys_scale * _area_scale
                t_cx = x_brain_right - t_w / 2.0
        elif force_hemi == "right_mirrored":
            # Flipped atlas: x=0=lateral, x=a_mid_x=medial. Keep legacy formula.
            full_brain_px = a_w * phys_scale
            t_cx = float(rw) / 2.0 - full_brain_px / 4.0
        else:  # left
            full_brain_px = a_w * phys_scale
            t_cx = float(rw) / 2.0 + full_brain_px / 4.0

        # For right_flipped: restore tissue_mask from the brain silhouette we detected above,
        # so that finalize_registered_label clips the atlas to the actual brain boundary.
        # Use the 4-sigma corners mask (_tissue_accurate_4sig) if available — it is much more
        # precise than _bmask_full (30th-percentile), which includes too much background and
        # causes _fill_outer_missing_tissue_nearest to expand the atlas beyond true tissue.
        if force_hemi == "right_flipped":
            if "_tissue_accurate_4sig" in dir():
                tissue_mask = _tissue_accurate_4sig
            elif "_bmask_full" in dir():
                tissue_mask = _bmask_full

    # Determine whether to skip optimizer before the candidate loop (used inside it)
    skip_linear_opt = force_hemi in ("right", "right_flipped", "right_mirrored", "left") and bool(
        _warp_param(warp_params, "physical_placement_skip_opt", True)
    )

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
        if name == "full" or skip_linear_opt:
            # For physical placement (skip_linear_opt), use zero rotation:
            # any rotation shifts the centroid away from t_cx due to the off-axis DV extent.
            init_angle = 0.0
        else:
            init_angle = _clamp_angle(float(tissue_orientation - _atlas_orientation(cand_label)))

        warped_init = _apply_warp(
            cand_label, float(init_s), float(init_angle), float(init_dx), float(init_dy), (rh, rw)
        )
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

    # skip_linear_opt was computed above (before candidate loop).
    # For physical placement (force_hemisphere), disable intensity-based nonlinear
    # (it gets confused by sparse cell signals) and use contour-conform instead.
    if skip_linear_opt:
        enable_nonlinear = False
    _enable_contour_conform = (
        skip_linear_opt
        and tissue_mask is not None
        and bool(_warp_param(warp_params, "enable_silhouette_conform", True))
    )

    ranked_idx = sorted(
        range(len(candidates)), key=lambda i: candidates[i]["init"]["objective"], reverse=True
    )
    for rank, idx in enumerate(ranked_idx):
        c = candidates[idx]
        c["best"] = dict(c["init"])
        c["best"]["state"] = "initial"
        if rank >= int(candidate_top_n):
            continue
        if skip_linear_opt:
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

    if _enable_contour_conform:
        # Contour-conforming TPS warp: fit atlas boundary to tissue outline.
        # Pass tissue_mask (= _tissue_accurate_4sig: 4-sigma corners + disk-closing) so
        # _contour_conform_warp uses it directly as the brain silhouette rather than
        # re-deriving from the raw image (unreliable for sparse cleared-brain fluorescence).
        warped, nl_meta = _contour_conform_warp(
            warped,
            real_u8,
            tissue_mask=tissue_mask,
            n_contour_pts=int(_warp_param(warp_params, "contour_n_pts", 72)),
            tps_smooth=float(_warp_param(warp_params, "contour_tps_smooth", 4.0)),
            max_disp_ratio=float(_warp_param(warp_params, "contour_max_disp_ratio", 0.12)),
            max_disp_min_px=float(_warp_param(warp_params, "contour_max_disp_min_px", 40.0)),
            fill_radius=int(_warp_param(warp_params, "contour_fill_radius", 40)),
            ds_size=int(_warp_param(warp_params, "contour_ds_size", 320)),
        )
    elif enable_nonlinear:
        warped, nl_meta = _refine_warp_nonlinear(
            real_u8,
            warped,
            tissue_mask=tissue_mask,
            warp_params=warp_params,
        )
    else:
        nl_meta = {"applied": False, "reason": "disabled"}

    score_left = next(
        (float(c["best"]["quality"]) for c in candidates if c["name"] == "left"), None
    )
    score_right = next(
        (float(c["best"]["quality"]) for c in candidates if c["name"] == "right_mirrored"), None
    )
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


def _align_shape_physical(
    label_slice: np.ndarray,
    target_shape: tuple[int, int],
    atlas_res_um: float = 25.0,
    real_res_um: float = 0.65,
    fit_mode: str = "contain",
):
    """Fallback: physical scale + fit mode + center crop/pad."""
    scale = atlas_res_um / real_res_um
    scaled_label = rescale(
        label_slice.astype(np.float32), scale, order=0, preserve_range=True, anti_aliasing=False
    ).astype(np.int32)
    th, tw = target_shape
    sh, sw = scaled_label.shape
    fh, fw = th / max(sh, 1), tw / max(sw, 1)
    mode = str(fit_mode or "contain").lower()
    fit = (
        min(fh, fw)
        if mode in ("contain",)
        else max(fh, fw)
        if mode == "cover"
        else fw
        if mode == "width-lock"
        else fh
        if mode == "height-lock"
        else min(fh, fw)
    )
    if abs(float(fit) - 1.0) > 1e-6:
        scaled_label = rescale(
            scaled_label.astype(np.float32),
            float(fit),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.int32)
    fitted_shape = (int(scaled_label.shape[0]), int(scaled_label.shape[1]))
    sh, sw = scaled_label.shape
    out = np.zeros((th, tw), dtype=scaled_label.dtype)
    min_h, min_w = min(th, sh), min(tw, sw)
    oy, ox = (th - min_h) // 2, (tw - min_w) // 2
    iy, ix = (sh - min_h) // 2, (sw - min_w) // 2
    out[oy : oy + min_h, ox : ox + min_w] = scaled_label[iy : iy + min_h, ix : ix + min_w]
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
    return (
        max(0, int(x0) - pad),
        max(0, int(y0) - pad),
        int(min(w, x1 + pad) - max(0, x0 - pad)),
        int(min(h, y1 + pad) - max(0, y0 - pad)),
    )


def _paint_boundaries(canvas: np.ndarray, mask: np.ndarray, color) -> None:
    """Paint region boundaries on canvas in-place using find_boundaries."""
    bd = find_boundaries(mask, mode="outer", connectivity=2)
    canvas[bd] = color


def _paint_contours_smooth(canvas: np.ndarray, mask: np.ndarray, color, thickness: int = 2) -> None:
    """Draw smooth anti-aliased contours.

    The mask is Gaussian-blurred before contour detection so that block-scaled
    atlas pixels produce smooth curves instead of staircase edges.
    """
    try:
        import cv2 as _cv2

        arr = (mask > 0).astype(np.uint8) * 255
        # Blur radius must be larger than the atlas voxel size in pixels to smooth staircase edges.
        # Atlas voxels are ~5-8px wide at typical scales; use ~3x that as sigma → ksize ≈ 25-45px.
        ksize = max(21, int(min(canvas.shape[:2]) * 0.025) | 1)  # odd number
        blurred = _cv2.GaussianBlur(arr, (ksize, ksize), 0)
        _, binary = _cv2.threshold(blurred, 127, 255, _cv2.THRESH_BINARY)
        contours, _ = _cv2.findContours(binary, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_TC89_KCOS)
        if contours:
            _cv2.drawContours(
                canvas,
                contours,
                contourIdx=-1,
                color=tuple(int(c) for c in color),
                thickness=thickness,
                lineType=_cv2.LINE_AA,
            )
        return
    except Exception:
        pass
    _paint_boundaries(canvas, mask, color)


def draw_contours(label_img: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    """Draw all inter-region boundaries (internal + outer) from a label map.

    Uses find_boundaries for a single-pass boundary extraction, then smooths
    each connected contour curve using moving-average point smoothing via cv2.
    """
    from skimage.segmentation import find_boundaries

    h, w = label_img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # Single-pass: all region boundaries at once
    boundary = find_boundaries(label_img, mode="inner", connectivity=2).astype(np.uint8) * 255
    try:
        import cv2 as _cv2

        # Find individual contour curves
        contours, _ = _cv2.findContours(boundary, _cv2.RETR_LIST, _cv2.CHAIN_APPROX_NONE)
        smoothed = []
        win = max(5, int(min(h, w) * 0.012))  # moving-average window ~1.2% of image
        for cnt in contours:
            if len(cnt) < 6:
                smoothed.append(cnt)
                continue
            pts = cnt[:, 0, :].astype(np.float32)  # (N, 2)
            n = len(pts)
            # Circular moving average on x and y
            tiled = np.tile(pts, (3, 1))
            sx = np.convolve(tiled[:, 0], np.ones(win) / win, "valid")[n : 2 * n]
            sy = np.convolve(tiled[:, 1], np.ones(win) / win, "valid")[n : 2 * n]
            smooth_cnt = np.round(np.column_stack([sx, sy])).astype(np.int32)
            smoothed.append(smooth_cnt.reshape(-1, 1, 2))
        color_bgr = tuple(int(c) for c in color)
        _cv2.polylines(
            canvas, smoothed, isClosed=True, color=color_bgr, thickness=1, lineType=_cv2.LINE_AA
        )
    except Exception:
        # Fallback: just paint the raw boundary pixels
        mask = boundary > 0
        canvas[mask] = color
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
        for rid in ids[np.argsort(counts)[::-1][: max(1, int(top_k))]]:
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
    display_gamma: float = 1.0,
):
    render_t0 = time.perf_counter()
    timings_ms: dict[str, float | dict[str, float]] = {}

    t0 = time.perf_counter()
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
    timings_ms["load_inputs"] = float((time.perf_counter() - t0) * 1000.0)

    label_shape_before = tuple(int(x) for x in label.shape)
    real_shape = tuple(int(x) for x in real.shape)

    # User pre-transform: rotate / flip atlas
    if rotate_deg != 0.0:
        label = rotate(
            label.astype(np.float32), rotate_deg, order=0, preserve_range=True, resize=True
        ).astype(np.int32)
    if flip_mode == "horizontal":
        label = np.fliplr(label)
    elif flip_mode == "vertical":
        label = np.flipud(label)

    # 鈹€鈹€ Tissue-guided registration 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    t0 = time.perf_counter()
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
            real,
            label,
            atlas_res_um=25.0,
            real_res_um=pixel_size_um,
            fit_mode=fit_mode,
            warp_params=warp_params,
        )

        # ── SimpleITK nonlinear refinement via Allen reference image ──────────
        # Uses the Allen Nissl thumbnail at this AP slice as a moving image,
        # registers it to the lightsheet image, and applies the same displacement
        # to the label map.  Only runs when enable_sitk_ref_refine=True.
        if (
            bool(_warp_param(warp_params, "enable_sitk_ref_refine", False))
            and label_z_index is not None
        ):
            t0_sitk = time.perf_counter()
            _total_scale = float(warp_meta.get("total_scale", 1.0))
            _angle_rad = float(np.deg2rad(warp_meta.get("angle_deg", 0.0)))
            _tx, _ty = warp_meta.get("translation", [0.0, 0.0])
            _hemi = str(warp_meta.get("hemisphere_chosen", "right_flipped"))
            _placed_ref = _load_placed_allen_ref(
                atlas_z=int(label_z_index),
                total_scale=_total_scale,
                angle_rad=_angle_rad,
                dx=float(_tx),
                dy=float(_ty),
                out_shape=(real.shape[0], real.shape[1]),
                hemisphere=_hemi,
            )
            _sitk_meta: dict = {"applied": False}
            if _placed_ref is not None and _placed_ref.max() > 0.01:
                real_u8_for_sitk = _norm_u8_robust(real)
                _flow = _sitk_nonlinear_register(
                    placed_ref=_placed_ref,
                    real_u8=real_u8_for_sitk,
                    tissue_mask=warp_meta.get("tissue_mask"),
                    max_dim=int(_warp_param(warp_params, "sitk_max_dim", 512)),
                    mi_bins=int(_warp_param(warp_params, "sitk_mi_bins", 32)),
                    max_iter=int(_warp_param(warp_params, "sitk_max_iter", 60)),
                    mesh_size=int(_warp_param(warp_params, "sitk_mesh_size", 8)),
                    max_disp_frac=float(_warp_param(warp_params, "sitk_max_disp_frac", 0.12)),
                )
                if _flow is not None:
                    label = _warp_label_with_flow(label, _flow[0], _flow[1])
                    _sitk_meta = {
                        "applied": True,
                        "time_ms": float((time.perf_counter() - t0_sitk) * 1000),
                    }
            warp_meta["sitk_refine"] = _sitk_meta

    timings_ms["registration"] = float((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    try:
        label, postprocess_timings = finalize_registered_label(
            label.astype(np.int32),
            tissue_mask=warp_meta.get("tissue_mask"),
            mode=mode,
            edge_smooth_iter=edge_smooth_iter,
            warp_params=warp_params,
            cleanup_label_topology=_cleanup_label_topology,
            smooth_label_edges=_smooth_label_edges,
            gaussian_vote_boundary_smooth=_gaussian_vote_boundary_smooth,
        )
    except Exception:
        postprocess_timings = {}
    timings_ms["postprocess"] = {
        **(
            {k: float(v) for k, v in postprocess_timings.items()}
            if isinstance(postprocess_timings, dict)
            else {}
        ),
        "wall": float((time.perf_counter() - t0) * 1000.0),
    }
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    label_shape_after = tuple(int(x) for x in label.shape)
    roi_bbox = _roi_bbox_from_real(real)
    rx, ry, rw2, rh2 = roi_bbox
    cx_full = float(rx + rw2 / 2.0)
    cy_full = float(ry + rh2 / 2.0)

    real_u8 = _norm_u8_robust(real)
    # Gamma correction for display: values < 1.0 brighten dim cleared-tissue images.
    # Applied after robust stretch so the correction operates on the full 0-255 range.
    if display_gamma != 1.0:
        _g = float(np.clip(display_gamma, 0.2, 3.0))
        real_u8 = (255.0 * (real_u8.astype(np.float32) / 255.0) ** _g).astype(np.uint8)

    t0 = time.perf_counter()
    colored = colorize_label(label.astype(np.int32), structure_csv=structure_csv)
    timings_ms["colorize"] = float((time.perf_counter() - t0) * 1000.0)

    # Retrieve tissue mask for outer boundary (cached from registration)
    _tissue_mask_for_contour = warp_meta.get("tissue_mask")

    t0 = time.perf_counter()
    if mode == "contour":
        colored = draw_contours(label)
    elif mode == "contour-major":
        colored = draw_contours_major(
            label, top_k=major_top_k, tissue_mask=_tissue_mask_for_contour
        )
    elif mode == "fill+contour":
        # Filled regions + boundary lines on top (no clip artifacts)
        lines = draw_contours(label)
        line_mask = lines.any(axis=2)
        colored[line_mask] = lines[line_mask]
    timings_ms["render_mode"] = float((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    overlay = alpha_blend(real_u8, colored, alpha)
    timings_ms["alpha_blend"] = float((time.perf_counter() - t0) * 1000.0)

    # Quality guard (lower threshold for contour modes which are mostly black)
    effective_thr = (
        min_mean_threshold if mode == "fill" else min(float(min_mean_threshold) * 0.25, 2.0)
    )
    if float(np.mean(overlay)) < effective_thr:
        raise ValueError(
            f"overlay quality check failed: near-black output "
            f"(mean={float(np.mean(overlay)):.2f}, threshold={effective_thr:.2f})"
        )

    # 鈹€鈹€ Region labels 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    t0 = time.perf_counter()
    if mode in ("fill",):
        overlay = draw_region_labels(overlay, label)
    timings_ms["draw_region_labels"] = float((time.perf_counter() - t0) * 1000.0)

    # ── Hard tissue boundary constraint ──────────────────────────────────────
    # Atlas must never overflow the real brain boundary.  We clip the final
    # overlay with a feathered tissue mask (12-px fade) so there are no
    # hard-edge "hedgehog" spikes and no coloured regions outside the tissue.
    _hard_tissue = _tissue_mask_for_contour
    if _hard_tissue is None or _hard_tissue.shape != overlay.shape[:2]:
        # Fallback: recompute from raw image using 4-sigma corner-background threshold.
        _b2 = 80
        _rf = real.astype(np.float32)
        _c2 = np.concatenate(
            [
                _rf[:_b2, :_b2].ravel(),
                _rf[:_b2, -_b2:].ravel(),
                _rf[-_b2:, :_b2].ravel(),
                _rf[-_b2:, -_b2:].ravel(),
            ]
        )
        _thr2 = float(np.mean(_c2) + 4.0 * np.std(_c2))
        _hard_tissue = _rf > _thr2
        try:
            _hard_tissue = morphology.closing(_hard_tissue, morphology.disk(30))
            from skimage import measure as _skm3

            _lbl3 = _skm3.label(_hard_tissue)
            if _lbl3.max() > 0:
                _sz3 = np.bincount(_lbl3.ravel())
                _sz3[0] = 0
                _hard_tissue = _lbl3 == _sz3.argmax()
        except Exception:
            pass
    if _hard_tissue is not None and _hard_tissue.shape == overlay.shape[:2]:
        # Distance from tissue interior: 0 at boundary, grows inward.
        _dist_in = distance_transform_edt(_hard_tissue.astype(np.uint8))
        # Feather: 0→1 over 12 px, so the edge fades smoothly to background.
        _edge_alpha = np.clip(_dist_in / 12.0, 0.0, 1.0).astype(np.float32)
        _raw_rgb = np.stack([real_u8, real_u8, real_u8], axis=-1).astype(np.float32)
        _ea3 = _edge_alpha[:, :, np.newaxis]
        overlay = np.clip(
            overlay.astype(np.float32) * _ea3 + _raw_rgb * (1.0 - _ea3),
            0,
            255,
        ).astype(np.uint8)

    t0 = time.perf_counter()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_png), overlay)

    if warped_label_out is not None:
        warped_label_out.parent.mkdir(parents=True, exist_ok=True)
        imwrite(str(warped_label_out), label.astype(np.int32))
    timings_ms["save_outputs"] = float((time.perf_counter() - t0) * 1000.0)
    timings_ms["total"] = float((time.perf_counter() - render_t0) * 1000.0)

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
            "timings_ms": timings_ms,
            "warp": warp_meta,
        }

    return out_png
