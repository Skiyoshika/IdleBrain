from __future__ import annotations

from pathlib import Path

import numpy as np
from scripts.slice_select import select_real_slice_2d
from skimage import morphology
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import find_boundaries
from skimage.transform import resize as sk_resize
from tifffile import imread, imwrite


def _roi_bbox_from_real(real_img: np.ndarray, pad: int = 4) -> tuple[int, int, int, int]:
    x = real_img.astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    thr = float(np.percentile(x, 88))
    mask = x > thr
    h, w = x.shape
    # Use the total bounding box of ALL bright pixels (not just the largest component),
    # so sparse bright-cell images still capture the full tissue extent.
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (0, 0, int(w), int(h))
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    return (x0, y0, int(x1 - x0), int(y1 - y0))


def _center_pad_or_crop(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    th, tw = target_shape
    sh, sw = img.shape
    out = np.zeros((th, tw), dtype=np.float32)
    min_h = min(th, sh)
    min_w = min(tw, sw)
    oy = (th - min_h) // 2
    ox = (tw - min_w) // 2
    iy = (sh - min_h) // 2
    ix = (sw - min_w) // 2
    out[oy : oy + min_h, ox : ox + min_w] = img[iy : iy + min_h, ix : ix + min_w]
    return out


def _prep_real_features(
    real_img: np.ndarray, target_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    if real_img.ndim == 3:
        real_img = real_img[..., 0]
    x = _center_pad_or_crop(real_img.astype(np.float32), target_shape)
    edges = sobel(x)

    # Build a robust real tissue mask — compute percentile over non-zero pixels only
    # to avoid being dominated by zero-padding when tissue is small relative to canvas
    nonzero = x[x > 0]
    if nonzero.size > 50:
        p80 = float(np.percentile(nonzero, 80))
        p65 = float(np.percentile(nonzero, 65))
    else:
        p80 = float(np.percentile(x, 80))
        p65 = float(np.percentile(x, 65))
    mask = x > p80
    if float(np.mean(mask)) < 0.01:
        mask = x > p65
    return edges.astype(np.float32), mask.astype(bool)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return float(np.mean(ys)), float(np.mean(xs))


def _shift_with_zero(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = img.shape
    out = np.zeros_like(img)

    sy0 = max(0, -dy)
    sy1 = min(h, h - dy)  # exclusive
    sx0 = max(0, -dx)
    sx1 = min(w, w - dx)  # exclusive
    if sy1 <= sy0 or sx1 <= sx0:
        return out

    ty0 = sy0 + dy
    ty1 = sy1 + dy
    tx0 = sx0 + dx
    tx1 = sx1 + dx
    out[ty0:ty1, tx0:tx1] = img[sy0:sy1, sx0:sx1]
    return out


def _safe_dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    den = float(np.sum(aa) + np.sum(bb)) + 1e-6
    inter = float(np.sum(aa & bb))
    return 2.0 * inter / den


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    den = float(np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-6
    return float(np.dot(af, bf) / den)


def _normalized_shape_score(
    real_roi: np.ndarray, label_slice: np.ndarray, target_size: int = 128
) -> float:
    """Score atlas slice by normalizing both real tissue and atlas to the same canvas.

    Both images are cropped to their non-zero bounding boxes then resized to
    target_size × target_size before edge comparison.  This normalizes for the
    resolution mismatch (e.g. 5µm tissue vs 25µm atlas) and is much more
    discriminative than atlas-space SSIM when tissue coverage is tiny.
    """
    rf = real_roi.astype(np.float32)
    if rf.ndim == 3:
        rf = rf[..., 0]

    # Binary tissue mask: use low percentile to capture the full brain outline.
    # For dense-tissue images (nearly uniform intensity, e.g. Gaussian-blurred atlas),
    # p15 of nonzero pixels equals the dominant value → only outliers pass.
    # Detect this case via low coefficient-of-variation and fall back to p5 of all pixels.
    nz = rf[rf > 0]
    if nz.size > 20:
        cv = float(np.std(nz) / max(float(np.mean(nz)), 1e-6))
        if cv < 0.15:
            # Dense tissue: threshold = 5th percentile of ALL pixels (captures full tissue blob)
            thr = float(np.percentile(rf, 5))
        else:
            thr = float(np.percentile(nz, 15))
    else:
        thr = 0.0
    real_bin = (rf > thr).astype(np.float32)
    ry, rx = np.nonzero(real_bin)
    if len(ry) < 50:
        return -1.0
    # Crop real to non-zero bounding box
    real_crop = real_bin[ry.min() : ry.max() + 1, rx.min() : rx.max() + 1]

    aH, aW = label_slice.shape
    a_mid = aW // 2
    T = int(target_size)

    def _score_candidate(atlas_candidate):
        """Score a candidate atlas mask (binary float32) against the real tissue crop."""
        ay, ax = np.nonzero(atlas_candidate > 0)
        if len(ay) < 30:
            return -1.0
        # Crop atlas candidate to non-zero bounding box before resizing
        at_crop = atlas_candidate[ay.min() : ay.max() + 1, ax.min() : ax.max() + 1].astype(
            np.float32
        )
        re_n = sk_resize(real_crop, (T, T), order=1, preserve_range=True)
        at_n = sk_resize(at_crop, (T, T), order=0, preserve_range=True)
        re_n /= max(float(re_n.max()), 1e-6)
        at_n = (at_n > 0.5).astype(np.float32)
        ae = sobel(re_n)
        be = sobel(at_n)
        dr = float(max(np.ptp(ae), np.ptp(be), 1e-6))
        try:
            return float(ssim(ae, be, data_range=dr))
        except Exception:
            return float(np.corrcoef(ae.ravel(), be.ravel())[0, 1])

    # Use region boundaries (inner edges) for more discriminative atlas fingerprint:
    # atlas with many sub-regions has denser edge pattern → better z discrimination.
    label_slice.astype(np.float32)
    atlas_inner = find_boundaries(
        label_slice.astype(np.int32), mode="inner", connectivity=2
    ).astype(np.float32)
    atlas_outer = (label_slice > 0).astype(np.float32)
    atlas_feat = np.clip(atlas_outer + atlas_inner * 2.0, 0.0, 1.0)

    # Try left half, right half, and full atlas (bilateral) — use best score
    left_s = _score_candidate(atlas_feat[:, :a_mid])
    right_s = _score_candidate(atlas_feat[:, a_mid:])
    full_s = _score_candidate(atlas_feat)
    valid = [s for s in [left_s, right_s, full_s] if s > -0.5]
    return float(max(valid)) if valid else -1.0


def _label_edge_score(
    real_edges: np.ndarray, real_mask: np.ndarray, label_slice: np.ndarray
) -> float:
    # Skip nearly-empty atlas slices
    coverage = float(np.mean(label_slice > 0))
    if coverage < 0.015:
        return -1.0

    atlas_mask = (label_slice > 0).astype(np.float32)
    outer = sobel(atlas_mask)
    inner = find_boundaries(label_slice.astype(np.int32), mode="inner", connectivity=2).astype(
        np.float32
    )
    atlas_feat = 0.55 * outer + 1.00 * inner

    # Coarse centroid alignment to reduce translation sensitivity during scoring
    c_real = _centroid(real_mask)
    c_atlas = _centroid(atlas_mask > 0)
    if c_real is not None and c_atlas is not None:
        dy = int(round(c_real[0] - c_atlas[0]))
        dx = int(round(c_real[1] - c_atlas[1]))
        atlas_feat = _shift_with_zero(atlas_feat, dy=dy, dx=dx)
        atlas_mask = _shift_with_zero(atlas_mask, dy=dy, dx=dx)
        inner = _shift_with_zero(inner, dy=dy, dx=dx)

    real_signal = float(np.mean(real_edges > (np.percentile(real_edges, 70))))
    if real_signal < 0.004:
        return 0.6 * coverage + 0.4 * _safe_dice(real_mask, atlas_mask > 0)

    # When tissue covers < 15% of atlas canvas (e.g. high-mag real image at 5µm/px),
    # full-canvas SSIM is dominated by matched background zeros → score every z similarly.
    # Instead, crop comparison to the tissue bounding box for a meaningful local score.
    real_cov = float(np.mean(real_mask))
    if real_cov < 0.15:
        ys, xs = np.nonzero(real_mask)
        if len(ys) > 10:
            pad = 4
            yp = max(0, int(ys.min()) - pad)
            yq = min(real_mask.shape[0], int(ys.max()) + pad)
            xp = max(0, int(xs.min()) - pad)
            xq = min(real_mask.shape[1], int(xs.max()) + pad)
            re_crop = real_edges[yp:yq, xp:xq]
            at_crop = atlas_feat[yp:yq, xp:xq]
            rm_crop = real_mask[yp:yq, xp:xq]
            at_m_crop = (atlas_mask > 0.0)[yp:yq, xp:xq]
            if re_crop.size > 25:
                dr_local = float(max(np.ptp(re_crop), np.ptp(at_crop), 1e-6))
                try:
                    ssim_local = float(ssim(re_crop, at_crop, data_range=dr_local))
                except Exception:
                    ssim_local = float(np.corrcoef(re_crop.ravel(), at_crop.ravel())[0, 1])
                dice_local = _safe_dice(rm_crop, at_m_crop)
                inner_corr_local = _norm_corr(re_crop, at_crop)
                return float(
                    0.55 * ssim_local
                    + 0.30 * dice_local
                    + 0.15 * inner_corr_local
                    + 0.02 * coverage
                )

    dr = float(max(np.ptp(real_edges), np.ptp(atlas_feat), 1e-6))
    ssim_score = float(ssim(real_edges, atlas_feat, data_range=dr))
    shape_dice = _safe_dice(real_mask, atlas_mask > 0)
    inner_corr = _norm_corr(real_edges, inner)
    return float(0.70 * ssim_score + 0.20 * shape_dice + 0.10 * inner_corr + 0.03 * coverage)


def _candidate_zs_from_coarse(
    coarse_scores: list[tuple[int, float]],
    z_dim: int,
    base_step: int,
    max_seeds: int = 10,
) -> list[int]:
    if not coarse_scores:
        return [z_dim // 2]
    sorted_scores = sorted(coarse_scores, key=lambda x: x[1], reverse=True)
    seeds = [int(z) for z, _ in sorted_scores[: max(1, int(max_seeds))]]
    radius = max(1, int(base_step))
    cand = set()
    for z in seeds:
        for dz in range(-radius, radius + 1):
            zz = z + dz
            if 0 <= zz < z_dim:
                cand.add(int(zz))
    return sorted(cand)


def refine_atlas_z_by_size(
    real_img: np.ndarray,
    vol: np.ndarray,
    z_estimate: int,
    search_range: int = 80,
    pixel_size_um: float = 5.0,
    hemisphere: str = "right",
) -> tuple[int, float]:
    """Refine atlas AP index using size-aware shape matching.

    For each AP candidate, computes:
      1. Size ratio = min(brain_area, atlas_area) / max(brain_area, atlas_area)
      2. Normalized shape score (SSIM of outlines at 96×96)
      3. Combined = size_ratio^1.5 × shape_score

    Uses coarse size-only pass (step=2) to shortlist top-8 candidates,
    then refines with shape scoring. Total overhead ~0.5s per slice.
    Returns (best_z, best_combined_score).
    """
    from skimage.transform import rescale

    real_f = real_img.astype(np.float32)
    if real_f.ndim == 3:
        real_f = real_f[..., 0]

    # Scale brain to atlas space (5µm → 25µm)
    scale = pixel_size_um / 25.0
    real_scaled = rescale(real_f, scale, order=1, preserve_range=True, anti_aliasing=True)
    nz = real_scaled[real_scaled > 0]
    thr = float(np.percentile(nz, 15)) if nz.size > 20 else 0.0
    brain_area = float(np.sum(real_scaled > thr))

    z_dim = vol.shape[0]
    aW = vol.shape[2]
    a_mid = aW // 2
    z_start = max(0, z_estimate - search_range)
    z_end = min(z_dim, z_estimate + search_range + 1)

    # --- Coarse pass: size ratio only (very fast) ---
    size_scores: list[tuple[int, float]] = []
    for z in range(z_start, z_end, 2):
        if hemisphere == "right":
            atlas_area = float(np.sum(vol[z, :, a_mid:] > 0))
        elif hemisphere == "left":
            atlas_area = float(np.sum(vol[z, :, :a_mid] > 0))
        else:
            atlas_area = float(np.sum(vol[z] > 0))
        if atlas_area < 10:
            size_scores.append((z, 0.0))
            continue
        ratio = min(brain_area, atlas_area) / max(brain_area, atlas_area)
        size_scores.append((z, ratio))

    if not size_scores:
        return int(z_estimate), 0.0

    # --- Refined pass: top-8 by size ratio → add shape score ---
    top_candidates = sorted(size_scores, key=lambda x: x[1], reverse=True)[:8]

    best_z = int(z_estimate)
    best_score = -1.0
    for z, size_r in top_candidates:
        atlas_slice = vol[int(z), :, :]
        shape_s = _normalized_shape_score(real_img, atlas_slice, target_size=96)
        combined = float((size_r**1.5) * max(shape_s, 0.0))
        if combined > best_score:
            best_score = combined
            best_z = int(z)

    return best_z, float(best_score)


def autopick_best_z(
    real_path: Path,
    annotation_nii: Path,
    out_label_tif: Path,
    z_step: int = 1,
    pixel_size_um: float = 0.65,
    slicing_plane: str = "coronal",
    roi_mode: str = "auto",
    real_z_index: int | None = None,
    progress_cb=None,
    z_range: list | None = None,
) -> dict:
    import nibabel as nib
    from skimage.transform import rescale

    real_raw = imread(str(real_path))
    real, real_slice_meta = select_real_slice_2d(
        real_raw,
        z_index=real_z_index,
        source_path=real_path,
    )
    nii = nib.load(str(annotation_nii))
    vol = np.asarray(nii.get_fdata(), dtype=np.int32)

    if progress_cb:
        progress_cb(2, 100, "Scanning atlas slices (coarse)...")

    plane = str(slicing_plane or "coronal").lower()

    def _get_slice(v: np.ndarray, z: int, p: str) -> np.ndarray:
        if p == "coronal":
            return v[z, :, :]
        if p == "sagittal":
            return v[:, :, z]
        if p in ("horizontal", "axial"):
            return v[:, z, :]
        raise ValueError(f"unsupported slicing_plane: {p}")

    if plane == "coronal":
        z_dim = vol.shape[0]
    elif plane == "sagittal":
        z_dim = vol.shape[2]
    elif plane in ("horizontal", "axial"):
        z_dim = vol.shape[1]
    else:
        raise ValueError(f"unsupported slicing_plane: {plane}")

    sample = _get_slice(vol, 0, plane)
    target_shape = sample.shape

    roi_bbox = (0, 0, int(real.shape[1]), int(real.shape[0]))
    roi = real
    if str(roi_mode or "off").lower() in ("auto", "on", "true"):
        x0, y0, rw, rh = _roi_bbox_from_real(real)
        roi_bbox = (int(x0), int(y0), int(rw), int(rh))
        roi = real[y0 : y0 + rh, x0 : x0 + rw]

    # Pre-scale real ROI to atlas space
    scale = pixel_size_um / 25.0
    real_f = roi.astype(np.float32)
    real_scaled = rescale(real_f, scale, order=1, preserve_range=True, anti_aliasing=True)
    real_edges, real_mask = _prep_real_features(real_scaled, target_shape)

    best_z, best_score = 0, -1.0
    coarse_scores: list[tuple[int, float]] = []
    step = max(1, int(z_step))
    tissue_coverage = float(real_mask.mean())

    # Apply z_range constraint if specified (prevents wrong selections outside biologically plausible range)
    z_start, z_end = 0, z_dim
    if z_range and len(z_range) >= 2:
        z_start = max(0, int(z_range[0]))
        z_end = min(z_dim, int(z_range[1]))

    # When tissue occupies < 15% of the atlas canvas (e.g. high-mag tissue in low-res atlas),
    # use normalized shape scoring: resize both tissue and atlas to the same canvas for comparison.
    use_normalized = tissue_coverage < 0.15

    for z in range(z_start, z_end, step):
        atlas_slice = _get_slice(vol, z, plane)
        if use_normalized:
            s = _normalized_shape_score(roi, atlas_slice)
        else:
            s = _label_edge_score(real_edges, real_mask, atlas_slice)
        coarse_scores.append((int(z), float(s)))
        if s > best_score:
            best_score = s
            best_z = z
        if progress_cb and z % (step * 10) == 0:
            progress_cb(
                5 + int((z - z_start) * 55 / max(z_end - z_start, 1)),
                100,
                f"Coarse scan: slice {z}/{z_end}",
            )

    if progress_cb:
        progress_cb(62, 100, "Refined scoring top candidates...")

    # Stage-2 refined z evaluation: run fast tissue-guided warp scoring on top candidates.
    # SKIP when tissue coverage is too low (< 15% of atlas canvas) — the refined warp quality
    # score becomes unreliable with tiny tissue and tends to select wrong z positions.
    refined_best_z = int(best_z)
    refined_best_score = float(best_score)
    refined_scores: list[tuple[int, float]] = []
    if tissue_coverage >= 0.15:
        try:
            from scripts.overlay_render import (
                _alignment_quality,
                _norm_u8_robust,
                _tissue_guided_warp,
            )

            real_u8 = _norm_u8_robust(real)
            candidates = _candidate_zs_from_coarse(
                coarse_scores,
                z_dim=z_end,
                base_step=step,
                max_seeds=10,
            )
            candidates = [c for c in candidates if z_start <= c < z_end]
            for idx, z in enumerate(candidates):
                atlas_slice = _get_slice(vol, int(z), plane).astype(np.int32)
                warped, meta = _tissue_guided_warp(
                    real,
                    atlas_slice,
                    atlas_res_um=25.0,
                    real_res_um=float(pixel_size_um),
                    fit_mode="contain",
                    enable_nonlinear=False,
                    opt_maxiter=70,
                )
                tissue_mask = meta.get("tissue_mask")
                if tissue_mask is not None:
                    clip = morphology.dilation(
                        tissue_mask.astype(np.uint8), morphology.disk(3)
                    ).astype(bool)
                    warped = np.where(clip, warped, 0).astype(np.int32)
                q = float(_alignment_quality(real_u8, warped, tissue_mask=tissue_mask))
                refined_scores.append((int(z), q))
                if q > refined_best_score:
                    refined_best_score = q
                    refined_best_z = int(z)
                if progress_cb:
                    progress_cb(
                        62 + int(idx * 33 / max(len(candidates), 1)),
                        100,
                        f"Refined scoring: candidate {idx + 1}/{len(candidates)}",
                    )
        except Exception:
            refined_scores = []

    if refined_scores:
        refined_best_z, refined_best_score = max(refined_scores, key=lambda x: x[1])
    best_z = int(refined_best_z)
    best_score = float(refined_best_score)

    best_slice = _get_slice(vol, best_z, plane).astype(np.int32)
    out_label_tif.parent.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(97, 100, "Saving result...")
    imwrite(str(out_label_tif), best_slice)

    return {
        "best_z": int(best_z),
        "best_score": float(best_score),
        "best_score_type": "refined_warp_quality" if refined_scores else "coarse_edge_score",
        "label_slice_tif": str(out_label_tif),
        "shape": [int(x) for x in vol.shape],
        "slicing_plane": plane,
        "slice_shape": [int(x) for x in best_slice.shape],
        "roi_mode": str(roi_mode),
        "roi_bbox": [int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])],
        "real_slice": real_slice_meta,
        "tissue_coverage": float(tissue_coverage),
        "coarse_top": [
            [int(z), float(s)]
            for z, s in sorted(coarse_scores, key=lambda x: x[1], reverse=True)[:8]
        ],
        "refined_top": [
            [int(z), float(s)]
            for z, s in sorted(refined_scores, key=lambda x: x[1], reverse=True)[:8]
        ],
    }
