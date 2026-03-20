"""
make_demo_panel.py  —  Generate a publication-quality demo panel from registered overlays.

Usage:
    cd project
    python scripts/make_demo_panel.py [--out outputs/demo_panel.png] [--n 9]
"""

from __future__ import annotations

import argparse
import colorsys
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

# ── annotated single-slice helper ────────────────────────────────────────────


_GENERIC_LABEL_ACRONYMS = {"?", "ROOT", "GREY", "BRAIN"}
_GENERIC_LABEL_NAMES = {
    "root",
    "brain",
    "basic cell groups and regions",
}


@lru_cache(maxsize=1)
def _load_structure_tree_lookup() -> dict[int, dict[str, str]]:
    candidates = [
        Path(__file__).resolve().parent.parent / "configs" / "allen_structure_tree.json",
        Path(__file__).resolve().parent / "allen_structure_tree.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            out: dict[int, dict[str, str]] = {}
            for key, info in data.items():
                try:
                    rid = int(key)
                except Exception:
                    continue
                if not isinstance(info, dict):
                    continue
                out[rid] = {
                    "acronym": str(info.get("acronym", "") or "").strip(),
                    "name": str(info.get("name", "") or "").strip(),
                    "color": str(info.get("color", "") or "").strip().lstrip("#"),
                }
            return out
    return {}


@lru_cache(maxsize=8)
def _load_structure_csv_lookup(structure_csv: str) -> dict[int, dict[str, str]]:
    import pandas as pd

    path = Path(structure_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(str(path))
    out: dict[int, dict[str, str]] = {}
    for _, row in df.iterrows():
        try:
            rid = int(row["id"])
        except Exception:
            continue
        out[rid] = {
            "acronym": str(row.get("acronym", "") or "").strip(),
            "name": str(row.get("name", "") or "").strip(),
            "color": str(row.get("color_hex_triplet", "") or "").strip().lstrip("#"),
        }
    return out


def _combined_structure_lookup(structure_csv: Path | None = None) -> dict[int, dict[str, str]]:
    lookup = dict(_load_structure_tree_lookup())
    if structure_csv is not None:
        csv_lookup = _load_structure_csv_lookup(str(Path(structure_csv).resolve()))
        for rid, info in csv_lookup.items():
            existing = lookup.get(rid, {})
            lookup[rid] = {
                "acronym": info.get("acronym") or existing.get("acronym", ""),
                "name": info.get("name") or existing.get("name", ""),
                "color": info.get("color") or existing.get("color", ""),
            }
    return lookup


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int] | None:
    hx = str(hex_color or "").strip().lstrip("#")
    if len(hx) != 6:
        return None
    try:
        return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
    except Exception:
        return None


def _fallback_color(region_id: int) -> tuple[int, int, int]:
    rng = (int(region_id) * 2654435761) & 0xFFFFFF
    return ((rng >> 16) & 0xFF, (rng >> 8) & 0xFF, rng & 0xFF)


def _family_color(
    region_id: int, structure_lookup: dict[int, dict[str, str]] | None = None
) -> tuple[int, int, int]:
    info = (structure_lookup or {}).get(int(region_id)) or _load_structure_tree_lookup().get(
        int(region_id), {}
    )
    base = _hex_to_rgb(info.get("color", "") if info else "") or _fallback_color(int(region_id))
    r, g, b = [v / 255.0 for v in base]
    h, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    jitter = (((int(region_id) * 37) % 9) - 4) * 0.028
    lightness = min(0.78, max(0.28, lightness + jitter))
    saturation = min(0.95, max(0.35, saturation * 1.08))
    rr, gg, bb = colorsys.hls_to_rgb(h, lightness, saturation)
    return int(rr * 255), int(gg * 255), int(bb * 255)


def _meaningful_region_label(
    region_id: int, structure_lookup: dict[int, dict[str, str]]
) -> tuple[str, str] | None:
    info = structure_lookup.get(int(region_id), {})
    acro = str(info.get("acronym", "") or "").strip()
    name = str(info.get("name", "") or "").strip()

    if not acro and not name:
        return None
    if re.fullmatch(r"\d+", acro or ""):
        return None
    if re.fullmatch(r"\d+", name or ""):
        return None
    if acro.upper() in _GENERIC_LABEL_ACRONYMS:
        return None
    if name.strip().lower() in _GENERIC_LABEL_NAMES:
        return None
    return acro or name, name or acro


def _representative_point(mask: np.ndarray) -> tuple[int, int]:
    from scipy.ndimage import distance_transform_edt

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0, 0
    dist = distance_transform_edt(mask.astype(np.uint8))
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(cy), int(cx)


def _select_region_annotations(
    label_arr: np.ndarray,
    *,
    structure_lookup: dict[int, dict[str, str]],
    top_n: int,
    min_pixels: int = 80,
) -> list[dict[str, object]]:
    ids, counts = np.unique(label_arr[label_arr > 0], return_counts=True)
    ranking = sorted(
        (
            (int(uid), int(count))
            for uid, count in zip(ids, counts, strict=True)
            if int(count) >= int(min_pixels)
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    selected: list[dict[str, object]] = []
    used_acronyms: set[str] = set()
    for uid, count in ranking:
        label = _meaningful_region_label(uid, structure_lookup)
        if label is None:
            continue
        acro, name = label
        acro_key = acro.upper()
        if acro_key in used_acronyms:
            continue
        mask = label_arr == uid
        cy, cx = _representative_point(mask)
        selected.append(
            {
                "region_id": uid,
                "count": count,
                "acro": acro,
                "name": name,
                "color": _family_color(uid, structure_lookup),
                "cy": cy,
                "cx": cx,
            }
        )
        used_acronyms.add(acro_key)
        if len(selected) >= int(top_n):
            break
    return selected


def make_annotated_slice(
    overlay_png: Path,
    label_tif: Path,
    raw_tif: Path | None,
    structure_csv: Path,
    out_path: Path,
    top_n: int = 10,
) -> Path:
    """Generate a labeled comparison image: raw | vibrant-colored atlas with region names.

    Regions are annotated with their Allen acronym + full name at centroid positions.
    """
    # --- load overlay and label ---
    ov_arr = np.array(Image.open(str(overlay_png)).convert("RGB"))
    lbl_arr = tifffile.imread(str(label_tif)).astype(np.int32)

    structure_lookup = _combined_structure_lookup(structure_csv)

    tissue_mask, tissue_alpha = (
        _tissue_support_from_raw(Path(str(raw_tif)))
        if raw_tif
        else (
            None,
            None,
        )
    )

    # --- vibrant recolor + smooth tissue clipping ---
    vib = _vibrant_recolor(ov_arr, lbl_arr, tissue_mask=tissue_mask)
    if tissue_alpha is not None:
        vib = _apply_tissue_alpha(vib, tissue_alpha)

    # --- crop to brain area ---
    r0, r1, c0, c1 = _crop_bounds(
        ov_arr.shape[:2],
        pad=20,
        mask=tissue_mask,
        image=ov_arr,
    )
    vib_crop = vib[r0 : r1 + 1, c0 : c1 + 1]
    lbl_crop = lbl_arr[r0 : r1 + 1, c0 : c1 + 1]

    # --- load raw image if available ---
    if raw_tif and Path(str(raw_tif)).exists():
        try:
            raw = tifffile.imread(str(raw_tif)).astype(np.float32)
            if raw.ndim == 3:
                raw = raw[..., 0]
            p1, p99 = np.percentile(raw[raw > 0], [1, 99]) if raw.max() > 0 else (0, 1)
            raw_norm = np.clip((raw - p1) / (p99 - p1 + 1e-6) * 255, 0, 255).astype(np.uint8)
            raw_rgb = np.stack([raw_norm] * 3, axis=-1)
            raw_rgb = raw_rgb[r0 : r1 + 1, c0 : c1 + 1]
        except Exception:
            raw_rgb = None
    else:
        raw_rgb = None

    H, W = vib_crop.shape[:2]
    TARGET_H = 560  # normalize height
    scale = TARGET_H / H

    def _resize(arr):
        img = Image.fromarray(arr)
        return np.array(img.resize((int(W * scale), TARGET_H), Image.LANCZOS))

    vib_r = _resize(vib_crop)
    lbl_r = np.array(
        Image.fromarray(lbl_crop.astype(np.float32)).resize(
            (int(W * scale), TARGET_H), Image.NEAREST
        )
    ).astype(np.int32)

    # --- draw region labels on the vibrant overlay ---
    result_img = Image.fromarray(vib_r)
    draw = ImageDraw.Draw(result_img)

    try:
        font_sm = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
        ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 14)
    except Exception:
        font_sm = ImageFont.load_default()

    region_entries = _select_region_annotations(
        lbl_r,
        structure_lookup=structure_lookup,
        top_n=top_n,
        min_pixels=90,
    )

    legend_entries = []  # (color, acro, full_name) for legend panel
    for entry in region_entries:
        cy, cx = int(entry["cy"]), int(entry["cx"])
        acro = str(entry["acro"])
        name = str(entry["name"])
        color = tuple(int(v) for v in entry["color"])
        legend_entries.append((color, acro, name))
        # Draw small circle at centroid
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(255, 255, 255))
        # Draw acronym label with drop-shadow
        tx, ty = cx + 7, cy - 8
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((tx + dx, ty + dy), acro, fill=(0, 0, 0), font=font_sm)
        draw.text((tx, ty), acro, fill=color, font=font_sm)

    annotated = np.array(result_img)

    # --- compose body ---
    if raw_rgb is not None:
        raw_r = _resize(raw_rgb)
        sep = np.full((TARGET_H, 6, 3), 30, dtype=np.uint8)
        body = np.concatenate([raw_r, sep, annotated], axis=1)
    else:
        body = annotated

    TW = body.shape[1]

    # --- legend panel below body ---
    LEGEND_ROW_H = 22
    LEGEND_PAD = 10
    legend_h = LEGEND_PAD * 2 + 20 + len(legend_entries) * LEGEND_ROW_H
    legend_panel = np.full((legend_h, TW, 3), 28, dtype=np.uint8)
    leg_img = Image.fromarray(legend_panel)
    leg_draw = ImageDraw.Draw(leg_img)
    try:
        font_leg = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
        font_leg_b = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 13)
    except Exception:
        font_leg = font_leg_b = ImageFont.load_default()
    leg_draw.text(
        (12, LEGEND_PAD - 1),
        "Dots mark representative anchor points for the labeled regions below.",
        fill=(150, 156, 170),
        font=font_leg,
    )
    for i, (color, acro, name) in enumerate(legend_entries):
        y = LEGEND_PAD + 18 + i * LEGEND_ROW_H + 2
        # color swatch
        leg_draw.rectangle([12, y + 1, 28, y + 15], fill=color)
        # acronym (bold) + full name
        leg_draw.text((36, y), acro, fill=color, font=font_leg_b)
        acro_w = (
            leg_draw.textlength(acro, font=font_leg_b)
            if hasattr(leg_draw, "textlength")
            else len(acro) * 8
        )
        disp_name = name if len(name) <= 40 else name[:38] + ".."
        leg_draw.text((36 + acro_w + 8, y), f"— {disp_name}", fill=(180, 180, 180), font=font_leg)
    legend_panel = np.array(leg_img)

    # --- header ---
    HDR_H = 50
    hdr = np.full((HDR_H, TW, 3), 18, dtype=np.uint8)
    full = Image.fromarray(np.concatenate([hdr, body, legend_panel], axis=0))
    draw2 = ImageDraw.Draw(full)
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
        ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()

    if raw_rgb is not None:
        draw2.text((12, 14), "Raw Lightsheet", fill=(200, 200, 200), font=font_title)
        draw2.text(
            (raw_r.shape[1] + 14, 14),
            "Brainfast — Allen CCFv3 Atlas Registration",
            fill=(200, 200, 200),
            font=font_title,
        )
    else:
        draw2.text(
            (12, 14),
            "Brainfast — Allen CCFv3 Atlas Registration",
            fill=(200, 200, 200),
            font=font_title,
        )

    full.save(str(out_path), quality=95)
    return out_path


# ── helpers ──────────────────────────────────────────────────────────────────


def _crop_bounds(
    shape: tuple[int, int],
    pad: int = 20,
    mask: np.ndarray | None = None,
    image: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    if mask is not None:
        if mask.shape != shape:
            mask = (
                np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (shape[1], shape[0]), Image.Resampling.NEAREST
                    )
                )
                > 127
            )
        work_mask = mask.astype(bool)
    elif image is not None:
        gray = image.mean(axis=2) if image.ndim == 3 else image
        work_mask = gray > 8
    else:
        work_mask = np.zeros(shape, dtype=bool)

    rows = np.any(work_mask, axis=1)
    cols = np.any(work_mask, axis=0)
    if not rows.any():
        return 0, shape[0] - 1, 0, shape[1] - 1
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    H, W = shape
    r0 = max(0, r0 - pad)
    r1 = min(H - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(W - 1, c1 + pad)
    return int(r0), int(r1), int(c0), int(c1)


def _crop_to_brain(
    img_arr: np.ndarray, pad: int = 20, mask: np.ndarray | None = None
) -> np.ndarray:
    """Crop away dark background, keeping only the brain region + padding."""
    r0, r1, c0, c1 = _crop_bounds(img_arr.shape[:2], pad=pad, mask=mask, image=img_arr)
    return img_arr[r0 : r1 + 1, c0 : c1 + 1]


def _load_raw_slice(raw_tif_path: Path) -> np.ndarray:
    raw = tifffile.imread(str(raw_tif_path)).astype(np.float32)
    if raw.ndim == 3:
        raw = raw[0] if raw.shape[0] < raw.shape[1] else raw[..., 0]
    return raw.astype(np.float32, copy=False)


def _resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    return (
        np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (shape[1], shape[0]),
                Image.Resampling.NEAREST,
            )
        )
        > 127
    )


def _resize_alpha(alpha: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if alpha.shape == shape:
        return np.clip(alpha.astype(np.float32), 0.0, 1.0)
    img = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8))
    arr = np.array(img.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)).astype(np.float32)
    return np.clip(arr / 255.0, 0.0, 1.0)


def _vibrant_recolor(
    img_arr: np.ndarray, label_arr: np.ndarray, tissue_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Re-colorize using a more vibrant, distinct palette while preserving brain texture.
    img_arr  : (H, W, 3) uint8 — existing fill-mode overlay
    label_arr: (H, W)    int32 — registered label (region IDs)
    """
    structure_lookup = _combined_structure_lookup()

    # Convert original image to float grayscale (brain texture)
    gray = img_arr.mean(axis=2).astype(np.float32) / 255.0
    # Gamma < 1 brightens dim cleared-tissue images so texture is visible through color.
    gray_bright = np.clip(gray**0.55, 0.0, 1.0)
    out = np.zeros((*img_arr.shape[:2], 3), dtype=np.uint8)

    ids = np.unique(label_arr)
    ids = ids[ids > 0]
    for uid in ids:
        mask = label_arr == uid
        color = _family_color(int(uid), structure_lookup)
        texture = gray_bright[mask]  # brightened tissue texture
        for ch, cv in enumerate(color):
            # Use the atlas family color, then keep the lightsheet texture visible under the tint.
            out[mask, ch] = np.clip(texture * 0.57 * 255 + cv * 0.43, 0, 255).astype(np.uint8)

    # Background: keep dark
    bg = label_arr == 0
    out[bg] = (img_arr[bg].astype(np.float32) * 0.3).astype(np.uint8)
    try:
        from scipy.ndimage import gaussian_filter
        from skimage import morphology
        from skimage.segmentation import find_boundaries

        support = tissue_mask if tissue_mask is not None else (label_arr > 0)
        if support.shape != label_arr.shape:
            support = (
                np.array(
                    Image.fromarray(support.astype(np.uint8) * 255).resize(
                        (label_arr.shape[1], label_arr.shape[0]),
                        Image.Resampling.NEAREST,
                    )
                )
                > 127
            )
        edge_band = morphology.dilation(
            find_boundaries(support.astype(bool), mode="outer", connectivity=2).astype(np.uint8),
            morphology.disk(3),
        ).astype(bool)
        if np.any(edge_band):
            blurred = gaussian_filter(out.astype(np.float32), sigma=(1.1, 1.1, 0.0))
            out = out.copy()
            out[edge_band] = np.clip(blurred[edge_band], 0, 255).astype(np.uint8)
    except Exception:
        pass
    return out


def _add_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple,
    font,
    text_color=(255, 255, 255),
    shadow_color=(0, 0, 0),
):
    x, y = xy
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((x + dx, y + dy), text, fill=shadow_color, font=font)
    draw.text(xy, text, fill=text_color, font=font)


# ── main ─────────────────────────────────────────────────────────────────────


def _tissue_support_from_raw(raw_tif_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a smooth tissue support from the brightened real slice.

    Returns a hard mask for cropping plus a soft alpha matte for display clipping.
    """
    try:
        from scipy.ndimage import binary_fill_holes as _fill_holes
        from scipy.ndimage import gaussian_filter as _gaussian_filter
        from skimage import filters as _filters
        from skimage import measure as _meas
        from skimage import morphology as _morph

        raw = _load_raw_slice(raw_tif_path)
        p1, p99 = np.percentile(raw, [1.0, 99.8])
        if p99 <= p1:
            p99 = p1 + 1.0
        norm = np.clip((raw - p1) / (p99 - p1), 0.0, 1.0)
        bright = np.clip(norm**0.58, 0.0, 1.0)

        sigma = max(1.4, min(raw.shape) * 0.0038)
        bright_smooth = _gaussian_filter(bright, sigma=sigma)

        b = max(16, min(80, min(raw.shape) // 10))
        corners = np.concatenate(
            [
                bright_smooth[:b, :b].ravel(),
                bright_smooth[:b, -b:].ravel(),
                bright_smooth[-b:, :b].ravel(),
                bright_smooth[-b:, -b:].ravel(),
            ]
        )
        corner_thr = float(np.mean(corners) + 3.2 * np.std(corners))
        otsu_thr = float(_filters.threshold_otsu(bright_smooth))
        corner_thr = min(corner_thr, otsu_thr + 0.18)
        hi_thr = float(np.percentile(bright_smooth, 92.0))
        thr = min(max(corner_thr, otsu_thr * 0.92), hi_thr)
        mask = bright_smooth > thr

        close_r = max(10, int(round(min(raw.shape) * 0.010)))
        open_r = max(2, int(round(close_r * 0.35)))
        mask = _morph.closing(mask, _morph.disk(close_r))
        mask = _fill_holes(mask).astype(bool)
        mask = _morph.opening(mask, _morph.disk(open_r))
        labeled = _meas.label(mask)
        if labeled.max() > 0:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            mask = labeled == counts.argmax()
        _obj_limit = max(256, raw.size // 400)
        try:
            mask = _morph.remove_small_objects(mask.astype(bool), max_size=_obj_limit)
        except TypeError:
            mask = _morph.remove_small_objects(mask.astype(bool), min_size=_obj_limit)
        mask = _fill_holes(mask).astype(bool)

        alpha = _gaussian_filter(mask.astype(np.float32), sigma=max(1.2, close_r * 0.18))
        alpha = np.clip((alpha - 0.08) / 0.84, 0.0, 1.0).astype(np.float32)
        return mask.astype(bool), alpha
    except Exception:
        return None, None


def _tissue_mask_from_raw(raw_tif_path: Path) -> np.ndarray | None:
    mask, _ = _tissue_support_from_raw(raw_tif_path)
    return mask


def _clip_to_tissue(
    img_arr: np.ndarray, tissue_mask: np.ndarray, feather_px: int = 10
) -> np.ndarray:
    """Clip a colourised overlay to the actual tissue boundary with a smooth feathered edge."""
    from scipy.ndimage import distance_transform_edt as _dt_edt
    from skimage import morphology as _morph

    if tissue_mask.shape != img_arr.shape[:2]:
        # Resize mask to match image
        from PIL import Image as _Im

        tm_img = _Im.fromarray(tissue_mask.astype(np.uint8) * 255)
        tm_img = tm_img.resize((img_arr.shape[1], img_arr.shape[0]), _Im.BILINEAR)
        tissue_mask = np.array(tm_img) > 96

    tissue_mask = _morph.closing(
        tissue_mask.astype(np.uint8),
        _morph.disk(max(2, int(round(feather_px * 0.35)))),
    ).astype(bool)
    tissue_mask = _morph.opening(
        tissue_mask.astype(np.uint8),
        _morph.disk(max(1, int(round(feather_px * 0.18)))),
    ).astype(bool)
    _hole_limit = max(64, int((feather_px * 2) ** 2))
    try:
        tissue_mask = _morph.remove_small_holes(tissue_mask, max_size=_hole_limit)
    except TypeError:
        tissue_mask = _morph.remove_small_holes(tissue_mask, area_threshold=_hole_limit)
    dist_in = _dt_edt(tissue_mask.astype(np.uint8)).astype(np.float32)
    dist_out = _dt_edt((~tissue_mask).astype(np.uint8)).astype(np.float32)
    signed = dist_in - dist_out
    alpha = np.clip((signed + max(feather_px, 1)) / (2.0 * max(feather_px, 1)), 0.0, 1.0)[
        :, :, np.newaxis
    ]
    return np.clip(img_arr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _apply_tissue_alpha(img_arr: np.ndarray, tissue_alpha: np.ndarray) -> np.ndarray:
    alpha = _resize_alpha(tissue_alpha, img_arr.shape[:2])[:, :, np.newaxis]
    return np.clip(img_arr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def make_panel(
    reg_dir: Path,
    out_path: Path,
    n_slices: int = 9,
    cols: int = 3,
    thumb_size: int = 400,
    title: str = "Brainfast — Sample 35 Atlas Registration",
    slice_dir: Path | None = None,
) -> Path:
    overlay_files = sorted(reg_dir.glob("slice_*_overlay.png"))
    label_files = sorted(reg_dir.glob("slice_*_registered_label.tif"))
    # Pre-sort raw slice files for tissue-mask clipping
    raw_files: list[Path] = (
        sorted(slice_dir.glob("*.tif")) if slice_dir and slice_dir.exists() else []
    )

    if not overlay_files:
        raise FileNotFoundError(f"No overlay PNGs in {reg_dir}")

    # Sample evenly across available slices, displayed anterior→posterior
    # (reversed from scan order: scan goes posterior→anterior as z increases,
    # so the last slice in sorted order is the most anterior).
    total = len(overlay_files)
    indices_raw = np.linspace(0, total - 1, min(n_slices, total), dtype=int)
    indices = indices_raw[::-1]  # show anterior slices first (conventional neuroanatomy order)

    rows = (len(indices) + cols - 1) // cols
    GAP = 8
    TITLE_H = 50
    panel_w = cols * (thumb_size + GAP) + GAP
    panel_h = rows * (thumb_size + GAP) + GAP + TITLE_H
    panel = Image.new("RGB", (panel_w, panel_h), (18, 18, 24))

    # Font setup
    font_big, font_sm = None, None
    for fname in [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]:
        try:
            font_big = ImageFont.truetype(fname, 22)
            font_sm = ImageFont.truetype(fname, 14)
            break
        except Exception:
            pass
    if font_big is None:
        font_big = ImageFont.load_default()
        font_sm = font_big

    draw_panel = ImageDraw.Draw(panel)
    # Title
    draw_panel.text((GAP, 10), title, fill=(230, 230, 230), font=font_big)

    for pos, idx in enumerate(indices):
        ov_path = overlay_files[idx]
        lbl_path = None
        tmask = None
        talpha = None
        # Match label file by slice number
        stem = ov_path.stem.replace("_overlay", "")  # slice_0017
        for lf in label_files:
            if lf.stem.startswith(stem):
                lbl_path = lf
                break

        slice_num = int(stem.split("_")[-1]) if stem.split("_")[-1].isdigit() else idx

        img_arr = np.array(Image.open(ov_path).convert("RGB"))

        if raw_files and slice_num < len(raw_files):
            tmask, talpha = _tissue_support_from_raw(raw_files[slice_num])

        # Re-colorize if label available
        if lbl_path and lbl_path.exists():
            try:
                lbl = tifffile.imread(str(lbl_path))
                img_arr = _vibrant_recolor(img_arr, lbl, tissue_mask=tmask)
            except Exception:
                pass

        # Clip using the smooth contour derived from the real slice itself.
        if talpha is not None:
            img_arr = _apply_tissue_alpha(img_arr, talpha)
        elif tmask is not None:
            img_arr = _clip_to_tissue(img_arr, tmask, feather_px=10)

        # Crop to brain
        cropped = _crop_to_brain(img_arr, pad=15, mask=tmask)
        h, w = cropped.shape[:2]

        # Fit into thumb_size × thumb_size while preserving aspect ratio
        scale = thumb_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        thumb = Image.fromarray(cropped).resize((nw, nh), Image.LANCZOS)

        # Place on dark background
        cell = Image.new("RGB", (thumb_size, thumb_size), (18, 18, 24))
        xoff = (thumb_size - nw) // 2
        yoff = (thumb_size - nh) // 2
        cell.paste(thumb, (xoff, yoff))

        # Slice label showing atlas AP position
        draw_cell = ImageDraw.Draw(cell)
        # atlas_z = int(z_file * (-0.2)) + 330 (reversed direction, offset=330)
        z_file = 50 + slice_num * 5
        atlas_z = max(0, min(527, int(z_file * -0.2) + 330))
        ap_mm = round(atlas_z * 0.025, 2)  # 25µm per atlas voxel → mm
        label_txt = f"Slice {slice_num:03d}  atlas_z={atlas_z}  AP={ap_mm}mm"
        _add_label(draw_cell, label_txt, (6, thumb_size - 22), font_sm)

        row_i = pos // cols
        col_i = pos % cols
        x = GAP + col_i * (thumb_size + GAP)
        y = TITLE_H + GAP + row_i * (thumb_size + GAP)
        panel.paste(cell, (x, y))

    panel.save(str(out_path), quality=92)
    print(f"Saved demo panel ({len(indices)} slices): {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg_dir", default="outputs/registered_slices")
    ap.add_argument("--out", default="outputs/demo_panel.jpg")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--size", type=int, default=420)
    args = ap.parse_args()
    make_panel(
        reg_dir=Path(args.reg_dir),
        out_path=Path(args.out),
        n_slices=args.n,
        cols=args.cols,
        thumb_size=args.size,
    )


if __name__ == "__main__":
    main()
