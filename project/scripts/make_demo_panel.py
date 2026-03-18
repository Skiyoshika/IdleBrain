"""
make_demo_panel.py  —  Generate a publication-quality demo panel from registered overlays.

Usage:
    cd project
    python scripts/make_demo_panel.py [--out outputs/demo_panel.png] [--n 9]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

# ── annotated single-slice helper ────────────────────────────────────────────


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
    import pandas as pd

    # --- load overlay and label ---
    ov_arr = np.array(Image.open(str(overlay_png)).convert("RGB"))
    lbl_arr = tifffile.imread(str(label_tif)).astype(np.int32)

    # --- load structure tree for region names ---
    try:
        df = pd.read_csv(str(structure_csv))
        id2name = dict(zip(df["id"].astype(int), df["name"].fillna("?"), strict=False))
        id2acro = dict(zip(df["id"].astype(int), df["acronym"].fillna("?"), strict=False))
    except Exception:
        id2name = {}
        id2acro = {}

    # --- vibrant recolor ---
    vib = _vibrant_recolor(ov_arr, lbl_arr)

    # --- crop to brain area ---
    vib_crop = _crop_to_brain(vib, pad=20)
    lbl_crop = _crop_to_brain(lbl_arr, pad=20)  # same crop applied

    # Re-compute crop coords to match
    gray = ov_arr.mean(axis=2)
    mask = gray > 8
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = max(0, np.where(rows)[0][0] - 20)
    r1 = min(ov_arr.shape[0] - 1, np.where(rows)[0][-1] + 20)
    c0 = max(0, np.where(cols)[0][0] - 20)
    c1 = min(ov_arr.shape[1] - 1, np.where(cols)[0][-1] + 20)
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
            raw_rgb = np.stack([raw_norm] * 3, axis=-1)[r0 : r1 + 1, c0 : c1 + 1]
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

    # --- find top regions by area ---
    ids, counts = np.unique(lbl_r[lbl_r > 0], return_counts=True)
    sorted_idx = np.argsort(counts)[::-1][:top_n]

    # --- draw region labels on the vibrant overlay ---
    PALETTE = [
        (255, 80, 80),
        (255, 165, 50),
        (255, 220, 50),
        (80, 200, 80),
        (50, 200, 200),
        (80, 130, 255),
        (180, 80, 255),
        (255, 100, 200),
        (255, 140, 80),
        (120, 220, 120),
    ]
    result_img = Image.fromarray(vib_r)
    draw = ImageDraw.Draw(result_img)

    try:
        font_sm = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
        ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 14)
    except Exception:
        font_sm = ImageFont.load_default()

    legend_entries = []  # (color, acro, full_name) for legend panel
    for rank, si in enumerate(sorted_idx):
        uid = int(ids[si])
        ys, xs = np.nonzero(lbl_r == uid)
        if len(ys) < 50:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        acro = id2acro.get(uid, str(uid))
        name = id2name.get(uid, acro)
        color = PALETTE[rank % len(PALETTE)]
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
    legend_h = LEGEND_PAD * 2 + len(legend_entries) * LEGEND_ROW_H
    legend_panel = np.full((legend_h, TW, 3), 28, dtype=np.uint8)
    leg_img = Image.fromarray(legend_panel)
    leg_draw = ImageDraw.Draw(leg_img)
    try:
        font_leg = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
        font_leg_b = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 13)
    except Exception:
        font_leg = font_leg_b = ImageFont.load_default()
    for i, (color, acro, name) in enumerate(legend_entries):
        y = LEGEND_PAD + i * LEGEND_ROW_H + 2
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


def _crop_to_brain(img_arr: np.ndarray, pad: int = 20) -> np.ndarray:
    """Crop away dark background, keeping only the brain region + padding."""
    gray = img_arr.mean(axis=2) if img_arr.ndim == 3 else img_arr
    mask = gray > 8  # threshold for "not black background"
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img_arr
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    H, W = img_arr.shape[:2]
    r0 = max(0, r0 - pad)
    r1 = min(H - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(W - 1, c1 + pad)
    return img_arr[r0 : r1 + 1, c0 : c1 + 1]


def _vibrant_recolor(img_arr: np.ndarray, label_arr: np.ndarray) -> np.ndarray:
    """
    Re-colorize using a more vibrant, distinct palette while preserving brain texture.
    img_arr  : (H, W, 3) uint8 — existing fill-mode overlay
    label_arr: (H, W)    int32 — registered label (region IDs)
    """
    # Vivid palette (HSL-spaced, high saturation)
    PALETTE = [
        (255, 80, 80),
        (255, 165, 50),
        (255, 220, 50),
        (80, 200, 80),
        (50, 200, 200),
        (80, 130, 255),
        (180, 80, 255),
        (255, 100, 200),
        (255, 140, 80),
        (120, 220, 120),
        (80, 200, 230),
        (160, 100, 255),
        (220, 200, 80),
        (100, 200, 160),
        (255, 80, 140),
        (200, 160, 80),
        (80, 180, 255),
        (255, 120, 50),
        (150, 255, 150),
        (200, 80, 255),
    ]

    # Convert original image to float grayscale (brain texture)
    gray = img_arr.mean(axis=2).astype(np.float32) / 255.0
    # Gamma < 1 brightens dim cleared-tissue images so texture is visible through color.
    gray_bright = np.clip(gray**0.55, 0.0, 1.0)
    out = np.zeros((*img_arr.shape[:2], 3), dtype=np.uint8)

    ids = np.unique(label_arr)
    ids = ids[ids > 0]
    for i, uid in enumerate(ids):
        mask = label_arr == uid
        color = PALETTE[i % len(PALETTE)]
        texture = gray_bright[mask]  # brightened tissue texture
        for ch, cv in enumerate(color):
            # 60% texture + 40% color: tissue structure clearly visible through tint
            out[mask, ch] = np.clip(texture * 0.60 * 255 + cv * 0.40, 0, 255).astype(np.uint8)

    # Background: keep dark
    bg = label_arr == 0
    out[bg] = (img_arr[bg].astype(np.float32) * 0.3).astype(np.uint8)
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


def _tissue_mask_from_raw(raw_tif_path: Path) -> np.ndarray | None:
    """Compute accurate tissue mask from a raw lightsheet TIF using 4-sigma corner threshold."""
    try:
        raw = tifffile.imread(str(raw_tif_path)).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[0] if raw.shape[0] < raw.shape[1] else raw[..., 0]
        b = 60
        corners = np.concatenate(
            [
                raw[:b, :b].ravel(),
                raw[:b, -b:].ravel(),
                raw[-b:, :b].ravel(),
                raw[-b:, -b:].ravel(),
            ]
        )
        thr = float(np.mean(corners) + 4.0 * np.std(corners))
        mask = raw > thr
        from skimage import measure as _meas
        from skimage import morphology as _morph

        mask = _morph.closing(mask, _morph.disk(20))
        labeled = _meas.label(mask)
        if labeled.max() > 0:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            mask = labeled == counts.argmax()
        return mask.astype(bool)
    except Exception:
        return None


def _clip_to_tissue(
    img_arr: np.ndarray, tissue_mask: np.ndarray, feather_px: int = 10
) -> np.ndarray:
    """Clip a colourised overlay to the actual tissue boundary with a smooth feathered edge."""
    from scipy.ndimage import distance_transform_edt as _dt_edt

    if tissue_mask.shape != img_arr.shape[:2]:
        # Resize mask to match image
        from PIL import Image as _Im

        tm_img = _Im.fromarray(tissue_mask.astype(np.uint8) * 255)
        tm_img = tm_img.resize((img_arr.shape[1], img_arr.shape[0]), _Im.NEAREST)
        tissue_mask = np.array(tm_img) > 127
    dist_in = _dt_edt(tissue_mask.astype(np.uint8)).astype(np.float32)
    alpha = np.clip(dist_in / max(feather_px, 1), 0.0, 1.0)[:, :, np.newaxis]
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
        # Match label file by slice number
        stem = ov_path.stem.replace("_overlay", "")  # slice_0017
        for lf in label_files:
            if lf.stem.startswith(stem):
                lbl_path = lf
                break

        slice_num = int(stem.split("_")[-1]) if stem.split("_")[-1].isdigit() else idx

        img_arr = np.array(Image.open(ov_path).convert("RGB"))

        # Re-colorize if label available
        if lbl_path and lbl_path.exists():
            try:
                lbl = tifffile.imread(str(lbl_path))
                img_arr = _vibrant_recolor(img_arr, lbl)
            except Exception:
                pass

        # Hard tissue boundary: clip using raw slice tissue mask
        if raw_files and slice_num < len(raw_files):
            tmask = _tissue_mask_from_raw(raw_files[slice_num])
            if tmask is not None:
                img_arr = _clip_to_tissue(img_arr, tmask, feather_px=10)

        # Crop to brain
        cropped = _crop_to_brain(img_arr, pad=15)
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
