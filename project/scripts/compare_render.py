from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scripts.slice_select import select_label_slice_2d, select_real_slice_2d
from skimage.transform import resize
from tifffile import imread


def _norm_u8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return ((img - img.min()) / (np.ptp(img) + 1e-6) * 255.0).astype(np.uint8)


def render_before_after(
    real_path: Path,
    before_label_path: Path,
    after_label_path: Path,
    out_path: Path,
    alpha: float = 0.45,
    before_score: float | None = None,
    after_score: float | None = None,
) -> Path:
    real = imread(str(real_path))
    b = imread(str(before_label_path))
    a = imread(str(after_label_path))

    real, _ = select_real_slice_2d(real, source_path=real_path)
    b, _ = select_label_slice_2d(b)
    a, _ = select_label_slice_2d(a)

    # ensure label maps match real slice size
    if b.shape != real.shape:
        b = resize(
            b.astype(np.float32), real.shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint16)
    if a.shape != real.shape:
        a = resize(
            a.astype(np.float32), real.shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint16)

    real_u8 = _norm_u8(real)
    base = np.stack([real_u8, real_u8, real_u8], axis=-1).astype(np.float32)

    lut = np.array(
        [
            [0, 0, 0],
            [0, 200, 255],
            [0, 255, 120],
            [255, 120, 180],
            [255, 70, 70],
            [220, 220, 80],
        ],
        dtype=np.uint8,
    )

    cb = lut[(b.astype(np.int32) % len(lut))].astype(np.float32)
    ca = lut[(a.astype(np.int32) % len(lut))].astype(np.float32)

    ob = np.clip((1 - alpha) * base + alpha * cb, 0, 255).astype(np.uint8)
    oa = np.clip((1 - alpha) * base + alpha * ca, 0, 255).astype(np.uint8)

    h, w, _ = ob.shape
    canvas = np.zeros((h, w * 2 + 8, 3), dtype=np.uint8)
    canvas[:, :w] = ob
    canvas[:, w + 8 :] = oa

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.rectangle([6, 6, 240, 26], fill=(0, 0, 0))
    draw.text((10, 10), "Before", fill=(255, 255, 255))
    draw.rectangle([w + 14, 6, w + 270, 26], fill=(0, 0, 0))
    draw.text((w + 18, 10), "After", fill=(255, 255, 255))
    if before_score is not None and after_score is not None:
        draw.rectangle([6, h - 26, w * 2, h - 4], fill=(0, 0, 0))
        draw.text(
            (10, h - 22),
            f"SSIM before={before_score:.4f}  after={after_score:.4f}",
            fill=(255, 255, 255),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
