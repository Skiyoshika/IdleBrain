from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paths import bootstrap_sys_path

PROJECT_DIR = bootstrap_sys_path()


def _fallback_color(region_id: int) -> tuple[int, int, int]:
    x = int(region_id) * 2654435761 & 0xFFFFFFFF
    return (
        80 + (x & 0x7F),
        80 + ((x >> 8) & 0x7F),
        80 + ((x >> 16) & 0x7F),
    )


def _render_brain_slice(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    from scripts.image_utils import norm_u8_robust

    x = arr.astype(np.float32, copy=False)
    use_mask = mask is not None and bool(np.any(mask))
    if use_mask:
        values = x[mask]
        p1 = float(np.percentile(values, 1.0))
        p99 = float(np.percentile(values, 99.0))
        if p99 <= p1:
            gray = norm_u8_robust(x)
        else:
            gray = np.clip((x - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
            gray = (gray * 255.0).astype(np.uint8)
            gray = gray.copy()
            gray[~mask] = 0
    else:
        gray = norm_u8_robust(x)
    rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    rgb[..., 0] = gray
    rgb[..., 1] = (gray.astype(np.float32) * 0.35).astype(np.uint8)
    rgb[..., 2] = (gray.astype(np.float32) * 0.1).astype(np.uint8)
    return rgb


def _render_annotation_slice(
    arr: np.ndarray, color_map: dict[int, tuple[int, int, int]]
) -> np.ndarray:
    out = np.full((*arr.shape, 3), (45, 130, 190), dtype=np.uint8)
    ids = np.unique(arr)
    ids = ids[ids > 0]
    for rid in ids:
        mask = arr == rid
        out[mask] = color_map.get(int(rid), _fallback_color(int(rid)))
    return out


def make_registration_overview(
    brain_volume: Path,
    annotation_volume: Path,
    out_path: Path,
    *,
    slices: list[int],
    structure_csv: Path | None = None,
) -> Path:
    from scripts.allen_colors import load_allen_color_map

    brain = np.asarray(nib.load(str(brain_volume)).dataobj)
    annotation = np.asarray(nib.load(str(annotation_volume)).dataobj)
    if brain.shape != annotation.shape:
        raise ValueError(f"brain/annotation shape mismatch: {brain.shape} vs {annotation.shape}")

    color_map = load_allen_color_map(structure_csv) if structure_csv else {}
    panels: list[np.ndarray] = []
    labels: list[str] = []
    target_h = 360

    for z in slices:
        zz = int(np.clip(z, 0, brain.shape[0] - 1))
        annotation_slice = np.asarray(annotation[zz], dtype=np.int32)
        brain_rgb = _render_brain_slice(
            np.asarray(brain[zz], dtype=np.float32),
            mask=annotation_slice > 0,
        )
        annotation_rgb = _render_annotation_slice(annotation_slice, color_map)
        for kind, arr in (
            ("brain", brain_rgb),
            ("annotation", annotation_rgb),
        ):
            img = Image.fromarray(arr)
            scale = target_h / max(img.height, 1)
            resized = img.resize((int(img.width * scale), target_h), Image.Resampling.LANCZOS)
            panels.append(np.array(resized))
            labels.append(f"{kind} z={zz}")

    widths = [panel.shape[1] for panel in panels]
    heights = [panel.shape[0] for panel in panels]
    gap = 14
    header_h = 34
    canvas = np.full(
        (header_h + max(heights), sum(widths) + gap * (len(panels) - 1), 3),
        20,
        dtype=np.uint8,
    )
    x = 0
    for panel in panels:
        canvas[header_h : header_h + panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1] + gap

    out_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(out_img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    x = 0
    for label, w in zip(labels, widths, strict=False):
        draw.text((x + 6, 8), label, fill=(220, 220, 220), font=font)
        x += w + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(str(out_path), quality=95)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a 3D registration overview image")
    ap.add_argument("--brain-volume", required=True, help="Registered brain volume NIfTI path")
    ap.add_argument("--annotation-volume", required=True, help="Registered annotation NIfTI path")
    ap.add_argument("--out", required=True, help="Output PNG/JPG path")
    ap.add_argument("--slices", default="200,300", help="Comma-separated z indices")
    ap.add_argument(
        "--structure-csv",
        default="configs/allen_mouse_structure_graph.csv",
        help="Allen structure CSV for colors",
    )
    args = ap.parse_args()

    slices = [int(v.strip()) for v in str(args.slices).split(",") if v.strip()]
    out = make_registration_overview(
        Path(args.brain_volume),
        Path(args.annotation_volume),
        Path(args.out),
        slices=slices,
        structure_csv=(PROJECT_DIR / args.structure_csv)
        if not Path(args.structure_csv).is_absolute()
        else Path(args.structure_csv),
    )
    print(f"Saved overview -> {out}")


if __name__ == "__main__":
    main()
