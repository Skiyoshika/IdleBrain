"""atlas_shape_match.py — AP-position finding by brain silhouette comparison.

Idea (from user 2026-03-16):
  Instead of comparing fluorescence intensity vs atlas annotation colour,
  compare the BRAIN OUTLINE shape between our lightsheet slice and the
  Allen reference section thumbnails.  Brain silhouette is modality-invariant
  (same coronal cross-section shape regardless of stain) and uniquely
  identifies the AP position.

Data source:
  Allen Brain Atlas experiment 100048576 — a full coronal series of the P56
  mouse brain.  Section number 1–528 maps 1:1 to atlas_z index 0–527 in the
  annotation_25.nii.gz CCFv3 volume.  Thumbnails are downloaded once and
  cached to configs/allen_ref_cache/.

Public API
----------
  find_ap_by_silhouette(real_img, search_z_min, search_z_max,
                        pixel_size_um, hemisphere, cache_dir) -> (atlas_z, score)

  build_silhouette_cache(z_min, z_max, cache_dir, step)  # pre-download
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from skimage import measure, morphology
from skimage.transform import resize as sk_resize

# Allen ISH experiment that covers the full brain, section_number == atlas_z
_ALLEN_DATASET_ID = 100048576
_ALLEN_IMG_API = "https://api.brain-map.org/api/v2/section_image_download/{img_id}"
_ALLEN_DATA_API = (
    "https://api.brain-map.org/api/v2/data/query.json"
    "?criteria=model%3A%3ASectionDataSet%2Crma%3A%3Acriteria"
    "%2C%5Bid%24eq{ds_id}%5D%2Crma%3A%3Ainclude%2Csection_images"
    "%5Bfailed%24eqfalse%5D&num_rows=1"
)

# Thumbnail downsample level (6 → ~150–170 px wide, ~3–4 KB each)
_DOWNSAMPLE = 6

# Standard size to which all silhouettes are normalised for comparison
_STD_SIZE = 128


# ── Silhouette extraction ────────────────────────────────────────────────────

# Allen ISH thumbnails at downsample=6: full-res resolution=1.047µm × 2^6 = 67.0µm/px
_ALLEN_THUMB_UM_PX = 1.047 * (2**_DOWNSAMPLE)


def _extract_silhouette(
    img: np.ndarray, hemisphere: str = "right", pixel_um: float | None = None
) -> tuple[np.ndarray | None, float]:
    """Return a binary brain-outline mask resized to _STD_SIZE × _STD_SIZE.

    Returns (silhouette_mask_128x128, area_mm2).
    area_mm2 is the estimated brain cross-section area in mm² (0 if pixel_um is None).

    Handles two image types automatically:
      • Bright-background (Nissl/ISH RGB thumbnails): tissue is DARKER — detected
        by inverting and thresholding above a low percentile.
      • Dark-background (fluorescence 16-bit): tissue is BRIGHTER — detected
        by thresholding above a low percentile relative to background.
    """
    if img is None:
        return None, 0.0
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1).astype(np.float32)  # RGB → gray
    else:
        img = img.astype(np.float32)
    h, w = img.shape

    # Hemisphere selection
    if hemisphere in ("right", "right_flipped"):
        img = img[:, w // 2 :]
    elif hemisphere == "left":
        img = img[:, : w // 2]
    # else "full"

    # Normalise to [0, 255] regardless of bit depth
    v_max = float(img.max())
    if v_max > 255.0:
        img = img * (255.0 / v_max)

    ih, iw = img.shape

    # Detect image type by median: >128 = bright background (Nissl); <=128 = dark (fluorescence)
    median_v = float(np.median(img))

    if median_v > 128.0:
        # Bright-background: tissue is darker than background.
        # Invert → tissue becomes bright, background near 0.
        img_work = 255.0 - img
        # Background pixels (original bright) → near 0 after inversion; skip them.
        fg = img_work[img_work > 20]
        if fg.size < 50:
            return None, 0.0
        thr = float(np.percentile(fg, 20))  # fairly generous — Nissl fills the whole section
        rough = img_work > thr
    else:
        # Dark-background: tissue is brighter than background.
        # Estimate background from lower 20th percentile of whole image.
        bg_low = img[img <= float(np.percentile(img, 20))]
        bg_med = float(np.median(bg_low)) if bg_low.size > 0 else 0.0
        bg_std = float(np.std(bg_low)) if bg_low.size > 0 else 1.0
        thr = bg_med + max(3.0, bg_std * 2.5)
        rough = img > thr

    # Morphological closing to fill gaps (large disk to bridge labelled cell spots)
    disk_r = max(4, min(ih, iw) // 15)
    filled = morphology.closing(rough, morphology.disk(disk_r))
    filled = morphology.dilation(filled, morphology.disk(max(2, disk_r // 3)))

    # Keep only the largest connected component
    lbl = measure.label(filled, connectivity=2)
    if lbl.max() == 0:
        return None, 0.0
    props = sorted(measure.regionprops(lbl), key=lambda p: p.area, reverse=True)
    sil = (lbl == props[0].label).astype(np.float32)

    if float(np.sum(sil)) < 30:
        return None, 0.0

    # Physical area in mm²
    area_mm2 = 0.0
    if pixel_um is not None and pixel_um > 0:
        area_px = float(np.sum(sil))
        area_mm2 = area_px * (pixel_um * 1e-3) ** 2  # µm → mm

    # Crop to bounding box, then resize to standard size
    ys, xs = np.nonzero(sil)
    crop = sil[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return None, 0.0
    norm = sk_resize(crop, (_STD_SIZE, _STD_SIZE), order=1, preserve_range=True)
    return (norm > 0.5).astype(np.float32), area_mm2


def _silhouette_score(
    sil_a: np.ndarray,
    sil_b: np.ndarray,
    area_a_mm2: float | None = None,
    area_b_mm2: float | None = None,
) -> float:
    """Multi-metric silhouette similarity score in [0, 1].

    Combines:
      • IoU of the two normalised silhouettes (shape match)
      • Aspect ratio similarity
      • Absolute area similarity (when physical areas in mm² are provided) —
        the most discriminative cue since brain cross-section area changes
        strongly with AP position
    """
    inter = float(np.sum(sil_a * sil_b))
    union = float(np.sum((sil_a + sil_b) > 0)) + 1e-6
    iou = inter / union

    # Aspect ratio
    ys_a, xs_a = np.nonzero(sil_a > 0.5)
    ys_b, xs_b = np.nonzero(sil_b > 0.5)
    asp_sim = 0.5  # neutral if we can't compute
    if len(ys_a) > 10 and len(ys_b) > 10:
        asp_a = (ys_a.max() - ys_a.min() + 1.0) / max(xs_a.max() - xs_a.min() + 1.0, 1.0)
        asp_b = (ys_b.max() - ys_b.min() + 1.0) / max(xs_b.max() - xs_b.min() + 1.0, 1.0)
        asp_sim = 1.0 - min(abs(asp_a - asp_b) / max(asp_a, asp_b, 0.1), 1.0)

    # Physical area similarity (strong cue when available)
    if area_a_mm2 is not None and area_b_mm2 is not None and area_a_mm2 > 0 and area_b_mm2 > 0:
        area_ratio = min(area_a_mm2, area_b_mm2) / max(area_a_mm2, area_b_mm2)
        # Sharpen: square the ratio so perfect match=1.0 and 50% difference → 0.25
        area_score = float(area_ratio**2)
        return float(0.40 * iou + 0.45 * area_score + 0.15 * asp_sim)

    return float(0.70 * iou + 0.30 * asp_sim)


# ── Cache management ─────────────────────────────────────────────────────────


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "allen_ref_cache"


def _section_id_map(cache_dir: Path) -> dict[int, int]:
    """Load or fetch the section_number → image_id mapping."""
    map_path = cache_dir / "section_id_map.json"
    if map_path.exists():
        return {int(k): int(v) for k, v in json.loads(map_path.read_text()).items()}

    import requests

    print("[atlas_shape_match] Fetching Allen section ID map ...")
    url = _ALLEN_DATA_API.format(ds_id=_ALLEN_DATASET_ID)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    imgs = r.json()["msg"][0]["section_images"]
    mapping = {int(im["section_number"]): int(im["id"]) for im in imgs}
    cache_dir.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(mapping))
    print(f"[atlas_shape_match] Saved section ID map ({len(mapping)} entries)")
    return mapping


def _thumb_path(z: int, cache_dir: Path) -> Path:
    return cache_dir / f"thumb_z{z:04d}.jpg"


def _download_thumb(z: int, img_id: int, cache_dir: Path) -> Path | None:
    import requests

    url = _ALLEN_IMG_API.format(img_id=img_id)
    params = {"downsample": _DOWNSAMPLE, "quality": 75}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        p = _thumb_path(z, cache_dir)
        p.write_bytes(r.content)
        return p
    except Exception as e:
        print(f"[atlas_shape_match] WARNING: failed to download z={z}: {e}")
        return None


def build_silhouette_cache(
    z_min: int = 130,
    z_max: int = 300,
    cache_dir: Path | None = None,
    step: int = 1,
) -> None:
    """Download Allen reference thumbnails and cache silhouettes.

    Call once before running the pipeline.  Saves thumbnails + silhouette
    arrays to cache_dir.  Skips already-downloaded entries.
    """
    from PIL import Image

    if cache_dir is None:
        cache_dir = _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    id_map = _section_id_map(cache_dir)
    zs = list(range(z_min, z_max + 1, step))
    missing = [z for z in zs if not _thumb_path(z, cache_dir).exists()]
    print(f"[atlas_shape_match] Downloading {len(missing)}/{len(zs)} thumbnails ...")

    sil_cache: dict[int, list] = {}
    sil_cache_path = cache_dir / "silhouettes.json"
    if sil_cache_path.exists():
        sil_cache = {int(k): v for k, v in json.loads(sil_cache_path.read_text()).items()}

    for i, z in enumerate(missing):
        if z not in id_map:
            continue
        p = _download_thumb(z, id_map[z], cache_dir)
        if p is not None and (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(missing)}")
        time.sleep(0.05)  # polite throttle

    # Build/update silhouette cache for all z in range
    updated = False
    for z in zs:
        if z in sil_cache:
            continue
        tp = _thumb_path(z, cache_dir)
        if not tp.exists():
            continue
        try:
            arr = np.array(Image.open(tp))
            # Compute silhouette + area for both full and right-half
            # Allen thumbnails are at _ALLEN_THUMB_UM_PX resolution
            sil_full, area_full = _extract_silhouette(arr, "full", _ALLEN_THUMB_UM_PX)
            sil_right, area_right = _extract_silhouette(arr, "right", _ALLEN_THUMB_UM_PX)
            sil_cache[z] = {
                "full": sil_full.tolist() if sil_full is not None else None,
                "right": sil_right.tolist() if sil_right is not None else None,
                "area_full": float(area_full),
                "area_right": float(area_right),
            }
            updated = True
        except Exception:
            pass

    if updated:
        sil_cache_path.write_text(json.dumps(sil_cache))
        print(f"[atlas_shape_match] Silhouette cache saved ({len(sil_cache)} entries)")


def _load_silhouette_cache(cache_dir: Path) -> dict[int, dict]:
    p = cache_dir / "silhouettes.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {int(k): v for k, v in raw.items()}


# ── Main API ─────────────────────────────────────────────────────────────────


def find_ap_by_silhouette(
    real_img: np.ndarray,
    search_z_min: int = 130,
    search_z_max: int = 300,
    pixel_size_um: float = 5.0,
    hemisphere: str = "right_flipped",
    cache_dir: Path | None = None,
    auto_download: bool = True,
) -> tuple[int, float]:
    """Find best atlas AP index by comparing brain outline shapes.

    Parameters
    ----------
    real_img : 2-D or 3-D uint16/uint8 array (H × W or H × W × C)
    search_z_min / search_z_max : atlas_z search range (inclusive)
    pixel_size_um : real image pixel size (µm)
    hemisphere : "right" / "right_flipped" / "left" / "full"
    cache_dir : where Allen thumbnails are stored
    auto_download : if True, download missing thumbnails on demand

    Returns
    -------
    (best_atlas_z, best_score)  — score in [0, 1], higher = better match
    """
    if cache_dir is None:
        cache_dir = _default_cache_dir()

    # Normalise hemisphere key for silhouette extraction
    hemi_key = "right" if hemisphere in ("right", "right_flipped") else hemisphere

    # Extract silhouette + physical area from real image
    sil_real, area_real_mm2 = _extract_silhouette(
        real_img, hemisphere=hemi_key, pixel_um=pixel_size_um
    )
    if sil_real is None:
        print("[atlas_shape_match] WARNING: could not extract brain silhouette from real image")
        return (search_z_min + search_z_max) // 2, 0.0
    print(f"[atlas_shape_match] Real brain area: {area_real_mm2:.2f} mm2")

    # Load cached silhouettes
    sil_db = _load_silhouette_cache(cache_dir)

    # Determine which z-values we need
    needed_zs = list(range(search_z_min, search_z_max + 1))
    missing_zs = [z for z in needed_zs if z not in sil_db]

    if missing_zs and auto_download:
        print(f"[atlas_shape_match] Cache missing {len(missing_zs)} entries; downloading ...")
        build_silhouette_cache(z_min=search_z_min, z_max=search_z_max, cache_dir=cache_dir, step=1)
        sil_db = _load_silhouette_cache(cache_dir)

    # Score each candidate z
    scores: list[tuple[int, float]] = []
    for z in needed_zs:
        entry = sil_db.get(z)
        if entry is None:
            continue
        sil_ref_raw = entry.get(hemi_key) or entry.get("right") or entry.get("full")
        area_ref_key = f"area_{hemi_key}" if f"area_{hemi_key}" in entry else "area_right"
        if sil_ref_raw is None:
            continue
        sil_ref = np.array(sil_ref_raw, dtype=np.float32)
        if sil_ref.shape != (_STD_SIZE, _STD_SIZE):
            continue
        area_ref_mm2 = float(entry.get(area_ref_key, 0.0))
        sc = _silhouette_score(sil_real, sil_ref, area_a_mm2=area_real_mm2, area_b_mm2=area_ref_mm2)
        scores.append((z, sc))

    if not scores:
        print("[atlas_shape_match] WARNING: no cached silhouettes in range; returning midpoint")
        return (search_z_min + search_z_max) // 2, 0.0

    best_z, best_sc = max(scores, key=lambda x: x[1])
    # Print top-5 for debugging
    top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    print(f"[atlas_shape_match] Top-5: {[(z, round(s, 4)) for z, s in top5]}")
    print(f"[atlas_shape_match] Best match: atlas_z={best_z}, score={best_sc:.4f}")
    return int(best_z), float(best_sc)


# ── CLI helper ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    from tifffile import imread as tiff_imread

    ap = argparse.ArgumentParser(description="Find atlas AP by brain silhouette")
    ap.add_argument("--slice", default=None, help="Path to real lightsheet TIFF slice")
    ap.add_argument("--z-min", type=int, default=130)
    ap.add_argument("--z-max", type=int, default=290)
    ap.add_argument("--hemi", default="right_flipped")
    ap.add_argument("--pixel-um", type=float, default=5.0)
    ap.add_argument(
        "--download-only", action="store_true", help="Just pre-download the cache and exit"
    )
    args = ap.parse_args()

    cache_dir = _default_cache_dir()
    if args.download_only:
        build_silhouette_cache(z_min=args.z_min, z_max=args.z_max, cache_dir=cache_dir)
        sys.exit(0)

    img = tiff_imread(args.slice)
    best_z, score = find_ap_by_silhouette(
        img,
        search_z_min=args.z_min,
        search_z_max=args.z_max,
        pixel_size_um=args.pixel_um,
        hemisphere=args.hemi,
        cache_dir=cache_dir,
    )
    print(f"Result: atlas_z={best_z}  score={score:.4f}")
