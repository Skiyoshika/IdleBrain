import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread, imwrite

from map_and_aggregate import map_cells_to_regions, aggregate_by_region, write_outputs
from atlas_mapper import map_cells_with_label_volume
from dedup import apply_dedup_kdtree, write_dedup_stats
from preprocess import merge_every_n_slices
from detect import detect_cells
from registration_adapter import bootstrap_registration_assets
from qc import export_slice_qc, export_overlays


def load_config(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _resolve_channel_index(cfg: dict) -> int:
    active = os.environ.get("BRAINCOUNT_ACTIVE_CHANNEL") or cfg.get("input", {}).get("active_channel", "red")
    cmap = cfg.get("input", {}).get("channel_map", {"red": 0, "green": 1, "farred": 2})
    return int(cmap.get(active, 0))


def _collect_slice_files(slice_dir: Path, glob_pattern: str) -> list[Path]:
    return sorted(slice_dir.glob(glob_pattern))


def _extract_channel_to_tmp(src_files: list[Path], out_dir: Path, ch_idx: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(src_files):
        arr = imread(str(p))
        if arr.ndim == 3:
            if arr.shape[-1] <= ch_idx:
                ch = arr[..., 0]
            else:
                ch = arr[..., ch_idx]
        else:
            ch = arr
        dst = out_dir / f"ch_{ch_idx}_{i:04d}.tif"
        imwrite(str(dst), ch.astype(np.uint16))
        out.append(dst)
    return out


def _make_sample_tiffs(out_dir: Path, n: int = 12, h: int = 256, w: int = 256) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(n):
        base = rng.normal(200, 20, size=(h, w, 3)).astype(np.float32)
        # red channel hotspots
        for _ in range(10):
            y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
            base[y - 2 : y + 3, x - 2 : x + 3, 0] += 250
        # green channel hotspots
        for _ in range(8):
            y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
            base[y - 2 : y + 3, x - 2 : x + 3, 1] += 180
        # farred channel hotspots
        for _ in range(6):
            y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
            base[y - 2 : y + 3, x - 2 : x + 3, 2] += 130

        arr = np.clip(base, 0, 65535).astype(np.uint16)
        imwrite(str(out_dir / f"slice_{i:04d}.tif"), arr)


def run_demo(cfg: dict):
    cells_csv = Path("notes/demo_cells.csv")
    atlas_csv = Path("notes/demo_atlas_map.csv")
    cells = pd.read_csv(cells_csv)

    px_um = float(cfg.get("input", {}).get("pixel_size_um_xy", 1.0) if cfg.get("input", {}).get("pixel_size_um_xy") != "TODO" else 1.0)
    spacing_um = float(cfg.get("input", {}).get("slice_spacing_um", 25.0) if cfg.get("input", {}).get("slice_spacing_um") != "TODO" else 25.0)
    neighbor = int(cfg.get("dedup", {}).get("neighbor_slices", 1))
    rxy = float(cfg.get("dedup", {}).get("r_xy_um", 6.0))

    deduped, stats = apply_dedup_kdtree(
        cells,
        neighbor_slices=neighbor,
        pixel_size_um=px_um,
        slice_spacing_um=spacing_um,
        r_xy_um=rxy,
    )
    deduped.to_csv("outputs/demo_cells_dedup.csv", index=False)
    write_dedup_stats(stats, Path("outputs"))

    mapped = map_cells_to_regions(Path("outputs/demo_cells_dedup.csv"), atlas_csv)
    leaf, hierarchy = aggregate_by_region(mapped)
    write_outputs(leaf, hierarchy, Path("outputs"))
    print("Demo pipeline complete: kdtree dedup + mapping + aggregation outputs generated")


def run_real_input(cfg: dict, input_dir: Path):
    slice_glob = cfg.get("input", {}).get("slice_glob", "*.tif")
    files = _collect_slice_files(input_dir, slice_glob)
    if not files:
        raise FileNotFoundError(f"No slice files found in {input_dir} with pattern {slice_glob}")

    ch_idx = _resolve_channel_index(cfg)
    ch_dir = Path("outputs/tmp_channel")
    channel_files = _extract_channel_to_tmp(files, ch_dir, ch_idx)

    n_merge = int(cfg.get("input", {}).get("slice_interval_n", 5))
    merged_files = merge_every_n_slices(channel_files, Path("outputs/tmp_merged"), n=n_merge)

    rows = []
    next_id = 1
    for sid, mp in enumerate(merged_files):
        det = detect_cells(mp, cfg)
        if det.empty:
            continue
        det = det.copy()
        det["slice_id"] = sid
        det["cell_id"] = range(next_id, next_id + len(det))
        next_id += len(det)
        rows.append(det[["cell_id", "slice_id", "x", "y", "score"]])

    if not rows:
        print("No detections found in real-input run.")
        return

    cells = pd.concat(rows, ignore_index=True)
    cells.to_csv("outputs/cells_detected.csv", index=False)

    px_um_raw = cfg.get("input", {}).get("pixel_size_um_xy", 1.0)
    px_um = float(px_um_raw if px_um_raw != "TODO" else 1.0)
    spacing_raw = cfg.get("input", {}).get("slice_spacing_um", 25.0)
    spacing_um = float(spacing_raw if spacing_raw != "TODO" else 25.0)
    neighbor = int(cfg.get("dedup", {}).get("neighbor_slices", 1))
    rxy = float(cfg.get("dedup", {}).get("r_xy_um", 6.0))

    deduped, stats = apply_dedup_kdtree(
        cells,
        neighbor_slices=neighbor,
        pixel_size_um=px_um,
        slice_spacing_um=spacing_um,
        r_xy_um=rxy,
    )
    deduped.to_csv("outputs/cells_dedup.csv", index=False)
    write_dedup_stats(stats, Path("outputs"))

    label_nii = Path("outputs/registration/annotation_25.nii.gz")
    structure_csv = Path("outputs/registration/structure_tree.csv")
    mapped = map_cells_with_label_volume(
        deduped,
        label_nii=label_nii,
        structure_csv=structure_csv,
        pixel_size_um=px_um,
        slice_spacing_um=spacing_um,
        atlas_voxel_um=25.0,
    )
    mapped.to_csv("outputs/cells_mapped.csv", index=False)
    leaf, hierarchy = aggregate_by_region(mapped)
    write_outputs(leaf, hierarchy, Path("outputs"))

    cells_mapped_path = Path("outputs/cells_mapped.csv")
    slice_qc_path = Path("outputs/slice_qc.csv")
    export_slice_qc(cells_mapped_path, slice_qc_path)
    export_overlays(input_dir, Path("outputs/cells_dedup.csv"), Path("outputs/qc_overlays"), max_images=5)

    print(
        f"Real-input end-to-end complete: detected={len(cells)}, dedup={len(deduped)} -> outputs/cell_counts_leaf.csv + QC"
    )


def main():
    parser = argparse.ArgumentParser(description="Brain atlas cell count MVP pipeline entry")
    parser.add_argument("--config", required=True, help="Path to run config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print plan only")
    parser.add_argument("--demo-map", action="store_true", help="Run mapping/aggregation demo with sample CSVs")
    parser.add_argument("--run-real-input", type=str, default="", help="Run preprocess+detect on real input folder")
    parser.add_argument("--make-sample-tiff", type=str, default="", help="Create synthetic 3-channel TIFF slices into folder")
    parser.add_argument("--init-registration", action="store_true", help="Bootstrap registration assets from legacy repo")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    steps = [
        "prepare_input",
        "register_slices",
        "detect_cells",
        "dedup_cells_kdtree",
        "map_cells_to_regions",
        "aggregate_by_region",
        "export_qc",
    ]

    print("Loaded config for project:", cfg.get("project", {}).get("name", "unknown"))
    print("Active channel:", cfg.get("input", {}).get("active_channel", "unknown"))
    print("Planned steps:")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    if args.dry_run:
        print("Dry-run complete.")
        return

    if args.demo_map:
        run_demo(cfg)
        return

    if args.make_sample_tiff:
        _make_sample_tiffs(Path(args.make_sample_tiff))
        print(f"Sample TIFFs generated at {args.make_sample_tiff}")
        return

    if args.init_registration:
        assets = bootstrap_registration_assets(Path(__file__).resolve().parents[1])
        out = Path("outputs/registration_assets.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(assets, indent=2), encoding="utf-8")
        print(f"Registration assets initialized -> {out}")
        return

    if args.run_real_input:
        run_real_input(cfg, Path(args.run_real_input))
        return

    print("MVP skeleton only: registration/mapping integration pending.")


if __name__ == "__main__":
    main()
