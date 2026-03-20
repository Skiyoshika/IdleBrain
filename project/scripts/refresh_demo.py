"""
refresh_demo.py  —  Regenerate all demo visuals after the pipeline finishes.

Usage (run from project/ directory):
    python scripts/refresh_demo.py [--slice <N>]

What it does:
    1. Generates demo_panel.jpg  (12-slice whole-brain grid)
    2. Generates demo_best_slice.jpg  (raw vs atlas side-by-side)
    3. Generates cell_count_chart.png  (bar + pie chart)
    4. Prints a summary of results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paths import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _make_cell_chart():
    hier_path = OUTPUT_DIR / "cell_counts_hierarchy.csv"
    chart_path = OUTPUT_DIR / "cell_count_chart.png"
    if not hier_path.exists():
        print("  [skip] No hierarchy CSV found.")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    hier = pd.read_csv(hier_path)
    d2 = hier[hier["depth"] == 2].sort_values("count", ascending=False)
    d2 = d2[d2["count"] > 0]
    if d2.empty:
        print("  [skip] No depth-2 regions found.")
        return

    colors = [
        "#E57373",
        "#FF9800",
        "#FFEB3B",
        "#66BB6A",
        "#26C6DA",
        "#5C6BC0",
        "#AB47BC",
        "#EC407A",
        "#26A69A",
        "#8D6E63",
        "#78909C",
        "#FFA726",
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0e0e16")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#141420")
        ax.tick_params(colors="#cccccc", labelsize=9)
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    bars = ax1.barh(
        range(len(d2)),
        d2["count"].values,
        color=[colors[i % len(colors)] for i in range(len(d2))],
        edgecolor="none",
        height=0.7,
    )
    ax1.set_yticks(range(len(d2)))
    ax1.set_yticklabels(
        [f"{r['acronym']} ({r['region_name'][:20]})" for _, r in d2.iterrows()],
        fontsize=8.5,
        color="#cccccc",
    )
    ax1.set_xlabel("Cell Count", color="#cccccc", fontsize=10)
    ax1.set_title("Cell Counts by Major Brain Region", color="white", fontsize=12, pad=10)
    for bar, row in zip(bars, d2.itertuples(), strict=False):
        ax1.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row.count):,}",
            va="center",
            ha="left",
            fontsize=8,
            color="#aaa",
        )
    total = d2["count"].sum()
    pcts = d2["count"].values / total * 100
    labels = [
        f"{r['acronym']} {p:.1f}%" if p > 3 else ""
        for (_, r), p in zip(d2.iterrows(), pcts, strict=False)
    ]
    ax2.pie(
        d2["count"].values,
        colors=[colors[i % len(colors)] for i in range(len(d2))],
        labels=labels,
        labeldistance=1.1,
        textprops={"fontsize": 8, "color": "#cccccc"},
        startangle=90,
        wedgeprops={"edgecolor": "#0e0e16", "linewidth": 1.5},
    )
    ax2.set_title("Proportion by Brain Region", color="white", fontsize=12, pad=10)
    root = hier[hier["depth"] == 0]["count"].values[0]
    fig.suptitle(
        f"Sample 35 — Whole Brain Cell Counts  |  Total: {int(root):,} cells  |  {len(hier)} regions mapped",
        color="white",
        fontsize=11,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        str(chart_path), dpi=120, bbox_inches="tight", facecolor="#0e0e16", edgecolor="none"
    )
    plt.close()
    print(f"  cell_count_chart.png  (total {int(root):,} cells, {len(hier)} regions)")


def _make_best_slice(idx: int):
    from PIL import Image as Im
    from PIL import ImageDraw as ID
    from PIL import ImageFont as IF
    from scripts.make_demo_panel import (
        _apply_tissue_alpha,
        _crop_to_brain,
        _tissue_support_from_raw,
        _vibrant_recolor,
    )

    reg_dir = OUTPUT_DIR / "registered_slices"
    ov_path = reg_dir / f"slice_{idx:04d}_overlay.png"
    lbl_path = reg_dir / f"slice_{idx:04d}_registered_label.tif"
    raw_files = sorted((PROJECT_ROOT / "data" / "35_C0_demo").glob("*.tif"))
    if not ov_path.exists():
        print(f"  [skip] slice_{idx:04d}_overlay.png not found.")
        return

    ov = np.array(Im.open(str(ov_path)).convert("RGB"))
    lbl = (
        tifffile.imread(str(lbl_path))
        if lbl_path.exists()
        else np.zeros(ov.shape[:2], dtype=np.int32)
    )
    raw = (
        tifffile.imread(str(raw_files[min(idx, len(raw_files) - 1)]))
        if raw_files
        else np.zeros(ov.shape[:2], dtype=np.uint8)
    )
    p1, p99 = np.percentile(raw[raw > 0], [1, 99]) if raw.max() > 0 else (0, 1)
    raw_norm = np.clip((raw.astype(np.float32) - p1) / (p99 - p1 + 1e-6) * 255, 0, 255).astype(
        np.uint8
    )
    raw_rgb = np.stack([raw_norm] * 3, axis=-1)
    tmask, talpha = (None, None)
    if raw_files:
        try:
            tmask, talpha = _tissue_support_from_raw(raw_files[min(idx, len(raw_files) - 1)])
        except Exception:
            tmask, talpha = (None, None)

    vib = _vibrant_recolor(ov, lbl, tissue_mask=tmask)
    if talpha is not None:
        vib = _apply_tissue_alpha(vib, talpha)
    raw_crop = _crop_to_brain(raw_rgb, pad=25, mask=tmask)
    vib_crop = _crop_to_brain(vib, pad=25, mask=tmask)
    H = max(raw_crop.shape[0], vib_crop.shape[0])

    def rh(arr, h):
        img = Im.fromarray(arr)
        sc = h / img.height
        return np.array(img.resize((int(img.width * sc), h), Im.LANCZOS))

    raw_r = rh(raw_crop, H)
    vib_r = rh(vib_crop, H)
    div = np.full((H, 8, 3), 35, dtype=np.uint8)
    body_arr = np.concatenate([raw_r, div, vib_r], axis=1)
    W = body_arr.shape[1]
    hdr = np.full((55, W, 3), 18, dtype=np.uint8)
    body = Im.fromarray(np.concatenate([hdr, body_arr], axis=0))
    draw = ID.Draw(body)
    try:
        font = IF.truetype("C:/Windows/Fonts/arial.ttf", 26)
        sm = IF.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except Exception:
        font = sm = IF.load_default()
    draw.text((12, 14), "Raw Lightsheet", fill=(200, 200, 200), font=font)
    draw.text(
        (raw_r.shape[1] + 14, 14),
        "Brainfast — Atlas Registration (Allen CCFv3)",
        fill=(200, 200, 200),
        font=font,
    )
    draw.text((W - 260, 36), f"Slice {idx} · Allen CCFv3", fill=(100, 100, 100), font=sm)
    out = OUTPUT_DIR / "demo_best_slice.jpg"
    body.save(str(out), quality=95)
    print(f"  demo_best_slice.jpg  (slice {idx}, {body.size[0]}×{body.size[1]})")


def main():
    ap = argparse.ArgumentParser(description="Regenerate all demo visuals")
    ap.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Slice index for best-slice image (default: auto-pick)",
    )
    ap.add_argument("--panel-n", type=int, default=12)
    args = ap.parse_args()

    reg_overlays = sorted((OUTPUT_DIR / "registered_slices").glob("slice_*_overlay.png"))
    total = len(reg_overlays)
    if total == 0:
        print("ERROR: No registered slices found. Run the pipeline first.")
        sys.exit(1)

    print(f"\n=== refresh_demo.py  ({total} registered slices) ===\n")

    # 1. Demo panel
    from make_demo_panel import make_panel

    make_panel(
        reg_dir=OUTPUT_DIR / "registered_slices",
        out_path=OUTPUT_DIR / "demo_panel.jpg",
        n_slices=args.panel_n,
        cols=4,
        thumb_size=380,
        slice_dir=PROJECT_ROOT / "data" / "35_C0_demo",
    )
    print(f"  demo_panel.jpg  ({args.panel_n} slices)")

    # 2. Best-slice comparison (auto-pick = most colorful slice)
    if args.slice is not None:
        best_idx = args.slice
    else:
        from PIL import Image

        best_idx, best_score = 0, -1.0
        for i, f in enumerate(reg_overlays):
            arr = np.array(Image.open(str(f)).convert("RGB"))
            score = (abs(arr[:, :, 0].astype(int) - arr[:, :, 1].astype(int)) > 15).mean()
            brain = (arr.mean(axis=2) > 8).mean()
            combined = score * brain
            if combined > best_score:
                best_score, best_idx = combined, i
        print(f"  auto-selected slice {best_idx} (colorfulness*coverage = {best_score:.3f})")
    _make_best_slice(best_idx)

    # 2b. Annotated single-slice example (with region names)
    reg_dir = OUTPUT_DIR / "registered_slices"
    best_ov = reg_overlays[best_idx]
    best_lbl = reg_dir / best_ov.name.replace("_overlay.png", "_registered_label.tif")
    raw_files = sorted((PROJECT_ROOT / "data" / "35_C0_demo").glob("*.tif"))
    best_raw = raw_files[min(best_idx, len(raw_files) - 1)] if raw_files else None
    structure_csv = PROJECT_ROOT / "configs" / "allen_mouse_structure_graph.csv"
    if not structure_csv.exists():
        structure_csv = PROJECT_ROOT / "outputs" / "registration" / "structure_tree.csv"
    if best_lbl.exists() and structure_csv.exists():
        from make_demo_panel import make_annotated_slice

        ann_out = OUTPUT_DIR / "demo_annotated_slice.jpg"
        make_annotated_slice(
            overlay_png=best_ov,
            label_tif=best_lbl,
            raw_tif=best_raw,
            structure_csv=structure_csv,
            out_path=ann_out,
            top_n=12,
        )
        print(
            f"  demo_annotated_slice.jpg  (slice {best_idx}, {len(np.unique(tifffile.imread(str(best_lbl)))) - 1} regions)"
        )

    # 3. Cell count chart
    _make_cell_chart()

    # 4. Summary
    print("\n=== Results Summary ===")
    hier_path = OUTPUT_DIR / "cell_counts_hierarchy.csv"
    if hier_path.exists():
        import pandas as pd

        hier = pd.read_csv(hier_path)
        root = hier[hier["depth"] == 0]
        if not root.empty:
            total_cells = int(root["count"].values[0])
            print(f"  Total cells : {total_cells:,}")
            print(f"  Regions     : {len(hier)}")
            d2 = hier[hier["depth"] == 2].sort_values("count", ascending=False).head(5)
            for _, r in d2.iterrows():
                pct = r["count"] / total_cells * 100
                print(
                    f"    {r['acronym']:<8} {r['region_name'][:30]:<30}  {int(r['count']):>8,}  ({pct:.1f}%)"
                )
    print("\nDone. Open http://127.0.0.1:8787 and go to QC tab to view results.\n")


if __name__ == "__main__":
    main()
