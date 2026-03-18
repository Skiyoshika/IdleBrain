from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tifffile import imread


def export_slice_qc(cells_mapped_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cells_mapped_csv)
    grp = df.groupby("slice_id", as_index=False).size().rename(columns={"size": "cell_count"})
    grp["status"] = grp["cell_count"].map(lambda n: "ok" if n > 0 else "empty")
    grp.to_csv(out_csv, index=False)
    return grp


def export_overlays(
    sample_input_dir: Path, cells_csv: Path, out_dir: Path, max_images: int = 5
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(cells_csv)
    files = sorted(sample_input_dir.glob("*.tif"))[:max_images]
    for i, fp in enumerate(files):
        img = imread(str(fp))
        if img.ndim == 3:
            img = img[..., 0]
        sid = i
        sub = cells[cells["slice_id"] == sid]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap="gray")
        if not sub.empty:
            ax.scatter(sub["x"], sub["y"], s=8, c="lime")
        ax.set_title(f"slice={sid}, n={len(sub)}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_{sid:03d}.png", dpi=120)
        plt.close(fig)
