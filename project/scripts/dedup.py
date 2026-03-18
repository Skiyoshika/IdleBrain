from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.spatial import cKDTree


def _ensure_um_columns(cells: pd.DataFrame, pixel_size_um: float) -> pd.DataFrame:
    out = cells.copy()
    if "x_um" not in out.columns:
        out["x_um"] = out["x"] * pixel_size_um
    if "y_um" not in out.columns:
        out["y_um"] = out["y"] * pixel_size_um
    if "z_um" not in out.columns:
        out["z_um"] = out["slice_id"] * 1.0
    return out


def apply_dedup_kdtree(
    cells: pd.DataFrame,
    *,
    neighbor_slices: int = 1,
    pixel_size_um: float = 1.0,
    slice_spacing_um: float = 25.0,
    r_xy_um: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    KDTree anisotropic dedup.
    Keeps highest-score point among neighbors where:
      (dx/r_xy)^2 + (dy/r_xy)^2 + (dz/r_z)^2 < 1
    and |slice_id_i - slice_id_j| <= neighbor_slices.
    """
    if cells.empty:
        stats = pd.DataFrame(
            [
                {
                    "before_count": 0,
                    "after_count": 0,
                    "merged_count": 0,
                    "merge_ratio": 0.0,
                    "method": "kdtree_anisotropic",
                    "neighbor_slices": neighbor_slices,
                    "r_xy_um": r_xy_um,
                    "r_z_um": slice_spacing_um * 0.5,
                }
            ]
        )
        return cells.copy(), stats

    r_z_um = slice_spacing_um * 0.5
    df = _ensure_um_columns(cells, pixel_size_um)

    order = df["score"].fillna(0).sort_values(ascending=False).index.to_list()
    coords = df[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)

    # Scale coordinates to unit sphere threshold radius 1
    scaled = coords.copy()
    scaled[:, 0] /= max(r_xy_um, 1e-6)
    scaled[:, 1] /= max(r_xy_um, 1e-6)
    scaled[:, 2] /= max(r_z_um, 1e-6)
    tree = cKDTree(scaled)

    removed: set[int] = set()
    keep_indices: list[int] = []

    for idx in order:
        if idx in removed:
            continue
        keep_indices.append(idx)
        nbrs = tree.query_ball_point(scaled[df.index.get_loc(idx)], r=1.0)
        base_slice = int(df.loc[idx, "slice_id"])
        for npos in nbrs:
            nidx = int(df.index[npos])
            if nidx == idx or nidx in removed:
                continue
            if abs(int(df.loc[nidx, "slice_id"]) - base_slice) <= neighbor_slices:
                removed.add(nidx)

    deduped = (
        df.loc[sorted(keep_indices)].sort_values(["slice_id", "cell_id"]).reset_index(drop=True)
    )
    before = len(df)
    after = len(deduped)
    stats = pd.DataFrame(
        [
            {
                "before_count": before,
                "after_count": after,
                "merged_count": before - after,
                "merge_ratio": (before - after) / before if before else 0.0,
                "method": "kdtree_anisotropic",
                "neighbor_slices": neighbor_slices,
                "r_xy_um": r_xy_um,
                "r_z_um": r_z_um,
            }
        ]
    )
    return deduped, stats


def write_dedup_stats(stats: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "dedup_stats.csv", index=False)
