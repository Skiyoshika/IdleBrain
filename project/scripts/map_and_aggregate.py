from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.structure_tree import load_structure_table, parse_structure_id_path
except Exception:
    from structure_tree import load_structure_table, parse_structure_id_path


def map_cells_to_regions(cells_csv: Path, atlas_map_csv: Path) -> pd.DataFrame:
    """
    MVP placeholder mapping by nearest slice-level lookup.
    Expected cells_csv cols: cell_id,slice_id,x,y,score
    Expected atlas_map_csv cols: slice_id,region_id,region_name,hemisphere
    """
    cells = pd.read_csv(cells_csv)
    atlas = pd.read_csv(atlas_map_csv)
    merged = cells.merge(atlas, on="slice_id", how="left")
    merged["region_id"] = merged["region_id"].fillna(0).astype(int)
    merged["region_name"] = merged["region_name"].fillna("OUTSIDE")
    merged["hemisphere"] = merged["hemisphere"].fillna("unknown")
    return merged


def aggregate_by_region(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mapped.empty:
        empty_leaf = pd.DataFrame(
            columns=["region_id", "region_name", "acronym", "hemisphere", "count", "confidence"]
        )
        empty_hierarchy = pd.DataFrame(
            columns=[
                "region_id",
                "region_name",
                "acronym",
                "parent_structure_id",
                "hemisphere",
                "depth",
                "count",
                "confidence",
            ]
        )
        return empty_leaf, empty_hierarchy

    total_cells = max(int(len(mapped)), 1)

    # Base count aggregation
    leaf = (
        mapped.groupby(["region_id", "region_name", "acronym", "hemisphere"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    leaf["confidence"] = leaf["count"].astype(float) / float(total_cells)

    # Garwood Poisson confidence intervals (95%)
    # ci_low = 0.5 * chi2(2*n, alpha/2), ci_high = 0.5 * chi2(2*(n+1), 1-alpha/2)
    # Approximated via inverse-CDF on Poisson: ppf(0.025, n) and ppf(0.975, n+1)
    def _poisson_ci_low(n: int) -> float:
        if n == 0:
            return 0.0
        # Wilson-Hilferty normal approximation for lower bound
        lam = float(n)
        z = 1.96
        return max(0.0, lam * (1 - 1 / (9 * lam) - z / (3 * np.sqrt(lam))) ** 3)

    def _poisson_ci_high(n: int) -> float:
        lam = float(n) + 1.0
        z = 1.96
        return lam * (1 - 1 / (9 * lam) + z / (3 * np.sqrt(lam))) ** 3

    leaf["ci_low"] = leaf["count"].apply(_poisson_ci_low).round(1)
    leaf["ci_high"] = leaf["count"].apply(_poisson_ci_high).round(1)

    # Optional morphology aggregation (if columns present)
    _morph_cols = [c for c in ("area_px", "elongation", "mean_intensity") if c in mapped.columns]
    if _morph_cols:
        _agg_fns = {c: "mean" for c in _morph_cols}
        _morph = (
            mapped.groupby(["region_id", "hemisphere"], as_index=False)
            .agg(_agg_fns)
            .rename(columns={c: f"mean_{c}" for c in _morph_cols})
        )
        leaf = leaf.merge(_morph, on=["region_id", "hemisphere"], how="left")

    if "structure_source" not in mapped.columns:
        empty_hierarchy = pd.DataFrame(
            columns=[
                "region_id",
                "region_name",
                "acronym",
                "parent_structure_id",
                "hemisphere",
                "depth",
                "count",
                "confidence",
            ]
        )
        return leaf, empty_hierarchy
    structure_sources = [
        x for x in mapped["structure_source"].dropna().astype(str).unique().tolist() if x
    ]
    if not structure_sources:
        empty_hierarchy = pd.DataFrame(
            columns=[
                "region_id",
                "region_name",
                "acronym",
                "parent_structure_id",
                "hemisphere",
                "depth",
                "count",
                "confidence",
            ]
        )
        return leaf, empty_hierarchy

    structure_df = load_structure_table(Path(structure_sources[0]))
    structure_meta = structure_df[
        ["id", "name", "acronym", "parent_structure_id", "depth", "graph_order"]
    ].rename(columns={"id": "region_id", "name": "region_name"})

    hierarchy_rows: list[dict] = []
    for row in mapped[["structure_id_path", "hemisphere"]].itertuples(index=False):
        for region_id in parse_structure_id_path(row.structure_id_path):
            hierarchy_rows.append({"region_id": int(region_id), "hemisphere": row.hemisphere})

    hierarchy = pd.DataFrame(hierarchy_rows)
    if hierarchy.empty:
        hierarchy = pd.DataFrame(
            columns=[
                "region_id",
                "region_name",
                "acronym",
                "parent_structure_id",
                "hemisphere",
                "depth",
                "count",
                "confidence",
            ]
        )
    else:
        hierarchy = (
            hierarchy.groupby(["region_id", "hemisphere"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        hierarchy = hierarchy.merge(structure_meta, on="region_id", how="left")
        hierarchy["confidence"] = hierarchy["count"].astype(float) / float(total_cells)
        hierarchy["ci_low"] = hierarchy["count"].apply(_poisson_ci_low).round(1)
        hierarchy["ci_high"] = hierarchy["count"].apply(_poisson_ci_high).round(1)
        hierarchy = hierarchy.sort_values(["depth", "region_id", "hemisphere"]).reset_index(
            drop=True
        )

    leaf = (
        leaf.merge(
            structure_meta[["region_id", "parent_structure_id", "depth", "graph_order"]],
            on="region_id",
            how="left",
        )
        .sort_values(["graph_order", "region_id", "hemisphere"])
        .reset_index(drop=True)
    )

    return leaf, hierarchy


def write_outputs(leaf: pd.DataFrame, hierarchy: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    leaf.to_csv(out_dir / "cell_counts_leaf.csv", index=False)
    hierarchy.to_csv(out_dir / "cell_counts_hierarchy.csv", index=False)
