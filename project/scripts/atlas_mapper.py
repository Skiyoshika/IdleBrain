from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    from scripts.structure_tree import load_structure_table
except Exception:
    from structure_tree import load_structure_table


def _to_voxel(x_um: float, y_um: float, z_um: float, voxel_um: float) -> tuple[int, int, int]:
    return int(round(x_um / voxel_um)), int(round(y_um / voxel_um)), int(round(z_um / voxel_um))


def _normalize_region_id(region_id: int) -> int:
    rid = int(region_id)
    return 0 if rid == 997 else rid


def _attach_structure_metadata(out: pd.DataFrame, structure_source: Path) -> pd.DataFrame:
    st = load_structure_table(Path(structure_source))
    st = st[
        [
            "id",
            "name",
            "acronym",
            "parent_structure_id",
            "structure_id_path",
            "color_hex_triplet",
            "depth",
            "graph_order",
            "hemisphere_id",
        ]
    ].copy()
    st = st.rename(columns={"id": "region_id", "name": "region_name"})
    out = out.merge(st, on="region_id", how="left")

    name_by_id = st.set_index("region_id")["region_name"].to_dict()
    out["region_name"] = out["region_name"].fillna(
        out["region_id"].map(
            lambda rid: "OUTSIDE_ATLAS" if int(rid) == 0 else f"UNKNOWN_REGION_{int(rid)}"
        )
    )
    out["acronym"] = out["acronym"].fillna(
        out["region_id"].map(lambda rid: "OUT" if int(rid) == 0 else f"RID_{int(rid)}")
    )
    out["parent_name"] = out["parent_structure_id"].map(name_by_id).fillna("")
    out["hemisphere"] = (
        out["hemisphere_id"].map({1: "left", 2: "right", 3: "bilateral"}).fillna("unknown")
    )
    out["structure_source"] = str(Path(structure_source).resolve())
    return out


def map_cells_with_registered_label_slice(
    cells: pd.DataFrame,
    *,
    registered_label_tif: Path,
    structure_csv: Path,
    atlas_slice_index: int,
    slicing_plane: str = "coronal",
    registration_score: float | None = None,
    registration_method: str = "registered_slice_label",
) -> pd.DataFrame:
    if not Path(registered_label_tif).exists():
        raise FileNotFoundError(f"registered label slice not found: {registered_label_tif}")
    if not Path(structure_csv).exists():
        raise FileNotFoundError(f"structure ontology source not found: {structure_csv}")

    label = imread(str(registered_label_tif))
    if label.ndim == 3:
        label = label[..., 0]
    label = label.astype(np.int32, copy=False)
    h, w = label.shape[:2]

    out = cells.copy()
    region_ids: list[int] = []
    mapping_status: list[str] = []
    for _, row in out.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        if 0 <= x < w and 0 <= y < h:
            rid = _normalize_region_id(int(label[y, x]))
            region_ids.append(rid)
            mapping_status.append("ok" if rid > 0 else "outside_registered_slice")
        else:
            region_ids.append(0)
            mapping_status.append("outside_slice_bounds")

    out["region_id"] = region_ids
    out["mapping_status"] = mapping_status
    out["atlas_slice_index"] = int(atlas_slice_index)
    out["slicing_plane"] = str(slicing_plane)
    out["registration_method"] = str(registration_method)
    out["registered_label_path"] = str(Path(registered_label_tif).resolve())
    if registration_score is not None:
        out["registration_score"] = float(registration_score)
    return _attach_structure_metadata(out, structure_csv)


def map_cells_with_label_volume(
    cells: pd.DataFrame,
    *,
    label_nii: Path | None,
    structure_csv: Path | None,
    pixel_size_um: float,
    slice_spacing_um: float,
    atlas_voxel_um: float = 25.0,
) -> pd.DataFrame:
    """
    Atlas label-volume mapping interface.
    This function is intentionally fail-fast: missing atlas assets should stop the run.
    """
    if label_nii is None or not Path(label_nii).exists():
        raise FileNotFoundError(
            "registered atlas label volume not found; generate registration outputs before region mapping"
        )
    if structure_csv is None or not Path(structure_csv).exists():
        raise FileNotFoundError(
            "structure ontology source not found; provide a CSV or JSON structure table"
        )
    if nib is None:
        raise RuntimeError("nibabel is required for atlas label volume mapping")

    out = cells.copy()
    out["x_um"] = out["x"] * pixel_size_um
    out["y_um"] = out["y"] * pixel_size_um
    out["z_um"] = out["slice_id"] * slice_spacing_um

    img = nib.load(str(label_nii))
    data = np.asarray(img.get_fdata(), dtype=np.int32)
    sx, sy, sz = data.shape

    region_ids: list[int] = []
    atlas_x: list[int] = []
    atlas_y: list[int] = []
    atlas_z: list[int] = []
    mapping_status: list[str] = []
    for _, r in out.iterrows():
        vx, vy, vz = _to_voxel(float(r.x_um), float(r.y_um), float(r.z_um), atlas_voxel_um)
        atlas_x.append(vx)
        atlas_y.append(vy)
        atlas_z.append(vz)
        if 0 <= vx < sx and 0 <= vy < sy and 0 <= vz < sz:
            rid = _normalize_region_id(int(data[vx, vy, vz]))
            region_ids.append(rid)
            mapping_status.append("ok" if rid > 0 else "outside_atlas")
        else:
            region_ids.append(0)
            mapping_status.append("outside_volume")

    out["atlas_x_voxel"] = atlas_x
    out["atlas_y_voxel"] = atlas_y
    out["atlas_z_voxel"] = atlas_z
    out["region_id"] = region_ids
    out["mapping_status"] = mapping_status

    return _attach_structure_metadata(out, Path(structure_csv))
