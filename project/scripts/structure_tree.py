from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd


def parse_structure_id_path(value: str | float | int | None) -> list[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p for p in text.strip("/").split("/") if p]
    out: list[int] = []
    for part in parts:
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


@lru_cache(maxsize=8)
def _load_structure_table_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"structure source not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"structure json must contain an object: {path}")
        rows = []
        for raw_id, info in data.items():
            if not isinstance(info, dict):
                continue
            try:
                rid = int(raw_id)
            except Exception:
                continue
            rows.append(
                {
                    "id": rid,
                    "name": str(info.get("name", "")).strip(),
                    "acronym": str(info.get("acronym", "")).strip(),
                    "color_hex_triplet": str(info.get("color", "")).strip(),
                    "parent_structure_id": pd.NA,
                    "structure_id_path": f"/{rid}/",
                    "depth": 0,
                    "graph_order": rid,
                    "hemisphere_id": pd.NA,
                }
            )
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"unsupported structure source: {path}")

    rename_map = {
        "safe_name": "safe_name",
        "name": "name",
        "acronym": "acronym",
        "color": "color_hex_triplet",
        "color_hex_triplet": "color_hex_triplet",
        "parent": "parent_structure_id",
        "parent_id": "parent_structure_id",
        "parent_structure_id": "parent_structure_id",
        "structure_id_path": "structure_id_path",
        "depth": "depth",
        "graph_order": "graph_order",
        "hemisphere_id": "hemisphere_id",
        "id": "id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required_defaults = {
        "id": pd.Series(dtype="Int64"),
        "name": pd.Series(dtype="string"),
        "acronym": pd.Series(dtype="string"),
        "color_hex_triplet": pd.Series(dtype="string"),
        "parent_structure_id": pd.Series(dtype="Int64"),
        "structure_id_path": pd.Series(dtype="string"),
        "depth": pd.Series(dtype="Int64"),
        "graph_order": pd.Series(dtype="Int64"),
        "hemisphere_id": pd.Series(dtype="Int64"),
    }
    for col, empty in required_defaults.items():
        if col not in df.columns:
            df[col] = empty

    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(int)
    df["parent_structure_id"] = pd.to_numeric(df["parent_structure_id"], errors="coerce").astype(
        "Int64"
    )
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce").fillna(0).astype(int)
    df["graph_order"] = (
        pd.to_numeric(df["graph_order"], errors="coerce").fillna(df["id"]).astype(int)
    )
    df["hemisphere_id"] = pd.to_numeric(df["hemisphere_id"], errors="coerce").astype("Int64")
    df["name"] = df["name"].fillna("").astype(str)
    df["acronym"] = df["acronym"].fillna("").astype(str)
    df["color_hex_triplet"] = df["color_hex_triplet"].fillna("").astype(str)
    df["structure_id_path"] = (
        df["structure_id_path"].fillna(df["id"].map(lambda x: f"/{x}/")).astype(str)
    )
    return df.sort_values(["graph_order", "id"]).reset_index(drop=True)


def load_structure_table(path: Path | str) -> pd.DataFrame:
    return _load_structure_table_cached(str(Path(path).resolve())).copy()
