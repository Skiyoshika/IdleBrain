from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


def load_allen_color_map(structure_csv: Path) -> dict[int, tuple[int, int, int]]:
    """Load Allen-like color mapping from CSV if rgb columns exist.
    Falls back to deterministic palette elsewhere in renderer.
    """
    return _load_allen_color_map_cached(str(Path(structure_csv).resolve()))


@lru_cache(maxsize=8)
def _load_allen_color_map_cached(structure_csv: str) -> dict[int, tuple[int, int, int]]:
    path = Path(structure_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("region_id")
    r_col = cols.get("r") or cols.get("red")
    g_col = cols.get("g") or cols.get("green")
    b_col = cols.get("b") or cols.get("blue")
    hex_col = cols.get("color_hex_triplet")
    if not id_col:
        return {}

    cmap: dict[int, tuple[int, int, int]] = {}
    for _, row in df.iterrows():
        try:
            rid = int(row[id_col])
            if r_col and g_col and b_col:
                cmap[rid] = (int(row[r_col]), int(row[g_col]), int(row[b_col]))
            elif hex_col:
                hx = str(row[hex_col]).strip().lstrip("#")
                if len(hx) == 6:
                    cmap[rid] = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
        except Exception:
            continue
    return cmap
