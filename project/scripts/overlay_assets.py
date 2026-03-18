from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from scripts.allen_colors import load_allen_color_map
except Exception:
    from allen_colors import load_allen_color_map


@lru_cache(maxsize=1)
def load_structure_tree() -> dict:
    candidates = [
        Path(__file__).resolve().parent.parent / "configs" / "allen_structure_tree.json",
        Path(__file__).resolve().parent / "allen_structure_tree.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return {}


def _deterministic_color(region_id: int) -> tuple[int, int, int]:
    rng = (int(region_id) * 2654435761) & 0xFFFFFF
    return ((rng >> 16) & 0xFF, (rng >> 8) & 0xFF, rng & 0xFF)


def colorize_label(label: np.ndarray, structure_csv: Path | None = None) -> np.ndarray:
    tree = load_structure_tree()
    csv_colors = load_allen_color_map(Path(structure_csv)) if structure_csv is not None else {}

    colored = np.zeros((*label.shape, 3), dtype=np.uint8)
    for rid in np.unique(label.astype(np.int32)):
        if int(rid) == 0:
            continue

        color = None
        info = tree.get(str(int(rid)))
        if info and info.get("color") and len(str(info["color"])) == 6:
            hx = str(info["color"])
            color = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
        elif int(rid) in csv_colors:
            color = tuple(int(v) for v in csv_colors[int(rid)])
        else:
            color = _deterministic_color(int(rid))

        colored[label == rid] = np.array(color, dtype=np.uint8)
    return colored
