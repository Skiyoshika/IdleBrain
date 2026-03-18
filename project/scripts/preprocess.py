from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import imread, imwrite


def merge_every_n_slices(input_files: list[Path], out_dir: Path, n: int = 5) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for i in range(0, len(input_files), n):
        chunk = input_files[i : i + n]
        if not chunk:
            continue
        imgs = [imread(str(p)).astype(np.float32) for p in chunk]
        merged = np.mean(np.stack(imgs, axis=0), axis=0)
        out = out_dir / f"merged_{i // n:04d}.tif"
        imwrite(str(out), merged.astype(np.uint16))
        outputs.append(out)
    return outputs
