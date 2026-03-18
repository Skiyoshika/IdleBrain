from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
from skimage import morphology


def finalize_registered_label(
    label: np.ndarray,
    *,
    tissue_mask: np.ndarray | None,
    mode: str,
    edge_smooth_iter: int,
    warp_params: dict | None,
    cleanup_label_topology: Callable[..., np.ndarray],
    smooth_label_edges: Callable[..., np.ndarray],
    gaussian_vote_boundary_smooth: Callable[..., np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    timings_ms: dict[str, float] = {}
    out = label.astype(np.int32, copy=False)

    t0 = perf_counter()
    if tissue_mask is not None:
        tissue_clip = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(3)).astype(
            bool
        )
        out = np.where(tissue_clip, out, 0).astype(np.int32)
    timings_ms["clip_to_tissue"] = float((perf_counter() - t0) * 1000.0)

    t0 = perf_counter()
    cleanup_params = dict(warp_params or {})
    if str(mode or "fill").lower() != "fill":
        cleanup_params["cleanup_fill_outer_missing"] = False
    out = cleanup_label_topology(
        out.astype(np.int32),
        tissue_mask=tissue_mask,
        warp_params=cleanup_params,
    )
    timings_ms["cleanup_topology"] = float((perf_counter() - t0) * 1000.0)

    t0 = perf_counter()
    out = smooth_label_edges(
        out.astype(np.int32),
        tissue_mask=tissue_mask,
        iterations=max(0, int(edge_smooth_iter)),
    )
    timings_ms["smooth_edges"] = float((perf_counter() - t0) * 1000.0)

    t0 = perf_counter()
    if str(mode or "fill").lower() in ("fill",):
        out = gaussian_vote_boundary_smooth(
            out.astype(np.int32),
            tissue_mask=tissue_mask,
            warp_params=warp_params,
        )
    timings_ms["gaussian_vote_fill"] = float((perf_counter() - t0) * 1000.0)
    timings_ms["total"] = float(sum(timings_ms.values()))
    return out.astype(np.int32), timings_ms
