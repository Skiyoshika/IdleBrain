from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import cKDTree
from skimage import exposure, feature, filters, measure, morphology


@dataclass(slots=True)
class LaplacianConfig:
    axis: int = 0
    rtol: float = 1e-2
    maxiter: int = 1000
    normal_radius: int = 3
    degree_thresh_deg: float = 5.0
    distance_neighbours: int = 30
    template_min_component: int = 25
    data_min_component: int = 100
    print_every: int = 10
    chunk_slices: int = 16


@dataclass(slots=True)
class LaplacianResult:
    final_registered_path: Path
    deformation_field_path: Path
    fpoints_path: Path
    mpoints_path: Path
    boundary_csv_path: Path
    boundary_npy_path: Path
    correspondences: int
    unique_boundary_voxels: int
    solve_seconds: float
    residual_history_u: list[float]
    residual_history_v: list[float]


def _norm01(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _cleanup_edges(edges: np.ndarray, min_component: int) -> np.ndarray:
    labeled = measure.label(edges)
    if labeled.max() <= 0:
        return edges.astype(bool, copy=False)
    out = edges.astype(bool, copy=True)
    for label in range(1, int(labeled.max()) + 1):
        if int(np.sum(labeled == label)) < min_component:
            out[labeled == label] = False
    return morphology.thin(out)


def get_template_contours(section: np.ndarray, min_component: int) -> tuple[np.ndarray, np.ndarray]:
    section = _norm01(section)
    thresh = filters.threshold_otsu(section)
    binary = section > thresh
    edges = feature.canny(binary.astype(np.float32), sigma=3.0)
    edges = _cleanup_edges(edges, min_component=min_component)
    return edges, binary


def get_data_contours(section: np.ndarray, min_component: int) -> tuple[np.ndarray, np.ndarray]:
    section = np.asarray(section, dtype=np.float32)
    section = np.clip(section, 0.0, float(np.percentile(section, 99.5)))
    section = exposure.equalize_adapthist(_norm01(section))
    thresh = filters.threshold_otsu(section)
    binary = section > thresh
    edges = feature.canny(binary.astype(np.float32), sigma=3.0)
    edges = _cleanup_edges(edges, min_component=min_component)
    return edges, binary


def estimate_normal(point: np.ndarray, neighbours: np.ndarray) -> np.ndarray | None:
    centroid = np.mean(neighbours, axis=0)
    centered = neighbours - centroid
    point_centered = point - centroid
    try:
        vh = np.linalg.svd(centered - point_centered, full_matrices=False)[-1]
    except np.linalg.LinAlgError:
        return None
    return np.asarray(vh[-1], dtype=np.float32)


def orient_2d_normals(
    points: np.ndarray, normals: np.ndarray, section: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, normals

    section_flat = np.asarray(section, dtype=np.float32).ravel()
    height = int(section.shape[0])
    width = int(section.shape[1])
    keep = np.ones(len(points), dtype=bool)

    def _flat_index(coords: np.ndarray) -> np.ndarray:
        rows = np.clip(coords[:, 0].astype(int), 0, height - 1)
        cols = np.clip(coords[:, 1].astype(int), 0, width - 1)
        return rows * width + cols

    for sign in (1.0, -1.0):
        probe = points + (9.0 * sign) * normals
        keep &= probe[:, 0] >= 0
        keep &= probe[:, 0] < section.shape[0]
        keep &= probe[:, 1] >= 0
        keep &= probe[:, 1] < section.shape[1]

    points = points[keep]
    normals = normals[keep]
    if len(points) == 0:
        return points, normals

    left_sum = np.zeros(len(points), dtype=np.float32)
    right_sum = np.zeros(len(points), dtype=np.float32)
    for step in range(1, 10):
        probe_left = np.round(points + step * normals).astype(int)
        probe_right = np.round(points - step * normals).astype(int)
        left_sum += section_flat[_flat_index(probe_left)]
        right_sum += section_flat[_flat_index(probe_right)]

    direction = np.ones(len(points), dtype=np.float32)
    direction[left_sum >= right_sum] = -1.0
    normals = normals * direction[:, None]
    return points, normals


def estimate_2d_normals(
    points: np.ndarray,
    binary_section: np.ndarray,
    *,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points.astype(np.float32), np.zeros((0, 2), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)
    normals = np.zeros(points.shape, dtype=np.float32)
    tree = cKDTree(points)

    valid = np.ones(len(points), dtype=bool)
    for idx, point in enumerate(points):
        neighbour_ids = tree.query_ball_point(point, radius)
        if len(neighbour_ids) < 4:
            valid[idx] = False
            continue
        normal = estimate_normal(point, points[np.asarray(neighbour_ids, dtype=int)])
        if normal is None or normal.shape != (2,):
            valid[idx] = False
            continue
        normals[idx] = normal

    points = points[valid]
    normals = normals[valid]
    if len(points) == 0:
        return points, normals
    return orient_2d_normals(points, normals, binary_section)


def _get_correspondence(
    source_point: np.ndarray,
    source_normal: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    tree: cKDTree,
    *,
    degree_thresh_deg: float,
    distance_neighbours: int,
) -> int:
    if len(target_points) == 0:
        return -1

    k = int(min(max(1, distance_neighbours), len(target_points)))
    distances, indices = tree.query(source_point, k=k)
    distances = np.atleast_1d(distances)
    indices = np.atleast_1d(indices)
    finite = np.isfinite(distances)
    if not np.any(finite):
        return -1
    distances = distances[finite]
    indices = indices[finite]
    if len(indices) == 0:
        return -1

    cutoff = np.percentile(distances, 90.0)
    keep = distances <= cutoff
    indices = indices[keep]
    if len(indices) == 0:
        return -1

    similarity = target_normals[indices] @ source_normal
    degree_thresh = np.cos(np.deg2rad(degree_thresh_deg))
    valid = np.where(similarity >= degree_thresh)[0]
    if len(valid) == 0:
        return -1
    return int(indices[valid[0]])


def get_2d_correspondences(
    fixed_edges: np.ndarray,
    moving_edges: np.ndarray,
    fixed_binary: np.ndarray,
    moving_binary: np.ndarray,
    *,
    cfg: LaplacianConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mpoints, mnormals = estimate_2d_normals(
        np.column_stack(np.nonzero(moving_edges)),
        moving_binary,
        radius=cfg.normal_radius,
    )
    fpoints, fnormals = estimate_2d_normals(
        np.column_stack(np.nonzero(fixed_edges)),
        fixed_binary,
        radius=cfg.normal_radius,
    )

    if len(fpoints) == 0 or len(mpoints) == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)

    moving_tree = cKDTree(mpoints)
    correspondences = np.array(
        [
            _get_correspondence(
                point,
                fnormals[idx],
                mpoints,
                mnormals,
                moving_tree,
                degree_thresh_deg=cfg.degree_thresh_deg,
                distance_neighbours=cfg.distance_neighbours,
            )
            for idx, point in enumerate(fpoints)
        ],
        dtype=np.int32,
    )
    valid = correspondences >= 0
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)

    fixed_pts = np.round(fpoints[valid]).astype(np.int32)
    moving_pts = np.round(mpoints[correspondences[valid]]).astype(np.int32)
    disp = moving_pts - fixed_pts
    keep = np.ones(len(fixed_pts), dtype=bool)
    for col in range(2):
        abs_disp = np.abs(disp[:, col])
        cutoff = max(10.0, float(np.percentile(abs_disp, 90.0)))
        keep &= abs_disp < cutoff

    fixed_pts = fixed_pts[keep]
    moving_pts = moving_pts[keep]
    return fixed_pts, moving_pts


def _slice_to_volume_points(points_2d: np.ndarray, slice_index: int, axis: int) -> np.ndarray:
    if len(points_2d) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    slice_column = np.full((len(points_2d), 1), int(slice_index), dtype=np.int32)
    if axis == 0:
        return np.hstack([slice_column, points_2d.astype(np.int32)])
    if axis == 1:
        return np.column_stack([points_2d[:, 0], slice_column[:, 0], points_2d[:, 1]]).astype(
            np.int32
        )
    if axis == 2:
        return np.column_stack([points_2d, slice_column[:, 0]]).astype(np.int32)
    raise ValueError(f"unsupported axis: {axis}")


def find_volume_correspondences(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    cfg: LaplacianConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if fixed.shape != moving.shape:
        raise ValueError(f"shape mismatch: fixed={fixed.shape}, moving={moving.shape}")

    fpoints_all: list[np.ndarray] = []
    mpoints_all: list[np.ndarray] = []
    total_slices = fixed.shape[cfg.axis]
    print(f"  Finding slice-to-slice correspondences for {total_slices} slices...")

    for slice_index in range(total_slices):
        fixed_section = np.take(fixed, slice_index, axis=cfg.axis)
        moving_section = np.take(moving, slice_index, axis=cfg.axis)
        fixed_edges, fixed_binary = get_template_contours(fixed_section, cfg.template_min_component)
        moving_edges, moving_binary = get_data_contours(moving_section, cfg.data_min_component)
        fixed_pts, moving_pts = get_2d_correspondences(
            fixed_edges,
            moving_edges,
            fixed_binary,
            moving_binary,
            cfg=cfg,
        )
        if len(fixed_pts) == 0:
            continue
        fpoints_all.append(_slice_to_volume_points(fixed_pts, slice_index, cfg.axis))
        mpoints_all.append(_slice_to_volume_points(moving_pts, slice_index, cfg.axis))
        if (slice_index + 1) % 50 == 0 or slice_index + 1 == total_slices:
            print(f"    processed {slice_index + 1}/{total_slices} slices")

    if not fpoints_all:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.int32)
    return np.vstack(fpoints_all), np.vstack(mpoints_all)


def _aggregate_unique_correspondences(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear = np.ravel_multi_index(fixed_points.T, shape)
    unique_linear, inverse = np.unique(linear, return_inverse=True)
    fixed_unique = np.column_stack(np.unravel_index(unique_linear, shape)).astype(np.int32)
    moving_sum = np.zeros((len(unique_linear), 3), dtype=np.float32)
    counts = np.zeros(len(unique_linear), dtype=np.float32)
    np.add.at(moving_sum, inverse, moving_points.astype(np.float32))
    np.add.at(counts, inverse, 1.0)
    moving_unique = np.round(moving_sum / counts[:, None]).astype(np.int32)
    return fixed_unique, moving_unique, unique_linear.astype(np.int64)


class WeightedDirichletLaplacian:
    def __init__(
        self,
        shape: tuple[int, int, int],
        boundary_mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        self.shape = tuple(int(v) for v in shape)
        self.boundary_mask = np.asarray(boundary_mask, dtype=bool)
        self.free_mask = ~self.boundary_mask
        self.spacing = tuple(float(v) for v in spacing)
        self.weights = tuple(np.float32(1.0 / max(v * v, 1e-12)) for v in self.spacing)
        self.size = int(np.prod(self.shape))
        self._pairs = tuple(
            self._make_axis_pair(axis, weight) for axis, weight in enumerate(self.weights)
        )
        self.diagonal = self._build_diagonal()
        self.operator = LinearOperator(
            shape=(self.size, self.size),
            matvec=self._matvec,
            dtype=np.float32,
        )
        inv_diag = np.where(self.diagonal > 1e-8, 1.0 / self.diagonal, 1.0).astype(np.float32)
        self.preconditioner = LinearOperator(
            shape=(self.size, self.size),
            matvec=lambda x: inv_diag * np.asarray(x, dtype=np.float32),
            dtype=np.float32,
        )

    def _make_axis_pair(
        self, axis: int, weight: np.float32
    ) -> tuple[tuple[slice, ...], tuple[slice, ...], np.float32]:
        head = [slice(None)] * 3
        tail = [slice(None)] * 3
        head[axis] = slice(1, None)
        tail[axis] = slice(None, -1)
        return tuple(head), tuple(tail), weight

    def _build_diagonal(self) -> np.ndarray:
        diag = self.boundary_mask.astype(np.float32)
        for head, tail, weight in self._pairs:
            free_head = self.free_mask[head]
            free_tail = self.free_mask[tail]
            diag_head = diag[head]
            diag_tail = diag[tail]
            diag_head[free_head] += weight
            diag_tail[free_tail] += weight
        return diag.ravel()

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x3 = np.asarray(x, dtype=np.float32).reshape(self.shape)
        y = np.zeros(self.shape, dtype=np.float32)
        y[self.boundary_mask] = x3[self.boundary_mask]

        for head, tail, weight in self._pairs:
            x_head = x3[head]
            x_tail = x3[tail]
            y_head = y[head]
            y_tail = y[tail]
            free_head = self.free_mask[head]
            free_tail = self.free_mask[tail]

            free_free = free_head & free_tail
            if np.any(free_free):
                diff = (x_head - x_tail) * weight
                y_head[free_free] += diff[free_free]
                y_tail[free_free] -= diff[free_free]

            free_boundary = free_head & (~free_tail)
            if np.any(free_boundary):
                y_head[free_boundary] += weight * x_head[free_boundary]

            boundary_free = (~free_head) & free_tail
            if np.any(boundary_free):
                y_tail[boundary_free] += weight * x_tail[boundary_free]

        return y.ravel()

    def build_rhs(self, boundary_values: np.ndarray) -> np.ndarray:
        rhs = np.zeros(self.shape, dtype=np.float32)
        rhs[self.boundary_mask] = boundary_values[self.boundary_mask]
        for head, tail, weight in self._pairs:
            rhs_head = rhs[head]
            rhs_tail = rhs[tail]
            values_head = boundary_values[head]
            values_tail = boundary_values[tail]
            free_head = self.free_mask[head]
            free_tail = self.free_mask[tail]

            free_boundary = free_head & (~free_tail)
            if np.any(free_boundary):
                rhs_head[free_boundary] += weight * values_tail[free_boundary]

            boundary_free = (~free_head) & free_tail
            if np.any(boundary_free):
                rhs_tail[boundary_free] += weight * values_head[boundary_free]
        return rhs.ravel()


def _solve_field(
    laplacian: WeightedDirichletLaplacian,
    rhs: np.ndarray,
    x0: np.ndarray,
    *,
    label: str,
    rtol: float,
    maxiter: int,
    print_every: int,
) -> tuple[np.ndarray, list[float]]:
    rhs = np.asarray(rhs, dtype=np.float32)
    x0 = np.asarray(x0, dtype=np.float32)
    rhs_norm = max(float(np.linalg.norm(rhs)), 1e-12)
    started = time.time()
    residual_history: list[float] = []
    iteration = 0

    def callback(xk: np.ndarray) -> None:
        nonlocal iteration
        iteration += 1
        if iteration == 1 or iteration % max(1, print_every) == 0:
            resid = rhs - laplacian.operator.matvec(xk)
            rel = float(np.linalg.norm(resid) / rhs_norm)
            residual_history.append(rel)
            elapsed = time.time() - started
            print(
                f"    {label}: iter {iteration}/{maxiter}, rel_resid={rel:0.2e}, {elapsed:0.0f}s elapsed"
            )

    solution, info = cg(
        laplacian.operator,
        rhs,
        x0=x0,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
        M=laplacian.preconditioner,
        callback=callback,
    )
    final_resid = rhs - laplacian.operator.matvec(solution)
    residual_history.append(float(np.linalg.norm(final_resid) / rhs_norm))
    if info != 0:
        raise RuntimeError(f"Laplacian solve for {label} did not converge (info={info})")
    return solution.reshape(laplacian.shape), residual_history


def apply_deformation_field(
    volume: np.ndarray,
    deformation_field: np.ndarray,
    *,
    chunk_slices: int = 16,
) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    deformation_field = np.asarray(deformation_field, dtype=np.float32)
    if deformation_field.shape != (3,) + volume.shape:
        raise ValueError(
            f"expected deformation field shape {(3,) + volume.shape}, got {deformation_field.shape}"
        )

    out = np.zeros_like(volume, dtype=np.float32)
    y_grid = np.arange(volume.shape[1], dtype=np.float32)[None, :, None]
    z_grid = np.arange(volume.shape[2], dtype=np.float32)[None, None, :]
    for start in range(0, volume.shape[0], max(1, chunk_slices)):
        stop = min(volume.shape[0], start + max(1, chunk_slices))
        x_grid = np.arange(start, stop, dtype=np.float32)[:, None, None]
        coords = [
            x_grid + deformation_field[0, start:stop],
            y_grid + deformation_field[1, start:stop],
            z_grid + deformation_field[2, start:stop],
        ]
        out[start:stop] = ndimage.map_coordinates(volume, coords, order=1, mode="nearest")
    return out


def _save_boundary_conditions(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    *,
    csv_path: Path,
    npy_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack(
        [
            fixed_points[:, 0],
            fixed_points[:, 1],
            fixed_points[:, 2],
            moving_points[:, 0],
            moving_points[:, 1],
            moving_points[:, 2],
            moving_points[:, 1] - fixed_points[:, 1],
            moving_points[:, 2] - fixed_points[:, 2],
        ]
    ).astype(np.float32)
    np.save(npy_path, rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fx", "fy", "fz", "mx", "my", "mz", "disp_y", "disp_z"])
        writer.writerows(rows.tolist())


def refine_volume_with_laplacian(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    *,
    cfg: LaplacianConfig,
) -> LaplacianResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    params_dir = out_dir / "parameters"
    params_dir.mkdir(parents=True, exist_ok=True)

    fixed_img = nib.load(str(fixed_path))
    moving_img = nib.load(str(moving_path))
    fixed = np.asarray(fixed_img.dataobj, dtype=np.float32)
    moving = np.asarray(moving_img.dataobj, dtype=np.float32)
    shape = tuple(min(a, b) for a, b in zip(fixed.shape, moving.shape, strict=False))
    fixed = fixed[: shape[0], : shape[1], : shape[2]]
    moving = moving[: shape[0], : shape[1], : shape[2]]

    print(f"Laplacian refinement: Processing volume of shape {shape} along axis {cfg.axis}")
    started = time.time()
    fixed_points, moving_points = find_volume_correspondences(fixed, moving, cfg=cfg)
    if len(fixed_points) == 0:
        raise RuntimeError("Laplacian refinement found no valid correspondence points")

    fixed_unique, moving_unique, unique_linear = _aggregate_unique_correspondences(
        fixed_points, moving_points, shape
    )
    print(
        f"Laplacian refinement: Found {len(fixed_points)} correspondence points "
        f"({len(unique_linear)} unique boundary voxels)."
    )

    boundary_mask = np.zeros(shape, dtype=bool)
    boundary_mask.reshape(-1)[unique_linear] = True
    spacing = tuple(float(v) for v in fixed_img.header.get_zooms()[:3])
    print(f"  Using spacing-weighted stencil: {spacing}")
    laplacian = WeightedDirichletLaplacian(shape, boundary_mask, spacing)

    field_u = np.zeros(shape, dtype=np.float32)
    field_v = np.zeros(shape, dtype=np.float32)
    disp_u = moving_unique[:, 1].astype(np.float32) - fixed_unique[:, 1].astype(np.float32)
    disp_v = moving_unique[:, 2].astype(np.float32) - fixed_unique[:, 2].astype(np.float32)
    field_u.reshape(-1)[unique_linear] = disp_u
    field_v.reshape(-1)[unique_linear] = disp_v

    rhs_u = laplacian.build_rhs(field_u)
    rhs_v = laplacian.build_rhs(field_v)
    print("  Solving for in-plane displacement fields (PCG+Jacobi)...")
    solved_u, residual_u = _solve_field(
        laplacian,
        rhs_u,
        field_u.ravel(),
        label="disp_y",
        rtol=cfg.rtol,
        maxiter=cfg.maxiter,
        print_every=cfg.print_every,
    )
    solved_v, residual_v = _solve_field(
        laplacian,
        rhs_v,
        field_v.ravel(),
        label="disp_z",
        rtol=cfg.rtol,
        maxiter=cfg.maxiter,
        print_every=cfg.print_every,
    )

    deformation_field = np.zeros((3,) + shape, dtype=np.float32)
    if cfg.axis != 0:
        raise NotImplementedError("current Brainfast Laplacian refinement only supports axis=0")
    deformation_field[1] = solved_u
    deformation_field[2] = solved_v
    warped = apply_deformation_field(moving, deformation_field, chunk_slices=cfg.chunk_slices)

    final_registered_path = out_dir / "final_registered.nii.gz"
    nib.save(
        nib.Nifti1Image(warped, fixed_img.affine, fixed_img.header), str(final_registered_path)
    )

    deformation_field_path = params_dir / "laplacian_deformation_field.npy"
    np.save(deformation_field_path, deformation_field)
    fpoints_path = params_dir / "fpoints.npy"
    np.save(fpoints_path, fixed_unique.astype(np.int32))
    mpoints_path = params_dir / "mpoints.npy"
    np.save(mpoints_path, moving_unique.astype(np.int32))
    boundary_csv_path = params_dir / "boundary_conditions.csv"
    boundary_npy_path = params_dir / "boundary_conditions.npy"
    _save_boundary_conditions(
        fixed_unique,
        moving_unique,
        csv_path=boundary_csv_path,
        npy_path=boundary_npy_path,
    )

    return LaplacianResult(
        final_registered_path=final_registered_path,
        deformation_field_path=deformation_field_path,
        fpoints_path=fpoints_path,
        mpoints_path=mpoints_path,
        boundary_csv_path=boundary_csv_path,
        boundary_npy_path=boundary_npy_path,
        correspondences=int(len(fixed_points)),
        unique_boundary_voxels=int(len(unique_linear)),
        solve_seconds=float(time.time() - started),
        residual_history_u=residual_u,
        residual_history_v=residual_v,
    )
