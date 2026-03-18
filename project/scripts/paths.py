"""
Centralized path management for Brainfast pipelines.

Usage:
    from scripts.paths import RunPaths

    paths = RunPaths.from_project_root(project_root, cfg)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    cells_csv = paths.cells_detected
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def bootstrap_sys_path() -> Path:
    """Ensure PROJECT_ROOT is on sys.path. Idempotent; frozen-EXE aware.

    Call this at the top of any standalone script that needs to import from
    ``scripts.*`` or ``project.*``.  Returns PROJECT_ROOT.

    Example (at top of script, before any project imports)::

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.paths import bootstrap_sys_path
        PROJECT_ROOT = bootstrap_sys_path()
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        project_root = Path(sys._MEIPASS)
    else:
        # __file__ is scripts/paths.py → parents[1] is project/
        project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


@dataclass
class RunPaths:
    """All file/directory paths for a single pipeline run."""

    # ── Root dirs ─────────────────────────────────────────────────────────────
    project_root: Path
    outputs: Path
    registered_slices: Path
    qc_dir: Path

    # ── Atlas asset ───────────────────────────────────────────────────────────
    annotation_nii: Path
    structure_csv: Path

    # ── Intermediate CSVs ─────────────────────────────────────────────────────
    cells_detected: Path
    cells_dedup: Path
    cells_mapped: Path
    dedup_stats: Path

    # ── Final outputs ─────────────────────────────────────────────────────────
    cell_counts_leaf: Path
    cell_counts_hierarchy: Path
    slice_registration_qc: Path
    slice_qc: Path

    # ── Tuning / training ─────────────────────────────────────────────────────
    tuned_params: Path
    trainset_tuned_params: Path

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
        cfg: dict[str, Any],
    ) -> RunPaths:
        outputs_cfg = cfg.get("outputs", {})
        outputs = project_root / "outputs"

        leaf_csv = outputs_cfg.get("leaf_csv", "outputs/cell_counts_leaf.csv")
        hierarchy_csv = outputs_cfg.get("hierarchy_csv", "outputs/cell_counts_hierarchy.csv")
        qc_dir_cfg = outputs_cfg.get("qc_dir", "outputs/qc")

        # Resolve relative paths from project root
        def _resolve(p: str) -> Path:
            pp = Path(p)
            if pp.is_absolute():
                return pp
            return project_root / pp

        structure_csv = project_root / "configs" / "allen_mouse_structure_graph.csv"
        structure_csv_fallback = outputs / "registration" / "structure_tree.csv"
        if structure_csv_fallback.exists():
            structure_csv = structure_csv_fallback

        return cls(
            project_root=project_root,
            outputs=outputs,
            registered_slices=outputs / "registered_slices",
            qc_dir=_resolve(qc_dir_cfg),
            annotation_nii=project_root / "annotation_25.nii.gz",
            structure_csv=structure_csv,
            cells_detected=outputs / "cells_detected.csv",
            cells_dedup=outputs / "cells_dedup.csv",
            cells_mapped=outputs / "cells_mapped.csv",
            dedup_stats=outputs / "dedup_stats.csv",
            cell_counts_leaf=_resolve(leaf_csv),
            cell_counts_hierarchy=_resolve(hierarchy_csv),
            slice_registration_qc=outputs / "slice_registration_qc.csv",
            slice_qc=outputs / "slice_qc.csv",
            tuned_params=outputs / "tuned_params.json",
            trainset_tuned_params=outputs / "trainset_tuned_params.json",
        )

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in (self.outputs, self.registered_slices, self.qc_dir):
            d.mkdir(parents=True, exist_ok=True)

    def registered_slice_overlay(self, idx: int) -> Path:
        return self.registered_slices / f"slice_{idx:04d}_overlay.png"

    def registered_label(self, idx: int) -> Path:
        return self.registered_slices / f"slice_{idx:04d}_registered_label.tif"

    def auto_label(self, idx: int) -> Path:
        return self.registered_slices / f"slice_{idx:04d}_auto_label.tif"
