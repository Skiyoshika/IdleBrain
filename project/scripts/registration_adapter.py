from __future__ import annotations

import shutil
from pathlib import Path


def locate_legacy_registration_repo(root: Path) -> Path:
    repo = root / "repos" / "uci-allen-brainrepositorycodegui"
    if not repo.exists():
        raise FileNotFoundError(f"Legacy repo not found: {repo}")
    return repo


def prepare_registration_workspace(project_root: Path) -> Path:
    reg_dir = project_root / "outputs" / "registration"
    reg_dir.mkdir(parents=True, exist_ok=True)
    return reg_dir


def copy_elastix_params(legacy_repo: Path, reg_dir: Path) -> tuple[Path, Path]:
    p1 = legacy_repo / "001_parameters_Rigid.txt"
    p2 = legacy_repo / "002_parameters_BSpline.txt"
    if not p1.exists() or not p2.exists():
        raise FileNotFoundError("Elastix parameter files missing in legacy repo")

    dst1 = reg_dir / p1.name
    dst2 = reg_dir / p2.name
    shutil.copy2(p1, dst1)
    shutil.copy2(p2, dst2)
    return dst1, dst2


def bootstrap_registration_assets(project_root: Path) -> dict:
    workspace_root = project_root.parent  # D:/brain-atlas-cellcount-work
    legacy = locate_legacy_registration_repo(workspace_root)
    reg_dir = prepare_registration_workspace(project_root)
    rigid, bspline = copy_elastix_params(legacy, reg_dir)
    return {
        "legacy_repo": str(legacy),
        "registration_dir": str(reg_dir),
        "rigid_param": str(rigid),
        "bspline_param": str(bspline),
    }
