from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from config_validation import load_config, validate_runtime_config

REQUIRED_MODULES = (
    "flask",
    "numpy",
    "pandas",
    "scipy",
    "skimage",
    "tifffile",
    "PIL",
    "nibabel",
    "matplotlib",
)

OPTIONAL_MODULES = (
    "cellpose",
    "SimpleITK",
    "pystray",
    "nrrd",
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _print_status(ok: bool, kind: str, label: str, detail: str = "") -> None:
    state = "OK" if ok else kind
    line = f"[{state}] {label}"
    if detail:
        line = f"{line}: {detail}"
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Brainfast runtime environment")
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent.parent / "configs" / "run_config.template.json"
        ),
        help="Config file to validate",
    )
    parser.add_argument(
        "--require-input-dir",
        action="store_true",
        help="Fail if input.slice_dir is missing or still a placeholder",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    failures = 0

    py_ok = sys.version_info >= (3, 10)
    _print_status(py_ok, "FAIL", "python", f"{sys.version.split()[0]} (need >= 3.10)")
    if not py_ok:
        failures += 1

    for name in REQUIRED_MODULES:
        ok = _module_available(name)
        _print_status(ok, "FAIL", f"python module '{name}'")
        if not ok:
            failures += 1

    for name in OPTIONAL_MODULES:
        ok = _module_available(name)
        _print_status(ok, "WARN", f"optional module '{name}'")

    required_assets = [
        project_root / "annotation_25.nii.gz",
        project_root / "configs" / "allen_structure_tree.json",
        project_root / "configs" / "allen_mouse_structure_graph.csv",
        project_root / "frontend" / "index.html",
        project_root / "frontend" / "server.py",
    ]
    for path in required_assets:
        ok = path.exists()
        _print_status(ok, "FAIL", "asset", str(path))
        if not ok:
            failures += 1

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        _print_status(False, "FAIL", "config", str(cfg_path))
        failures += 1
    else:
        _print_status(True, "OK", "config", str(cfg_path))
        try:
            cfg = load_config(cfg_path)
            issues = validate_runtime_config(cfg, require_input_dir=bool(args.require_input_dir))
            if issues:
                failures += len(issues)
                for issue in issues:
                    _print_status(False, "FAIL", "config", issue)
            else:
                _print_status(True, "OK", "config", "runtime fields validated")
        except Exception as exc:
            failures += 1
            _print_status(False, "FAIL", "config", str(exc))

    if failures:
        print(f"Environment check finished with {failures} failure(s).")
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
