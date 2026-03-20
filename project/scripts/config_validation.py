from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).parent.parent / "configs" / "run_config.schema.json"


def validate_against_schema(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Validate *cfg* against run_config.schema.json using jsonschema.

    Returns a list of issue dicts (same format as collect_runtime_config_issues).
    Returns an empty list if jsonschema is not installed (graceful degradation).
    """
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return []
    if not _SCHEMA_PATH.exists():
        return []
    try:
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(cfg), key=lambda e: list(e.absolute_path))
        issues = []
        for err in errors:
            path = ".".join(str(p) for p in err.absolute_path) or "<root>"
            issues.append({"field": path, "severity": "error", "message": err.message})
        return issues
    except Exception:
        return []


_PLACEHOLDER_STRINGS = {"", "todo", "tbd", "changeme", "required"}


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a JSON object: {path}")
    return data


def is_placeholder_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_STRINGS
    return False


def _get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _make_issue(field: str, severity: str, message: str) -> dict[str, str]:
    return {"field": field, "severity": severity, "message": message}


def _append_issue(
    issues: list[dict[str, str]],
    *,
    field: str,
    severity: str,
    message: str,
) -> None:
    issues.append(_make_issue(field, severity, message))


def _require_text(cfg: dict[str, Any], dotted_key: str, issues: list[dict[str, str]]) -> None:
    value = _get(cfg, dotted_key)
    if is_placeholder_value(value):
        _append_issue(
            issues,
            field=dotted_key,
            severity="error",
            message=f"{dotted_key} is required and must not be a placeholder value",
        )
        return
    if not isinstance(value, str):
        _append_issue(
            issues,
            field=dotted_key,
            severity="error",
            message=f"{dotted_key} must be a string",
        )


def _require_positive_number(
    cfg: dict[str, Any], dotted_key: str, issues: list[dict[str, str]]
) -> None:
    value = _get(cfg, dotted_key)
    if is_placeholder_value(value):
        _append_issue(
            issues,
            field=dotted_key,
            severity="error",
            message=f"{dotted_key} is required and must not be a placeholder value",
        )
        return
    try:
        num = float(value)
    except Exception:
        _append_issue(
            issues,
            field=dotted_key,
            severity="error",
            message=f"{dotted_key} must be numeric, got {value!r}",
        )
        return
    if num <= 0:
        _append_issue(
            issues,
            field=dotted_key,
            severity="error",
            message=f"{dotted_key} must be > 0, got {num}",
        )


def _require_nonnegative_int(
    cfg: dict[str, Any], dotted_key: str, issues: list[dict[str, str]]
) -> None:
    value = _get(cfg, dotted_key)
    if is_placeholder_value(value):
        _append_issue(
            issues,
            field=dotted_key,
            severity="error",
            message=f"{dotted_key} is required and must not be a placeholder value",
        )
        return
    try:
        num = int(value)
    except Exception:
        _append_issue(
            issues,
            field=dotted_key,
            severity="error",
            message=f"{dotted_key} must be an integer, got {value!r}",
        )
        return
    if num < 0:
        _append_issue(
            issues,
            field=dotted_key,
            severity="error",
            message=f"{dotted_key} must be >= 0, got {num}",
        )


def collect_runtime_config_issues(
    cfg: dict[str, Any],
    *,
    require_input_dir: bool = False,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = validate_against_schema(cfg)

    _require_text(cfg, "project.name", issues)
    _require_text(cfg, "input.slice_glob", issues)
    _require_positive_number(cfg, "input.pixel_size_um_xy", issues)
    _require_positive_number(cfg, "input.slice_spacing_um", issues)
    _require_text(cfg, "detection.primary_model", issues)
    _require_nonnegative_int(cfg, "dedup.neighbor_slices", issues)
    _require_positive_number(cfg, "dedup.r_xy_um", issues)
    _require_text(cfg, "outputs.leaf_csv", issues)
    _require_text(cfg, "outputs.hierarchy_csv", issues)
    _require_text(cfg, "outputs.qc_dir", issues)

    if require_input_dir:
        _require_text(cfg, "input.slice_dir", issues)

    channel_map = _get(cfg, "input.channel_map", {})
    if not isinstance(channel_map, dict) or not channel_map:
        _append_issue(
            issues,
            field="input.channel_map",
            severity="error",
            message="input.channel_map must be a non-empty object",
        )
    else:
        active = _get(cfg, "input.active_channel", "red")
        if active and active not in channel_map:
            _append_issue(
                issues,
                field="input.active_channel",
                severity="error",
                message=f"input.active_channel '{active}' is not defined in input.channel_map",
            )

    try:
        refine_range = int(_get(cfg, "registration.atlas_z_refine_range", 0) or 0)
    except Exception:
        refine_range = 0
    if refine_range == 0:
        _append_issue(
            issues,
            field="registration.atlas_z_refine_range",
            severity="warning",
            message=(
                "registration.atlas_z_refine_range is 0. AP refinement around the initial atlas guess is disabled."
            ),
        )

    return issues


def validate_runtime_config(
    cfg: dict[str, Any],
    *,
    require_input_dir: bool = False,
) -> list[str]:
    issues = collect_runtime_config_issues(cfg, require_input_dir=require_input_dir)
    return [issue["message"] for issue in issues if issue.get("severity") == "error"]


def config_value(cfg: dict[str, Any], dotted_key: str) -> Any:
    return _get(cfg, dotted_key)
