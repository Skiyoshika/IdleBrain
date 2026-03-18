from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _require_text(cfg: dict[str, Any], dotted_key: str, issues: list[str]) -> None:
    value = _get(cfg, dotted_key)
    if is_placeholder_value(value):
        issues.append(f"{dotted_key} is required and must not be a placeholder value")
        return
    if not isinstance(value, str):
        issues.append(f"{dotted_key} must be a string")


def _require_positive_number(cfg: dict[str, Any], dotted_key: str, issues: list[str]) -> None:
    value = _get(cfg, dotted_key)
    if is_placeholder_value(value):
        issues.append(f"{dotted_key} is required and must not be a placeholder value")
        return
    try:
        num = float(value)
    except Exception:
        issues.append(f"{dotted_key} must be numeric, got {value!r}")
        return
    if num <= 0:
        issues.append(f"{dotted_key} must be > 0, got {num}")


def _require_nonnegative_int(cfg: dict[str, Any], dotted_key: str, issues: list[str]) -> None:
    value = _get(cfg, dotted_key)
    if is_placeholder_value(value):
        issues.append(f"{dotted_key} is required and must not be a placeholder value")
        return
    try:
        num = int(value)
    except Exception:
        issues.append(f"{dotted_key} must be an integer, got {value!r}")
        return
    if num < 0:
        issues.append(f"{dotted_key} must be >= 0, got {num}")


def validate_runtime_config(
    cfg: dict[str, Any],
    *,
    require_input_dir: bool = False,
) -> list[str]:
    issues: list[str] = []

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
        issues.append("input.channel_map must be a non-empty object")
    else:
        active = _get(cfg, "input.active_channel", "red")
        if active and active not in channel_map:
            issues.append(f"input.active_channel '{active}' is not defined in input.channel_map")

    return issues


def config_value(cfg: dict[str, Any], dotted_key: str) -> Any:
    return _get(cfg, dotted_key)
