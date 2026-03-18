"""
Brainfast structured exception hierarchy.

Usage:
    from scripts.exceptions import RegistrationError, AtlasError, PipelineError

    raise RegistrationError("No tissue mask found", slice_idx=42)
"""

from __future__ import annotations


class BrainfastError(Exception):
    """Base class for all Brainfast errors. Always catch this in top-level handlers."""


# Backwards-compat alias
IdleBrainError = BrainfastError


# ── Pipeline ──────────────────────────────────────────────────────────────────


class PipelineError(BrainfastError):
    """Raised when the overall pipeline cannot continue."""


class ConfigError(BrainfastError):
    """Raised for invalid or missing configuration values."""

    def __init__(self, message: str, issues: list[str] | None = None) -> None:
        if issues:
            full = message + "\n  - " + "\n  - ".join(issues)
        else:
            full = message
        super().__init__(full)
        self.issues = issues or []


# ── Registration ──────────────────────────────────────────────────────────────


class RegistrationError(BrainfastError):
    """Raised when atlas registration fails for a slice."""

    def __init__(self, message: str, slice_idx: int | None = None) -> None:
        prefix = f"[slice {slice_idx}] " if slice_idx is not None else ""
        super().__init__(prefix + message)
        self.slice_idx = slice_idx


class AtlasError(BrainfastError):
    """Raised when the atlas file is missing, corrupt, or out of range."""


class AlignmentScoreError(RegistrationError):
    """Raised when registration score is below the configured threshold."""

    def __init__(self, score: float, threshold: float, slice_idx: int | None = None) -> None:
        msg = f"score {score:.3f} < threshold {threshold:.3f}"
        super().__init__(msg, slice_idx=slice_idx)
        self.score = score
        self.threshold = threshold


# ── Detection ─────────────────────────────────────────────────────────────────


class DetectionError(BrainfastError):
    """Raised when cell detection fails for a slice."""

    def __init__(self, message: str, slice_idx: int | None = None) -> None:
        prefix = f"[slice {slice_idx}] " if slice_idx is not None else ""
        super().__init__(prefix + message)
        self.slice_idx = slice_idx


# ── I/O ───────────────────────────────────────────────────────────────────────


class InputError(BrainfastError):
    """Raised for missing or unreadable input files / directories."""


class OutputError(BrainfastError):
    """Raised when an output file cannot be written."""
