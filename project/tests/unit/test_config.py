"""Unit tests for config_validation and exceptions modules."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_validation import collect_runtime_config_issues, validate_runtime_config, load_config
from scripts.exceptions import (
    ConfigError,
    RegistrationError,
    AlignmentScoreError,
    DetectionError,
    AtlasError,
    PipelineError,
    InputError,
    OutputError,
    BrainfastError,
)


def _minimal_valid_cfg() -> dict:
    return {
        "project": {"name": "test_project"},
        "input": {
            "slice_dir": "data/slices",
            "slice_glob": "z*.tif",
            "pixel_size_um_xy": 5.0,
            "slice_spacing_um": 25.0,
            "channel_map": {"red": 0},
            "active_channel": "red",
        },
        "detection": {"primary_model": "fallback"},
        "dedup": {"neighbor_slices": 1, "r_xy_um": 8.0},
        "outputs": {
            "leaf_csv": "outputs/leaf.csv",
            "hierarchy_csv": "outputs/hierarchy.csv",
            "qc_dir": "outputs/qc",
        },
    }


class TestValidateRuntimeConfig(unittest.TestCase):
    def test_valid_config_returns_no_issues(self):
        issues = validate_runtime_config(_minimal_valid_cfg())
        self.assertEqual(issues, [])

    def test_missing_project_name(self):
        cfg = _minimal_valid_cfg()
        cfg["project"]["name"] = ""
        issues = validate_runtime_config(cfg)
        self.assertTrue(any("project.name" in i for i in issues))

    def test_nonpositive_pixel_size(self):
        cfg = _minimal_valid_cfg()
        cfg["input"]["pixel_size_um_xy"] = 0
        issues = validate_runtime_config(cfg)
        self.assertTrue(any("pixel_size_um_xy" in i for i in issues))

    def test_active_channel_not_in_map(self):
        cfg = _minimal_valid_cfg()
        cfg["input"]["active_channel"] = "green"
        issues = validate_runtime_config(cfg)
        self.assertTrue(any("active_channel" in i for i in issues))

    def test_empty_channel_map(self):
        cfg = _minimal_valid_cfg()
        cfg["input"]["channel_map"] = {}
        issues = validate_runtime_config(cfg)
        self.assertTrue(any("channel_map" in i for i in issues))

    def test_placeholder_name_rejected(self):
        for placeholder in ("todo", "TBD", "changeme", ""):
            cfg = _minimal_valid_cfg()
            cfg["project"]["name"] = placeholder
            issues = validate_runtime_config(cfg)
            self.assertTrue(
                any("project.name" in i for i in issues),
                f"Expected issue for placeholder {placeholder!r}",
            )

    def test_negative_neighbor_slices(self):
        cfg = _minimal_valid_cfg()
        cfg["dedup"]["neighbor_slices"] = -1
        issues = validate_runtime_config(cfg)
        self.assertTrue(any("neighbor_slices" in i for i in issues))

    def test_collect_runtime_config_issues_warns_on_zero_refine_range(self):
        cfg = _minimal_valid_cfg()
        cfg["registration"] = {"atlas_z_refine_range": 0}
        issues = collect_runtime_config_issues(cfg)
        self.assertTrue(
            any(
                issue["field"] == "registration.atlas_z_refine_range"
                and issue["severity"] == "warning"
                for issue in issues
            )
        )


class TestExceptionHierarchy(unittest.TestCase):
    def test_all_errors_are_idlebrain_errors(self):
        for cls in (
            ConfigError,
            RegistrationError,
            AlignmentScoreError,
            DetectionError,
            AtlasError,
            PipelineError,
            InputError,
            OutputError,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, BrainfastError))

    def test_alignment_score_error_has_score_attributes(self):
        err = AlignmentScoreError(score=0.05, threshold=0.1, slice_idx=42)
        self.assertAlmostEqual(err.score, 0.05)
        self.assertAlmostEqual(err.threshold, 0.1)
        self.assertEqual(err.slice_idx, 42)
        self.assertIn("42", str(err))
        self.assertIn("0.050", str(err))

    def test_config_error_formats_issue_list(self):
        err = ConfigError("Validation failed", issues=["field_a is required", "field_b must be > 0"])
        msg = str(err)
        self.assertIn("field_a", msg)
        self.assertIn("field_b", msg)

    def test_registration_error_slice_prefix(self):
        err = RegistrationError("no tissue found", slice_idx=7)
        self.assertIn("[slice 7]", str(err))
        self.assertEqual(err.slice_idx, 7)

    def test_registration_error_no_slice(self):
        err = RegistrationError("general failure")
        self.assertIsNone(err.slice_idx)

    def test_detection_error_slice_prefix(self):
        err = DetectionError("model crash", slice_idx=0)
        self.assertIn("[slice 0]", str(err))

    def test_can_catch_as_base(self):
        """All subclasses must be catchable as BrainfastError."""
        with self.assertRaises(BrainfastError):
            raise AlignmentScoreError(score=0.0, threshold=0.1)


if __name__ == "__main__":
    unittest.main()
