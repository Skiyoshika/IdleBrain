"""Unit tests for map_and_aggregate.py — core research output.

Tests use synthetic DataFrames so no atlas files are required.
"""
from __future__ import annotations

import pandas as pd
import pytest

# Allow import from project root
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from map_and_aggregate import aggregate_by_region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mapped(n_cortex=10, n_hp=5, include_morph=False) -> pd.DataFrame:
    """Synthetic mapped DataFrame with two regions."""
    rows_ctx = [
        {
            "region_id": 100,
            "region_name": "Cortex",
            "acronym": "CTX",
            "hemisphere": "left",
            "slice_id": i,
            "score": 0.9,
        }
        for i in range(n_cortex)
    ]
    rows_hp = [
        {
            "region_id": 200,
            "region_name": "Hippocampus",
            "acronym": "HP",
            "hemisphere": "right",
            "slice_id": i,
            "score": 0.8,
        }
        for i in range(n_hp)
    ]
    rows = rows_ctx + rows_hp
    df = pd.DataFrame(rows)
    if include_morph:
        df["area_px"] = 50.0
        df["elongation"] = 1.5
        df["mean_intensity"] = 200.0
    return df


# ---------------------------------------------------------------------------
# aggregate_by_region — empty input
# ---------------------------------------------------------------------------

def test_aggregate_empty_returns_empty_frames():
    empty = pd.DataFrame(
        columns=["region_id", "region_name", "acronym", "hemisphere"]
    )
    leaf, hierarchy = aggregate_by_region(empty)
    assert leaf.empty
    assert hierarchy.empty


# ---------------------------------------------------------------------------
# aggregate_by_region — basic counts
# ---------------------------------------------------------------------------

def test_aggregate_leaf_counts():
    mapped = _make_mapped(n_cortex=10, n_hp=5)
    leaf, _ = aggregate_by_region(mapped)

    ctx_row = leaf[leaf["acronym"] == "CTX"].iloc[0]
    hp_row = leaf[leaf["acronym"] == "HP"].iloc[0]

    assert int(ctx_row["count"]) == 10
    assert int(hp_row["count"]) == 5


def test_aggregate_confidence_sums_to_one():
    mapped = _make_mapped(n_cortex=8, n_hp=2)
    leaf, _ = aggregate_by_region(mapped)
    total_confidence = leaf["confidence"].sum()
    assert abs(total_confidence - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Garwood / Wilson-Hilferty Poisson CI
# ---------------------------------------------------------------------------

def test_ci_low_zero_for_zero_count():
    """ci_low must be 0 when count is 0."""
    mapped = _make_mapped(n_cortex=0, n_hp=1)
    leaf, _ = aggregate_by_region(mapped)
    hp_row = leaf[leaf["acronym"] == "HP"].iloc[0]
    assert float(hp_row["ci_low"]) == 0.0


def test_ci_bounds_order():
    """ci_low < count < ci_high for all non-zero rows."""
    mapped = _make_mapped(n_cortex=50, n_hp=3)
    leaf, _ = aggregate_by_region(mapped)
    for _, row in leaf.iterrows():
        if row["count"] > 0:
            assert row["ci_low"] <= row["count"], f"ci_low={row['ci_low']} > count={row['count']}"
            assert row["ci_high"] >= row["count"], f"ci_high={row['ci_high']} < count={row['count']}"


def test_ci_columns_present():
    mapped = _make_mapped(n_cortex=10, n_hp=5)
    leaf, _ = aggregate_by_region(mapped)
    assert "ci_low" in leaf.columns
    assert "ci_high" in leaf.columns


# ---------------------------------------------------------------------------
# Morphology columns (optional path)
# ---------------------------------------------------------------------------

def test_morph_columns_absent_when_no_morph_data():
    mapped = _make_mapped(n_cortex=5, n_hp=3, include_morph=False)
    leaf, _ = aggregate_by_region(mapped)
    assert "mean_area_px" not in leaf.columns
    assert "mean_elongation" not in leaf.columns


def test_morph_columns_present_when_morph_data_available():
    mapped = _make_mapped(n_cortex=5, n_hp=3, include_morph=True)
    leaf, _ = aggregate_by_region(mapped)
    assert "mean_area_px" in leaf.columns
    assert "mean_elongation" in leaf.columns
    assert "mean_mean_intensity" in leaf.columns


def test_morph_values_correct():
    mapped = _make_mapped(n_cortex=5, n_hp=3, include_morph=True)
    leaf, _ = aggregate_by_region(mapped)
    ctx_row = leaf[leaf["acronym"] == "CTX"].iloc[0]
    assert abs(float(ctx_row["mean_area_px"]) - 50.0) < 0.01
    assert abs(float(ctx_row["mean_elongation"]) - 1.5) < 0.01


# ---------------------------------------------------------------------------
# No structure_source column → hierarchy is empty
# ---------------------------------------------------------------------------

def test_hierarchy_empty_without_structure_source():
    mapped = _make_mapped(n_cortex=10, n_hp=5)
    # No structure_source column → hierarchy should be empty DataFrame
    _, hierarchy = aggregate_by_region(mapped)
    assert hierarchy.empty or len(hierarchy) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_cell():
    row = pd.DataFrame([{
        "region_id": 999,
        "region_name": "TestRegion",
        "acronym": "TST",
        "hemisphere": "left",
        "slice_id": 0,
        "score": 1.0,
    }])
    leaf, _ = aggregate_by_region(row)
    assert len(leaf) == 1
    assert leaf.iloc[0]["count"] == 1
    assert leaf.iloc[0]["confidence"] == pytest.approx(1.0)


def test_all_outside_region():
    rows = [
        {"region_id": 0, "region_name": "OUTSIDE", "acronym": "OUT", "hemisphere": "unknown",
         "slice_id": i, "score": 0.5}
        for i in range(20)
    ]
    mapped = pd.DataFrame(rows)
    leaf, _ = aggregate_by_region(mapped)
    assert len(leaf) == 1
    assert leaf.iloc[0]["count"] == 20
