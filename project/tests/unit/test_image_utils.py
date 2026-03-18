"""Unit tests for scripts/image_utils.py — no IO, no atlas required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.image_utils import norm_u8_robust, alpha_blend, to_gray_u8

pytestmark = pytest.mark.unit


class TestNormU8Robust:
    def test_output_dtype(self):
        arr = np.array([0, 100, 500, 1000, 65535], dtype=np.uint16)
        out = norm_u8_robust(arr)
        assert out.dtype == np.uint8

    def test_output_range(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 65535, (64, 64), dtype=np.uint16)
        out = norm_u8_robust(arr)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_constant_array_no_crash(self):
        arr = np.full((10, 10), 1000, dtype=np.uint16)
        out = norm_u8_robust(arr)
        assert out.dtype == np.uint8

    def test_sparse_fluorescence(self, sparse_real_array):
        """Sparse image: a few bright cells should not collapse entire range to 0."""
        out = norm_u8_robust(sparse_real_array)
        # Background pixels should still have some signal, not all zero
        assert int(out.mean()) > 0
        assert out.dtype == np.uint8

    def test_shape_preserved(self):
        arr = np.arange(200, dtype=np.float32).reshape(10, 20)
        out = norm_u8_robust(arr)
        assert out.shape == (10, 20)

    def test_stretches_low_contrast(self):
        arr = np.array([1000, 1001, 1002, 1003], dtype=np.uint16)
        out = norm_u8_robust(arr)
        # Should span most of 0-255 range after stretch
        assert int(out.max()) - int(out.min()) > 100


class TestAlphaBlend:
    def test_alpha_zero_returns_gray(self):
        gray = np.full((4, 4), 128, dtype=np.uint8)
        color = np.zeros((4, 4, 3), dtype=np.uint8)
        color[..., 0] = 255  # pure red
        out = alpha_blend(gray, color, alpha=0.0)
        assert out.shape == (4, 4, 3)
        assert out[0, 0, 0] == 128
        assert out[0, 0, 1] == 128
        assert out[0, 0, 2] == 128

    def test_alpha_one_returns_color(self):
        gray = np.zeros((4, 4), dtype=np.uint8)
        color = np.full((4, 4, 3), 200, dtype=np.uint8)
        out = alpha_blend(gray, color, alpha=1.0)
        assert int(out[0, 0, 0]) == 200

    def test_output_dtype_uint8(self):
        gray = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        color = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        out = alpha_blend(gray, color, alpha=0.5)
        assert out.dtype == np.uint8

    def test_output_shape(self):
        gray = np.zeros((16, 32), dtype=np.uint8)
        color = np.zeros((16, 32, 3), dtype=np.uint8)
        out = alpha_blend(gray, color, alpha=0.5)
        assert out.shape == (16, 32, 3)


class TestToGrayU8:
    def test_2d_input(self):
        arr = np.arange(100, dtype=np.uint16).reshape(10, 10) * 655
        out = to_gray_u8(arr)
        assert out.shape == (10, 10)
        assert out.dtype == np.uint8

    def test_3d_uses_first_channel(self):
        # Channel 0 has a gradient (non-constant), channels 1 and 2 are zero.
        arr = np.zeros((10, 10, 3), dtype=np.uint16)
        arr[..., 0] = np.tile(np.arange(10, dtype=np.uint16) * 3000, (10, 1))
        out = to_gray_u8(arr)
        assert out.shape == (10, 10)
        # Non-constant channel 0 should produce a non-trivial range after stretch
        assert int(out.max()) > int(out.min())
