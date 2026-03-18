from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries
from tifffile import imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.atlas_autopick import autopick_best_z
from scripts.atlas_mapper import map_cells_with_registered_label_slice
from scripts.dedup import apply_dedup_kdtree
from scripts.learn_from_trainset import _load_target_for_sample, _pair_ids
from scripts.map_and_aggregate import aggregate_by_region
from scripts.overlay_render import render_overlay
from scripts.structure_tree import load_structure_table, parse_structure_id_path
from scripts.detect import detect_cells


def _write_gray_png(path: Path, arr: np.ndarray) -> None:
    gray = np.asarray(arr, dtype=np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    Image.fromarray(rgb).save(path)


class MappingAggregationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.structure_csv = PROJECT_ROOT / "configs" / "allen_mouse_structure_graph.csv"
        cls.structure_df = load_structure_table(cls.structure_csv)
        candidates = cls.structure_df[
            (cls.structure_df["id"] != 997)
            & cls.structure_df["structure_id_path"].astype(str).str.len().gt(8)
        ].sort_values(["depth", "graph_order"], ascending=[False, True])
        if candidates.empty:
            raise RuntimeError("failed to find a non-root structure row for regression tests")
        cls.region = candidates.iloc[0].to_dict()

    def test_registered_slice_mapping_uses_official_structure_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            label_path = tmpdir / "registered_label.tif"
            label = np.zeros((20, 20), dtype=np.int32)
            label[2:8, 2:8] = int(self.region["id"])
            label[10:14, 10:14] = 997
            imwrite(str(label_path), label)

            cells = pd.DataFrame(
                [
                    {"cell_id": 1, "slice_id": 0, "x": 4.0, "y": 4.0, "score": 10.0},
                    {"cell_id": 2, "slice_id": 0, "x": 11.0, "y": 11.0, "score": 9.0},
                    {"cell_id": 3, "slice_id": 0, "x": 25.0, "y": 25.0, "score": 8.0},
                ]
            )

            mapped = map_cells_with_registered_label_slice(
                cells,
                registered_label_tif=label_path,
                structure_csv=self.structure_csv,
                atlas_slice_index=123,
            )

            hit = mapped.loc[mapped["cell_id"] == 1].iloc[0]
            root = mapped.loc[mapped["cell_id"] == 2].iloc[0]
            oob = mapped.loc[mapped["cell_id"] == 3].iloc[0]

            self.assertEqual(int(hit["region_id"]), int(self.region["id"]))
            self.assertEqual(str(hit["region_name"]), str(self.region["name"]))
            self.assertEqual(str(hit["structure_id_path"]), str(self.region["structure_id_path"]))
            self.assertEqual(root["mapping_status"], "outside_registered_slice")
            self.assertEqual(int(root["region_id"]), 0)
            self.assertEqual(oob["mapping_status"], "outside_slice_bounds")
            self.assertTrue(Path(str(hit["structure_source"])).exists())

    def test_aggregate_by_region_uses_structure_id_path(self) -> None:
        mapped = pd.DataFrame(
            [
                {
                    "cell_id": 1,
                    "slice_id": 0,
                    "x": 4.0,
                    "y": 4.0,
                    "score": 10.0,
                    "region_id": int(self.region["id"]),
                    "region_name": str(self.region["name"]),
                    "acronym": str(self.region["acronym"]),
                    "hemisphere": "left",
                    "structure_id_path": str(self.region["structure_id_path"]),
                    "structure_source": str(self.structure_csv.resolve()),
                },
                {
                    "cell_id": 2,
                    "slice_id": 0,
                    "x": 5.0,
                    "y": 5.0,
                    "score": 11.0,
                    "region_id": int(self.region["id"]),
                    "region_name": str(self.region["name"]),
                    "acronym": str(self.region["acronym"]),
                    "hemisphere": "left",
                    "structure_id_path": str(self.region["structure_id_path"]),
                    "structure_source": str(self.structure_csv.resolve()),
                },
            ]
        )

        leaf, hierarchy = aggregate_by_region(mapped)

        self.assertEqual(int(leaf.iloc[0]["count"]), 2)
        self.assertAlmostEqual(float(leaf.iloc[0]["confidence"]), 1.0)
        self.assertTrue(hierarchy["region_id"].isin(parse_structure_id_path(self.region["structure_id_path"])).all())
        self.assertIn(int(self.region["id"]), hierarchy["region_id"].tolist())


class TrainsetLearningRegressionTests(unittest.TestCase):
    def test_target_loading_prefers_label_and_keeps_show_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            train_dir = Path(td)

            gray = np.full((24, 24), 120, dtype=np.uint8)
            _write_gray_png(train_dir / "1_Ori.png", gray)
            _write_gray_png(train_dir / "2_Ori.png", gray)
            _write_gray_png(train_dir / "3_Ori.png", gray)

            label = np.zeros((24, 24), dtype=np.int32)
            label[5:18, 6:19] = 315
            imwrite(str(train_dir / "1_Label.tif"), label)

            show = np.zeros((24, 24, 3), dtype=np.uint8)
            show[6:18, 6:18, 1] = 220
            show[6:18, 6:18, 2] = 220
            Image.fromarray(show).save(train_dir / "1_Show.png")
            Image.fromarray(show).save(train_dir / "2_Show.png")

            ids = _pair_ids(train_dir)
            self.assertEqual(ids, ["1", "2"])

            label_target = _load_target_for_sample(train_dir, "1")
            show_target = _load_target_for_sample(train_dir, "2")

            self.assertEqual(label_target["target_type"], "label")
            self.assertTrue(np.any(label_target["label"] > 0))
            self.assertTrue(str(label_target["path"]).endswith("1_Label.tif"))
            self.assertEqual(show_target["target_type"], "show_boundary")
            self.assertTrue(np.any(show_target["mask"]))
            self.assertTrue(str(show_target["path"]).endswith("2_Show.png"))


class SyntheticPipelineSmokeTests(unittest.TestCase):
    def test_synthetic_slice_end_to_end(self) -> None:
        annotation = PROJECT_ROOT / "annotation_25.nii.gz"
        structure_csv = PROJECT_ROOT / "configs" / "allen_mouse_structure_graph.csv"
        if not annotation.exists():
            self.skipTest(f"missing atlas annotation: {annotation}")
        if not structure_csv.exists():
            self.skipTest(f"missing structure csv: {structure_csv}")

        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            real_path = tmpdir / "slice_0000.tif"
            auto_label_path = tmpdir / "slice_0000_auto_label.tif"
            registered_label_path = tmpdir / "slice_0000_registered_label.tif"
            overlay_png = tmpdir / "slice_0000_overlay.png"

            vol = np.asarray(nib.load(str(annotation)).get_fdata(), dtype=np.int32)
            nonzero_fraction = np.mean(vol > 0, axis=(1, 2))
            true_z = int(np.argmax(nonzero_fraction))
            atlas_slice = vol[true_z].astype(np.int32)

            tissue = (atlas_slice > 0).astype(np.float32)
            # Burn region boundaries into the synthetic image so the edge-matcher
            # finds a unique fingerprint at true_z and not at nearby atlas slices.
            region_boundaries = find_boundaries(atlas_slice.astype(np.int32), mode="inner", connectivity=2).astype(np.float32)
            real = gaussian_filter(tissue, sigma=2.4) * 150.0 + region_boundaries * 200.0
            imwrite(str(real_path), np.clip(real, 0, 255).astype(np.uint16))

            cfg = {
                "input": {"pixel_size_um_xy": 25.0},
                "compute": {"device": "cpu"},
                "detection": {
                    "primary_model": "disabled",
                    "secondary_model": "disabled",
                    "fallback_model": "threshold",
                    "fallback_threshold": 220.0,
                    "fallback_min_distance": 6,
                    "within_slice_dedup_px": 4.0,
                },
            }

            detections = detect_cells(real_path, cfg)
            self.assertGreater(len(detections), 0)

            auto_meta = autopick_best_z(
                real_path=real_path,
                annotation_nii=annotation,
                out_label_tif=auto_label_path,
                z_step=2,
                pixel_size_um=25.0,
                slicing_plane="coronal",
                roi_mode="auto",
            )

            self.assertTrue(auto_label_path.exists())
            self.assertGreater(float(auto_meta.get("best_score", 0.0)), 0.0)
            self.assertLessEqual(abs(int(auto_meta["best_z"]) - true_z), 12)

            _, diagnostic = render_overlay(
                real_slice_path=real_path,
                label_slice_path=auto_label_path,
                out_png=overlay_png,
                alpha=0.72,
                mode="contour-major",
                pixel_size_um=25.0,
                major_top_k=28,
                fit_mode="cover",
                edge_smooth_iter=0,
                warp_params={},
                return_meta=True,
                warped_label_out=registered_label_path,
            )

            self.assertTrue(registered_label_path.exists())
            self.assertTrue(overlay_png.exists())
            self.assertIn("warp", diagnostic)
            self.assertIn("timings_ms", diagnostic)
            self.assertGreater(float(diagnostic["timings_ms"].get("total", 0.0)), 0.0)

            detections = detections.copy()
            detections["slice_id"] = 0
            mapped = map_cells_with_registered_label_slice(
                detections,
                registered_label_tif=registered_label_path,
                structure_csv=structure_csv,
                atlas_slice_index=int(auto_meta["best_z"]),
                registration_score=float(auto_meta.get("best_score", 0.0)),
                registration_method=str(diagnostic.get("warp", {}).get("method", "registered_slice_label")),
            )

            deduped, _stats = apply_dedup_kdtree(
                mapped,
                neighbor_slices=1,
                pixel_size_um=0.65,
                slice_spacing_um=25.0,
                r_xy_um=6.0,
            )
            leaf, hierarchy = aggregate_by_region(deduped)

            self.assertGreater(len(deduped), 0)
            self.assertGreater(len(leaf), 0)
            self.assertFalse((leaf["region_id"] == 997).any())
            self.assertTrue((mapped["mapping_status"] == "ok").any())
            self.assertGreater(len(hierarchy), 0)


if __name__ == "__main__":
    unittest.main()
