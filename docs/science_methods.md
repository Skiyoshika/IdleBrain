# Brainfast — Scientific Methods Reference

> Algorithmic documentation for publication Methods sections and peer review.

---

## 1. Atlas AP Slice Selection

### Filename-based Z Assignment

When `atlas_z_from_filename: true`, each input image's Z position is extracted
from its filename (e.g., `slice_0042.tif` → Z = 42). The atlas AP index is
computed as:

```
atlas_z = int(z_filename × atlas_z_z_scale) + atlas_z_offset
```

Default coefficients for 25 µm atlas, 5 µm/pixel microscopy, 25 µm Z-step:
- `atlas_z_z_scale = -0.2`  (negative → posterior images map to lower AP)
- `atlas_z_offset = 330`

This maps filenames z=50–600 to AP indices 320–210, covering the posterior
hemisphere of the Allen CCFv3 atlas.

### Auto-pick (atlas_autopick.py)

When filename-based assignment is disabled, the system uses template matching:
normalized cross-correlation between the tissue silhouette and atlas annotation
cross-sections, evaluated over a configurable AP search range.

---

## 2. Section Registration

### Linear Stage (Affine)

A 6-DOF affine transformation (rotation, scale, shear, translation) is fitted
by maximising edge-SSIM between the registered atlas outline and the tissue
boundary. The initial placement uses a candidate pool:

- `right_flipped`: atlas mirrored for right-hemisphere samples
- Lateral offset: external face of tissue aligned to atlas lateral boundary

`skip_linear_opt: true` fixes the initial placement without further optimisation
(recommended when Z-step and pixel size are reliably known).

### Nonlinear Stage (TPS — Thin-Plate Spline)

A tissue mask is derived by Gaussian blurring the 16-bit raw image (σ = 40 px),
subtracting a corner-estimated background, and thresholding at CV > 0.5.

A set of control points on the tissue boundary and atlas silhouette boundary
are matched, and a thin-plate spline warp field is computed. This corrects
for local tissue deformation, folding, and sectioning artefacts.

### Quality Assessment

For each registered slice, edge-SSIM is computed between:
1. The registered atlas annotation boundary (1-px edge dilation)
2. The tissue edge derived from the raw image

SSIM scores are stored in `slice_registration_qc.csv` and displayed in the
QC tab gallery.

---

## 3. Z-Continuity Analysis

After registration, the AP index assigned to each slice is expected to vary
monotonically (or slowly) along the Z-axis. Brainfast detects outlier slices
using:

1. **Smoothing**: a rolling median (window = 5 slices) of AP indices
2. **Deviation**: `|AP_original − AP_smoothed|`
3. **Outlier threshold**: deviation > 2 × median absolute deviation (MAD)

Results are exported via `GET /api/outputs/z-continuity` and visualised in the
QC tab as a dual-line chart (original vs. smoothed) with outlier markers.

---

## 4. Cell Detection

### Laplacian of Gaussian (LoG) — Default

The built-in detector applies a multi-scale LoG filter:

```
L(x,y,σ) = σ² ∇²G(x,y,σ) ★ I(x,y)
```

Blob candidates are extracted at σ ∈ {2, 3, 4, 5} px (configurable). A local
maximum with score ≥ `min_score` is retained as a detection.

### Cellpose (Optional)

When `primary_model: "cellpose"` and Cellpose is installed, the Cellpose 2.0
cyto2 model is used for instance segmentation. Each detected mask yields one
cell centroid.

Fallback to LoG is automatic when Cellpose fails or is unavailable
(`allow_fallback: true`).

---

## 5. Cross-Slice Deduplication

Because Z-step spacing is typically coarser than cell diameter, the same cell
body may appear in two adjacent slices. Brainfast deduplicates by clustering
detections in 3-D (x, y, z_µm) space:

- `z_µm = slice_id × slice_spacing_um`
- DBSCAN clustering with `eps = cell_diameter_um` (default 15 µm)
- One representative detection is kept per cluster (highest score)

Deduplication statistics are written to `dedup_stats.json`.

---

## 6. Atlas Region Mapping

Each deduplicated cell centroid (x_px, y_px, slice_id) is mapped to atlas
voxel coordinates using the slice-specific affine + TPS transform. The
corresponding annotation label in `annotation_25.nii.gz` determines the
brain region (Allen CCFv3 structure ontology).

Cells falling outside the atlas annotation mask are labelled `OUTSIDE`
and excluded from regional statistics.

---

## 7. Hierarchical Aggregation

Cell counts are rolled up the Allen structure ontology tree. For each cell
mapped to leaf region L, all ancestors of L (Cortex → Isocortex → VISp → …)
receive +1 count. The final hierarchy table therefore reports cumulative
counts at every level.

### Morphology Statistics

When instance segmentation is used (Cellpose), additional morphology columns
are computed per cell and averaged per region:

| Column | Definition |
|--------|-----------|
| `mean_area_px` | Mean mask area in pixels |
| `mean_elongation` | Aspect ratio of fitted ellipse (major/minor axis) |
| `mean_mean_intensity` | Mean pixel intensity within mask |

---

## 8. Statistical Confidence Intervals (Garwood / Wilson-Hilferty)

For each region with count *n*, a 95% Poisson confidence interval is computed
using the Wilson-Hilferty normal approximation to the chi-squared quantiles
(equivalent to the Garwood method for large *n*):

```
CI_low  = max(0, λ · (1 − 1/(9λ) − 1.96/(3√λ))³)     [λ = n]
CI_high = λ' · (1 − 1/(9λ') + 1.96/(3√λ'))³            [λ' = n + 1]
```

For n = 0, CI_low = 0 and CI_high is computed with λ' = 1.

These intervals are reported as `ci_low` and `ci_high` in both leaf and
hierarchy CSV outputs, and displayed as `[lo–hi]` badges in the Results table.

---

## 9. Atlas Reproducibility Fingerprint

On each pipeline run, the SHA-256 hash (first 12 hex characters) of the atlas
annotation file is computed and stored in `detection_summary.json`:

```json
{ "atlas_sha256": "a1b2c3d4e5f6" }
```

This fingerprint is included in the generated Methods paragraph to allow
precise atlas version tracking across publications and lab computers.

---

## Methods Paragraph Template (English)

> Brain atlas registration was performed using Brainfast v{version} (run: {timestamp}).
> Microscopy images were acquired at {pixel_size} µm/pixel.
> Section registration was carried out against the Allen Mouse Brain Atlas
> (CCFv3, annotation_25.nii.gz, 25 µm voxel spacing, sha256: {atlas_sha256})
> using {affine/nonlinear TPS} transformation.
> Alignment quality was evaluated by edge-SSIM.
> Cell counting used {detector} on {single/merged} slices, followed by
> deduplication and hierarchical atlas aggregation.
> 95% Poisson confidence intervals were computed using the Garwood method.
> Channels: {channels}.
