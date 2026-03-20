# Brainfast User Guide

> Version 0.5 — Allen CCFv3 mouse brain atlas registration + cell counting

---

## Table of Contents

1. [Installation](#installation)
2. [First Run](#first-run)
3. [Workflow Overview](#workflow-overview)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Configuration Reference](#configuration-reference)
6. [Results Interpretation](#results-interpretation)
7. [Projects & Batch Processing](#projects--batch-processing)
8. [FAQ](#faq)

---

## Installation

### Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or 3.11 |
| OS | Windows 10/11 (tested), Linux (via Docker) |
| RAM | 16 GB recommended (32 GB for full-resolution datasets) |
| Atlas file | `annotation_25.nii.gz` — Allen CCFv3, 25 µm isotropic |

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<org>/Brainfast.git
cd Brainfast

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# 3. Install core dependencies
pip install -e .

# 4. (Optional) Install advanced dependencies — Cellpose, scipy
pip install -e ".[advanced]"

# 5. Place the atlas file
copy annotation_25.nii.gz project\

# 6. Start the server
cd project
python frontend/server.py
# Open: http://127.0.0.1:8787
```

### Windows Quick-Start

Double-click `project/frontend/StartIdleBrainTrial.bat`.
The browser opens automatically at `http://127.0.0.1:8787`.

---

## First Run

1. Open the browser at `http://127.0.0.1:8787`
2. In **Step 1 (Paths)**: set the input image folder and output folder
3. In **Step 2 (Atlas)**: verify the atlas file path (auto-filled if standard location)
4. In **Step 3 (Calibration)**: choose "Auto" alignment
5. In **Step 4 (Detection)**: leave defaults or adjust confidence threshold
6. Click **Run Pipeline**

Results appear in the **Results** tab after completion.

---

## Workflow Overview

```
Input images (Z-stack TIFF)
        │
        ▼
  Auto AP selection (atlas_autopick.py)
        │
        ▼
  Registration: Affine → TPS (nonlinear)
        │
        ▼
  Manual review / Liquify correction
        │
        ▼
  Cell detection (LoG or Cellpose)
        │
        ▼
  Deduplication (cross-slice)
        │
        ▼
  Atlas region mapping
        │
        ▼
  Hierarchical aggregation + CI computation
        │
        ▼
  Results: CSV, Excel, Methods text
```

---

## Step-by-Step Guide

### Step 1 — Paths

| Field | Description |
|-------|-------------|
| Input folder | Directory containing `.tif` slice images |
| Output folder | Where results are written (created if needed) |
| Config file | (Optional) Pre-saved JSON config to load |

### Step 2 — Atlas

| Field | Description |
|-------|-------------|
| Atlas file | Path to `annotation_25.nii.gz` |
| Hemisphere | `right_flipped` for cleared half-brain samples |
| Pixel size (µm) | Physical pixel size of your microscope |
| Z step (µm) | Distance between Z-stack slices |

### Step 3 — Alignment

- **Affine** (default): rigid rotation + scale + shear. Fast, robust.
- **Nonlinear (TPS)**: thin-plate spline warp. Use for curved/deformed tissue.
- **Calibration pairs**: train the auto-alignment from manually corrected examples.

### Step 4 — Detection

| Field | Description |
|-------|-------------|
| Primary model | `log` (built-in) or `cellpose` (requires install) |
| Channels | `red`, `green`, `farred` |
| Confidence threshold | Minimum detection score 0–1 (slider) |
| Sampling mode | `single` (one pass) or `merge` (average adjacent slices) |

---

## Configuration Reference

Configs are JSON files in `project/configs/`. Key fields:

```json
{
  "input": {
    "slice_glob": "*.tif",
    "sampling_mode": "single",
    "slice_interval_n": 1
  },
  "atlas": {
    "annotation_path": "annotation_25.nii.gz",
    "voxel_size_um": 25,
    "hemisphere": "right_flipped"
  },
  "registration": {
    "align_mode": "nonlinear",
    "atlas_z_from_filename": true,
    "atlas_z_z_scale": -0.2,
    "skip_linear_opt": true
  },
  "detection": {
    "primary_model": "log",
    "channels": ["red"],
    "min_score": 0.0,
    "allow_fallback": true
  },
  "output": {
    "output_dir": "outputs/default"
  }
}
```

See `project/configs/run_config_35.json` for a complete working example.

---

## Results Interpretation

After a run completes, the **Results** tab shows:

### Result Snapshot

Summary cards with:
- **Cells detected** — raw count before deduplication
- **Cells deduplicated** — final count after cross-slice dedup
- **Slices registered** — number of atlas-registered sections
- **Coverage** — fraction of cells successfully mapped to atlas regions

### Region Table

Hierarchical table of cell counts per brain region:
- **Count**: absolute cell number in that region
- **%**: fraction of total mapped cells
- **CI [low–high]**: 95% Garwood Poisson confidence interval
- **Elongation / Area / Intensity**: morphology columns (toggle "Show Morphology")

### QC Tab

- **Z-continuity chart**: AP-axis continuity across slices; red dots = outlier slices
- **Registration gallery**: side-by-side raw vs. atlas overlays per slice

### Export Options

| Button | Output |
|--------|--------|
| Export CSV | `cell_counts_leaf.csv` (per-region leaf counts) |
| Export Excel | 3-sheet workbook: Hierarchy / Leaf / Run parameters |
| Export Methods Text | LaTeX-ready Methods paragraph for publications |

---

## Projects & Batch Processing

The **Projects** tab (folder icon) manages multi-sample workflows:

1. **Create project** — name + optional description
2. **Add samples** — config path + output path per sample
3. **Run sample** — loads config into Step 1 and switches to Workflow tab
4. **Batch queue** — enqueue multiple samples; processed sequentially in background

---

## FAQ

**Q: The atlas overlay looks mirror-flipped.**
A: Set `hemisphere: "right_flipped"` in the config (for right-hemisphere cleared samples with the medial side on the right).

**Q: All slices converge to the same AP position.**
A: Disable AP refinement: set `atlas_z_refine_range: 0` in the config.

**Q: Cellpose is not detecting any cells.**
A: Ensure Cellpose is installed (`pip install cellpose`) and the model path is correct. LoG fallback activates automatically when Cellpose fails.

**Q: The server won't start.**
A: Check that port 8787 is not in use. Run `python frontend/server.py --port 8788` to use a different port.

**Q: Where are my results?**
A: In the output directory you specified in Step 1. Key files:
- `cell_counts_leaf.csv` — per-region leaf-level counts
- `cell_counts_hierarchy.csv` — hierarchical aggregation
- `detection_summary.json` — run metadata including atlas SHA256
- `registered_slices/` — overlay images
