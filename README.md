[中文文档 →](README.zh-CN.md)

---

# Brainfast

**Brain atlas registration + cell counting for cleared-tissue microscopy.**

Brainfast is a local, privacy-first desktop tool that registers fluorescence microscopy sections against the [Allen Mouse Brain Atlas (CCFv3)](https://atlas.brain-map.org/) and counts labelled cells per anatomical region — with a browser UI, no cloud upload, and full reproducibility metadata.

> Built for neuroscience labs working with lightsheet or confocal TIFF stacks from cleared half-brain or whole-brain samples.

---

## What it does

| Step | What happens |
|------|-------------|
| **Auto AP selection** | Matches each section to its atlas coronal plane by filename Z-coordinate or template cross-correlation |
| **Registration** | Affine placement → thin-plate spline (TPS) nonlinear warp to conform tissue boundary to atlas outline |
| **Cell detection** | Multi-scale LoG (built-in) or Cellpose instance segmentation, configurable per channel |
| **Deduplication** | 3-D KD-tree clustering removes cross-slice duplicates before counting |
| **Region mapping** | Each cell centroid is mapped into the Allen CCFv3 annotation volume |
| **Hierarchical counts** | Counts roll up through the full Allen ontology tree (leaf → area → division → …) |
| **QC & export** | Z-continuity chart, edge-SSIM per slice, Excel/CSV export, auto-generated Methods paragraph |

---

## Key features

- **Browser UI** — 4-tab single-page app (Workflow / QC / Results / Projects), no install beyond Python
- **Bilingual** — full EN/ZH interface toggle, all labels and hints translated
- **Garwood 95% CI** — Wilson-Hilferty Poisson confidence intervals on every region count
- **Atlas fingerprint** — SHA-256 of `annotation_25.nii.gz` written to `detection_summary.json` for reproducibility
- **Projects & batch queue** — SQLite-backed sample management, FIFO batch worker
- **Cross-sample comparison** — merge leaf CSVs from multiple runs into a pivot table
- **Multi-channel co-expression** — per-region red/green channel counts side-by-side
- **3D volume pipeline** — full volumetric registration via ANTs or Elastix with HTML run reports
- **Light/dark theme** — localStorage-persisted theme toggle
- **Docker-ready** — `Dockerfile` + `docker-compose.yml` for headless Linux server deployment
- **97 unit tests**, CI on GitHub Actions (Windows + Ubuntu)

---

## Quick start

### Requirements

- Python 3.10 or 3.11
- Windows 10/11 (primary) · Linux via Docker
- `annotation_25.nii.gz` — Allen CCFv3 25 µm annotation (place in `project/`)
- NVIDIA GPU recommended for Cellpose; LoG detector works on CPU

### Install

```powershell
git clone https://github.com/Skiyoshika/Brainfast.git
cd Brainfast
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[advanced,dev]"
```

### Run

```powershell
# Windows — double-click or:
.\Start_Brainfast.bat

# Or directly:
python project\frontend\server.py
```

Open **http://127.0.0.1:8787** in your browser.

### Docker (Linux server)

```bash
docker compose up -d
# Open http://localhost:8787
```

Set `BRAINFAST_HEADLESS=1` (already default in Docker) to disable the tkinter file-browser dialog.

---

## Workflow overview

```
Input TIFF slices (Z-stack)
      │
      ▼
Auto AP slice selection ──── atlas_autopick.py
      │
      ▼
Registration: Affine → TPS nonlinear warp
      │
      ▼
Manual review / liquify correction  ← browser UI
      │
      ▼
Cell detection  (LoG · Cellpose · per-channel)
      │
      ▼
3-D deduplication  (KD-tree, configurable radius)
      │
      ▼
Atlas region mapping  (CCFv3 annotation lookup)
      │
      ▼
Hierarchical aggregation + Garwood 95% CI
      │
      ▼
cell_counts_leaf.csv · cell_counts_hierarchy.csv
Excel export · Methods paragraph · Z-continuity chart
```

---

## Output files

| File | Contents |
|------|----------|
| `cell_counts_leaf.csv` | Per-region leaf counts with `ci_low`, `ci_high`, morphology columns |
| `cell_counts_hierarchy.csv` | Counts rolled up through the full Allen ontology tree |
| `cells_dedup.csv` | Deduplicated cell centroids (x, y, z_µm, score, region_id) |
| `detection_summary.json` | Detector choice, sampling mode, totals, `atlas_sha256` |
| `slice_registration_qc.csv` | Edge-SSIM per slice |
| `z_smoothness_report.json` | AP-axis continuity analysis (outlier flags) |
| `brainfast_results.xlsx` | 3-sheet Excel: Hierarchy / Leaf / Run parameters |

UI job outputs live in `project/outputs/jobs/<job_id>/`.

---

## API (REST)

The Flask server exposes a documented REST API.
See [docs/api_reference.md](docs/api_reference.md) for all endpoints with `curl` examples.

Key endpoints:

```bash
POST /api/run               # start pipeline
GET  /api/poll?job=...      # unified status + log tail + errors (replaces 3 polling loops)
GET  /api/outputs/excel     # download Excel workbook
GET  /api/outputs/z-continuity   # Z-axis continuity JSON
POST /api/compare/regions   # cross-sample pivot table
```

All error responses include a machine-readable `error_code` constant (e.g. `PIPELINE_ALREADY_RUNNING`, `CONFIG_PATH_DENIED`).

---

## Science methods

Algorithmic detail — registration stages, Garwood CI derivation, Z-continuity detection, atlas fingerprinting — is documented in [docs/science_methods.md](docs/science_methods.md).

**Methods paragraph template** (auto-generated by the UI):

> Brain atlas registration was performed using Brainfast v0.5 (run: …). Microscopy images were acquired at N µm/pixel. Section registration was carried out against the Allen Mouse Brain Atlas (CCFv3, annotation_25.nii.gz, 25 µm voxel spacing, sha256: …) using nonlinear (thin-plate spline) transformation. Alignment quality was evaluated by edge-SSIM. Cell counting used LoG on native single slices, followed by deduplication and hierarchical atlas aggregation. 95% Poisson confidence intervals were computed using the Garwood method. Channels: red.

---

## Architecture

```
project/
├── frontend/
│   ├── server.py              Flask entry point (70-line orchestration layer)
│   ├── server_context.py      Shared run state, job isolation, GC
│   ├── blueprints/            11 API blueprints
│   │   ├── api_pipeline.py    run / cancel / poll / preflight / methods-text
│   │   ├── api_outputs.py     CSV / Excel / Z-continuity / AP-density / coexpression
│   │   ├── api_projects.py    project + sample CRUD (SQLite)
│   │   ├── api_batch.py       FIFO batch queue
│   │   ├── api_compare.py     cross-sample region comparison
│   │   └── …
│   ├── index.html             Single-page UI (4 tabs, bilingual)
│   ├── app.js                 Frontend JS (~3500 lines, full i18n)
│   └── styles.css             Dark/light theme CSS variables
├── scripts/
│   ├── main.py                2D pipeline entry point
│   ├── detect.py              LoG + Cellpose detection
│   ├── map_and_aggregate.py   Region mapping + hierarchical counts + Garwood CI
│   ├── z_smoothness.py        AP-axis continuity analysis
│   └── …
└── tests/
    ├── unit/                  97 tests, no atlas file required
    └── integration/           Requires annotation_25.nii.gz
```

**Security invariants (v0.5+):**
- Config paths are containment-checked against `PROJECT_ROOT/configs` and `OUTPUT_DIR` — no arbitrary filesystem access
- `running = True` is set inside `_run_state_lock` before thread start — no race condition on concurrent `/api/run`
- `_job_states` is capped at 200 entries with LRU eviction — no unbounded memory growth

---

## Tests

```powershell
# Unit tests (no atlas file needed)
python -m pytest project/tests/unit -v

# With coverage (CI enforces ≥60%)
python -m pytest project/tests/unit --cov=project/scripts --cov-report=term-missing

# Lint
ruff check project/scripts/ project/frontend/blueprints/
```

CI: [GitHub Actions](.github/workflows/test.yml) — unit tests on Windows + Ubuntu, ruff lint on every push/PR to `main`.

Releases: [release workflow](.github/workflows/release.yml) — tag `v*.*.*` → auto-build Windows EXE → upload to GitHub Releases.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `project/configs/` | Run configs, Allen metadata, sample presets |
| `project/frontend/` | Flask app, UI assets, desktop launcher |
| `project/scripts/` | Registration, detection, mapping, aggregation, utility scripts |
| `project/tests/` | Unit and integration test suites |
| `project/train_data_set/` | Manual calibration pairs (17 samples) |
| `docs/` | [User guide](docs/user_guide.md) · [API reference](docs/api_reference.md) · [Science methods](docs/science_methods.md) |

---

## License

See [LICENSE](LICENSE).
