# Brainfast API Reference

Base URL: `http://127.0.0.1:8787`

All endpoints return JSON with at minimum `{"ok": true|false}`.
Error responses include `"error"` (human message) and `"error_code"` (machine constant).

---

## Pipeline

### `GET /api/info`
Server metadata and defaults.

```bash
curl http://127.0.0.1:8787/api/info
```
```json
{ "version": "0.5.1", "outputs": "outputs/jobs/default", "defaults": { "atlasPath": "..." } }
```

### `GET /api/validate`
Validate that the server is reachable.

### `POST /api/pipeline/preflight`
Validate config before running. Returns structured error list.

```bash
curl -X POST http://127.0.0.1:8787/api/pipeline/preflight \
  -H 'Content-Type: application/json' \
  -d '{"configPath": "configs/run_config_35.json"}'
```
```json
{ "ok": true, "issues": [] }
```

### `POST /api/run`
Start a pipeline run.

```bash
curl -X POST http://127.0.0.1:8787/api/run \
  -H 'Content-Type: application/json' \
  -d '{
    "configPath": "configs/run_config_35.json",
    "inputDir": "data/35_C0_demo",
    "outputDir": "outputs/jobs/sample35",
    "jobId": "sample35",
    "alignMode": "nonlinear",
    "pixelSizeUm": 5.0,
    "channels": ["red"],
    "confidenceThreshold": 0.0
  }'
```
```json
{ "ok": true, "jobId": "sample35" }
```

Error codes: `PIPELINE_ALREADY_RUNNING`, `CONFIG_NOT_FOUND`, `CONFIG_PATH_DENIED`,
`INVALID_INPUT`, `PREFLIGHT_FAILED`

### `GET /api/poll?job=<jobId>`
**Unified polling endpoint** — replaces `/api/status` + `/api/logs` + `/api/error-log`.
Poll at 500 ms when active, 30 s when idle.

```bash
curl 'http://127.0.0.1:8787/api/poll?job=sample35'
```
```json
{
  "ok": true,
  "jobId": "sample35",
  "running": false,
  "done": true,
  "error": null,
  "slicesDone": 111,
  "slicesTotal": 111,
  "logTail": ["[done] pipeline completed"],
  "errors": [],
  "progress": { "phase": "done", "stepCurrent": 6, "stepTotal": 6, "message": "" }
}
```

### `GET /api/status?job=<jobId>`
Full status including channels and start epoch.

### `GET /api/logs?job=<jobId>`
Full log array (all lines).

### `GET /api/error-log?job=<jobId>`
Structured error list with step, source, recoverable fields.

### `POST /api/cancel`
Cancel a running pipeline.

```bash
curl -X POST http://127.0.0.1:8787/api/cancel \
  -H 'Content-Type: application/json' \
  -d '{"jobId": "sample35"}'
```

### `GET /api/history?job=<jobId>`
Previous run history for this job.

### `GET /api/export/methods-text?job=<jobId>`
Auto-generated Methods paragraph (Chinese + English).

---

## Outputs

All output endpoints accept `?job=<jobId>` (defaults to `"default"`).

### `GET /api/outputs/hierarchy`
`cell_counts_hierarchy.csv` as JSON rows.

### `GET /api/outputs/leaf`
`cell_counts_leaf.csv` as JSON rows (includes `ci_low`, `ci_high`).

### `GET /api/outputs/leaf/<channel>`
Per-channel leaf CSV (`red`, `green`, `farred`).

### `GET /api/outputs/excel`
Download 3-sheet Excel workbook.

```bash
curl -OJ 'http://127.0.0.1:8787/api/outputs/excel?job=sample35'
# saves brainfast_results.xlsx
```

### `GET /api/outputs/z-continuity`
Z-axis continuity analysis.

```json
{
  "ok": true,
  "slices": [
    { "slice_id": 0, "original_z": 295, "smoothed_z": 293.0,
      "deviation": 2.0, "is_outlier": false }
  ],
  "outlier_count": 1,
  "total_slices": 111
}
```

### `GET /api/outputs/ap-density`
Cell counts per AP coordinate.

```json
{
  "ok": true,
  "ap_slices": [
    { "ap_index": 280, "slice_id": 42, "cell_count": 512 }
  ],
  "total_slices": 111
}
```

### `GET /api/outputs/coexpression`
Multi-channel co-expression per atlas region.

```json
{
  "ok": true,
  "regions": [
    { "acronym": "VISp", "name": "Primary visual area",
      "count_red": 340, "count_green": 120 }
  ],
  "channel_red_available": true,
  "channel_green_available": true
}
```

### `GET /api/outputs/registration-qc`
Registration quality CSV.

### `GET /api/outputs/reg-slice-list`
List of registered slice overlay filenames.

### `GET /api/outputs/reg-slice/<filename>`
Serve a registered slice overlay image.

### `GET /api/outputs/file-list`
All output file paths.

---

## Atlas

### `POST /api/atlas/autopick`
Auto-select best AP slice for a given input image.

```bash
curl -X POST http://127.0.0.1:8787/api/atlas/autopick \
  -H 'Content-Type: application/json' \
  -d '{"imagePath": "data/slice_0042.tif", "atlasPath": "annotation_25.nii.gz"}'
```

### `GET /api/atlas/slice-preview/<ap_index>`
Render an atlas cross-section at AP index as PNG.

---

## Projects

### `GET /api/projects`
List all projects.

### `POST /api/projects`
Create a project.
```bash
curl -X POST http://127.0.0.1:8787/api/projects \
  -H 'Content-Type: application/json' \
  -d '{"name": "Experiment A", "description": "Visual cortex injection"}'
```

### `GET /api/projects/<id>`
Get project details.

### `DELETE /api/projects/<id>`
Delete a project.

### `GET /api/projects/<id>/samples`
List samples in a project.

### `POST /api/projects/<id>/samples`
Add a sample.
```bash
curl -X POST http://127.0.0.1:8787/api/projects/1/samples \
  -H 'Content-Type: application/json' \
  -d '{"name": "Mouse35", "config_path": "configs/run_config_35.json", "output_path": "outputs/jobs/sample35"}'
```

### `DELETE /api/projects/<project_id>/samples/<sample_id>`
Remove a sample.

---

## Batch Queue

### `POST /api/batch/enqueue`
Add a sample to the processing queue.
```bash
curl -X POST http://127.0.0.1:8787/api/batch/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"jobId": "sample35", "configPath": "configs/run_config_35.json", "inputDir": "data/35_C0_demo", "outputDir": "outputs/jobs/sample35"}'
```

### `GET /api/batch/status`
Current queue status and item states.

### `POST /api/batch/cancel`
Cancel a queued or running batch item.

---

## Compare

### `POST /api/compare/regions`
Cross-sample region comparison (CSV merge).

```bash
curl -X POST http://127.0.0.1:8787/api/compare/regions \
  -H 'Content-Type: application/json' \
  -d '{
    "dirs": [
      {"path": "outputs/jobs/sample35", "label": "Mouse35"},
      {"path": "outputs/jobs/sample36", "label": "Mouse36"}
    ]
  }'
```
```json
{
  "ok": true,
  "columns": ["acronym", "Mouse35", "Mouse36"],
  "rows": [
    { "acronym": "VISp", "Mouse35": 340, "Mouse36": 280 }
  ]
}
```

---

## Error Codes

| Code | HTTP | Meaning |
|------|------|---------|
| `PIPELINE_ALREADY_RUNNING` | 409 | Run is active; cancel first |
| `PIPELINE_NOT_RUNNING` | 409 | No active run to cancel |
| `PIPELINE_START_FAILED` | 500 | Could not spawn pipeline subprocess |
| `INVALID_INPUT` | 400 | Missing or malformed request field |
| `MISSING_FIELD` | 400 | Required field absent |
| `CONFIG_NOT_FOUND` | 404 | Config file does not exist |
| `CONFIG_PATH_DENIED` | 403 | Config path outside allowed directories |
| `PREFLIGHT_FAILED` | 400 | Pre-run validation failed |
| `NOT_FOUND` | 404 | Generic resource not found |
| `FILE_NOT_FOUND` | 404 | Specific file not found |
| `JOB_NOT_FOUND` | 404 | Job ID unknown |
| `PROJECT_NOT_FOUND` | 404 | Project ID unknown |
| `SAMPLE_NOT_FOUND` | 404 | Sample ID unknown |
| `ALREADY_EXISTS` | 409 | Resource already exists |
| `TASK_CONFLICT` | 409 | Conflicting background task |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `IO_ERROR` | 500 | File I/O failure |
| `DEPENDENCY_ERROR` | 500 | Missing Python dependency |
