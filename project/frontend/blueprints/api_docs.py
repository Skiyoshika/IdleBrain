"""api_docs.py — OpenAPI 3.0 spec + embedded Swagger UI.

Endpoints:
  GET /api/openapi.json   — machine-readable OpenAPI 3.0.3 spec
  GET /api/docs           — browser Swagger UI (no CDN, inline JS redirect to openapi.json)
"""

from __future__ import annotations

from flask import Blueprint, Response, jsonify

bp = Blueprint("api_docs", __name__)

# ---------------------------------------------------------------------------
# OpenAPI 3.0.3 specification (hand-authored, covers all major endpoints)
# ---------------------------------------------------------------------------

_SPEC: dict = {
    "openapi": "3.0.3",
    "info": {
        "title": "Brainfast API",
        "version": "0.5",
        "description": (
            "REST API for the Brainfast brain-atlas registration and cell-counting tool. "
            'All endpoints return JSON with at minimum `{"ok": true|false}`. '
            "Error responses also include `error_code` (machine constant) and `error` (human message)."
        ),
        "contact": {"name": "Brainfast", "url": "https://github.com/Skiyoshika/Brainfast"},
        "license": {"name": "See LICENSE"},
    },
    "servers": [{"url": "http://127.0.0.1:8787", "description": "Local desktop server"}],
    "tags": [
        {"name": "pipeline", "description": "Pipeline run lifecycle"},
        {"name": "outputs", "description": "Result files and charts"},
        {"name": "projects", "description": "Project and sample management"},
        {"name": "batch", "description": "Batch queue"},
        {"name": "compare", "description": "Cross-sample comparison"},
        {"name": "atlas", "description": "Atlas slice selection and preview"},
        {"name": "docs", "description": "API documentation and schema"},
    ],
    "paths": {
        "/api/info": {
            "get": {
                "tags": ["pipeline"],
                "summary": "Server metadata and defaults",
                "operationId": "getInfo",
                "responses": {
                    "200": {
                        "description": "Server info",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "ok": {"type": "boolean"},
                                        "version": {"type": "string"},
                                        "buildDate": {"type": "string"},
                                        "commit": {"type": "string"},
                                        "outputs": {"type": "string"},
                                        "defaults": {"type": "object"},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/config-schema": {
            "get": {
                "tags": ["docs"],
                "summary": "JSON Schema for run_config.json (Draft-07)",
                "operationId": "getConfigSchema",
                "responses": {
                    "200": {"description": "Schema object"},
                    "404": {"description": "Schema file not found"},
                },
            }
        },
        "/api/pipeline/preflight": {
            "post": {
                "tags": ["pipeline"],
                "summary": "Validate config before starting a run",
                "operationId": "preflight",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "configPath": {"type": "string"},
                                    "inputDir": {"type": "string"},
                                    "atlasPath": {"type": "string"},
                                    "channels": {"type": "array", "items": {"type": "string"}},
                                    "params": {"type": "object"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "Preflight result with structured issue list"},
                    "400": {"description": "Config path denied or invalid input"},
                },
            }
        },
        "/api/run": {
            "post": {
                "tags": ["pipeline"],
                "summary": "Start a pipeline run",
                "operationId": "runPipeline",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["configPath"],
                                "properties": {
                                    "configPath": {"type": "string"},
                                    "inputDir": {"type": "string"},
                                    "outputDir": {"type": "string"},
                                    "jobId": {"type": "string", "default": "default"},
                                    "alignMode": {
                                        "type": "string",
                                        "enum": ["affine", "nonlinear"],
                                    },
                                    "pixelSizeUm": {"type": "number"},
                                    "channels": {"type": "array", "items": {"type": "string"}},
                                    "confidenceThreshold": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "Run started"},
                    "409": {"description": "Pipeline already running (PIPELINE_ALREADY_RUNNING)"},
                    "400": {"description": "Invalid input or preflight failure"},
                    "403": {"description": "Config path denied (CONFIG_PATH_DENIED)"},
                },
            }
        },
        "/api/poll": {
            "get": {
                "tags": ["pipeline"],
                "summary": "Unified poll — status + log tail + errors in one request",
                "operationId": "poll",
                "parameters": [
                    {
                        "name": "job",
                        "in": "query",
                        "schema": {"type": "string", "default": "default"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Poll response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "ok": {"type": "boolean"},
                                        "jobId": {"type": "string"},
                                        "running": {"type": "boolean"},
                                        "done": {"type": "boolean"},
                                        "error": {"type": ["string", "null"]},
                                        "slicesDone": {"type": "integer"},
                                        "slicesTotal": {"type": "integer"},
                                        "logTail": {"type": "array", "items": {"type": "string"}},
                                        "errors": {"type": "array"},
                                        "progress": {"type": "object"},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/status": {
            "get": {
                "tags": ["pipeline"],
                "summary": "Full job status",
                "operationId": "getStatus",
                "parameters": [{"name": "job", "in": "query", "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Status object"}},
            }
        },
        "/api/cancel": {
            "post": {
                "tags": ["pipeline"],
                "summary": "Cancel a running pipeline",
                "operationId": "cancelPipeline",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"jobId": {"type": "string"}},
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Cancelled"},
                    "409": {"description": "No running process"},
                },
            }
        },
        "/api/export/methods-text": {
            "get": {
                "tags": ["pipeline"],
                "summary": "Auto-generated Methods paragraph (EN + ZH)",
                "operationId": "exportMethodsText",
                "parameters": [{"name": "job", "in": "query", "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Methods text object"}},
            }
        },
        "/api/outputs/hierarchy": {
            "get": {
                "tags": ["outputs"],
                "summary": "cell_counts_hierarchy.csv as JSON",
                "operationId": "outputsHierarchy",
                "parameters": [{"name": "job", "in": "query", "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Hierarchy rows"}},
            }
        },
        "/api/outputs/leaf": {
            "get": {
                "tags": ["outputs"],
                "summary": "cell_counts_leaf.csv as JSON (includes ci_low, ci_high, morphology)",
                "operationId": "outputsLeaf",
                "parameters": [{"name": "job", "in": "query", "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Leaf rows"}},
            }
        },
        "/api/outputs/excel": {
            "get": {
                "tags": ["outputs"],
                "summary": "Download 3-sheet Excel workbook (Hierarchy / Leaf / RunParams)",
                "operationId": "outputsExcel",
                "parameters": [{"name": "job", "in": "query", "schema": {"type": "string"}}],
                "responses": {
                    "200": {
                        "description": "Excel file download",
                        "content": {
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {}
                        },
                    }
                },
            }
        },
        "/api/outputs/z-continuity": {
            "get": {
                "tags": ["outputs"],
                "summary": "Z-axis continuity analysis (AP smoothness, outlier flags)",
                "operationId": "outputsZContinuity",
                "parameters": [{"name": "job", "in": "query", "schema": {"type": "string"}}],
                "responses": {
                    "200": {
                        "description": "Z-continuity result",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "ok": {"type": "boolean"},
                                        "slices": {"type": "array"},
                                        "outlier_count": {"type": "integer"},
                                        "total_slices": {"type": "integer"},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/api/outputs/ap-density": {
            "get": {
                "tags": ["outputs"],
                "summary": "Cell counts per AP coordinate",
                "operationId": "outputsApDensity",
                "parameters": [{"name": "job", "in": "query", "schema": {"type": "string"}}],
                "responses": {"200": {"description": "AP density array"}},
            }
        },
        "/api/outputs/coexpression": {
            "get": {
                "tags": ["outputs"],
                "summary": "Per-region red/green channel co-expression counts",
                "operationId": "outputsCoexpression",
                "parameters": [{"name": "job", "in": "query", "schema": {"type": "string"}}],
                "responses": {"200": {"description": "Coexpression table"}},
            }
        },
        "/api/projects": {
            "get": {
                "tags": ["projects"],
                "summary": "List all projects",
                "operationId": "listProjects",
                "responses": {"200": {"description": "Project list"}},
            },
            "post": {
                "tags": ["projects"],
                "summary": "Create a project",
                "operationId": "createProject",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "Created project"}},
            },
        },
        "/api/projects/{id}/samples": {
            "get": {
                "tags": ["projects"],
                "summary": "List samples in a project",
                "operationId": "listSamples",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}
                ],
                "responses": {"200": {"description": "Sample list"}},
            },
            "post": {
                "tags": ["projects"],
                "summary": "Add a sample to a project",
                "operationId": "addSample",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name", "config_path"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "config_path": {"type": "string"},
                                    "output_path": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "Added sample"}},
            },
        },
        "/api/batch/enqueue": {
            "post": {
                "tags": ["batch"],
                "summary": "Enqueue a sample for batch processing",
                "operationId": "batchEnqueue",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "jobId": {"type": "string"},
                                    "configPath": {"type": "string"},
                                    "inputDir": {"type": "string"},
                                    "outputDir": {"type": "string"},
                                },
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "Enqueued"}},
            }
        },
        "/api/batch/status": {
            "get": {
                "tags": ["batch"],
                "summary": "Batch queue status",
                "operationId": "batchStatus",
                "responses": {"200": {"description": "Queue status"}},
            }
        },
        "/api/compare/regions": {
            "post": {
                "tags": ["compare"],
                "summary": "Cross-sample region pivot table",
                "operationId": "compareRegions",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["dirs"],
                                "properties": {
                                    "dirs": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "path": {"type": "string"},
                                                "label": {"type": "string"},
                                            },
                                        },
                                    }
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "Pivot table (columns = labels, rows = regions)"}
                },
            }
        },
        "/api/openapi.json": {
            "get": {
                "tags": ["docs"],
                "summary": "This OpenAPI 3.0 specification",
                "operationId": "getOpenApiSpec",
                "responses": {"200": {"description": "OpenAPI spec"}},
            }
        },
        "/api/docs": {
            "get": {
                "tags": ["docs"],
                "summary": "Swagger UI for interactive API exploration",
                "operationId": "swaggerUi",
                "responses": {"200": {"description": "Swagger UI HTML page"}},
            }
        },
    },
    "components": {
        "schemas": {
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean", "example": False},
                    "error": {"type": "string"},
                    "error_code": {
                        "type": "string",
                        "enum": [
                            "PIPELINE_ALREADY_RUNNING",
                            "PIPELINE_NOT_RUNNING",
                            "PIPELINE_START_FAILED",
                            "INVALID_INPUT",
                            "MISSING_FIELD",
                            "CONFIG_NOT_FOUND",
                            "CONFIG_PATH_DENIED",
                            "PREFLIGHT_FAILED",
                            "NOT_FOUND",
                            "FILE_NOT_FOUND",
                            "JOB_NOT_FOUND",
                            "PROJECT_NOT_FOUND",
                            "SAMPLE_NOT_FOUND",
                            "ALREADY_EXISTS",
                            "TASK_CONFLICT",
                            "INTERNAL_ERROR",
                            "IO_ERROR",
                            "DEPENDENCY_ERROR",
                        ],
                    },
                },
            }
        }
    },
}

# ---------------------------------------------------------------------------
# Swagger UI HTML (no CDN — loads spec from /api/openapi.json)
# ---------------------------------------------------------------------------
_SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Brainfast API Docs</title>
  <link rel="stylesheet" type="text/css"
        href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #1a1c22; color: #dde1e9; }
    .swagger-ui .topbar { background: #252830; }
    .swagger-ui .info .title { color: #7ea3ff; }
    #swagger-ui { max-width: 1200px; margin: 0 auto; padding: 16px; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({
      url: "/api/openapi.json",
      dom_id: "#swagger-ui",
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
      layout: "BaseLayout",
      deepLinking: true,
      supportedSubmitMethods: ["get", "post", "put", "delete", "patch"],
      tryItOutEnabled: true,
    });
  </script>
</body>
</html>"""


@bp.get("/api/openapi.json")
def openapi_spec():
    return jsonify(_SPEC)


@bp.get("/api/docs")
def swagger_ui():
    return Response(_SWAGGER_HTML, mimetype="text/html")
