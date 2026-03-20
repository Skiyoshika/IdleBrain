"""server.py — Thin orchestrator: sets up paths, imports blueprints, creates Flask app.

All route logic lives in project/frontend/blueprints/*.
All shared state and helpers live in project/frontend/server_context.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ROOT = Path(sys._MEIPASS)
    _proj = Path(sys.executable).resolve().parent
    PROJECT_ROOT = _proj.parent if str(_proj).endswith("frontend") else _proj
else:
    ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure the parent of PROJECT_ROOT is on sys.path so that
# `import project.frontend.server_context` resolves correctly from blueprints.
_repo_root = str(PROJECT_ROOT.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Patch context paths before any blueprint imports so helpers resolve correctly.
import project.frontend.server_context as ctx
ctx.ROOT = ROOT
ctx.PROJECT_ROOT = PROJECT_ROOT
ctx.OUTPUT_DIR = PROJECT_ROOT / "outputs"
ctx.DEFAULT_STRUCTURE_SOURCE = PROJECT_ROOT / "configs" / "allen_mouse_structure_graph.csv"

from flask import Flask

from project.frontend.blueprints.api_pipeline import bp as pipeline_bp
from project.frontend.blueprints.api_atlas import bp as atlas_bp
from project.frontend.blueprints.api_overlay import bp as overlay_bp
from project.frontend.blueprints.api_alignment import bp as alignment_bp
from project.frontend.blueprints.api_outputs import bp as outputs_bp
from project.frontend.blueprints.api_training import bp as training_bp
from project.frontend.blueprints.api_demo import bp as demo_bp
from project.frontend.blueprints.api_browse import bp as browse_bp
from project.frontend.blueprints.api_projects import bp as projects_bp
from project.frontend.blueprints.api_batch import bp as batch_bp
from project.frontend.blueprints.api_compare import bp as compare_bp
from project.frontend.blueprints.api_docs import bp as docs_bp


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(atlas_bp)
    app.register_blueprint(overlay_bp)
    app.register_blueprint(alignment_bp)
    app.register_blueprint(outputs_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(demo_bp)
    app.register_blueprint(browse_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(batch_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(docs_bp)
    return app


app = create_app()


def main():
    import os
    port = int(os.environ.get("IDLEBRAIN_PORT", "8787"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
