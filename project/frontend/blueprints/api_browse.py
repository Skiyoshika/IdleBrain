"""api_browse.py — Browse folder/file via tkinter dialog.

In headless environments (e.g. Docker/Linux servers), set the environment
variable BRAINFAST_HEADLESS=1 to return a friendly error instead of crashing.
"""

from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

from project.frontend.api_errors import (
    ERR_INTERNAL,
)

bp = Blueprint("api_browse", __name__, url_prefix="/api")

_HEADLESS = os.environ.get("BRAINFAST_HEADLESS", "").strip() not in ("", "0", "false", "no")

_HEADLESS_MSG = (
    "File browser is not available in headless mode (BRAINFAST_HEADLESS=1). "
    "Enter the path manually."
)


@bp.post("/browse/folder")
def browse_folder():
    if _HEADLESS:
        return jsonify(
            {"ok": False, "error": _HEADLESS_MSG, "headless": True, "error_code": ERR_INTERNAL}
        ), 200
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askdirectory(title="选择文件夹")
        root.destroy()
        return jsonify({"ok": True, "path": path or ""})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "error_code": ERR_INTERNAL}), 500


@bp.post("/browse/file")
def browse_file():
    if _HEADLESS:
        return jsonify(
            {"ok": False, "error": _HEADLESS_MSG, "headless": True, "error_code": ERR_INTERNAL}
        ), 200
    try:
        import tkinter as tk
        from tkinter import filedialog

        payload = request.get_json(force=True) or {}
        filetypes_raw = payload.get("filetypes", "")
        if filetypes_raw:
            exts = [e.strip() for e in filetypes_raw.split(",")]
            filetypes = [(f".{e} 文件", f"*.{e}") for e in exts] + [("所有文件", "*.*")]
        else:
            filetypes = [("所有文件", "*.*")]
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askopenfilename(title="选择文件", filetypes=filetypes)
        root.destroy()
        return jsonify({"ok": True, "path": path or ""})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "error_code": ERR_INTERNAL}), 500
