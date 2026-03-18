"""api_browse.py — Browse folder/file via tkinter dialog."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

bp = Blueprint("api_browse", __name__, url_prefix="/api")


@bp.post("/browse/folder")
def browse_folder():
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
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.post("/browse/file")
def browse_file():
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
        return jsonify({"ok": False, "error": str(e)}), 500
