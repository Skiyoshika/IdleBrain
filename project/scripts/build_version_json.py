#!/usr/bin/env python3
"""build_version_json.py — Write project/version.json from GITHUB_REF_NAME.

Usage (called by release.yml):
    python project/scripts/build_version_json.py

Environment variables read:
    GITHUB_REF_NAME  — e.g. "v0.5.1" (set automatically by GitHub Actions)
    GITHUB_SHA       — full commit SHA (set automatically by GitHub Actions)
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
VERSION_JSON = REPO_ROOT / "project" / "version.json"


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")[:7]


def main() -> None:
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    version = ref_name.lstrip("v") if ref_name else "0.0.0-dev"

    commit = _git_short_sha()
    build_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    data = {
        "version": version,
        "build_date": build_date,
        "commit": commit,
    }

    VERSION_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[build_version_json] wrote {VERSION_JSON}: {data}")


if __name__ == "__main__":
    main()
