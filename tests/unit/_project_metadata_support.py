"""Shared project metadata test helpers."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def write_pyproject(tmp_path: Path, body: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(dedent(body).lstrip(), encoding="utf-8")
    return tmp_path
