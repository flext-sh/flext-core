"""Pytest collection policy for flext-core's public test suite."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pytest import Collector



@pytest.hookimpl(tryfirst=True)
def pytest_collect_file(file_path: Path, parent: Collector) -> pytest.Module | None:
    """Keep ``test_*.py`` under pytest's native collector before docstring scans."""
    if file_path.suffix == ".py" and file_path.name.startswith("test_"):
        return pytest.Module.from_parent(parent, path=file_path)
    return None
