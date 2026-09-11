"""Centralized project-metadata fixtures for all project test suites.

Provides:
- project_metadata: Fixture returning ``m.ProjectMetadata`` for the current project
- project_tool_flext: Fixture returning ``m.ProjectToolFlext`` for the current project

All fixtures derive data exclusively from the SSOT
(``u.read_project_metadata``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_tests import p, u


def _find_project_root() -> Path:
    """Walk up from CWD to find the nearest pyproject.toml with [project].name."""
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            return parent
    msg = f"no pyproject.toml found above {cwd}"
    raise FileNotFoundError(msg)


@pytest.fixture
def project_metadata() -> p.ProjectMetadata:
    """Return the canonical ``m.ProjectMetadata`` for the project under test."""
    return u.read_project_metadata(_find_project_root()).unwrap()


@pytest.fixture
def project_tool_flext() -> p.ProjectToolFlext:
    """Return the ``[tool.flext]`` config for the project under test."""
    return u.read_project_metadata(_find_project_root()).unwrap().flext


__all__: list[str] = ["project_metadata", "project_tool_flext"]
