"""Pytest fixtures for FLEXT infra tests.

Provides reusable fixtures using flext_tests base classes (c, m, WorkspaceFactory).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_tests import c
from tests.infra.workspace_factory import WorkspaceFactory


@pytest.fixture
def real_toml_project(tmp_path: Path) -> Path:
    """Create a real project with valid pyproject.toml using WorkspaceFactory."""
    factory = WorkspaceFactory()
    return factory.create_minimal(tmp_path=tmp_path, name="test-project")


@pytest.fixture
def real_makefile_project(tmp_path: Path) -> Path:
    """Create a real project with valid Makefile using WorkspaceFactory."""
    factory = WorkspaceFactory()
    project_root = factory.create_minimal(tmp_path=tmp_path, name="makefile-project")
    makefile_content = f"""\
.PHONY: {c.Infra.Cli.MAKE_HELP} {c.Infra.Cli.MAKE_SETUP} check test

{c.Infra.Cli.MAKE_HELP}:
\t@echo "Available targets"

{c.Infra.Cli.MAKE_SETUP}:
\t@echo "Setting up"

check:
\t@echo "Checking"

test:
\t@echo "Testing"
"""
    (project_root / "Makefile").write_text(makefile_content)
    return project_root


@pytest.fixture
def real_python_package(tmp_path: Path) -> Path:
    """Create a real Python package with src layout using WorkspaceFactory."""
    factory = WorkspaceFactory()
    project_root = factory.create_minimal(tmp_path=tmp_path, name="test-pkg")
    src_dir = project_root / "src" / "test_pkg"
    (src_dir / "__init__.py").write_text(
        f'"""Test package."""\n__version__ = "{factory.default_version}"\n'
    )
    return project_root


@pytest.fixture
def real_workspace(tmp_path: Path) -> Path:
    """Create a real multi-project workspace using WorkspaceFactory."""
    factory = WorkspaceFactory()
    return factory.create_workspace(tmp_path=tmp_path, projects=3)


@pytest.fixture
def real_docs_project(tmp_path: Path) -> Path:
    """Create a real project with documentation using WorkspaceFactory."""
    factory = WorkspaceFactory()
    return factory.create_full(tmp_path=tmp_path, name="docs-project")


__all__ = [
    "real_docs_project",
    "real_makefile_project",
    "real_python_package",
    "real_toml_project",
    "real_workspace",
]
