"""Tests for github module __init__.py lazy imports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib

import pytest

from flext_infra.github.pr import FlextInfraPrManager


class TestGithubInit:
    """Test github module __init__.py lazy imports."""

    def test_lazy_import_pr_manager(self) -> None:
        github_module = importlib.import_module("flext_infra.github")
        manager = github_module.FlextInfraPrManager()
        assert isinstance(manager, FlextInfraPrManager)

    def test_getattr_invalid_attribute(self) -> None:
        github_module = importlib.import_module("flext_infra.github")
        with pytest.raises(AttributeError, match=r"module.*has no attribute"):
            _ = github_module.NonexistentAttribute

    def test_dir_returns_all_exports(self) -> None:
        github_module = importlib.import_module("flext_infra.github")
        exports = dir(github_module)
        assert "FlextInfraPrManager" in exports
        assert "FlextInfraPrWorkspaceManager" in exports
        assert "FlextInfraWorkflowLinter" in exports
        assert "FlextInfraWorkflowSyncer" in exports
        assert "SyncOperation" in exports
