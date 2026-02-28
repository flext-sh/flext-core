"""Tests for FlextInfraExtraPathsManager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomlkit

from flext_infra.deps.extra_paths import (
    FlextInfraExtraPathsManager,
    get_dep_paths,
)


class TestFlextInfraExtraPathsManager:
    """Test suite for FlextInfraExtraPathsManager."""

    def test_manager_initialization(self) -> None:
        """Test that manager initializes without errors."""
        manager = FlextInfraExtraPathsManager()
        assert manager is not None

    def test_get_dep_paths_empty_doc(self, tmp_path: Path) -> None:
        """Test get_dep_paths with empty TOML document."""
        doc = tomlkit.document()
        paths = get_dep_paths(doc, is_root=False)
        assert paths == []

    def test_get_dep_paths_with_pep621_deps(self, tmp_path: Path) -> None:
        """Test get_dep_paths with PEP 621 path dependencies."""
        doc = tomlkit.document()
        doc["project"] = {"dependencies": ["flext-core @ file:../flext-core"]}
        paths = get_dep_paths(doc, is_root=False)
        assert len(paths) > 0

    def test_get_dep_paths_with_poetry_deps(self, tmp_path: Path) -> None:
        """Test get_dep_paths with Poetry path dependencies."""
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {
                "dependencies": {
                    "flext-core": {"path": "../flext-core"},
                },
            },
        }
        paths = get_dep_paths(doc, is_root=False)
        assert len(paths) > 0

    @patch("flext_infra.deps.extra_paths.PathResolver")
    def test_manager_with_mocked_resolver(
        self,
        mock_resolver: MagicMock,
    ) -> None:
        """Test manager with mocked path resolver."""
        manager = FlextInfraExtraPathsManager()
        assert manager is not None
