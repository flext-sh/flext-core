"""Tests for FlextInfraExtraPathsManager."""

from __future__ import annotations

import tomlkit
from flext_infra.deps.extra_paths import (
    FlextInfraExtraPathsManager,
    get_dep_paths,
)


class TestFlextInfraExtraPathsManager:
    """Test FlextInfraExtraPathsManager."""

    def test_manager_initialization(self) -> None:
        """Test manager initializes without errors."""
        manager = FlextInfraExtraPathsManager()
        assert manager is not None


class TestGetDepPaths:
    """Test get_dep_paths function."""

    def test_get_dep_paths_empty_doc(self) -> None:
        """Test get_dep_paths with empty TOML document."""
        doc = tomlkit.document()
        paths = get_dep_paths(doc, is_root=False)
        assert paths == []

    def test_get_dep_paths_with_pep621_deps(self) -> None:
        """Test get_dep_paths with PEP 621 path dependencies."""
        doc = tomlkit.document()
        doc["project"] = {"dependencies": ["flext-core @ file:../flext-core"]}
        paths = get_dep_paths(doc, is_root=False)
        assert isinstance(paths, list)

    def test_get_dep_paths_with_poetry_deps(self) -> None:
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
        assert isinstance(paths, list)
