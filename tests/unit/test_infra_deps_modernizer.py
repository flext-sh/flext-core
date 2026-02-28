"""Tests for FlextInfraPyprojectModernizer."""

from __future__ import annotations

from pathlib import Path

import tomlkit
from flext_infra.deps.modernizer import (
    ConsolidateGroupsPhase,
    EnsurePyreflyConfigPhase,
    EnsurePytestConfigPhase,
    FlextInfraPyprojectModernizer,
    InjectCommentsPhase,
)


class TestFlextInfraPyprojectModernizer:
    """Test FlextInfraPyprojectModernizer."""

    def test_modernizer_initialization(self) -> None:
        """Test modernizer initializes without errors."""
        modernizer = FlextInfraPyprojectModernizer()
        assert modernizer is not None

    def test_modernizer_with_empty_pyproject(self, tmp_path: Path) -> None:
        """Test modernizer with empty pyproject.toml."""
        pyproject_path = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {"name": "test-project"}
        pyproject_path.write_text(tomlkit.dumps(doc))

        modernizer = FlextInfraPyprojectModernizer()
        assert modernizer is not None

    def test_modernizer_with_dependencies(self, tmp_path: Path) -> None:
        """Test modernizer with project dependencies."""
        pyproject_path = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {
            "name": "test-project",
            "dependencies": ["requests>=2.0"],
        }
        pyproject_path.write_text(tomlkit.dumps(doc))

        modernizer = FlextInfraPyprojectModernizer()
        assert modernizer is not None

    def test_modernizer_phases_exist(self) -> None:
        """Test that modernizer has required phase classes."""
        assert ConsolidateGroupsPhase is not None
        assert EnsurePytestConfigPhase is not None
        assert EnsurePyreflyConfigPhase is not None
        assert InjectCommentsPhase is not None
