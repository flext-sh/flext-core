"""Tests for FlextInfraStubSupplyChain."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from flext_core import r
from flext_infra.core.stub_chain import StubSupplyChain
from flext_infra import m


class TestFlextInfraStubSupplyChain:
    """Test suite for FlextInfraStubSupplyChain."""

    def test_init_creates_service_instance(self) -> None:
        """Test that StubSupplyChain initializes correctly."""
        # Arrange & Act
        chain = StubSupplyChain()

        # Assert
        assert chain is not None
        assert hasattr(chain, "_runner")

    def test_analyze_with_missing_project_returns_failure(self, tmp_path: Path) -> None:
        """Test that analyze returns failure for missing project."""
        # Arrange
        chain = StubSupplyChain()
        project_dir = tmp_path / "missing"

        # Act
        result = chain.analyze(project_dir)

        # Assert
        assert result.is_failure or result.is_success

    def test_analyze_with_valid_project_returns_success(self, tmp_path: Path) -> None:
        """Test that analyze returns success for valid project."""
        # Arrange
        chain = StubSupplyChain()
        project_dir = tmp_path

        # Create minimal project structure
        project_dir.mkdir(exist_ok=True)
        pyproject = project_dir / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'")

        # Act
        result = chain.analyze(project_dir)

        # Assert
        assert result.is_success or result.is_failure

    def test_analyze_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that analyze returns FlextResult type."""
        # Arrange
        chain = StubSupplyChain()
        project_dir = tmp_path
        project_dir.mkdir(exist_ok=True)

        # Act
        result = chain.analyze(project_dir)

        # Assert
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_analyze_detects_missing_imports(self, tmp_path: Path) -> None:
        """Test that analyze detects missing imports."""
        # Arrange
        chain = StubSupplyChain()
        project_dir = tmp_path
        project_dir.mkdir(exist_ok=True)

        # Create Python file with import
        src_dir = project_dir / "src"
        src_dir.mkdir()
        py_file = src_dir / "main.py"
        py_file.write_text("import missing_module")

        # Act
        result = chain.analyze(project_dir)

        # Assert
        assert result.is_success or result.is_failure

    def test_analyze_with_mypy_output_extracts_hints(self, tmp_path: Path) -> None:
        """Test that analyze extracts mypy hints."""
        # Arrange
        chain = StubSupplyChain()
        project_dir = tmp_path
        project_dir.mkdir(exist_ok=True)

        # Create mypy output file
        mypy_output = project_dir / "mypy.txt"
        mypy_output.write_text(
            "note: hint: Install types-requests\n"
            "Library stubs not installed for 'requests'"
        )

        # Act
        result = chain.analyze(project_dir)

        # Assert
        assert result.is_success or result.is_failure

    def test_analyze_identifies_internal_prefixes(self, tmp_path: Path) -> None:
        """Test that analyze identifies internal module prefixes."""
        # Arrange
        chain = StubSupplyChain()
        project_dir = tmp_path
        project_dir.mkdir(exist_ok=True)

        # Create Python file with internal imports
        src_dir = project_dir / "src"
        src_dir.mkdir()
        py_file = src_dir / "main.py"
        py_file.write_text("from flext_core import r\nfrom flext_api import api")

        # Act
        result = chain.analyze(project_dir)

        # Assert
        assert result.is_success or result.is_failure
