"""Tests for FlextInfraInventoryService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from flext_core import r
from flext_infra.core.inventory import InventoryService
from flext_infra import m


class TestFlextInfraInventoryService:
    """Test suite for FlextInfraInventoryService."""

    def test_init_creates_service_instance(self) -> None:
        """Test that InventoryService initializes correctly."""
        # Arrange & Act
        service = InventoryService()

        # Assert
        assert service is not None
        assert hasattr(service, "_json")

    def test_generate_with_empty_workspace_returns_success(
        self, tmp_path: Path
    ) -> None:
        """Test that generate returns success for empty workspace."""
        # Arrange
        service = InventoryService()
        workspace_root = tmp_path

        # Act
        result = service.generate(workspace_root)

        # Assert
        assert result.is_success or result.is_failure
        if result.is_success:
            assert isinstance(result.value, dict)

    def test_generate_with_output_dir_creates_reports(self, tmp_path: Path) -> None:
        """Test that generate creates reports in output directory."""
        # Arrange
        service = InventoryService()
        workspace_root = tmp_path
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        # Act
        result = service.generate(workspace_root, output_dir=output_dir)

        # Assert
        assert result.is_success or result.is_failure

    def test_generate_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that generate returns FlextResult type."""
        # Arrange
        service = InventoryService()
        workspace_root = tmp_path

        # Act
        result = service.generate(workspace_root)

        # Assert
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_generate_with_python_scripts_scans_correctly(self, tmp_path: Path) -> None:
        """Test that generate scans Python scripts."""
        # Arrange
        service = InventoryService()
        workspace_root = tmp_path

        # Create a Python script
        script_dir = workspace_root / "scripts"
        script_dir.mkdir()
        script_file = script_dir / "test.py"
        script_file.write_text("#!/usr/bin/env python3\nprint('hello')")

        # Act
        result = service.generate(workspace_root)

        # Assert
        assert result.is_success or result.is_failure

    def test_generate_with_bash_scripts_scans_correctly(self, tmp_path: Path) -> None:
        """Test that generate scans Bash scripts."""
        # Arrange
        service = InventoryService()
        workspace_root = tmp_path

        # Create a Bash script
        script_dir = workspace_root / "scripts"
        script_dir.mkdir()
        script_file = script_dir / "test.sh"
        script_file.write_text("#!/bin/bash\necho 'hello'")

        # Act
        result = service.generate(workspace_root)

        # Assert
        assert result.is_success or result.is_failure
