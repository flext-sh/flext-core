"""Tests for FlextInfraDependencyDetectionService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flext_infra.deps.detection import (
    FlextInfraDependencyDetectionModels,
    FlextInfraDependencyDetectionService,
)


class TestFlextInfraDependencyDetectionService:
    """Test suite for FlextInfraDependencyDetectionService."""

    def test_service_initialization(self) -> None:
        """Test that service initializes without errors."""
        service = FlextInfraDependencyDetectionService()
        assert service is not None

    def test_deptry_report_model_creation(self) -> None:
        """Test DeptryReport model creation."""
        dm = FlextInfraDependencyDetectionModels
        report = dm.DeptryReport(
            missing=[],
            unused=[],
            transitive=[],
            dev_in_runtime=[],
            raw_count=0,
        )
        assert report.missing == []
        assert report.raw_count == 0

    def test_project_dependency_report_model(self) -> None:
        """Test ProjectDependencyReport model creation."""
        dm = FlextInfraDependencyDetectionModels
        deptry = dm.DeptryReport()
        report = dm.ProjectDependencyReport(
            project="test-project",
            deptry=deptry,
        )
        assert report.project == "test-project"

    def test_typings_report_model(self) -> None:
        """Test TypingsReport model creation."""
        dm = FlextInfraDependencyDetectionModels
        report = dm.TypingsReport(
            required_packages=["types-pyyaml"],
            hinted=[],
            missing_modules=[],
            current=[],
            to_add=[],
            to_remove=[],
        )
        assert "types-pyyaml" in report.required_packages

    @patch("flext_infra.deps.detection.CommandRunner")
    def test_service_with_mocked_runner(self, mock_runner: MagicMock) -> None:
        """Test service behavior with mocked external runner."""
        service = FlextInfraDependencyDetectionService()
        assert service is not None
