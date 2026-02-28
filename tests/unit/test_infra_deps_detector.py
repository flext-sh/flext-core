"""Tests for FlextInfraRuntimeDevDependencyDetector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flext_infra.deps.detector import (
    FlextInfraDependencyDetectorModels,
    FlextInfraRuntimeDevDependencyDetector,
)


class TestFlextInfraRuntimeDevDependencyDetector:
    """Test suite for FlextInfraRuntimeDevDependencyDetector."""

    def test_detector_initialization(self) -> None:
        """Test that detector initializes without errors."""
        detector = FlextInfraRuntimeDevDependencyDetector()
        assert detector is not None

    def test_dependency_limits_info_model(self) -> None:
        """Test DependencyLimitsInfo model creation."""
        ddm = FlextInfraDependencyDetectorModels
        info = ddm.DependencyLimitsInfo(
            python_version="3.11",
            limits_path="/path/to/limits.toml",
        )
        assert info.python_version == "3.11"
        assert info.limits_path == "/path/to/limits.toml"

    def test_pip_check_report_model(self) -> None:
        """Test PipCheckReport model creation."""
        ddm = FlextInfraDependencyDetectorModels
        report = ddm.PipCheckReport(ok=True, lines=[])
        assert report.ok is True
        assert report.lines == []

    def test_workspace_dependency_report_model(self) -> None:
        """Test WorkspaceDependencyReport model creation."""
        ddm = FlextInfraDependencyDetectorModels
        report = ddm.WorkspaceDependencyReport(
            workspace="test-workspace",
            projects={},
        )
        assert report.workspace == "test-workspace"
        assert report.projects == {}

    @patch("flext_infra.deps.detector.PathResolver")
    @patch("flext_infra.deps.detector.ReportingService")
    def test_detector_with_mocked_services(
        self,
        mock_reporting: MagicMock,
        mock_paths: MagicMock,
    ) -> None:
        """Test detector with mocked services."""
        detector = FlextInfraRuntimeDevDependencyDetector()
        assert detector is not None
