"""Tests for FlextInfraRuntimeDevDependencyDetector."""

from __future__ import annotations

from flext_infra.deps.detector import (
    FlextInfraRuntimeDevDependencyDetector,
    ddm,
)


class TestFlextInfraDependencyDetectorModels:
    """Test FlextInfraDependencyDetectorModels namespace."""

    def test_dependency_limits_info_creation(self) -> None:
        """Test DependencyLimitsInfo model creation."""
        info = ddm.DependencyLimitsInfo()
        assert info.python_version is None
        assert info.limits_path == ""

    def test_pip_check_report_creation(self) -> None:
        """Test PipCheckReport model creation."""
        report = ddm.PipCheckReport()
        assert report.ok is True
        assert report.lines == []

    def test_workspace_dependency_report_creation(self) -> None:
        """Test WorkspaceDependencyReport model creation."""
        report = ddm.WorkspaceDependencyReport(workspace="test-workspace")
        assert report.workspace == "test-workspace"
        assert report.projects == {}
        assert report.pip_check is None
        assert report.dependency_limits is None


class TestFlextInfraRuntimeDevDependencyDetector:
    """Test FlextInfraRuntimeDevDependencyDetector."""

    def test_detector_initialization(self) -> None:
        """Test detector initializes without errors."""
        detector = FlextInfraRuntimeDevDependencyDetector()
        assert detector is not None

    def test_detector_has_required_services(self) -> None:
        """Test detector has all required internal services."""
        detector = FlextInfraRuntimeDevDependencyDetector()
        assert hasattr(detector, "_paths")
        assert hasattr(detector, "_reporting")
        assert hasattr(detector, "_json")
        assert hasattr(detector, "_deps")
        assert hasattr(detector, "_runner")
