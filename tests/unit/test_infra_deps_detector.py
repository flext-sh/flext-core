"""Tests for FlextInfraRuntimeDevDependencyDetector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from flext_core import r
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

    def test_parser_all_arguments(self, tmp_path: Path) -> None:
        """Test parser accepts all arguments together."""
        parser = FlextInfraRuntimeDevDependencyDetector._parser(
            tmp_path / "limits.toml"
        )
        args = parser.parse_args([
            "--project",
            "test",
            "--no-pip-check",
            "--dry-run",
            "--json",
            "-o",
            "/tmp/out.json",
            "-q",
            "--no-fail",
            "--typings",
            "--apply-typings",
            "--limits",
            "/custom/limits.toml",
        ])
        assert args.project == "test"
        assert args.no_pip_check is True
        assert args.dry_run is True
        assert args.json_stdout is True
        assert args.output == "/tmp/out.json"
        assert args.quiet is True
        assert args.no_fail is True
        assert args.typings is True
        assert args.apply_typings is True
        assert args.limits == "/custom/limits.toml"


class TestFlextInfraRuntimeDevDependencyDetectorRunMethod:
    """Test FlextInfraRuntimeDevDependencyDetector.run() method."""

    def test_run_with_no_projects(self, tmp_path: Path) -> None:
        """Test run() when no projects are discovered."""
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([])

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                detector = FlextInfraRuntimeDevDependencyDetector()
                result = detector.run(["--no-pip-check"])
                assert result.is_success
                assert result.value == 2

    def test_run_with_deptry_missing(self, tmp_path: Path) -> None:
        """Test run() when deptry is not installed."""
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                with patch("pathlib.Path.exists", return_value=False):
                    detector = FlextInfraRuntimeDevDependencyDetector()
                    result = detector.run(["--no-pip-check"])
                    assert result.is_success
                    assert result.value == 3

    def test_run_with_projects_and_deptry(self, tmp_path: Path) -> None:
        """Test run() processes projects when deptry is available."""
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok((
            {"missing": [], "unused": []},
            0,
        ))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                with patch("pathlib.Path.exists", return_value=True):
                    detector = FlextInfraRuntimeDevDependencyDetector()
                    result = detector.run(["--no-pip-check", "--dry-run"])
                    assert result.is_success
                    assert result.value == 0
