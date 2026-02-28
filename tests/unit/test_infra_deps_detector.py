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

    def test_project_filter_with_single_project(self, tmp_path: Path) -> None:
        """Test _project_filter extracts single project (line 140).

        This tests the --project argument handling.
        """
        parser = FlextInfraRuntimeDevDependencyDetector._parser(
            tmp_path / "limits.toml"
        )
        args = parser.parse_args(["--project", "test-proj"])
        result = FlextInfraRuntimeDevDependencyDetector._project_filter(args)
        assert result == ["test-proj"]

    def test_project_filter_with_multiple_projects(self, tmp_path: Path) -> None:
        """Test _project_filter extracts multiple projects (line 142).

        This tests the --projects argument handling with comma-separated values.
        """
        parser = FlextInfraRuntimeDevDependencyDetector._parser(
            tmp_path / "limits.toml"
        )
        args = parser.parse_args(["--projects", "proj-a,proj-b,proj-c"])
        result = FlextInfraRuntimeDevDependencyDetector._project_filter(args)
        assert result == ["proj-a", "proj-b", "proj-c"]

    def test_project_filter_with_no_filter(self, tmp_path: Path) -> None:
        """Test _project_filter returns None when no filter specified.

        This tests the default behavior when no project filter is provided.
        """
        parser = FlextInfraRuntimeDevDependencyDetector._parser(
            tmp_path / "limits.toml"
        )
        args = parser.parse_args([])
        result = FlextInfraRuntimeDevDependencyDetector._project_filter(args)
        assert result is None


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

    def test_run_with_apply_typings_success(self, tmp_path: Path) -> None:
        """Test run() with --apply-typings adds typing packages (lines 230-245).

        This tests the apply_typings block that runs poetry add for each package.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        src_dir = tmp_path / "proj-a" / "src"
        src_dir.mkdir(parents=True)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )
        mock_deps.get_required_typings.return_value = r[object].ok(
            Mock(model_dump=Mock(return_value={"to_add": ["types-requests"]}))
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 0))

        mock_runner = Mock()
        mock_runner.run_raw.return_value = r[object].ok(Mock(exit_code=0))

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                with patch(
                    "flext_infra.deps.detector.FlextInfraCommandRunner",
                    return_value=mock_runner,
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        detector = FlextInfraRuntimeDevDependencyDetector()
                        result = detector.run([
                            "--typings",
                            "--apply-typings",
                            "--no-pip-check",
                        ])
                        assert result.is_success
                        # Verify poetry add was called
                        mock_runner.run_raw.assert_called_once()

    def test_run_with_apply_typings_non_string_package(self, tmp_path: Path) -> None:
        """Test run() skips non-string packages in to_add list (lines 236-237).

        This tests the isinstance(package, str) check.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        src_dir = tmp_path / "proj-a" / "src"
        src_dir.mkdir(parents=True)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )
        # Return typing dict with non-string package in to_add list
        mock_deps.get_required_typings.return_value = r[object].ok(
            Mock(
                model_dump=Mock(return_value={"to_add": ["types-requests", 123, None]})
            )
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 0))

        mock_runner = Mock()
        mock_runner.run_raw.return_value = r[object].ok(Mock(exit_code=0))

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                with patch(
                    "flext_infra.deps.detector.FlextInfraCommandRunner",
                    return_value=mock_runner,
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        detector = FlextInfraRuntimeDevDependencyDetector()
                        result = detector.run([
                            "--typings",
                            "--apply-typings",
                            "--no-pip-check",
                        ])
                        assert result.is_success
                        # Verify run_raw was called only for string packages
                        assert mock_runner.run_raw.call_count == 1

    def test_run_with_apply_typings_poetry_add_failure(self, tmp_path: Path) -> None:
        """Test run() logs warning when poetry add fails (lines 244-249).

        This tests the error handling when run_raw returns non-zero exit code.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        src_dir = tmp_path / "proj-a" / "src"
        src_dir.mkdir(parents=True)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )
        mock_deps.get_required_typings.return_value = r[object].ok(
            Mock(model_dump=Mock(return_value={"to_add": ["types-requests"]}))
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 0))

        mock_runner = Mock()
        # Simulate poetry add failure with non-zero exit code
        mock_runner.run_raw.return_value = r[object].ok(Mock(exit_code=1))

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                with patch(
                    "flext_infra.deps.detector.FlextInfraCommandRunner",
                    return_value=mock_runner,
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        detector = FlextInfraRuntimeDevDependencyDetector()
                        result = detector.run([
                            "--typings",
                            "--apply-typings",
                            "--no-pip-check",
                        ])
                        assert result.is_success

    def test_run_with_apply_typings_poetry_add_failure_result(
        self, tmp_path: Path
    ) -> None:
        """Test run() logs warning when poetry add returns failure (lines 244-249).

        This tests when run_raw returns a failure FlextResult.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        src_dir = tmp_path / "proj-a" / "src"
        src_dir.mkdir(parents=True)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )
        mock_deps.get_required_typings.return_value = r[object].ok(
            Mock(model_dump=Mock(return_value={"to_add": ["types-requests"]}))
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 0))

        mock_runner = Mock()
        # Simulate poetry add failure with failure result
        mock_runner.run_raw.return_value = r[object].fail("poetry add failed")

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                with patch(
                    "flext_infra.deps.detector.FlextInfraCommandRunner",
                    return_value=mock_runner,
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        detector = FlextInfraRuntimeDevDependencyDetector()
                        result = detector.run([
                            "--typings",
                            "--apply-typings",
                            "--no-pip-check",
                        ])
                        assert result.is_success

    def test_run_with_output_flag(self, tmp_path: Path) -> None:
        """Test run() with --output flag writes to custom path (line 270).

        This tests the custom output path handling.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 0))

        mock_json = Mock()
        mock_json.write.return_value = r[str].ok("written")

        custom_output = tmp_path / "custom_report.json"

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                with patch(
                    "flext_infra.deps.detector.FlextInfraJsonService",
                    return_value=mock_json,
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        detector = FlextInfraRuntimeDevDependencyDetector()
                        result = detector.run([
                            "--output",
                            str(custom_output),
                            "--no-pip-check",
                        ])
                        assert result.is_success
                        # Verify write was called with custom path
                        mock_json.write.assert_called_once()
                        call_args = mock_json.write.call_args
                        assert call_args[0][0] == custom_output

    def test_run_with_issues_and_pip_failure(self, tmp_path: Path) -> None:
        """Test run() returns 1 when issues exist or pip check fails (line 308).

        This tests the final return logic with both deptry issues and pip failures.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({"missing": ["pkg"]}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 5}})
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 1))

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
                    result = detector.run(["--dry-run", "--no-pip-check"])
                    assert result.is_success
                    assert result.value == 1  # Has issues and pip failed

    def test_run_with_workspace_root_resolution_failure(self) -> None:
        """Test run() when workspace root resolution fails (line 140).

        This tests the error handling when workspace root cannot be resolved.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].fail(
            "root not found"
        )

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            detector = FlextInfraRuntimeDevDependencyDetector()
            result = detector.run([])
            assert result.is_failure
            assert (
                "root not found" in result.error
                or "workspace root resolution failed" in result.error
            )

    def test_run_with_project_discovery_failure(self, tmp_path: Path) -> None:
        """Test run() when project discovery fails (line 142).

        This tests the error handling when projects cannot be discovered.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].fail(
            "discovery failed"
        )

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                detector = FlextInfraRuntimeDevDependencyDetector()
                result = detector.run([])
                assert result.is_failure
                assert (
                    "discovery failed" in result.error
                    or "project discovery failed" in result.error
                )

    def test_run_with_deptry_failure(self, tmp_path: Path) -> None:
        """Test run() when deptry execution fails (line 149).

        This tests the error handling when deptry fails.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].fail("deptry failed")

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
                    result = detector.run(["--no-pip-check"])
                    assert result.is_failure
                    assert (
                        "deptry failed" in result.error
                        or "deptry run failed" in result.error
                    )

    def test_run_with_typings_detection_failure(self, tmp_path: Path) -> None:
        """Test run() when typing detection fails (line 206).

        This tests the error handling when typing detection fails.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        src_dir = tmp_path / "proj-a" / "src"
        src_dir.mkdir(parents=True)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )
        mock_deps.get_required_typings.return_value = r[object].fail(
            "typing detection failed"
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
                    result = detector.run(["--typings", "--no-pip-check"])
                    assert result.is_failure
                    assert (
                        "typing detection failed" in result.error
                        or "typing dependency detection failed" in result.error
                    )

    def test_run_with_report_directory_creation_failure(self, tmp_path: Path) -> None:
        """Test run() when report directory creation fails (line 275-276).

        This tests the error handling when report directory cannot be created.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 0))

        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "readonly"

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                with patch(
                    "flext_infra.deps.detector.FlextInfraReportingService",
                    return_value=mock_reporting,
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch(
                            "pathlib.Path.mkdir",
                            side_effect=OSError("Permission denied"),
                        ):
                            detector = FlextInfraRuntimeDevDependencyDetector()
                            result = detector.run(["--no-pip-check"])
                            assert result.is_failure
                            assert "failed to create report directory" in result.error

    def test_run_with_json_write_failure(self, tmp_path: Path) -> None:
        """Test run() when JSON write fails (line 282).

        This tests the error handling when JSON write fails.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 0))

        mock_json = Mock()
        mock_json.write.return_value = r[str].fail("write failed")

        mock_reporting = Mock()
        mock_reporting.get_report_dir.return_value = tmp_path / "reports"

        with patch(
            "flext_infra.deps.detector.FlextInfraPathResolver",
            return_value=mock_paths,
        ):
            with patch(
                "flext_infra.deps.detector.FlextInfraDependencyDetectionService",
                return_value=mock_deps,
            ):
                with patch(
                    "flext_infra.deps.detector.FlextInfraJsonService",
                    return_value=mock_json,
                ):
                    with patch(
                        "flext_infra.deps.detector.FlextInfraReportingService",
                        return_value=mock_reporting,
                    ):
                        with patch("pathlib.Path.exists", return_value=True):
                            with patch("pathlib.Path.mkdir"):
                                detector = FlextInfraRuntimeDevDependencyDetector()
                                result = detector.run(["--no-pip-check"])
                                assert result.is_failure
                                assert (
                                    "write failed" in result.error
                                    or "failed to write report" in result.error
                                )

    def test_run_with_no_fail_flag_with_issues(self, tmp_path: Path) -> None:
        """Test run() with --no-fail always returns 0 even with issues (line 307).

        This tests the --no-fail flag behavior.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({"missing": ["pkg"]}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 5}})
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 1))

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
                    result = detector.run(["--no-fail", "--dry-run", "--no-pip-check"])
                    assert result.is_success
                    assert result.value == 0  # Always returns 0 with --no-fail

    def test_run_with_json_stdout_flag(self, tmp_path: Path) -> None:
        """Test run() with --json flag returns 0 without writing (line 266).

        This tests the --json flag behavior that prints to stdout only.
        """
        mock_paths = Mock()
        mock_paths.workspace_root_from_file.return_value = r[Path].ok(tmp_path)

        mock_deps = Mock()
        mock_deps.discover_projects.return_value = r[list[Path]].ok([
            tmp_path / "proj-a"
        ])
        mock_deps.run_deptry.return_value = r[tuple].ok(({}, 0))
        mock_deps.build_project_report.return_value = Mock(
            model_dump=Mock(return_value={"deptry": {"raw_count": 0}})
        )
        mock_deps.run_pip_check.return_value = r[tuple].ok(([], 0))

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
                    result = detector.run(["--json", "--no-pip-check"])
                    assert result.is_success
                    assert result.value == 0
