"""Tests for FlextInfraRuntimeDevDependencyDetector.

Uses flext_tests matchers (tm) for consistent assertions.
No unittest.mock — uses real service instances and monkeypatch.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import pytest

from flext_core import r
from flext_infra.deps import detector as det_mod
from flext_infra.deps.detector import FlextInfraRuntimeDevDependencyDetector, ddm
from flext_tests import tm


class TestFlextInfraDependencyDetectorModels:
    """Test FlextInfraDependencyDetectorModels namespace."""

    def test_dependency_limits_info_creation(self) -> None:
        """Test DependencyLimitsInfo model creation."""
        info = ddm.DependencyLimitsInfo()
        tm.that(info.python_version, none=True)
        tm.that(info.limits_path, eq="")

    def test_pip_check_report_creation(self) -> None:
        """Test PipCheckReport model creation."""
        report = ddm.PipCheckReport()
        tm.that(report.ok, eq=True)
        tm.that(report.lines, eq=[])

    def test_workspace_dependency_report_creation(self) -> None:
        """Test WorkspaceDependencyReport model creation."""
        report = ddm.WorkspaceDependencyReport(workspace="test-workspace")
        tm.that(report.workspace, eq="test-workspace")
        tm.that(report.projects, eq={})
        tm.that(report.pip_check, none=True)
        tm.that(report.dependency_limits, none=True)


class TestFlextInfraRuntimeDevDependencyDetector:
    """Test FlextInfraRuntimeDevDependencyDetector."""

    def test_detector_initialization(self) -> None:
        """Test detector initializes without errors."""
        detector = FlextInfraRuntimeDevDependencyDetector()
        tm.that(detector, none=False)

    def test_detector_has_required_services(self) -> None:
        """Test detector has all required internal services."""
        detector = FlextInfraRuntimeDevDependencyDetector()
        tm.that(hasattr(detector, "_paths"), eq=True)
        tm.that(hasattr(detector, "_reporting"), eq=True)
        tm.that(hasattr(detector, "_json"), eq=True)
        tm.that(hasattr(detector, "_deps"), eq=True)
        tm.that(hasattr(detector, "_runner"), eq=True)

    def test_parser_all_arguments(self, tmp_path: Path) -> None:
        """Test parser accepts all arguments together."""
        parser = FlextInfraRuntimeDevDependencyDetector._parser(
            tmp_path / "limits.toml",
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
        tm.that(args.project, eq="test")
        tm.that(args.no_pip_check, eq=True)
        tm.that(args.dry_run, eq=True)
        tm.that(args.json_stdout, eq=True)
        tm.that(args.output, eq="/tmp/out.json")
        tm.that(args.quiet, eq=True)
        tm.that(args.no_fail, eq=True)
        tm.that(args.typings, eq=True)
        tm.that(args.apply_typings, eq=True)
        tm.that(args.limits, eq="/custom/limits.toml")

    def test_project_filter_with_single_project(self, tmp_path: Path) -> None:
        """Test _project_filter extracts single project."""
        parser = FlextInfraRuntimeDevDependencyDetector._parser(
            tmp_path / "limits.toml",
        )
        args = parser.parse_args(["--project", "test-proj"])
        result = FlextInfraRuntimeDevDependencyDetector._project_filter(args)
        tm.that(result, eq=["test-proj"])

    def test_project_filter_with_multiple_projects(self, tmp_path: Path) -> None:
        """Test _project_filter extracts multiple projects."""
        parser = FlextInfraRuntimeDevDependencyDetector._parser(
            tmp_path / "limits.toml",
        )
        args = parser.parse_args(["--projects", "proj-a,proj-b,proj-c"])
        result = FlextInfraRuntimeDevDependencyDetector._project_filter(args)
        tm.that(result, eq=["proj-a", "proj-b", "proj-c"])

    def test_project_filter_with_no_filter(self, tmp_path: Path) -> None:
        """Test _project_filter returns None when no filter specified."""
        parser = FlextInfraRuntimeDevDependencyDetector._parser(
            tmp_path / "limits.toml",
        )
        args = parser.parse_args([])
        result = FlextInfraRuntimeDevDependencyDetector._project_filter(args)
        tm.that(result, none=True)


class TestFlextInfraRuntimeDevDependencyDetectorRunMethod:
    """Test FlextInfraRuntimeDevDependencyDetector.run() method.

    Uses monkeypatch for test isolation instead of unittest.mock.
    """

    def test_run_with_no_projects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test run() when no projects are discovered."""

        class FakePaths:
            def workspace_root_from_file(self: Self) -> r[Path]:
                return r[Path].ok(tmp_path)

        class FakeDeps:
            def discover_projects(
                self: Self, root: Path, project_filter: list[str] | None = None
            ) -> r[list[Path]]:
                return r[list[Path]].ok([])

        monkeypatch.setattr(det_mod, "FlextInfraUtilitiesPaths", FakePaths)
        monkeypatch.setattr(det_mod, "FlextInfraDependencyDetectionService", FakeDeps)
        detector = FlextInfraRuntimeDevDependencyDetector()
        result = detector.run(["--no-pip-check"])
        value = tm.ok(result)
        tm.that(value, eq=2)
