from __future__ import annotations

from pathlib import Path

from flext_infra import FlextInfraRuntimeDevDependencyDetector, u
from flext_tests import tm


class TestFlextInfraRuntimeDevDependencyDetectorInit:
    def test_detector_initialization(self) -> None:
        detector = FlextInfraRuntimeDevDependencyDetector()
        tm.that(
            detector.__class__.__name__, eq="FlextInfraRuntimeDevDependencyDetector"
        )

    def test_detector_has_required_services(self) -> None:
        detector = FlextInfraRuntimeDevDependencyDetector()
        tm.that(hasattr(detector, "paths"), eq=True)
        tm.that(hasattr(detector, "reporting"), eq=True)
        tm.that(hasattr(detector, "json"), eq=True)
        tm.that(hasattr(detector, "deps"), eq=True)
        tm.that(hasattr(detector, "runner"), eq=True)

    def test_parser_all_arguments(self, tmp_path: Path) -> None:
        parser = FlextInfraRuntimeDevDependencyDetector.parser(tmp_path / "limits.toml")
        args = parser.parse_args([
            "--project",
            "test",
            "--no-pip-check",
            "--dry-run",
            "--format",
            "json",
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
        tm.that(args.output_format, eq="json")
        tm.that(args.output, eq="/tmp/out.json")
        tm.that(args.quiet, eq=True)
        tm.that(args.no_fail, eq=True)
        tm.that(args.typings, eq=True)
        tm.that(args.apply_typings, eq=True)
        tm.that(args.limits, eq="/custom/limits.toml")

    def test_project_filter_with_single_project(self, tmp_path: Path) -> None:
        parser = FlextInfraRuntimeDevDependencyDetector.parser(tmp_path / "limits.toml")
        args = u.Infra.resolve(parser.parse_args(["--project", "test-proj"]))
        tm.that(
            FlextInfraRuntimeDevDependencyDetector.project_filter(args),
            eq=["test-proj"],
        )

    def test_project_filter_with_multiple_projects(self, tmp_path: Path) -> None:
        parser = FlextInfraRuntimeDevDependencyDetector.parser(tmp_path / "limits.toml")
        args = u.Infra.resolve(
            parser.parse_args(["--projects", "proj-a,proj-b,proj-c"])
        )
        tm.that(
            FlextInfraRuntimeDevDependencyDetector.project_filter(args),
            eq=["proj-a", "proj-b", "proj-c"],
        )

    def test_project_filter_with_no_filter(self, tmp_path: Path) -> None:
        parser = FlextInfraRuntimeDevDependencyDetector.parser(tmp_path / "limits.toml")
        args = u.Infra.resolve(parser.parse_args([]))
        tm.that(FlextInfraRuntimeDevDependencyDetector.project_filter(args), eq=None)
