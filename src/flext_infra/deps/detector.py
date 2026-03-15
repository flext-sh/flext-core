"""Runtime vs dev dependency detector CLI with deptry, pip-check, and typing analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import flext_infra as infra
from flext_core import FlextLogger, r
from flext_infra import m, u
from flext_infra.deps._detector_runtime import (
    DepsService,
    FlextInfraDependencyDetectorRuntime,
    JsonService,
    PathsService,
    ReportingService,
    RunnerService,
)

FlextInfraUtilitiesIo = infra.FlextInfraUtilitiesIo
FlextInfraUtilitiesPaths = infra.FlextInfraUtilitiesPaths
FlextInfraUtilitiesReporting = infra.FlextInfraUtilitiesReporting
FlextInfraUtilitiesSubprocess = infra.FlextInfraUtilitiesSubprocess
FlextInfraDependencyDetectionService = infra.FlextInfraDependencyDetectionService


class FlextInfraRuntimeDevDependencyDetector:
    """CLI tool for detecting runtime vs dev dependencies across workspace."""

    log = FlextLogger.create_module_logger(__name__)
    paths: PathsService
    reporting: ReportingService
    json: JsonService
    deps: DepsService
    runner: RunnerService

    def __init__(self) -> None:
        """Initialize detector runtime services."""
        super().__init__()
        paths_factory = self._resolve_factory(
            "FlextInfraUtilitiesPaths", FlextInfraUtilitiesPaths
        )
        reporting_factory = self._resolve_factory(
            "FlextInfraUtilitiesReporting", FlextInfraUtilitiesReporting
        )
        json_factory = self._resolve_factory(
            "FlextInfraUtilitiesIo", FlextInfraUtilitiesIo
        )
        deps_factory = self._resolve_factory(
            "FlextInfraDependencyDetectionService", FlextInfraDependencyDetectionService
        )
        runner_factory = self._resolve_factory(
            "FlextInfraUtilitiesSubprocess", FlextInfraUtilitiesSubprocess
        )
        paths_obj = paths_factory()
        reporting_obj = reporting_factory()
        json_obj = json_factory()
        deps_obj = deps_factory()
        runner_obj = runner_factory()
        if isinstance(paths_obj, PathsService):
            self.paths = paths_obj
        else:
            self.paths = FlextInfraUtilitiesPaths()
        if isinstance(reporting_obj, ReportingService):
            self.reporting = reporting_obj
        else:
            self.reporting = FlextInfraUtilitiesReporting()
        if isinstance(json_obj, JsonService):
            self.json = json_obj
        else:
            self.json = FlextInfraUtilitiesIo()
        if isinstance(deps_obj, DepsService):
            self.deps = deps_obj
        else:
            self.deps = FlextInfraDependencyDetectionService()
        if isinstance(runner_obj, RunnerService):
            self.runner = runner_obj
        else:
            self.runner = FlextInfraUtilitiesSubprocess()
        self.log = self.log

    @staticmethod
    def _resolve_factory(name: str, default: type[object]) -> type[object]:
        deps_module = sys.modules.get("flext_infra.deps")
        root_module = sys.modules.get("flext_infra")
        candidate = (
            getattr(deps_module, name, default) if deps_module is not None else default
        )
        if candidate is default and root_module is not None:
            root_candidate = getattr(root_module, name, default)
            if root_candidate is not default:
                return root_candidate
        return candidate

    @staticmethod
    def parser(default_limits_path: Path) -> argparse.ArgumentParser:
        """Create argument parser for CLI with deptry, pip-check, and typing options."""
        parser = u.Infra.create_parser(
            prog="flext-infra deps detect",
            description="Detect runtime vs dev dependencies (deptry + pip check).",
            include_apply=True,
            include_project=True,
            include_format=True,
        )
        _ = parser.add_argument(
            "--no-pip-check",
            action="store_true",
            help="Skip pip check (workspace-level).",
        )
        _ = parser.add_argument(
            "-o",
            "--output",
            metavar="FILE",
            help="Write report to this path (default: .reports/dependencies/detect-runtime-dev-latest.json).",
        )
        _ = parser.add_argument(
            "-q", "--quiet", action="store_true", help="Minimal output (summary only)."
        )
        _ = parser.add_argument(
            "--no-fail", action="store_true", help="Always exit 0 (report only)."
        )
        _ = parser.add_argument(
            "--typings",
            action="store_true",
            help="Detect required typing libraries (types-*).",
        )
        _ = parser.add_argument(
            "--apply-typings",
            action="store_true",
            help="Add missing typings with poetry add --group typings.",
        )
        _ = parser.add_argument(
            "--limits",
            metavar="FILE",
            default=str(default_limits_path),
            help="Path to dependency_limits.toml.",
        )
        return parser

    @staticmethod
    def project_filter(cli: u.Infra.CliArgs) -> list[str] | None:
        """Extract project filter list from parsed CLI arguments."""
        return cli.project_names()

    def run(
        self: FlextInfraRuntimeDevDependencyDetector,
        argv: list[str] | None = None,
    ) -> r[int]:
        """Execute dependency detection and generate workspace report."""
        runtime = FlextInfraDependencyDetectorRuntime(
            detector=self,
            workspace_report_factory=m.Infra.Deps.WorkspaceDependencyReport,
            dependency_limits_factory=m.Infra.Deps.DependencyLimitsInfo,
            pip_check_factory=m.Infra.Deps.PipCheckReport,
        )
        return runtime.run(argv=argv)


@runtime_checkable
class RunnableDetector(Protocol):
    def run(self, argv: list[str] | None = None) -> r[int]: ...


def main() -> int:
    """Entry point for dependency detector CLI."""
    detector_type: type[object] = FlextInfraRuntimeDevDependencyDetector
    deps_module = sys.modules.get("flext_infra.deps")
    if deps_module is not None:
        deps_type = getattr(
            deps_module,
            "FlextInfraRuntimeDevDependencyDetector",
            detector_type,
        )
        if isinstance(deps_type, type):
            detector_type = deps_type
    root_module = sys.modules.get("flext_infra")
    if root_module is not None:
        root_type = getattr(
            root_module,
            "FlextInfraRuntimeDevDependencyDetector",
            detector_type,
        )
        if isinstance(root_type, type):
            detector_type = root_type
    detector_obj = detector_type()
    if not isinstance(detector_obj, RunnableDetector):
        return 1
    detector = detector_obj
    result = detector.run()
    if result.is_failure:
        logger = getattr(detector, "log", None)
        if logger is not None and hasattr(logger, "error"):
            logger.error("deps_detector_failed", error=result.error or "unknown error")
        return 1
    return result.value


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FlextInfraRuntimeDevDependencyDetector",
    "FlextInfraUtilitiesIo",
    "FlextInfraUtilitiesPaths",
    "FlextInfraUtilitiesReporting",
    "FlextInfraUtilitiesSubprocess",
    "main",
]
