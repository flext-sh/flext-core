"""Runtime vs dev dependency detector CLI with deptry, pip-check, and typing analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from flext_core import FlextLogger, r
from flext_infra import (
    FlextInfraUtilitiesIo,
    FlextInfraUtilitiesPaths,
    FlextInfraUtilitiesReporting,
    FlextInfraUtilitiesSubprocess,
    m,
    p,
)
from flext_infra.deps._detector_runtime import FlextInfraDependencyDetectorRuntime
from flext_infra.deps.detection import FlextInfraDependencyDetectionService

logger = FlextLogger.create_module_logger(__name__)


class FlextInfraRuntimeDevDependencyDetector:
    """CLI tool for detecting runtime vs dev dependencies across workspace."""

    def __init__(self) -> None:
        """Initialize detector runtime services."""
        super().__init__()
        self.paths = FlextInfraUtilitiesPaths()
        self.reporting = FlextInfraUtilitiesReporting()
        self.json = FlextInfraUtilitiesIo()
        self.deps = FlextInfraDependencyDetectionService()
        self.runner: p.Infra.CommandRunner = FlextInfraUtilitiesSubprocess()
        self.log = logger

    @staticmethod
    def parser(default_limits_path: Path) -> argparse.ArgumentParser:
        """Create argument parser for CLI with deptry, pip-check, and typing options."""
        parser = argparse.ArgumentParser(
            description="Detect runtime vs dev dependencies (deptry + pip check).",
        )
        _ = parser.add_argument(
            "--project",
            metavar="NAME",
            help="Run only for this project (directory name).",
        )
        _ = parser.add_argument(
            "--projects", metavar="NAMES", help="Comma-separated list of project names."
        )
        _ = parser.add_argument(
            "--no-pip-check",
            action="store_true",
            help="Skip pip check (workspace-level).",
        )
        _ = parser.add_argument(
            "--dry-run", action="store_true", help="Do not write report files."
        )
        _ = parser.add_argument(
            "--json",
            action="store_true",
            dest="json_stdout",
            help="Print full report JSON to stdout only (no file write).",
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
    def project_filter(args: argparse.Namespace) -> list[str] | None:
        """Extract project filter list from parsed CLI arguments."""
        if args.project:
            return [args.project]
        if args.projects:
            return [name.strip() for name in args.projects.split(",") if name.strip()]
        return None

    def run(self, argv: list[str] | None = None) -> r[int]:
        """Execute dependency detection and generate workspace report."""
        runtime = FlextInfraDependencyDetectorRuntime(
            detector=self,
            workspace_report_factory=m.Infra.Deps.WorkspaceDependencyReport,
            dependency_limits_factory=m.Infra.Deps.DependencyLimitsInfo,
            pip_check_factory=m.Infra.Deps.PipCheckReport,
        )
        return runtime.run(argv=argv)


def main() -> int:
    """Entry point for dependency detector CLI."""
    result = FlextInfraRuntimeDevDependencyDetector().run()
    if result.is_failure:
        logger.error("deps_detector_failed", error=result.error or "unknown error")
        return 1
    return result.value


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FlextInfraRuntimeDevDependencyDetector", "main"]
