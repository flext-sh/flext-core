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
    u,
)
from flext_infra.deps._detector_runtime import FlextInfraDependencyDetectorRuntime
from flext_infra.deps.detection import FlextInfraDependencyDetectionService


class FlextInfraRuntimeDevDependencyDetector:
    """CLI tool for detecting runtime vs dev dependencies across workspace."""

    log = FlextLogger.create_module_logger(__name__)

    def __init__(self) -> None:
        """Initialize detector runtime services."""
        super().__init__()
        self.paths = FlextInfraUtilitiesPaths()
        self.reporting = FlextInfraUtilitiesReporting()
        self.json = FlextInfraUtilitiesIo()
        self.deps = FlextInfraDependencyDetectionService()
        self.runner: p.Infra.CommandRunner = FlextInfraUtilitiesSubprocess()
        self.log = self.log

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
        if cli.project:
            return [cli.project]
        if cli.projects:
            return [name.strip() for name in cli.projects.split(",") if name.strip()]
        return None

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


def main() -> int:
    """Entry point for dependency detector CLI."""
    detector = FlextInfraRuntimeDevDependencyDetector()
    result = detector.run()
    if result.is_failure:
        logger = getattr(detector, "log", None)
        if logger is not None and hasattr(logger, "error"):
            logger.error("deps_detector_failed", error=result.error or "unknown error")
        return 1
    return result.value


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FlextInfraRuntimeDevDependencyDetector", "main"]
