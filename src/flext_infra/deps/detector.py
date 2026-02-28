"""Runtime vs dev dependency detector CLI with deptry, pip-check, and typing analysis."""

from __future__ import annotations

import argparse
import os
from collections.abc import MutableMapping
from pathlib import Path

from flext_core import FlextLogger, FlextResult, r, t
from pydantic import Field

from flext_infra import (
    FlextInfraCommandRunner,
    FlextInfraJsonService,
    FlextInfraPathResolver,
    FlextInfraReportingService,
    c,
    m,
)
from flext_infra.deps.detection import FlextInfraDependencyDetectionService

logger = FlextLogger.create_module_logger(__name__)


class FlextInfraDependencyDetectorModels(m):
    """Pydantic models for dependency detector reports and configuration."""

    class DependencyLimitsInfo(m.ArbitraryTypesModel):
        """Dependency limits configuration metadata."""

        python_version: str | None = None
        limits_path: str = Field(default="")

    class PipCheckReport(m.ArbitraryTypesModel):
        """Pip check execution report with status and output lines."""

        ok: bool = True
        lines: list[str] = Field(default_factory=list)

    class WorkspaceDependencyReport(m.ArbitraryTypesModel):
        """Workspace-level dependency analysis report aggregating all projects."""

        workspace: str
        projects: MutableMapping[str, MutableMapping[str, t.ConfigMapValue]] = Field(
            default_factory=dict,
        )
        pip_check: MutableMapping[str, t.ScalarValue] | None = None
        dependency_limits: MutableMapping[str, t.ScalarValue] | None = None


ddm = FlextInfraDependencyDetectorModels


class FlextInfraRuntimeDevDependencyDetector:
    """CLI tool for detecting runtime vs dev dependencies across workspace."""

    def __init__(self) -> None:
        """Initialize the detector with path resolver, reporting, JSON, deps, and runner services."""
        self._paths = FlextInfraPathResolver()
        self._reporting = FlextInfraReportingService()
        self._json = FlextInfraJsonService()
        self._deps = FlextInfraDependencyDetectionService()
        self._runner = FlextInfraCommandRunner()

    @staticmethod
    def _parser(default_limits_path: Path) -> argparse.ArgumentParser:
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
            "--projects",
            metavar="NAMES",
            help="Comma-separated list of project names.",
        )
        _ = parser.add_argument(
            "--no-pip-check",
            action="store_true",
            help="Skip pip check (workspace-level).",
        )
        _ = parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Do not write report files.",
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
            help=(
                "Write report to this path "
                "(default: .reports/dependencies/detect-runtime-dev-latest.json)."
            ),
        )
        _ = parser.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="Minimal output (summary only).",
        )
        _ = parser.add_argument(
            "--no-fail",
            action="store_true",
            help="Always exit 0 (report only).",
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
    def _project_filter(args: argparse.Namespace) -> list[str] | None:
        """Extract project filter list from parsed CLI arguments."""
        if args.project:
            return [args.project]
        if args.projects:
            return [name.strip() for name in args.projects.split(",") if name.strip()]
        return None

    def run(self, argv: list[str] | None = None) -> FlextResult[int]:
        """Execute dependency detection and generate workspace report."""
        root_result = self._paths.workspace_root_from_file(__file__)
        if root_result.is_failure:
            return r[int].fail(root_result.error or "workspace root resolution failed")
        root = root_result.value

        venv_bin = root / c.Paths.VENV_BIN_REL
        limits_default = Path(__file__).resolve().parent / "dependency_limits.toml"
        parser = self._parser(limits_default)
        args = parser.parse_args(argv)

        projects_result = self._deps.discover_projects(
            root,
            projects_filter=self._project_filter(args),
        )
        if projects_result.is_failure:
            return r[int].fail(projects_result.error or "project discovery failed")
        projects = projects_result.value
        if not projects:
            logger.error("deps_no_projects_found")
            return r[int].ok(2)

        if not (venv_bin / "deptry").exists():
            logger.error("deps_deptry_missing", path=str(venv_bin / "deptry"))
            return r[int].ok(3)

        apply_typings = bool(args.apply_typings)
        do_typings = bool(args.typings) or apply_typings
        limits_path = Path(args.limits) if args.limits else limits_default

        projects_report: MutableMapping[str, MutableMapping[str, t.ConfigMapValue]] = {}
        report_model = ddm.WorkspaceDependencyReport(
            workspace=str(root),
            projects=projects_report,
            pip_check=None,
            dependency_limits=None,
        )

        if do_typings:
            limits_data = self._deps.load_dependency_limits(limits_path)
            if limits_data:
                python_cfg = limits_data.get("python")
                python_version = (
                    str(python_cfg.get("version"))
                    if isinstance(python_cfg, dict)
                    and python_cfg.get("version") is not None
                    else None
                )
                report_model.dependency_limits = ddm.DependencyLimitsInfo(
                    python_version=python_version,
                    limits_path=str(limits_path),
                ).model_dump()

        for project_path in projects:
            project_name = project_path.name
            if not args.quiet:
                logger.info("deps_deptry_running", project=project_name)

            deptry_result = self._deps.run_deptry(project_path, venv_bin)
            if deptry_result.is_failure:
                return r[int].fail(deptry_result.error or "deptry run failed")
            issues, _ = deptry_result.value
            project_payload = self._deps.build_project_report(project_name, issues)
            project_dict = project_payload.model_dump()
            projects_report[project_name] = project_dict

            if do_typings and (project_path / c.Paths.DEFAULT_SRC_DIR).is_dir():
                if not args.quiet:
                    logger.info("deps_typings_detect_running", project=project_name)
                typings_result = self._deps.get_required_typings(
                    project_path,
                    venv_bin,
                    limits_path=limits_path,
                )
                if typings_result.is_failure:
                    return r[int].fail(
                        typings_result.error or "typing dependency detection failed",
                    )
                typing_dict = typings_result.value.model_dump()
                projects_report[project_name]["typings"] = typing_dict

                to_add_obj = typing_dict.get("to_add")
                to_add = to_add_obj if isinstance(to_add_obj, list) else []
                if apply_typings and to_add and not args.dry_run:
                    env = {
                        **os.environ,
                        "VIRTUAL_ENV": str(venv_bin.parent),
                        "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
                    }
                    for package in to_add:
                        if not isinstance(package, str):
                            continue
                        run = self._runner.run_raw(
                            ["poetry", "add", "--group", "typings", package],
                            cwd=project_path,
                            timeout=120,
                            env=env,
                        )
                        if run.is_failure or run.value.exit_code != 0:
                            logger.warning(
                                "deps_typings_add_failed",
                                project=project_name,
                                package=package,
                            )

        if not args.no_pip_check:
            if not args.quiet:
                logger.info("deps_pip_check_running")
            pip_result = self._deps.run_pip_check(root, venv_bin)
            if pip_result.is_failure:
                return r[int].fail(pip_result.error or "pip check failed")
            pip_lines, pip_exit = pip_result.value
            report_model.pip_check = ddm.PipCheckReport(
                ok=pip_exit == 0,
                lines=pip_lines,
            ).model_dump()

        report_payload = report_model.model_dump()

        if args.json_stdout:
            return r[int].ok(0)

        out_path: Path | None = None
        if args.output:
            out_path = Path(args.output)
        elif not args.dry_run:
            report_dir = self._reporting.get_report_dir(root, "project", "dependencies")
            try:
                report_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return r[int].fail(f"failed to create report directory: {exc}")
            out_path = report_dir / "detect-runtime-dev-latest.json"

        if out_path is not None and not args.dry_run:
            write_result = self._json.write(out_path, report_payload)
            if write_result.is_failure:
                return r[int].fail(write_result.error or "failed to write report")
            if not args.quiet:
                logger.info("deps_report_written", path=str(out_path))

        total_issues = 0
        for payload in projects_report.values():
            deptry_obj = payload.get("deptry")
            if isinstance(deptry_obj, dict):
                raw_count = deptry_obj.get("raw_count", 0)
                if isinstance(raw_count, int):
                    total_issues += raw_count

        pip_ok = True
        if isinstance(report_model.pip_check, dict):
            pip_ok = bool(report_model.pip_check.get("ok", True))

        if not args.quiet:
            logger.info(
                "deps_summary",
                projects=len(projects),
                deptry_issues=total_issues,
                pip_check="ok" if pip_ok else "FAIL",
            )

        if args.no_fail:
            return r[int].ok(0)
        return r[int].ok(0 if total_issues == 0 and pip_ok else 1)


def main() -> int:
    """Entry point for dependency detector CLI."""
    result = FlextInfraRuntimeDevDependencyDetector().run()
    if result.is_failure:
        logger.error("deps_detector_failed", error=result.error or "unknown error")
        return 1
    return result.value


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FlextInfraDependencyDetectorModels",
    "FlextInfraRuntimeDevDependencyDetector",
    "ddm",
    "main",
]
