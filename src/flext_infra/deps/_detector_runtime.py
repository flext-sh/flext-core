"""Runtime execution for dependency detector CLI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from flext_core import FlextLogger, r
from flext_infra import (
    FlextInfraUtilitiesIo,
    FlextInfraUtilitiesPaths,
    FlextInfraUtilitiesReporting,
    c,
    m,
    p,
    t,
)
from flext_infra.deps.detection import FlextInfraDependencyDetectionService

if TYPE_CHECKING:
    from flext_infra.deps.detector import FlextInfraDependencyDetectorModels as _DDM


class _DetectorModelsProtocol(Protocol):
    """Protocol for detector model namespace constructors."""

    def WorkspaceDependencyReport(
        self,
        *,
        workspace: str,
        projects: dict[str, dict[str, t.ContainerValue]],
        pip_check: _DDM.PipCheckReport | None,
        dependency_limits: _DDM.DependencyLimitsInfo | None,
    ) -> _DDM.WorkspaceDependencyReport:
        """Create workspace report model instance."""
        ...

    def DependencyLimitsInfo(
        self,
        *,
        python_version: str | None,
        limits_path: str,
    ) -> _DDM.DependencyLimitsInfo:
        """Create dependency limits model instance."""
        ...

    def PipCheckReport(self, *, ok: bool, lines: list[str]) -> _DDM.PipCheckReport:
        """Create pip-check report model instance."""
        ...


class _DetectorRuntimeProtocol(Protocol):
    """Protocol for detector runtime service dependencies."""

    paths: FlextInfraUtilitiesPaths
    reporting: FlextInfraUtilitiesReporting
    json: FlextInfraUtilitiesIo
    deps: FlextInfraDependencyDetectionService
    runner: p.Infra.CommandRunner
    log: FlextLogger

    @staticmethod
    def parser(default_limits_path: Path) -> argparse.ArgumentParser:
        """Create detector CLI parser."""
        ...

    @staticmethod
    def project_filter(args: argparse.Namespace) -> list[str] | None:
        """Resolve project filter list from parsed args."""
        ...


def run_detector(
    detector: _DetectorRuntimeProtocol,
    models: _DetectorModelsProtocol,
    argv: list[str] | None = None,
) -> r[int]:
    """Execute dependency detection and generate workspace report."""
    root_result = detector.paths.workspace_root_from_file(__file__)
    if root_result.is_failure:
        return r[int].fail(root_result.error or "workspace root resolution failed")
    root: Path = root_result.value
    venv_bin = root / c.Infra.Paths.VENV_BIN_REL
    limits_default = Path(__file__).resolve().parent / "dependency_limits.toml"
    parser = detector.parser(limits_default)
    args = parser.parse_args(argv)
    projects_result = detector.deps.discover_project_paths(
        root,
        projects_filter=detector.project_filter(args),
    )
    if projects_result.is_failure:
        return r[int].fail(projects_result.error or "project discovery failed")
    projects: list[Path] = projects_result.value
    if not projects:
        detector.log.error("deps_no_projects_found")
        return r[int].ok(2)
    if not (venv_bin / c.Infra.Toml.DEPTRY).exists():
        detector.log.error(
            "deps_deptry_missing", path=str(venv_bin / c.Infra.Toml.DEPTRY)
        )
        return r[int].ok(3)
    apply_typings = bool(args.apply_typings)
    do_typings = bool(args.typings) or apply_typings
    limits_path = Path(args.limits) if args.limits else limits_default
    projects_report: dict[str, dict[str, t.ContainerValue]] = {}
    report_model = models.WorkspaceDependencyReport(
        workspace=str(root),
        projects=projects_report,
        pip_check=None,
        dependency_limits=None,
    )
    if do_typings:
        limits_data = detector.deps.load_dependency_limits(limits_path)
        if limits_data:
            python_cfg = limits_data.get(c.Infra.Toml.PYTHON)
            python_version = (
                str(python_cfg.get(c.Infra.Toml.VERSION))
                if isinstance(python_cfg, dict)
                and python_cfg.get(c.Infra.Toml.VERSION) is not None
                else None
            )
            report_model.dependency_limits = models.DependencyLimitsInfo(
                python_version=python_version,
                limits_path=str(limits_path),
            )
    for project_path in projects:
        project_name = project_path.name
        if not args.quiet:
            detector.log.info("deps_deptry_running", project=project_name)
        deptry_result = detector.deps.run_deptry(project_path, venv_bin)
        if deptry_result.is_failure:
            return r[int].fail(deptry_result.error or "deptry run failed")
        issues, _ = deptry_result.value
        project_payload = detector.deps.build_project_report(project_name, issues)
        projects_report[project_name] = project_payload.model_dump()
        if do_typings and (project_path / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir():
            if not args.quiet:
                detector.log.info("deps_typings_detect_running", project=project_name)
            typings_result = detector.deps.get_required_typings(
                project_path,
                venv_bin,
                limits_path=limits_path,
            )
            if typings_result.is_failure:
                return r[int].fail(
                    typings_result.error or "typing dependency detection failed",
                )
            typings_report = typings_result.value
            projects_report[project_name][c.Infra.Directories.TYPINGS] = (
                typings_report.model_dump()
            )
            to_add: list[str] = typings_report.to_add
            if apply_typings and to_add and (not args.dry_run):
                env = {
                    **os.environ,
                    "VIRTUAL_ENV": str(venv_bin.parent),
                    "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
                }
                for package in to_add:
                    run = detector.runner.run_raw(
                        [
                            c.Infra.Cli.POETRY,
                            "add",
                            "--group",
                            c.Infra.Directories.TYPINGS,
                            package,
                        ],
                        cwd=project_path,
                        timeout=c.Infra.Timeouts.MEDIUM,
                        env=env,
                    )
                    if run.is_failure:
                        detector.log.warning(
                            "deps_typings_add_failed",
                            project=project_name,
                            package=package,
                        )
                    else:
                        run_output: m.Infra.Core.CommandOutput = run.value
                        if run_output.exit_code != 0:
                            detector.log.warning(
                                "deps_typings_add_failed",
                                project=project_name,
                                package=package,
                            )
    pip_ok = True
    if not args.no_pip_check:
        if not args.quiet:
            detector.log.info("deps_pip_check_running")
        pip_result = detector.deps.run_pip_check(root, venv_bin)
        if pip_result.is_failure:
            return r[int].fail(pip_result.error or "pip check failed")
        pip_lines, pip_exit = pip_result.value
        pip_ok = pip_exit == 0
        report_model.pip_check = models.PipCheckReport(
            ok=pip_ok,
            lines=pip_lines,
        )
    report_payload = report_model.model_dump()
    if args.json_stdout:
        return r[int].ok(0)
    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output)
    elif not args.dry_run:
        report_dir = detector.reporting.get_report_dir(
            root,
            c.Infra.Toml.PROJECT,
            c.Infra.Toml.DEPENDENCIES,
        )
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return r[int].fail(f"failed to create report directory: {exc}")
        out_path = report_dir / "detect-runtime-dev-latest.json"
    if out_path is not None and (not args.dry_run):
        write_result = detector.json.write_json(out_path, report_payload)
        if write_result.is_failure:
            return r[int].fail(write_result.error or "failed to write report")
        if not args.quiet:
            detector.log.info("deps_report_written", path=str(out_path))
    total_issues = 0
    for payload in projects_report.values():
        deptry_obj = payload.get(c.Infra.Toml.DEPTRY)
        if isinstance(deptry_obj, dict):
            raw_count = deptry_obj.get("raw_count", 0)
            if isinstance(raw_count, int):
                total_issues += raw_count
    if not args.quiet:
        detector.log.info(
            "deps_summary",
            projects=len(projects),
            deptry_issues=total_issues,
            pip_check=c.Infra.ReportKeys.OK if pip_ok else "FAIL",
        )
    if args.no_fail:
        return r[int].ok(0)
    return r[int].ok(0 if total_issues == 0 and pip_ok else 1)


__all__ = ["run_detector"]
