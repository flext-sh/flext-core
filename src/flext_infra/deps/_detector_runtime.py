"""Runtime execution for dependency detector CLI."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import TypeAdapter, ValidationError

from flext_core import FlextLogger, r
from flext_infra import c, m, t, u


class _WorkspaceReport(Protocol):
    """Protocol for workspace dependency report model contract."""

    pip_check: m.Infra.Deps.PipCheckReport | None
    dependency_limits: m.Infra.Deps.DependencyLimitsInfo | None

    def model_dump(self) -> dict[str, t.Infra.InfraValue]:
        """Serialize report model payload."""
        ...


@runtime_checkable
class PathsService(Protocol):
    def workspace_root_from_file(self, file: str | Path) -> r[Path]: ...


@runtime_checkable
class ReportingService(Protocol):
    def get_report_dir(self, workspace_root: Path, scope: str, verb: str) -> Path: ...


@runtime_checkable
class JsonService(Protocol):
    def write_json(
        self,
        path: Path,
        payload: dict[str, t.Infra.InfraValue],
    ) -> r[bool]: ...


@runtime_checkable
class ProjectReportLike(Protocol):
    def model_dump(self) -> dict[str, t.Infra.InfraValue]: ...


@runtime_checkable
class DepsService(Protocol):
    def discover_project_paths(
        self,
        workspace_root: Path,
        *,
        projects_filter: list[str] | None = None,
    ) -> r[list[Path]]: ...

    def run_deptry(
        self,
        project_path: Path,
        venv_bin: Path,
    ) -> r[tuple[list[t.Infra.IssueMap], int]]: ...

    def build_project_report(
        self,
        project_name: str,
        deptry_issues: list[t.Infra.IssueMap],
    ) -> ProjectReportLike: ...


@runtime_checkable
class TypingsDepsService(Protocol):
    def load_dependency_limits(
        self,
        limits_path: Path | None = None,
    ) -> Mapping[str, t.Infra.TomlValue]: ...

    def get_required_typings(
        self,
        project_path: Path,
        venv_bin: Path,
        limits_path: Path | None = None,
        *,
        include_mypy: bool = True,
    ) -> r[m.Infra.Deps.TypingsReport]: ...


@runtime_checkable
class PipCheckDepsService(Protocol):
    def run_pip_check(
        self,
        workspace_root: Path,
        venv_bin: Path,
    ) -> r[tuple[list[str], int]]: ...


@runtime_checkable
class RunnerService(Protocol):
    def run_raw(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> r[m.Infra.Core.CommandOutput]: ...


class _DetectorRuntime(Protocol):
    """Protocol for detector runtime service dependencies."""

    paths: PathsService
    reporting: ReportingService
    json: JsonService
    deps: DepsService
    runner: RunnerService
    log: FlextLogger

    @staticmethod
    def parser(default_limits_path: Path) -> argparse.ArgumentParser: ...

    @staticmethod
    def project_filter(cli: u.Infra.CliArgs) -> list[str] | None:
        """Resolve project filter list from parsed args."""
        ...


class FlextInfraDependencyDetectorRuntime:
    """Runtime executor for dependency detection pipeline."""

    def __init__(
        self,
        detector: _DetectorRuntime,
        workspace_report_factory: Callable[..., _WorkspaceReport],
        dependency_limits_factory: Callable[..., m.Infra.Deps.DependencyLimitsInfo],
        pip_check_factory: Callable[..., m.Infra.Deps.PipCheckReport],
    ) -> None:
        self._detector = detector
        self._workspace_report_factory = workspace_report_factory
        self._dependency_limits_factory = dependency_limits_factory
        self._pip_check_factory = pip_check_factory

    def run(self, argv: list[str] | None = None) -> r[int]:
        """Execute dependency detection and generate workspace report."""
        detector = self._detector
        limits_default = Path(__file__).resolve().parent / "dependency_limits.toml"
        parser = detector.parser(limits_default)
        args = parser.parse_args(argv)
        cli = u.Infra.resolve(args)
        root: Path = cli.workspace
        venv_bin = root / c.Infra.Paths.VENV_BIN_REL
        projects_result = detector.deps.discover_project_paths(
            root,
            projects_filter=detector.project_filter(cli),
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
        projects_report: dict[str, dict[str, t.Infra.InfraValue]] = {}
        report_model = self._workspace_report_factory(
            workspace=str(root),
            projects=projects_report,
            pip_check=None,
            dependency_limits=None,
        )
        deps_service = detector.deps
        typing_deps: TypingsDepsService | None = None
        if do_typings:
            if not isinstance(deps_service, TypingsDepsService):
                return r[int].fail("typing dependency detection service unavailable")
            typing_deps = deps_service
            limits_data = typing_deps.load_dependency_limits(limits_path)
            if limits_data:
                python_cfg = limits_data.get(c.Infra.Toml.PYTHON)
                python_version = (
                    str(python_cfg.get(c.Infra.Toml.VERSION))
                    if isinstance(python_cfg, dict)
                    and python_cfg.get(c.Infra.Toml.VERSION) is not None
                    else None
                )
                report_model.dependency_limits = self._dependency_limits_factory(
                    python_version=python_version,
                    limits_path=str(limits_path),
                )
        for project_path in projects:
            project_name = project_path.name
            if not args.quiet:
                detector.log.info("deps_deptry_running", project=project_name)
            deptry_result = deps_service.run_deptry(project_path, venv_bin)
            if deptry_result.is_failure:
                return r[int].fail(deptry_result.error or "deptry run failed")
            issues, _ = deptry_result.value
            project_payload = deps_service.build_project_report(project_name, issues)
            projects_report[project_name] = project_payload.model_dump()
            if do_typings and (project_path / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir():
                if typing_deps is None:
                    return r[int].fail(
                        "typing dependency detection service unavailable"
                    )
                if not args.quiet:
                    detector.log.info(
                        "deps_typings_detect_running", project=project_name
                    )
                typings_result = typing_deps.get_required_typings(
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
                if apply_typings and to_add and cli.apply:
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
            if not isinstance(deps_service, PipCheckDepsService):
                return r[int].fail("pip-check dependency detection service unavailable")
            if not args.quiet:
                detector.log.info("deps_pip_check_running")
            pip_result = deps_service.run_pip_check(root, venv_bin)
            if pip_result.is_failure:
                return r[int].fail(pip_result.error or "pip check failed")
            pip_lines, pip_exit = pip_result.value
            pip_ok = pip_exit == 0
            report_model.pip_check = self._pip_check_factory(ok=pip_ok, lines=pip_lines)
        report_payload = report_model.model_dump()
        if cli.output_format == "json":
            return r[int].ok(0)
        out_path: Path | None = None
        if args.output:
            out_path = Path(args.output)
        elif cli.apply:
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
        if out_path is not None and not cli.dry_run:
            write_result = detector.json.write_json(out_path, report_payload)
            if write_result.is_failure:
                return r[int].fail(write_result.error or "failed to write report")
            if not args.quiet:
                detector.log.info("deps_report_written", path=str(out_path))
        total_issues = 0
        for payload in projects_report.values():
            deptry_obj = payload.get(c.Infra.Toml.DEPTRY)
            if isinstance(deptry_obj, dict):
                try:
                    deptry_payload = TypeAdapter(
                        dict[str, t.Infra.InfraValue]
                    ).validate_python(deptry_obj)
                except ValidationError:
                    continue
                raw_count_obj: t.Infra.InfraValue = deptry_payload.get("raw_count", 0)
                if isinstance(raw_count_obj, int):
                    total_issues += raw_count_obj
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

    @staticmethod
    def run_detector(
        detector: _DetectorRuntime,
        workspace_report_factory: Callable[..., _WorkspaceReport],
        dependency_limits_factory: Callable[..., m.Infra.Deps.DependencyLimitsInfo],
        pip_check_factory: Callable[..., m.Infra.Deps.PipCheckReport],
        argv: list[str] | None = None,
    ) -> r[int]:
        """Execute dependency detection and generate workspace report."""
        runtime = FlextInfraDependencyDetectorRuntime(
            detector=detector,
            workspace_report_factory=workspace_report_factory,
            dependency_limits_factory=dependency_limits_factory,
            pip_check_factory=pip_check_factory,
        )
        return runtime.run(argv=argv)


__all__ = ["FlextInfraDependencyDetectorRuntime"]
