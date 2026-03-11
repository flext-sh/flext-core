"""Runtime execution for dependency detector CLI."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

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


class _WorkspaceReportProtocol(Protocol):
    """Protocol for workspace dependency report model contract."""

    def model_dump(self) -> dict[str, t.JsonValue]: ...


class _ProjectReportProtocol(Protocol):
    def model_dump(self) -> dict[str, t.JsonValue]: ...


class _DetectorRuntimeProtocol(Protocol):
    """Protocol for detector runtime service dependencies."""

    paths: FlextInfraUtilitiesPaths
    reporting: FlextInfraUtilitiesReporting
    json: FlextInfraUtilitiesIo
    deps: FlextInfraDependencyDetectionService
    runner: p.Infra.CommandRunner
    log: FlextLogger

    @staticmethod
    def parser(default_limits_path: Path) -> argparse.ArgumentParser: ...

    @staticmethod
    def project_filter(args: argparse.Namespace) -> list[str] | None: ...


class FlextInfraDependencyDetectorRuntime:
    """Service to execute dependency detection and generate reports."""

    _CONTAINER_LIST_ADAPTER: TypeAdapter[list[t.Infra.TomlValue]] = TypeAdapter(
        list[t.Infra.TomlValue]
    )

    def __init__(
        self,
        detector: _DetectorRuntimeProtocol,
        workspace_report_factory: Callable[..., _WorkspaceReportProtocol],
        dependency_limits_factory: Callable[..., m.Infra.Deps.DependencyLimitsInfo],
        pip_check_factory: Callable[..., m.Infra.Deps.PipCheckReport],
    ) -> None:
        """Initialize the detector runtime with required services and model factories."""
        self.detector = detector
        self.workspace_report_factory = workspace_report_factory
        self.dependency_limits_factory = dependency_limits_factory
        self.pip_check_factory = pip_check_factory

    @staticmethod
    def _deptry_from_payload(
        report_payload: dict[str, t.JsonValue],
    ) -> m.Infra.Deps.DeptryReport:
        deptry_payload = report_payload.get("deptry")
        if not isinstance(deptry_payload, dict):
            return m.Infra.Deps.DeptryReport()
        missing = deptry_payload.get("missing")
        unused = deptry_payload.get("unused")
        transitive = deptry_payload.get("transitive")
        dev_in_runtime = deptry_payload.get("dev_in_runtime")
        raw_count = deptry_payload.get("raw_count")
        missing_values = missing if isinstance(missing, list) else []
        unused_values = unused if isinstance(unused, list) else []
        transitive_values = transitive if isinstance(transitive, list) else []
        dev_in_runtime_values = (
            dev_in_runtime if isinstance(dev_in_runtime, list) else []
        )
        return m.Infra.Deps.DeptryReport(
            missing=[item for item in missing_values if isinstance(item, str)],
            unused=[item for item in unused_values if isinstance(item, str)],
            transitive=[item for item in transitive_values if isinstance(item, str)],
            dev_in_runtime=[
                item for item in dev_in_runtime_values if isinstance(item, str)
            ],
            raw_count=raw_count if isinstance(raw_count, int) else 0,
        )

    def run(self, argv: list[str] | None = None) -> r[int]:
        """Execute dependency detection and generate workspace report."""
        root_result = self.detector.paths.workspace_root_from_file(__file__)
        if root_result.is_failure:
            return r[int].fail(root_result.error or "workspace root resolution failed")
        root: Path = root_result.value
        venv_bin = root / c.Infra.Paths.VENV_BIN_REL
        limits_default = Path(__file__).resolve().parent / "dependency_limits.toml"
        parser = self.detector.parser(limits_default)
        args = parser.parse_args(argv)
        projects_result = self.detector.deps.discover_project_paths(
            root,
            projects_filter=self.detector.project_filter(args),
        )
        if projects_result.is_failure:
            return r[int].fail(projects_result.error or "project discovery failed")
        projects: list[Path] = projects_result.value
        if not projects:
            self.detector.log.error("deps_no_projects_found")
            return r[int].ok(2)
        if not (venv_bin / c.Infra.Toml.DEPTRY).exists():
            self.detector.log.error(
                "deps_deptry_missing", path=str(venv_bin / c.Infra.Toml.DEPTRY)
            )
            return r[int].ok(3)
        apply_typings = bool(args.apply_typings)
        do_typings = bool(args.typings) or apply_typings
        limits_path = Path(args.limits) if args.limits else limits_default
        projects_report: dict[str, m.Infra.Deps.ProjectRuntimeReport] = {}
        dependency_limits_model: m.Infra.Deps.DependencyLimitsInfo | None = None
        if do_typings:
            limits_data = self.detector.deps.load_dependency_limits(limits_path)
            if limits_data:
                python_cfg = limits_data.get(c.Infra.Toml.PYTHON)
                python_version = (
                    str(python_cfg.get(c.Infra.Toml.VERSION))
                    if isinstance(python_cfg, dict)
                    and python_cfg.get(c.Infra.Toml.VERSION) is not None
                    else None
                )
                dependency_limits_model = self.dependency_limits_factory(
                    python_version=python_version,
                    limits_path=str(limits_path),
                )
        for project_path in projects:
            project_name = project_path.name
            if not args.quiet:
                self.detector.log.info("deps_deptry_running", project=project_name)
            deptry_result = self.detector.deps.run_deptry(project_path, venv_bin)
            if deptry_result.is_failure:
                return r[int].fail(deptry_result.error or "deptry run failed")
            issues, _ = deptry_result.value
            project_dependency_report: _ProjectReportProtocol = (
                self.detector.deps.build_project_report(project_name, issues)
            )
            deptry_report = self._deptry_from_payload(
                project_dependency_report.model_dump()
            )
            project_report = m.Infra.Deps.ProjectRuntimeReport(deptry=deptry_report)
            if do_typings and (project_path / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir():
                if not args.quiet:
                    self.detector.log.info(
                        "deps_typings_detect_running", project=project_name
                    )
                typings_result = self.detector.deps.get_required_typings(
                    project_path,
                    venv_bin,
                    limits_path=limits_path,
                )
                if typings_result.is_failure:
                    return r[int].fail(
                        typings_result.error or "typing dependency detection failed",
                    )
                typings_payload = typings_result.value.model_dump()
                to_add_packages: list[str] = []
                to_add_value = typings_payload.get("to_add")
                if isinstance(to_add_value, list):
                    parsed_values = self._CONTAINER_LIST_ADAPTER.validate_python(
                        to_add_value
                    )
                    to_add_packages = [str(value_item) for value_item in parsed_values]
                typings_report = m.Infra.Deps.TypingsReport(
                    required_packages=[],
                    hinted=[],
                    missing_modules=[],
                    current=[],
                    to_add=[item for item in to_add_packages if item],
                    to_remove=[],
                    limits_applied=False,
                    python_version=None,
                )
                project_report.typings = typings_report
                if apply_typings and to_add_packages and (not args.dry_run):
                    env = {
                        **os.environ,
                        "VIRTUAL_ENV": str(venv_bin.parent),
                        "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
                    }
                    for package in to_add_packages:
                        run_res = self.detector.runner.run_raw(
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
                        if run_res.is_failure:
                            self.detector.log.warning(
                                "deps_typings_add_failed",
                                project=project_name,
                                package=package,
                            )
                        else:
                            run_output: m.Infra.Core.CommandOutput = run_res.value
                            if run_output.exit_code != 0:
                                self.detector.log.warning(
                                    "deps_typings_add_failed",
                                    project=project_name,
                                    package=package,
                                )
            projects_report[project_name] = project_report
        pip_ok = True
        pip_check_model: m.Infra.Deps.PipCheckReport | None = None
        if not args.no_pip_check:
            if not args.quiet:
                self.detector.log.info("deps_pip_check_running")
            pip_result = self.detector.deps.run_pip_check(root, venv_bin)
            if pip_result.is_failure:
                return r[int].fail(pip_result.error or "pip check failed")
            pip_lines, pip_exit = pip_result.value
            pip_ok = pip_exit == 0
            pip_check_model = self.pip_check_factory(ok=pip_ok, lines=pip_lines)
        report_model = self.workspace_report_factory(
            workspace=str(root),
            projects=projects_report,
            pip_check=pip_check_model,
            dependency_limits=dependency_limits_model,
        )
        report_payload = report_model.model_dump()
        if args.json_stdout:
            return r[int].ok(0)
        out_path: Path | None = None
        if args.output:
            out_path = Path(args.output)
        elif not args.dry_run:
            report_dir = self.detector.reporting.get_report_dir(
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
            write_result = self.detector.json.write_json(out_path, report_payload)
            if write_result.is_failure:
                return r[int].fail(write_result.error or "failed to write report")
            if not args.quiet:
                self.detector.log.info("deps_report_written", path=str(out_path))
        total_issues = 0
        for payload in projects_report.values():
            total_issues += payload.deptry.raw_count
        if not args.quiet:
            self.detector.log.info(
                "deps_summary",
                projects=len(projects),
                deptry_issues=total_issues,
                pip_check=c.Infra.ReportKeys.OK if pip_ok else "FAIL",
            )
        if args.no_fail:
            return r[int].ok(0)
        return r[int].ok(0 if total_issues == 0 and pip_ok else 1)


def run_detector(
    detector: _DetectorRuntimeProtocol,
    workspace_report_factory: Callable[..., _WorkspaceReportProtocol],
    dependency_limits_factory: Callable[..., m.Infra.Deps.DependencyLimitsInfo],
    pip_check_factory: Callable[..., m.Infra.Deps.PipCheckReport],
    argv: list[str] | None = None,
) -> r[int]:
    """Execute dependency detection and generate workspace report (backward compatible)."""
    runtime = FlextInfraDependencyDetectorRuntime(
        detector=detector,
        workspace_report_factory=workspace_report_factory,
        dependency_limits_factory=dependency_limits_factory,
        pip_check_factory=pip_check_factory,
    )
    return runtime.run(argv=argv)


__all__ = ["FlextInfraDependencyDetectorRuntime", "run_detector"]
