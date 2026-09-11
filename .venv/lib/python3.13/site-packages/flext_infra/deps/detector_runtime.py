"""Runtime execution for dependency detector CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, p, u
from flext_infra.deps._detector_runtime_steps import (
    FlextInfraDependencyDetectorRuntimeSteps,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping

    from flext_infra import m, t


class FlextInfraDependencyDetectorRuntime(FlextInfraDependencyDetectorRuntimeSteps):
    """Runtime executor for dependency detection pipeline."""

    def __init__(
        self,
        detector: p.Infra.DetectorRuntime,
        workspace_report_factory: Callable[..., p.Infra.WorkspaceReport],
        dependency_limits_factory: Callable[..., m.Infra.DependencyLimitsInfo],
        pip_check_factory: Callable[..., m.Infra.PipCheckReport],
    ) -> None:
        """Store runtime collaborators used by dependency detection orchestration."""
        self._detector = detector
        self._workspace_report_factory = workspace_report_factory
        self._dependency_limits_factory = dependency_limits_factory
        self._pip_check_factory = pip_check_factory

    def run(self, params: m.Infra.DetectCommand) -> p.Result[bool]:
        """Execute dependency detection and generate workspace report (orchestrator)."""
        detector = self._detector
        root = params.workspace_path
        venv_bin = root / c.Infra.VENV_BIN_REL
        env_result = self._validate_environment(params, root, venv_bin)
        if env_result.failure:
            return r[bool].fail(env_result.error or "environment validation failed")
        projects, limits_path = env_result.value
        do_typings = params.typings or params.apply_typings
        projects_report: MutableMapping[
            str, MutableMapping[str, t.Infra.InfraValue]
        ] = {}
        report_model = self._workspace_report_factory(
            workspace=str(root),
            projects=projects_report,
            pip_check=None,
            dependency_limits=None,
        )
        deps_service = detector.deps
        typing_deps = (
            deps_service
            if isinstance(deps_service, p.Infra.TypingsDepsService)
            else None
        )
        if do_typings:
            limits_setup = self._configure_typings_limits(
                typing_deps, limits_path, report_model
            )
            if limits_setup.failure:
                return r[bool].fail(
                    limits_setup.error or "typings limits configuration failed"
                )
        for project_path in projects:
            project_result = self._run_project_detection(
                project_path,
                deps_service=deps_service,
                typing_deps=typing_deps,
                venv_bin=venv_bin,
                limits_path=limits_path,
                params=params,
                do_typings=do_typings,
                projects_report=projects_report,
            )
            if project_result.failure:
                return r[bool].fail(project_result.error or "project detection failed")
        pip_check_result = self._run_pip_check(
            deps_service, root, venv_bin, params, report_model
        )
        if pip_check_result.failure:
            return r[bool].fail(pip_check_result.error or "pip check failed")
        pip_ok = pip_check_result.value
        if params.output_format == c.Cli.OutputFormats.JSON:
            return r[bool].ok(True)
        write_result = self._write_workspace_report(
            params, root, report_model, projects_report
        )
        if write_result.failure:
            return r[bool].fail(write_result.error or "failed to write report")
        return self._summarize_run(
            projects, projects_report, pip_ok=pip_ok, params=params
        )

    def _write_workspace_report(
        self,
        params: m.Infra.DetectCommand,
        root: Path,
        report_model: p.Infra.WorkspaceReport,
        projects_report: Mapping[str, Mapping[str, t.Infra.InfraValue]],
    ) -> p.Result[Path]:
        """Render and persist the canonical workspace dependency report JSON."""
        out_path: Path = params.output_path or u.Cli.resolve_report_path(
            root,
            c.Infra.PROJECT,
            c.Infra.DEPENDENCIES,
            "detect-runtime-dev-latest.json",
        )
        report_payload: t.JsonDict = {
            key: u.normalize_to_json_value(value)
            for key, value in report_model.model_dump().items()
        }
        report_payload["projects"] = {
            project_name: {
                key: u.normalize_to_json_value(value)
                for key, value in project_payload.items()
            }
            for project_name, project_payload in projects_report.items()
        }
        write_result = u.Cli.json_write(out_path, report_payload)
        if write_result.failure:
            return r[Path].fail(write_result.error or "failed to write report")
        if not params.quiet:
            self._detector.log.info("deps_report_written", path=str(out_path))
        return r[Path].ok(out_path)

    def _summarize_run(
        self,
        projects: t.SequenceOf[Path],
        projects_report: Mapping[str, Mapping[str, t.Infra.InfraValue]],
        *,
        pip_ok: bool,
        params: m.Infra.DetectCommand,
    ) -> p.Result[bool]:
        """Aggregate deptry counts, log the summary, decide overall pass/fail."""
        total_issues = sum(
            u.Cli.json_pick_int(
                u.Cli.json_as_mapping(payload.get(c.Infra.DEPTRY)), "raw_count"
            )
            for payload in projects_report.values()
        )
        if not params.quiet:
            self._detector.log.info(
                "deps_summary",
                projects=len(projects),
                deptry_issues=total_issues,
                pip_check=c.Infra.RK_OK if pip_ok else "FAIL",
            )
        if params.no_fail or (total_issues == 0 and pip_ok):
            return r[bool].ok(True)
        return r[bool].fail("dependency issues detected")


__all__: list[str] = ["FlextInfraDependencyDetectorRuntime"]
