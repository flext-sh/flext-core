"""Cohesive environment-setup + per-project execution mixin for the dependency detector runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, p, t

if TYPE_CHECKING:
    from flext_infra import m


class FlextInfraDependencyDetectorRuntimeSteps:
    """Mixin holding environment setup and per-project detection steps."""

    _detector: p.Infra.DetectorRuntime
    _dependency_limits_factory: Callable[..., m.Infra.DependencyLimitsInfo]
    _pip_check_factory: Callable[..., m.Infra.PipCheckReport]

    def _validate_environment(
        self, params: m.Infra.DetectCommand, root: Path, venv_bin: Path
    ) -> p.Result[tuple[t.SequenceOf[Path], Path]]:
        """Discover projects and verify deptry binary; return ``(projects, limits_path)``."""
        detector = self._detector
        projects_result = detector.deps.discover_project_paths(
            root, projects_filter=params.project_names
        )
        if projects_result.failure:
            return r[tuple[t.SequenceOf[Path], Path]].fail(
                projects_result.error or "project discovery failed"
            )
        projects: t.SequenceOf[Path] = projects_result.value
        if not projects:
            detector.log.error("deps_no_projects_found")
            return r[tuple[t.SequenceOf[Path], Path]].fail("no projects found")
        deptry_path = venv_bin / c.Infra.DEPTRY
        if not deptry_path.exists():
            detector.log.error("deps_deptry_missing", path=str(deptry_path))
            return r[tuple[t.SequenceOf[Path], Path]].fail(
                f"Deptry executable not found at {deptry_path}"
            )
        limits_default = (
            Path(__file__).resolve().parent / c.Infra.DEPENDENCY_LIMITS_FILENAME
        )
        limits_path = params.limits_path or limits_default
        return r[tuple[t.SequenceOf[Path], Path]].ok((projects, limits_path))

    def _configure_typings_limits(
        self,
        typing_deps: p.Infra.TypingsDepsService | None,
        limits_path: Path,
        report_model: p.Infra.WorkspaceReport,
    ) -> p.Result[bool]:
        """Load dependency-limits TOML and seed the workspace report's limits info."""
        if typing_deps is None:
            return r[bool].fail("typing dependency detection service unavailable")
        limits_data = typing_deps.load_dependency_limits(limits_path)
        if not limits_data:
            return r[bool].ok(False)
        python_payload = limits_data.get(c.Infra.PYTHON)
        python_cfg: t.JsonMapping = (
            t.Infra.INFRA_MAPPING_ADAPTER.validate_python(python_payload)
            if isinstance(python_payload, Mapping)
            else {}
        )
        version_value = python_cfg.get(c.Infra.VERSION)
        python_version = str(version_value) if version_value is not None else None
        report_model.dependency_limits = self._dependency_limits_factory(
            python_version=python_version, limits_path=str(limits_path)
        )
        return r[bool].ok(True)

    def _run_project_detection(
        self,
        project_path: Path,
        *,
        deps_service: p.Infra.DepsService,
        typing_deps: p.Infra.TypingsDepsService | None,
        venv_bin: Path,
        limits_path: Path,
        params: m.Infra.DetectCommand,
        do_typings: bool,
        projects_report: MutableMapping[str, MutableMapping[str, t.Infra.InfraValue]],
    ) -> p.Result[bool]:
        """Run deptry + optional typings detection/apply for one project."""
        detector = self._detector
        project_name = project_path.name
        if not params.quiet:
            detector.log.info("deps_deptry_running", project=project_name)
        deptry_result = deps_service.run_deptry(project_path, venv_bin)
        if deptry_result.failure:
            return r[bool].fail(deptry_result.error or "deptry run failed")
        issues, _ = deptry_result.value
        project_payload = deps_service.build_project_report(project_name, issues)
        projects_report[project_name] = dict(project_payload.model_dump())
        run_typings_for_project = (
            do_typings
            and typing_deps is not None
            and (project_path / c.Infra.DEFAULT_SRC_DIR).is_dir()
        )
        if not run_typings_for_project:
            return r[bool].ok(True)
        return self._run_project_typings(
            project_path,
            typing_deps=typing_deps,
            venv_bin=venv_bin,
            limits_path=limits_path,
            params=params,
            projects_report=projects_report,
        )

    def _run_project_typings(
        self,
        project_path: Path,
        *,
        typing_deps: p.Infra.TypingsDepsService | None,
        venv_bin: Path,
        limits_path: Path,
        params: m.Infra.DetectCommand,
        projects_report: MutableMapping[str, MutableMapping[str, t.Infra.InfraValue]],
    ) -> p.Result[bool]:
        """Detect required typings for a project and optionally add them via poetry."""
        detector = self._detector
        if typing_deps is None:
            return r[bool].fail("typing dependency detection service unavailable")
        project_name = project_path.name
        if not params.quiet:
            detector.log.info("deps_typings_detect_running", project=project_name)
        typings_result = typing_deps.get_required_typings(
            project_path, limits_path=limits_path
        )
        if typings_result.failure:
            return r[bool].fail(
                typings_result.error or "typing dependency detection failed"
            )
        typings_report = typings_result.value
        projects_report[project_name][c.Infra.DIR_TYPINGS] = typings_report.model_dump()
        to_add: t.StrSequence = typings_report.to_add
        if not (params.apply_typings and to_add and params.apply):
            return r[bool].ok(True)
        env = {"VIRTUAL_ENV": str(venv_bin.parent)}
        poetry = venv_bin / c.Infra.POETRY
        for package in to_add:
            run_outcome = detector.runner.run_raw(
                [str(poetry), "add", "--group", c.Infra.DIR_TYPINGS, package],
                cwd=project_path,
                timeout=c.Infra.TIMEOUT_MEDIUM,
                env=env,
            )
            poetry_failed = run_outcome.failure or run_outcome.value.exit_code != 0
            if poetry_failed:
                detector.log.warning(
                    "deps_typings_add_failed", project=project_name, package=package
                )
        return r[bool].ok(True)

    def _run_pip_check(
        self,
        deps_service: p.Infra.DepsService,
        root: Path,
        venv_bin: Path,
        params: m.Infra.DetectCommand,
        report_model: p.Infra.WorkspaceReport,
    ) -> p.Result[bool]:
        """Execute the workspace ``pip check`` and stamp the report; ``r.ok(pip_ok)``."""
        if params.no_pip_check:
            return r[bool].ok(True)
        if not isinstance(deps_service, p.Infra.PipCheckDepsService):
            return r[bool].fail("pip-check dependency detection service unavailable")
        detector = self._detector
        if not params.quiet:
            detector.log.info("deps_pip_check_running")
        pip_result = deps_service.run_pip_check(root, venv_bin)
        if pip_result.failure:
            return r[bool].fail(pip_result.error or "pip check failed")
        pip_lines, pip_exit = pip_result.value
        pip_ok = pip_exit == 0
        report_model.pip_check = self._pip_check_factory(ok=pip_ok, lines=pip_lines)
        return r[bool].ok(pip_ok)


__all__: list[str] = ["FlextInfraDependencyDetectorRuntimeSteps"]
