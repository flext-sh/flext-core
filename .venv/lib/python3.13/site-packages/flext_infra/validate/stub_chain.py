"""Typed dependency validation service.

Validates typing dependency hints and unresolved imports for workspace projects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.util as importlib_util
from pathlib import Path
from typing import Annotated, override

from flext_core import r
from flext_infra import c, m, p, t, u
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase

type _StubChainInitValue = (
    t.GuardInput | p.Settings | p.Context | t.SettingsClass | None
)
type _StubChainRuntimeState = dict[str, _StubChainInitValue]


class FlextInfraStubSupplyChain(FlextInfraProjectSelectionServiceBase[bool]):
    """Validate typed dependency supply chain for workspace projects.

    Coordinates mypy install hints, pyrefly missing imports, and installed
    typing-package validation.
    """

    all_projects: Annotated[
        bool, m.Field(alias="all", description="Validate all projects")
    ] = False
    _runner: p.Cli.CommandRunner | None = m.PrivateAttr(default=None)

    def __init__(
        self,
        *,
        repository_root: Path | None = None,
        apply_changes: bool = False,
        check_only: bool = False,
        dry_run: bool = False,
        output_format: str = "text",
        project_filter: str | None = None,
        target_module: str | None = None,
        target_namespace: str | None = None,
        report_path: Path | None = None,
        output_dir: Path | None = None,
        selected_projects: t.StrSequence | None = None,
        all_projects: bool = False,
        runner: p.Cli.CommandRunner | None = None,
        settings_type: t.SettingsClass | None = None,
        runtime_settings: p.Settings | None = None,
        settings_overrides: t.JsonMapping | None = None,
        initial_context: p.Context | None = None,
    ) -> None:
        """Initialize with an internal command runner dependency."""
        model_data: _StubChainRuntimeState = {
            "repository_root": repository_root or Path.cwd(),
            "apply_changes": apply_changes,
            "check_only": check_only,
            "dry_run": dry_run,
            "output_format": output_format,
            "project_filter": project_filter,
            "target_module": target_module,
            "target_namespace": target_namespace,
            "report_path": report_path,
            "output_dir": output_dir,
            "selected_projects": selected_projects,
            "all_projects": all_projects,
            # NOTE (multi-agent): flext-i6nq.12 — FlextMixins bootstrap inputs are now
            # native Pydantic fields validated with the rest of model_data.
            "settings_type": settings_type,
            "runtime_settings": runtime_settings,
            "settings_overrides": settings_overrides,
            "initial_context": initial_context,
        }
        self.__pydantic_validator__.validate_python(model_data, self_instance=self)
        self._runner = runner

    @property
    def runner(self) -> p.Cli.CommandRunner | None:
        """Optional command runner dependency for tests and command execution."""
        return self._runner

    @override
    @property
    def project_dirs(self) -> t.SequenceOf[Path] | None:
        """Resolved project directories for targeted validation."""
        names: t.StrSequence | None = self.project_names
        if self.all_projects or names is None:
            return None
        return [self.repository_root / name for name in names]

    def _discover_typed_projects(self, repository_root: Path) -> t.SequenceOf[Path]:
        """Discover projects that should participate in typed dependency checks."""
        _ = self
        return [
            project_root
            for project_root in u.Infra.discover_project_roots(repository_root)
            if (project_root / c.Infra.DEFAULT_SRC_DIR).is_dir()
        ]

    def _is_internal(self, module_name: str, project_name: str) -> bool:
        """Check if a module is an internal project module."""
        _ = self
        root_mod = module_name.split(".", 1)[0]
        project_root = project_name.replace("-", "_")
        if root_mod.startswith(c.Infra.INTERNAL_PREFIXES):
            return True
        return root_mod == project_root

    def _module_resolved(self, module_name: str) -> bool:
        """Check if an external module is installed in the active typed environment."""
        _ = self
        root_module = module_name.split(".", maxsplit=1)[0]
        try:
            return importlib_util.find_spec(root_module) is not None
        except c.EXC_OS_TYPE_VALUE:
            return False

    def analyze(
        self, project_dir: Path, repository_root: Path
    ) -> p.Result[m.Infra.StubAnalysisReport]:
        """Analyze a project for missing typed dependencies.

        Runs mypy for hints and pyrefly for missing imports, then
        classifies each as internal, resolved, or unresolved.

        Args:
            project_dir: Path to the project directory.
            repository_root: Root of the workspace.

        Returns:
            r with analysis report dict.

        """
        try:
            return self._analyze_project(project_dir, repository_root)
        except c.EXC_OS_TYPE_VALUE as exc:
            return r[m.Infra.StubAnalysisReport].fail(
                f"typed dependency analysis failed for {project_dir.name}: {exc}"
            )

    def build_report(
        self, repository_root: Path, project_dirs: t.SequenceOf[Path] | None = None
    ) -> p.Result[m.Infra.ValidationReport]:
        """Validate typed dependency supply chain across projects.

        Args:
            repository_root: Root directory of the workspace.
            project_dirs: Optional specific projects; discovers all if None.

        Returns:
            r with ValidationReport indicating overall status.

        """
        try:
            return self._build_typed_dependency_report(repository_root, project_dirs)
        except c.EXC_OS_TYPE_VALUE as exc:
            return r[m.Infra.ValidationReport].fail_op(
                "typed dependency validation", exc
            )

    def _classify_missing_imports(
        self, missing_imports: t.StrSequence, project_name: str
    ) -> tuple[t.StrSequence, t.StrSequence]:
        """Split missing imports into internal and unresolved external groups."""
        internal = tuple(
            module_name
            for module_name in missing_imports
            if self._is_internal(module_name, project_name)
        )
        external = tuple(
            module_name
            for module_name in missing_imports
            if not self._is_internal(module_name, project_name)
        )
        unresolved = tuple(
            module_name
            for module_name in external
            if not self._module_resolved(module_name)
        )
        return internal, unresolved

    def _analyze_project(
        self, project_dir: Path, repository_root: Path
    ) -> p.Result[m.Infra.StubAnalysisReport]:
        """Analyze one project after path resolution."""
        _ = repository_root
        proj = project_dir.resolve()
        mypy_result = self._run_mypy_hints(proj)
        if mypy_result.failure:
            return r[m.Infra.StubAnalysisReport].fail(
                mypy_result.error or "bounded Mypy analysis failed"
            )
        mypy_hints = mypy_result.value
        pyrefly_result = self._run_pyrefly_missing(proj)
        if pyrefly_result.failure:
            return r[m.Infra.StubAnalysisReport].fail(
                pyrefly_result.error or "Pyrefly analysis failed"
            )
        missing_imports = pyrefly_result.value
        internal, unresolved = self._classify_missing_imports(
            missing_imports, proj.name
        )
        return r[m.Infra.StubAnalysisReport].ok(
            m.Infra.StubAnalysisReport(
                project=proj.name,
                mypy_hints=mypy_hints,
                internal_missing=internal,
                unresolved_missing=unresolved,
                total_missing=len(missing_imports),
            )
        )

    def _project_violations(
        self, project_dir: Path, repository_root: Path
    ) -> t.StrSequence:
        """Return typed-dependency violations for one project."""
        result = self.analyze(project_dir, repository_root)
        if result.failure:
            return (f"{project_dir.name}: {result.error}",)
        data = result.value
        violations: t.MutableSequenceOf[str] = []
        if data.mypy_hints:
            violations.append(
                f"{project_dir.name}: {len(data.mypy_hints)} missing typing packages"
            )
        if data.internal_missing:
            violations.append(
                f"{project_dir.name}: {len(data.internal_missing)} internal missing imports"
            )
        if data.unresolved_missing:
            violations.append(
                f"{project_dir.name}: {len(data.unresolved_missing)} unresolved imports"
            )
        return tuple(violations)

    def _typed_dependency_violations(
        self, projects: t.SequenceOf[Path], repository_root: Path
    ) -> t.StrSequence:
        """Collect typed-dependency violations for all selected projects."""
        violations: t.MutableSequenceOf[str] = []
        for project_dir in projects:
            violations.extend(self._project_violations(project_dir, repository_root))
        return tuple(violations)

    def _build_typed_dependency_report(
        self, repository_root: Path, project_dirs: t.SequenceOf[Path] | None
    ) -> p.Result[m.Infra.ValidationReport]:
        """Build the workspace typed-dependency validation report."""
        root = repository_root.resolve()
        if not root.is_dir():
            return r[m.Infra.ValidationReport].fail(
                f"typed dependency workspace does not exist: {root}"
            )
        projects = project_dirs or self._discover_typed_projects(root)
        violations = self._typed_dependency_violations(projects, root)
        return r[m.Infra.ValidationReport].ok(
            m.Infra.ValidationReport(
                passed=not violations,
                violations=violations,
                summary=(
                    f"typed dependency chain: {len(projects)} projects, {len(violations)} issues"
                ),
            )
        )

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the typed-dependency validation CLI flow."""
        report_result = self.build_report(
            self.repository_root, project_dirs=self.project_dirs
        )
        if report_result.failure:
            return r[bool].fail(
                report_result.error or "typed dependency validation failed"
            )
        report = report_result.unwrap()
        return r[bool].ok(True) if report.passed else r[bool].fail(report.summary)

    def _run_mypy_hints(self, project_dir: Path) -> p.Result[t.StrSequence]:
        """Run mypy and extract install-package hints."""
        runner = self.runner or u.Cli()
        result = runner.run_raw(
            u.Infra.mypy_limited_command((
                c.Infra.POETRY,
                c.Infra.VERB_RUN,
                c.Infra.MYPY,
                c.Infra.DEFAULT_SRC_DIR,
                "--config-file",
                c.Infra.PYPROJECT_FILENAME,
                "--no-error-summary",
            )),
            cwd=project_dir,
            timeout=u.Infra.mypy_runner_timeout(),
        )
        if result.failure:
            return r[t.StrSequence].fail(
                u.Infra.mypy_launch_failure_diagnostic(
                    result.error or "Mypy process launch failed"
                )
            )
        cmd_output: p.Cli.CommandOutput = result.value
        if resource_diagnostic := u.Infra.mypy_failure_diagnostic(cmd_output):
            return r[t.StrSequence].fail(resource_diagnostic)
        return r[t.StrSequence].ok(
            sorted({
                match.group(1).strip()
                for match in c.Infra.MYPY_HINT_RE.finditer(cmd_output.stdout)
                if match.group(1).strip()
            })
        )

    def _run_pyrefly_missing(self, project_dir: Path) -> p.Result[t.StrSequence]:
        """Run pyrefly check and extract missing imports."""
        runner = self.runner or u.Cli()
        result = runner.run(
            [
                c.Infra.POETRY,
                c.Infra.VERB_RUN,
                c.Infra.PYREFLY,
                c.Infra.CHECK,
                c.Infra.DEFAULT_SRC_DIR,
                "--config",
                c.Infra.PYPROJECT_FILENAME,
            ],
            cwd=project_dir,
        )
        if result.failure:
            return r[t.StrSequence].fail(
                result.error or f"Pyrefly process launch failed for {project_dir.name}"
            )
        cmd_output: p.Cli.CommandOutput = result.value
        output = cmd_output.stdout
        seen: t.Infra.StrSet = set()
        ordered: t.MutableSequenceOf[str] = []
        for match in c.Infra.MISSING_IMPORT_RE.finditer(output):
            name = match.group(1).strip()
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return r[t.StrSequence].ok(ordered)


__all__: list[str] = ["FlextInfraStubSupplyChain"]
