"""Dependency detection and analysis service for deptry, pip-check, and typing stubs."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from pydantic import Field

from flext_core import r
from flext_infra import (
    FlextInfraCommandRunner,
    FlextInfraPatterns,
    FlextInfraProjectSelector,
    FlextInfraTomlService,
    c,
    m,
    p,
    t,
    u,
)


def _to_infra_value(value: t.ContainerValue) -> t.Infra.InfraValue | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        converted: list[t.Infra.InfraValue] = []
        for item in value:
            converted_item = _to_infra_value(item)
            if converted_item is None and item is not None:
                return None
            converted.append(converted_item)
        return converted
    if isinstance(value, Mapping):
        converted_map: MutableMapping[str, t.Infra.InfraValue] = {}
        for key, item in value.items():
            converted_item = _to_infra_value(item)
            if converted_item is None and item is not None:
                return None
            converted_map[str(key)] = converted_item
        return converted_map
    return None


class FlextInfraDependencyDetectionModels(m):
    """Pydantic models for dependency detection reports and analysis results."""

    class DeptryIssueGroups(m.ArbitraryTypesModel):
        """Deptry issue grouping model by error code (DEP001-DEP004)."""

        dep001: list[t.Infra.IssueMap] = Field(
            default_factory=lambda: list[t.Infra.IssueMap]()
        )
        dep002: list[t.Infra.IssueMap] = Field(
            default_factory=lambda: list[t.Infra.IssueMap]()
        )
        dep003: list[t.Infra.IssueMap] = Field(
            default_factory=lambda: list[t.Infra.IssueMap]()
        )
        dep004: list[t.Infra.IssueMap] = Field(
            default_factory=lambda: list[t.Infra.IssueMap]()
        )

    class DeptryReport(m.ArbitraryTypesModel):
        """Deptry analysis report with missing, unused, transitive, and dev-in-runtime issues."""

        missing: list[str | None] = Field(default_factory=lambda: list[str | None]())
        unused: list[str | None] = Field(default_factory=lambda: list[str | None]())
        transitive: list[str | None] = Field(default_factory=lambda: list[str | None]())
        dev_in_runtime: list[str | None] = Field(
            default_factory=lambda: list[str | None]()
        )
        raw_count: int = Field(default=0, ge=0)

    class ProjectDependencyReport(m.ArbitraryTypesModel):
        """Project-level dependency report combining deptry results."""

        project: str = Field(min_length=1)
        deptry: FlextInfraDependencyDetectionModels.DeptryReport

    class TypingsReport(m.ArbitraryTypesModel):
        """Typing stubs analysis report with required, current, and delta packages."""

        required_packages: list[str] = Field(default_factory=list)
        hinted: list[str] = Field(default_factory=list)
        missing_modules: list[str] = Field(default_factory=list)
        current: list[str] = Field(default_factory=list)
        to_add: list[str] = Field(default_factory=list)
        to_remove: list[str] = Field(default_factory=list)
        limits_applied: bool = False
        python_version: str | None = None


dm = FlextInfraDependencyDetectionModels


class FlextInfraDependencyDetectionService:
    """Runtime vs dev dependency detector using deptry, pip-check, and mypy stub analysis."""

    DEFAULT_MODULE_TO_TYPES_PACKAGE: Mapping[str, str] = (
        c.Infra.Deps.DEFAULT_MODULE_TO_TYPES_PACKAGE
    )

    def __init__(self) -> None:
        """Initialize the dependency detection service with selector, toml, and runner."""
        self.selector = FlextInfraProjectSelector()
        self.toml = FlextInfraTomlService()
        self.runner: p.Infra.CommandRunner = FlextInfraCommandRunner()

    @staticmethod
    def classify_issues(issues: list[t.Infra.IssueMap]) -> dm.DeptryIssueGroups:
        """Classify deptry issues by error code (DEP001-DEP004)."""
        groups = dm.DeptryIssueGroups()
        for item in issues:
            error_obj = item.get(c.Infra.Toml.ERROR)
            if not isinstance(error_obj, Mapping):
                continue
            code = error_obj.get(c.Infra.Toml.CODE)
            if code == "DEP001":
                groups.dep001.append(item)
            elif code == "DEP002":
                groups.dep002.append(item)
            elif code == "DEP003":
                groups.dep003.append(item)
            elif code == "DEP004":
                groups.dep004.append(item)
        return groups

    def build_project_report(
        self, project_name: str, deptry_issues: list[t.Infra.IssueMap]
    ) -> dm.ProjectDependencyReport:
        """Build a project dependency report from classified deptry issues."""
        classified = self.classify_issues(deptry_issues)

        def _module_name(item: t.Infra.IssueMap) -> str | None:
            val = item.get(c.Infra.Toml.MODULE)
            return str(val) if val is not None else None

        return dm.ProjectDependencyReport(
            project=project_name,
            deptry=dm.DeptryReport(
                missing=[_module_name(item) for item in classified.dep001],
                unused=[_module_name(item) for item in classified.dep002],
                transitive=[_module_name(item) for item in classified.dep003],
                dev_in_runtime=[_module_name(item) for item in classified.dep004],
                raw_count=len(deptry_issues),
            ),
        )

    def discover_projects(
        self, workspace_root: Path, projects_filter: list[str] | None = None
    ) -> r[list[Path]]:
        """Discover projects with pyproject.toml in workspace."""
        names = projects_filter or []
        result = self.selector.resolve_projects(workspace_root, names)
        if result.is_failure:
            return r[list[Path]].fail(result.error or "project resolution failed")
        projects_info: list[m.Infra.Workspace.ProjectInfo] = result.value
        projects = [
            project.path
            for project in projects_info
            if (project.path / c.Infra.Files.PYPROJECT_FILENAME).exists()
        ]
        return r[list[Path]].ok(sorted(projects))

    def get_current_typings_from_pyproject(self, project_path: Path) -> list[str]:
        """Extract currently declared typing packages from project pyproject.toml."""
        pyproject = project_path / c.Infra.Files.PYPROJECT_FILENAME
        read_result = self.toml.read(pyproject)
        if read_result.is_failure:
            return []
        data: t.Infra.ContainerDict = read_result.value
        if not data:
            return []
        names: set[str] = set()
        tool = data.get(c.Infra.Toml.TOOL)
        if tool is not None and isinstance(tool, Mapping):
            poetry = tool.get(c.Infra.Toml.POETRY)
            if poetry is not None and isinstance(poetry, Mapping):
                group = poetry.get(c.Infra.Toml.GROUP)
                if group is not None and isinstance(group, Mapping):
                    typings_group = group.get(c.Infra.Directories.TYPINGS)
                    if typings_group is not None and isinstance(typings_group, Mapping):
                        deps = typings_group.get(c.Infra.Toml.DEPENDENCIES)
                        if deps is not None and isinstance(deps, Mapping):
                            names.update(str(key) for key in deps)
        project = data.get(c.Infra.Toml.PROJECT)
        if project is not None and isinstance(project, Mapping):
            optional = project.get(c.Infra.Toml.OPTIONAL_DEPENDENCIES)
            if optional is not None and isinstance(optional, Mapping):
                typings = optional.get(c.Infra.Directories.TYPINGS)
                if isinstance(typings, list):
                    for spec in typings:
                        if isinstance(spec, str):
                            names.add(
                                spec.split("[")[0].split(">=")[0].split("==")[0].strip()
                            )
                elif typings is not None and isinstance(typings, Mapping):
                    names.update(str(key) for key in typings)
        return sorted(names)

    def get_required_typings(
        self,
        project_path: Path,
        venv_bin: Path,
        limits_path: Path | None = None,
        *,
        include_mypy: bool = True,
    ) -> r[dm.TypingsReport]:
        """Analyze project and generate typing stubs requirements report."""
        limits = self.load_dependency_limits(limits_path)
        exclude_set: set[str] = set()
        typing_libraries = limits.get(c.Infra.Toml.TYPING_LIBRARIES)
        if typing_libraries is not None and isinstance(typing_libraries, Mapping):
            excluded = typing_libraries.get(c.Infra.Toml.EXCLUDE)
            if isinstance(excluded, list):
                exclude_set = {str(item) for item in excluded}
        hinted: list[str] = []
        missing_modules: list[str] = []
        if include_mypy:
            hints_result = self.run_mypy_stub_hints(project_path, venv_bin)
            if hints_result.is_failure:
                return r[dm.TypingsReport].fail(
                    hints_result.error or "typing hint detection failed"
                )
            typed_hints: tuple[list[str], list[str]] = hints_result.value
            hinted, missing_modules = typed_hints
        required_set: set[str] = set(hinted)
        for module_name in missing_modules:
            package = self.module_to_types_package(module_name, limits)
            if package:
                required_set.add(package)
        required_set -= exclude_set
        current = self.get_current_typings_from_pyproject(project_path)
        current_set = set(current)
        python_cfg = limits.get(c.Infra.Toml.PYTHON)
        python_version = (
            str(python_cfg.get(c.Infra.Toml.VERSION))
            if python_cfg is not None
            and isinstance(python_cfg, Mapping)
            and (python_cfg.get(c.Infra.Toml.VERSION) is not None)
            else None
        )
        report = dm.TypingsReport(
            required_packages=sorted(required_set),
            hinted=hinted,
            missing_modules=missing_modules,
            current=current,
            to_add=sorted(required_set - current_set),
            to_remove=sorted(current_set - required_set),
            limits_applied=bool(limits),
            python_version=python_version,
        )
        return r[dm.TypingsReport].ok(report)

    def load_dependency_limits(
        self, limits_path: Path | None = None
    ) -> Mapping[str, t.Infra.InfraValue]:
        """Load dependency limits configuration from TOML file."""
        path = limits_path or Path(__file__).resolve().parent / "dependency_limits.toml"
        result = self.toml.read(path)
        if result.is_failure:
            return {}
        limits: MutableMapping[str, t.Infra.InfraValue] = {}
        toml_data: t.Infra.ContainerDict = result.value
        for key, value in toml_data.items():
            converted = _to_infra_value(value)
            if converted is not None or value is None:
                limits[str(key)] = converted
        return limits

    def module_to_types_package(
        self, module_name: str, limits: Mapping[str, t.Infra.InfraValue]
    ) -> str | None:
        """Map a module name to its corresponding types-* package."""
        root = module_name.split(".", 1)[0]
        if root.startswith(FlextInfraPatterns.INTERNAL_PREFIXES):
            return None
        typing_libraries = limits.get(c.Infra.Toml.TYPING_LIBRARIES)
        if typing_libraries is not None and isinstance(typing_libraries, Mapping):
            module_to_package = typing_libraries.get(c.Infra.Toml.MODULE_TO_PACKAGE)
            if (
                module_to_package is not None
                and isinstance(module_to_package, Mapping)
                and (root in module_to_package)
            ):
                value = module_to_package[root]
                return str(value)
        return self.DEFAULT_MODULE_TO_TYPES_PACKAGE.get(root.lower())

    def run_deptry(
        self,
        project_path: Path,
        venv_bin: Path,
        *,
        config_path: Path | None = None,
        json_output_path: Path | None = None,
        extend_exclude: list[str] | None = None,
    ) -> r[tuple[list[t.Infra.IssueMap], int]]:
        """Run deptry analysis on a project and parse JSON output."""
        config = config_path or project_path / c.Infra.Files.PYPROJECT_FILENAME
        if not config.exists():
            return r[tuple[list[t.Infra.IssueMap], int]].ok(([], 0))
        out_file = json_output_path or project_path / ".deptry-report.json"
        cmd: list[str] = [
            str(venv_bin / c.Infra.Toml.DEPTRY),
            ".",
            "--config",
            str(config),
            "--json-output",
            str(out_file),
            "--no-ansi",
        ]
        if extend_exclude:
            for excluded in extend_exclude:
                cmd.extend(["--extend-exclude", excluded])
        result = self.runner.run_raw(
            cmd, cwd=project_path, timeout=c.Infra.Timeouts.MEDIUM
        )
        if result.is_failure:
            return r[tuple[list[t.Infra.IssueMap], int]].fail(
                result.error or "deptry execution failed"
            )
        issues: list[t.Infra.IssueMap] = []
        if out_file.exists():
            raw = out_file.read_text(encoding=c.Infra.Encoding.DEFAULT)
            loaded_result = u.Infra.Io.parse(raw) if raw.strip() else None
            loaded_payload: t.ContainerValue = (
                loaded_result.value
                if loaded_result is not None and loaded_result.is_success
                else []
            )
            if isinstance(loaded_payload, list):
                normalized_issues: list[t.Infra.IssueMap] = []
                for item in loaded_payload:
                    if not isinstance(item, dict):
                        continue
                    converted_issue: dict[str, t.Infra.InfraValue] = {}
                    valid = True
                    for key, value in item.items():
                        converted = _to_infra_value(value)
                        if converted is None and value is not None:
                            valid = False
                            break
                        converted_issue[str(key)] = converted
                    if valid:
                        normalized_issues.append(converted_issue)
                issues = normalized_issues
            if json_output_path is None:
                with contextlib.suppress(OSError):
                    out_file.unlink()
        cmd_result: m.Infra.Core.CommandOutput = result.value
        return r[tuple[list[t.Infra.IssueMap], int]].ok((issues, cmd_result.exit_code))

    def run_mypy_stub_hints(
        self,
        project_path: Path,
        venv_bin: Path,
        *,
        timeout: int = c.Infra.Timeouts.DEFAULT,
    ) -> r[tuple[list[str], list[str]]]:
        """Run mypy to detect missing type stubs and hinted packages."""
        mypy_bin = venv_bin / c.Infra.Toml.MYPY
        if not mypy_bin.exists():
            return r[tuple[list[str], list[str]]].ok(([], []))
        cmd: list[str] = [
            str(mypy_bin),
            c.Infra.Paths.DEFAULT_SRC_DIR,
            "--config-file",
            c.Infra.Files.PYPROJECT_FILENAME,
            "--no-error-summary",
        ]
        env = {
            **os.environ,
            "VIRTUAL_ENV": str(venv_bin.parent),
            "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
        }
        result = self.runner.run_raw(cmd, cwd=project_path, timeout=timeout, env=env)
        if result.is_failure:
            return r[tuple[list[str], list[str]]].fail(
                result.error or "mypy execution failed"
            )
        cmd_result: m.Infra.Core.CommandOutput = result.value
        output = f"{cmd_result.stdout}\n{cmd_result.stderr}"
        hinted = {
            match.group(1).strip()
            for match in FlextInfraPatterns.MYPY_HINT_RE.finditer(output)
            if match.group(1).strip()
        }
        missing = {
            match.group(1).strip()
            for match in FlextInfraPatterns.MYPY_STUB_RE.finditer(output)
            if match.group(1).strip()
        }
        return r[tuple[list[str], list[str]]].ok((sorted(hinted), sorted(missing)))

    def run_pip_check(
        self, workspace_root: Path, venv_bin: Path
    ) -> r[tuple[list[str], int]]:
        """Run pip check to detect dependency conflicts in workspace."""
        pip = venv_bin / "pip"
        if not pip.exists():
            return r[tuple[list[str], int]].ok(([], 0))
        env = {**os.environ, "VIRTUAL_ENV": str(venv_bin.parent)}
        result = self.runner.run_raw(
            [str(pip), c.Infra.Verbs.CHECK],
            cwd=workspace_root,
            timeout=c.Infra.Timeouts.SHORT,
            env=env,
        )
        if result.is_failure:
            return r[tuple[list[str], int]].fail(result.error or "pip check failed")
        cmd_result: m.Infra.Core.CommandOutput = result.value
        output = cmd_result.stdout
        lines = output.strip().splitlines() if output else []
        return r[tuple[list[str], int]].ok((lines, cmd_result.exit_code))


_service = FlextInfraDependencyDetectionService()


def discover_projects(
    workspace_root: Path, projects_filter: list[str] | None = None
) -> r[list[Path]]:
    """Discover projects with pyproject.toml in workspace."""
    return _service.discover_projects(workspace_root, projects_filter=projects_filter)


def run_deptry(
    project_path: Path,
    venv_bin: Path,
    *,
    config_path: Path | None = None,
    json_output_path: Path | None = None,
    extend_exclude: list[str] | None = None,
) -> r[tuple[list[t.Infra.IssueMap], int]]:
    """Run deptry analysis on a project and parse JSON output."""
    return _service.run_deptry(
        project_path,
        venv_bin,
        config_path=config_path,
        json_output_path=json_output_path,
        extend_exclude=extend_exclude,
    )


def run_pip_check(workspace_root: Path, venv_bin: Path) -> r[tuple[list[str], int]]:
    """Run pip check to detect dependency conflicts in workspace."""
    return _service.run_pip_check(workspace_root, venv_bin)


def classify_issues(issues: list[t.Infra.IssueMap]) -> dm.DeptryIssueGroups:
    """Classify deptry issues by error code (DEP001-DEP004)."""
    return _service.classify_issues(issues)


def build_project_report(
    project_name: str, deptry_issues: list[t.Infra.IssueMap]
) -> dm.ProjectDependencyReport:
    """Build a project dependency report from classified deptry issues."""
    return _service.build_project_report(project_name, deptry_issues)


def load_dependency_limits(
    limits_path: Path | None = None,
) -> Mapping[str, t.Infra.InfraValue]:
    """Load dependency limits configuration from TOML file."""
    return _service.load_dependency_limits(limits_path)


def run_mypy_stub_hints(
    project_path: Path, venv_bin: Path, *, timeout: int = c.Infra.Timeouts.DEFAULT
) -> r[tuple[list[str], list[str]]]:
    """Run mypy to detect missing type stubs and hinted packages."""
    return _service.run_mypy_stub_hints(project_path, venv_bin, timeout=timeout)


def module_to_types_package(
    module_name: str, limits: Mapping[str, t.Infra.InfraValue]
) -> str | None:
    """Map a module name to its corresponding types-* package."""
    return _service.module_to_types_package(module_name, limits)


def get_current_typings_from_pyproject(project_path: Path) -> list[str]:
    """Extract currently declared typing packages from project pyproject.toml."""
    return _service.get_current_typings_from_pyproject(project_path)


def get_required_typings(
    project_path: Path,
    venv_bin: Path,
    limits_path: Path | None = None,
    *,
    include_mypy: bool = True,
) -> r[dm.TypingsReport]:
    """Analyze project and generate typing stubs requirements report."""
    return _service.get_required_typings(
        project_path, venv_bin, limits_path=limits_path, include_mypy=include_mypy
    )


__all__ = [
    "FlextInfraDependencyDetectionModels",
    "FlextInfraDependencyDetectionService",
    "build_project_report",
    "classify_issues",
    "discover_projects",
    "dm",
    "get_current_typings_from_pyproject",
    "get_required_typings",
    "load_dependency_limits",
    "module_to_types_package",
    "run_deptry",
    "run_mypy_stub_hints",
    "run_pip_check",
]
