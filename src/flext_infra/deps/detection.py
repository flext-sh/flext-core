"""Dependency detection and analysis service for deptry, pip-check, and typing stubs."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from flext_core import FlextLogger, r
from flext_infra import (
    FlextInfraUtilitiesPatterns,
    FlextInfraUtilitiesSelection,
    c,
    m,
    p,
    t,
    u,
)


class FlextInfraDependencyDetectionService:
    """Runtime vs dev dependency detector using deptry, pip-check, and mypy stub analysis."""

    _log = FlextLogger.create_module_logger(__name__)

    DEFAULT_MODULE_TO_TYPES_PACKAGE: Mapping[str, str] = (
        c.Infra.Deps.DEFAULT_MODULE_TO_TYPES_PACKAGE
    )

    def __init__(self) -> None:
        """Initialize the dependency detection service with selector, toml, and runner."""
        self.selector: FlextInfraUtilitiesSelection | None = None
        self.toml: p.Infra.TomlReader | None = None
        self.runner: p.Infra.CommandRunner | None = None

    def _resolve_projects(
        self,
        workspace_root: Path,
        names: list[str],
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        if self.selector is not None:
            return self.selector.resolve_projects(workspace_root, names)
        return u.Infra.resolve_projects(workspace_root, names)

    def _read_plain(self, path: Path) -> r[t.Infra.TomlConfig]:
        if self.toml is not None:
            return self.toml.read_plain(path)
        return u.Infra.read_plain(path)

    def _run_raw(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> r[m.Infra.Core.CommandOutput]:
        if self.runner is not None:
            return self.runner.run_raw(cmd, cwd=cwd, timeout=timeout, env=env)
        return u.Infra.run_raw(cmd, cwd=cwd, timeout=timeout, env=env)

    @staticmethod
    def to_infra_value(
        value: t.Infra.InfraValue | None,
    ) -> t.Infra.InfraValue:
        """Convert container value to namespaced infra value."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            try:
                sequence: list[JsonValue] = TypeAdapter(
                    list[JsonValue],
                ).validate_python(value)
            except ValidationError:
                return None
            converted: list[t.Infra.InfraValue] = []
            for item in sequence:
                converted_item = FlextInfraDependencyDetectionService.to_infra_value(
                    item
                )
                if converted_item is None and item is not None:
                    return None
                if not isinstance(converted_item, (str, int, float, bool, type(None))):
                    return None
                converted.append(converted_item)
            return converted
        try:
            mapping_value = TypeAdapter(dict[str, t.Infra.InfraValue]).validate_python(
                value
            )
        except ValidationError:
            return None
        converted_map: dict[str, t.Infra.InfraValue] = {}
        for key, item in mapping_value.items():
            converted_item = FlextInfraDependencyDetectionService.to_infra_value(item)
            if converted_item is None and item is not None:
                return None
            if not isinstance(converted_item, (str, int, float, bool, type(None))):
                return None
            converted_map[str(key)] = converted_item
        return converted_map

    @staticmethod
    def _mapping_from_value(
        value: t.Infra.TomlValue | t.Infra.InfraValue | None,
    ) -> t.Infra.TomlConfig:
        if value is None:
            return {}
        mapped_value = u.Infra.as_toml_mapping(value)
        if mapped_value is None:
            return {}
        normalized: t.Infra.TomlConfig = {}
        for key, raw_value in mapped_value.items():
            converted = FlextInfraDependencyDetectionService.to_infra_value(raw_value)
            if converted is None and raw_value is not None:
                continue
            normalized[key] = converted
        return normalized

    @staticmethod
    def classify_issues(
        issues: list[t.Infra.IssueMap],
    ) -> m.Infra.Deps.DeptryIssueGroups:
        """Classify deptry issues by error code (DEP001-DEP004)."""
        groups = m.Infra.Deps.DeptryIssueGroups.model_validate({
            "dep001": [],
            "dep002": [],
            "dep003": [],
            "dep004": [],
        })
        for item in issues:
            normalized_item: dict[str, str] = {}
            for key, raw_value in item.items():
                if raw_value is None:
                    normalized_item[str(key)] = ""
                    continue
                if isinstance(raw_value, (str, int, float, bool)):
                    normalized_item[str(key)] = str(raw_value)
            error_obj = item.get(c.Infra.Toml.ERROR)
            if not isinstance(error_obj, Mapping):
                continue
            try:
                error_data = TypeAdapter(dict[str, t.Infra.InfraValue]).validate_python(
                    error_obj
                )
            except ValidationError:
                continue
            code = error_data.get(c.Infra.Toml.CODE)
            if code == "DEP001":
                groups.dep001.append(normalized_item)
            elif code == "DEP002":
                groups.dep002.append(normalized_item)
            elif code == "DEP003":
                groups.dep003.append(normalized_item)
            elif code == "DEP004":
                groups.dep004.append(normalized_item)
        return groups

    @staticmethod
    def _to_toml_config(
        payload: Mapping[str, t.Infra.TomlValue | t.Infra.InfraValue],
    ) -> t.Infra.TomlConfig:
        normalized: t.Infra.TomlConfig = {}
        for key, value in payload.items():
            converted = FlextInfraDependencyDetectionService.to_infra_value(value)
            if converted is None and value is not None:
                continue
            normalized[str(key)] = converted
        return normalized

    def build_project_report(
        self,
        project_name: str,
        deptry_issues: list[t.Infra.IssueMap],
    ) -> m.Infra.Deps.ProjectDependencyReport:
        """Build a project dependency report from classified deptry issues."""
        classified = self.classify_issues(deptry_issues)

        def _module_name(item: Mapping[str, t.Infra.InfraValue]) -> str | None:
            val = item.get(c.Infra.Toml.MODULE)
            if val is None:
                return None
            return str(val)

        missing: list[str] = []
        unused: list[str] = []
        transitive: list[str] = []
        dev_in_runtime: list[str] = []

        for item in classified.dep001:
            name = _module_name(item)
            if name is not None:
                missing.append(name)
        for item in classified.dep002:
            name = _module_name(item)
            if name is not None:
                unused.append(name)
        for item in classified.dep003:
            name = _module_name(item)
            if name is not None:
                transitive.append(name)
        for item in classified.dep004:
            name = _module_name(item)
            if name is not None:
                dev_in_runtime.append(name)

        return m.Infra.Deps.ProjectDependencyReport(
            project=project_name,
            deptry=m.Infra.Deps.DeptryReport(
                missing=missing,
                unused=unused,
                transitive=transitive,
                dev_in_runtime=dev_in_runtime,
                raw_count=len(deptry_issues),
            ),
        )

    def discover_project_paths(
        self,
        workspace_root: Path,
        projects_filter: list[str] | None = None,
    ) -> r[list[Path]]:
        """Discover project paths with pyproject.toml in workspace.

        Returns only the Path objects, filtered to those with pyproject.toml.
        For full ProjectInfo metadata, use u.Infra.discover_projects().
        """
        names = projects_filter or []
        result = self._resolve_projects(workspace_root, names)
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
        read_result = self._read_plain(pyproject)
        if read_result.is_failure:
            return []
        data = self._to_toml_config(read_result.value)
        if not data:
            return []
        names: set[str] = set()
        tool = self._mapping_from_value(data.get(c.Infra.Toml.TOOL))
        poetry = self._mapping_from_value(tool.get(c.Infra.Toml.POETRY))
        group = self._mapping_from_value(poetry.get(c.Infra.Toml.GROUP))
        typings_group = self._mapping_from_value(group.get(c.Infra.Directories.TYPINGS))
        deps = self._mapping_from_value(typings_group.get(c.Infra.Toml.DEPENDENCIES))
        names.update(str(key) for key in deps)
        project = self._mapping_from_value(data.get(c.Infra.Toml.PROJECT))
        optional = self._mapping_from_value(
            project.get(c.Infra.Toml.OPTIONAL_DEPENDENCIES)
        )
        typings = optional.get(c.Infra.Directories.TYPINGS)
        if isinstance(typings, list):
            try:
                typed_typings = TypeAdapter(list[str]).validate_python([
                    str(s) for s in typings
                ])
            except ValidationError:
                typed_typings: list[str] = []
            for spec in typed_typings:
                spec_text = str(spec)
                names.add(
                    spec_text
                    .split("[", maxsplit=1)[0]
                    .split(">=", maxsplit=1)[0]
                    .split("==", maxsplit=1)[0]
                    .strip(),
                )
        elif isinstance(typings, Mapping):
            try:
                typed_typings_map = TypeAdapter(dict[str, str]).validate_python({
                    k: str(v) for k, v in typings.items()
                })
            except ValidationError:
                typed_typings_map: dict[str, str] = {}
            names.update(typed_typings_map.keys())
        return sorted(names)

    def get_required_typings(
        self,
        project_path: Path,
        venv_bin: Path,
        limits_path: Path | None = None,
        *,
        include_mypy: bool = True,
    ) -> r[m.Infra.Deps.TypingsReport]:
        """Analyze project and generate typing stubs requirements report."""
        limits = self.load_dependency_limits(limits_path)
        exclude_set: set[str] = set()
        typing_libraries = limits.get(c.Infra.Toml.TYPING_LIBRARIES)
        if typing_libraries is not None and isinstance(typing_libraries, Mapping):
            excluded = typing_libraries.get(c.Infra.Toml.EXCLUDE)
            if isinstance(excluded, list):
                try:
                    typed_excluded = TypeAdapter(list[str]).validate_python([
                        str(e) for e in excluded
                    ])
                except ValidationError:
                    typed_excluded: list[str] = []
                exclude_set = set(typed_excluded)
        hinted: list[str] = []
        missing_modules: list[str] = []
        if include_mypy:
            hints_result = self.run_mypy_stub_hints(project_path, venv_bin)
            if hints_result.is_failure:
                return r[m.Infra.Deps.TypingsReport].fail(
                    hints_result.error or "typing hint detection failed",
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
        report = m.Infra.Deps.TypingsReport(
            required_packages=sorted(required_set),
            hinted=hinted,
            missing_modules=missing_modules,
            current=current,
            to_add=sorted(required_set - current_set),
            to_remove=sorted(current_set - required_set),
            limits_applied=bool(limits),
            python_version=python_version,
        )
        return r[m.Infra.Deps.TypingsReport].ok(report)

    def load_dependency_limits(
        self,
        limits_path: Path | None = None,
    ) -> Mapping[str, t.Infra.TomlValue]:
        """Load dependency limits configuration from TOML file."""
        path = limits_path or Path(__file__).resolve().parent / "dependency_limits.toml"
        result = self._read_plain(path)
        if result.is_failure:
            return {}
        limits: MutableMapping[str, t.Infra.TomlValue] = {}
        toml_data = self._to_toml_config(result.value)
        for key, value in toml_data.items():
            converted = FlextInfraDependencyDetectionService.to_infra_value(value)
            if converted is not None or value is None:
                limits[str(key)] = converted
        return limits

    def module_to_types_package(
        self,
        module_name: str,
        limits: Mapping[str, t.Infra.TomlValue],
    ) -> str | None:
        """Map a module name to its corresponding types-* package."""
        root = module_name.split(".", 1)[0]
        if root.startswith(FlextInfraUtilitiesPatterns.INTERNAL_PREFIXES):
            return None
        typing_libraries = limits.get(c.Infra.Toml.TYPING_LIBRARIES)
        if typing_libraries is not None and isinstance(typing_libraries, Mapping):
            module_to_package = typing_libraries.get(c.Infra.Toml.MODULE_TO_PACKAGE)
            if (
                module_to_package is not None
                and isinstance(module_to_package, Mapping)
                and (root in module_to_package)
            ):
                try:
                    module_to_package_map = TypeAdapter(dict[str, str]).validate_python({
                        k: str(v) for k, v in module_to_package.items()
                    })
                except ValidationError:
                    module_to_package_map: dict[str, str] = {}
                value = module_to_package_map.get(root)
                if value is None:
                    return None
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
        result = self._run_raw(
            cmd,
            cwd=project_path,
            timeout=c.Infra.Timeouts.MEDIUM,
        )
        if result.is_failure:
            return r[tuple[list[t.Infra.IssueMap], int]].fail(
                result.error or "deptry execution failed",
            )
        issues: list[t.Infra.IssueMap] = []
        if out_file.exists():
            raw = out_file.read_text(encoding=c.Infra.Encoding.DEFAULT)
            loaded_result = u.Infra.parse(raw) if raw.strip() else None
            if (
                loaded_result is not None
                and loaded_result.is_success
                and isinstance(loaded_result.value, list)
            ):
                normalized_issues: list[t.Infra.IssueMap] = []
                for item in loaded_result.value:
                    if not isinstance(item, dict):
                        continue
                    converted_issue: dict[str, t.Infra.TomlValue] = {}
                    valid = True
                    for key, value in item.items():
                        converted = FlextInfraDependencyDetectionService.to_infra_value(
                            value
                        )
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
        result = self._run_raw(cmd, cwd=project_path, timeout=timeout, env=env)
        if result.is_failure:
            return r[tuple[list[str], list[str]]].fail(
                result.error or "mypy execution failed",
            )
        cmd_result: m.Infra.Core.CommandOutput = result.value
        output = f"{cmd_result.stdout}\n{cmd_result.stderr}"
        hinted = {
            match.group(1).strip()
            for match in FlextInfraUtilitiesPatterns.MYPY_HINT_RE.finditer(output)
            if match.group(1).strip()
        }
        missing = {
            match.group(1).strip()
            for match in FlextInfraUtilitiesPatterns.MYPY_STUB_RE.finditer(output)
            if match.group(1).strip()
        }
        return r[tuple[list[str], list[str]]].ok((sorted(hinted), sorted(missing)))

    def run_pip_check(
        self,
        workspace_root: Path,
        venv_bin: Path,
    ) -> r[tuple[list[str], int]]:
        """Run pip check to detect dependency conflicts in workspace."""
        pip = venv_bin / "pip"
        if not pip.exists():
            return r[tuple[list[str], int]].ok(([], 0))
        env = {**os.environ, "VIRTUAL_ENV": str(venv_bin.parent)}
        result = self._run_raw(
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
dm = m.Infra.Deps


def discover_project_paths(
    workspace_root: Path,
    projects_filter: list[str] | None = None,
) -> r[list[Path]]:
    """Discover project paths with pyproject.toml in workspace."""
    return _service.discover_project_paths(
        workspace_root, projects_filter=projects_filter
    )


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


def classify_issues(issues: list[t.Infra.IssueMap]) -> m.Infra.Deps.DeptryIssueGroups:
    """Classify deptry issues by error code (DEP001-DEP004)."""
    return _service.classify_issues(issues)


def build_project_report(
    project_name: str,
    deptry_issues: list[t.Infra.IssueMap],
) -> m.Infra.Deps.ProjectDependencyReport:
    """Build a project dependency report from classified deptry issues."""
    return _service.build_project_report(project_name, deptry_issues)


def load_dependency_limits(
    limits_path: Path | None = None,
) -> Mapping[str, t.Infra.TomlValue]:
    """Load dependency limits configuration from TOML file."""
    return _service.load_dependency_limits(limits_path)


def run_mypy_stub_hints(
    project_path: Path,
    venv_bin: Path,
    *,
    timeout: int = c.Infra.Timeouts.DEFAULT,
) -> r[tuple[list[str], list[str]]]:
    """Run mypy to detect missing type stubs and hinted packages."""
    return _service.run_mypy_stub_hints(project_path, venv_bin, timeout=timeout)


def module_to_types_package(
    module_name: str,
    limits: Mapping[str, t.Infra.TomlValue],
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
) -> r[m.Infra.Deps.TypingsReport]:
    """Analyze project and generate typing stubs requirements report."""
    return _service.get_required_typings(
        project_path,
        venv_bin,
        limits_path=limits_path,
        include_mypy=include_mypy,
    )


__all__ = [
    "FlextInfraDependencyDetectionService",
    "build_project_report",
    "classify_issues",
    "discover_project_paths",
    "dm",
    "get_current_typings_from_pyproject",
    "get_required_typings",
    "load_dependency_limits",
    "module_to_types_package",
    "run_deptry",
    "run_mypy_stub_hints",
    "run_pip_check",
]
