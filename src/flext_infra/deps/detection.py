from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from flext_core.result import FlextResult, r
from flext_core.typings import t
from pydantic import Field

from flext_infra.constants import ic
from flext_infra.models import im
from flext_infra.patterns import InfraPatterns
from flext_infra.selection import ProjectSelector
from flext_infra.subprocess import CommandRunner
from flext_infra.toml_io import TomlService

type IssueMap = dict[str, t.GeneralValueType]


class DependencyDetectionModels(im):
    class DeptryIssueGroups(im.ArbitraryTypesModel):
        dep001: list[IssueMap] = Field(default_factory=list)
        dep002: list[IssueMap] = Field(default_factory=list)
        dep003: list[IssueMap] = Field(default_factory=list)
        dep004: list[IssueMap] = Field(default_factory=list)

    class DeptryReport(im.ArbitraryTypesModel):
        missing: list[t.GeneralValueType] = Field(default_factory=list)
        unused: list[t.GeneralValueType] = Field(default_factory=list)
        transitive: list[t.GeneralValueType] = Field(default_factory=list)
        dev_in_runtime: list[t.GeneralValueType] = Field(default_factory=list)
        raw_count: int = Field(default=0, ge=0)

    class ProjectDependencyReport(im.ArbitraryTypesModel):
        project: str = Field(min_length=1)
        deptry: "DependencyDetectionModels.DeptryReport"

    class TypingsReport(im.ArbitraryTypesModel):
        required_packages: list[str] = Field(default_factory=list)
        hinted: list[str] = Field(default_factory=list)
        missing_modules: list[str] = Field(default_factory=list)
        current: list[str] = Field(default_factory=list)
        to_add: list[str] = Field(default_factory=list)
        to_remove: list[str] = Field(default_factory=list)
        limits_applied: bool = False
        python_version: str | None = None


dm = DependencyDetectionModels


class DependencyDetectionService:
    DEFAULT_MODULE_TO_TYPES_PACKAGE: Mapping[str, str] = {
        "yaml": "types-pyyaml",
        "ldap3": "types-ldap3",
        "redis": "types-redis",
        "requests": "types-requests",
        "setuptools": "types-setuptools",
        "toml": "types-toml",
        "dateutil": "types-python-dateutil",
        "psutil": "types-psutil",
        "psycopg2": "types-psycopg2",
        "protobuf": "types-protobuf",
        "pyyaml": "types-pyyaml",
        "decorator": "types-decorator",
        "jsonschema": "types-jsonschema",
        "openpyxl": "types-openpyxl",
        "xlrd": "types-xlrd",
    }

    def __init__(self) -> None:
        self._selector = ProjectSelector()
        self._toml = TomlService()
        self._runner = CommandRunner()

    def discover_projects(
        self,
        workspace_root: Path,
        projects_filter: list[str] | None = None,
    ) -> FlextResult[list[Path]]:
        names = projects_filter or []
        result = self._selector.resolve_projects(workspace_root, names)
        if result.is_failure:
            return r[list[Path]].fail(result.error or "project resolution failed")

        projects = [
            project.path
            for project in result.value
            if (project.path / ic.Files.PYPROJECT_FILENAME).exists()
        ]
        return r[list[Path]].ok(sorted(projects))

    def run_deptry(
        self,
        project_path: Path,
        venv_bin: Path,
        *,
        config_path: Path | None = None,
        json_output_path: Path | None = None,
        extend_exclude: list[str] | None = None,
    ) -> FlextResult[tuple[list[IssueMap], int]]:
        config = config_path or (project_path / ic.Files.PYPROJECT_FILENAME)
        if not config.exists():
            return r[tuple[list[IssueMap], int]].ok(([], 0))

        out_file = json_output_path or (project_path / ".deptry-report.json")
        cmd: list[str] = [
            str(venv_bin / "deptry"),
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

        result = self._runner.run_raw(cmd, cwd=project_path, timeout=120)
        if result.is_failure:
            return r[tuple[list[IssueMap], int]].fail(
                result.error or "deptry execution failed"
            )

        issues: list[IssueMap] = []
        if out_file.exists():
            try:
                raw = out_file.read_text(encoding=ic.Encoding.DEFAULT)
                loaded = json.loads(raw) if raw.strip() else []
                if isinstance(loaded, list):
                    issues = [
                        cast("IssueMap", item)
                        for item in loaded
                        if isinstance(item, dict)
                    ]
            except (json.JSONDecodeError, OSError):
                issues = []
            if json_output_path is None:
                with contextlib.suppress(OSError):
                    out_file.unlink()

        return r[tuple[list[IssueMap], int]].ok((issues, result.value.exit_code))

    def run_pip_check(
        self,
        workspace_root: Path,
        venv_bin: Path,
    ) -> FlextResult[tuple[list[str], int]]:
        pip = venv_bin / "pip"
        if not pip.exists():
            return r[tuple[list[str], int]].ok(([], 0))

        env = {**os.environ, "VIRTUAL_ENV": str(venv_bin.parent)}
        result = self._runner.run_raw(
            [str(pip), "check"],
            cwd=workspace_root,
            timeout=60,
            env=env,
        )
        if result.is_failure:
            return r[tuple[list[str], int]].fail(result.error or "pip check failed")

        output = result.value.stdout
        lines = output.strip().splitlines() if output else []
        return r[tuple[list[str], int]].ok((lines, result.value.exit_code))

    @staticmethod
    def classify_issues(
        issues: list[IssueMap],
    ) -> dm.DeptryIssueGroups:
        groups = dm.DeptryIssueGroups()
        for item in issues:
            error_obj = item.get("error")
            if not isinstance(error_obj, Mapping):
                continue
            code = error_obj.get("code")
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
        self,
        project_name: str,
        deptry_issues: list[IssueMap],
    ) -> dm.ProjectDependencyReport:
        classified = self.classify_issues(deptry_issues)
        return dm.ProjectDependencyReport(
            project=project_name,
            deptry=dm.DeptryReport(
                missing=[item.get("module") for item in classified.dep001],
                unused=[item.get("module") for item in classified.dep002],
                transitive=[item.get("module") for item in classified.dep003],
                dev_in_runtime=[item.get("module") for item in classified.dep004],
                raw_count=len(deptry_issues),
            ),
        )

    def load_dependency_limits(
        self,
        limits_path: Path | None = None,
    ) -> dict[str, t.GeneralValueType]:
        path = limits_path or (
            Path(__file__).resolve().parent / "dependency_limits.toml"
        )
        result = self._toml.read(path)
        if result.is_failure:
            return {}
        return cast("dict[str, t.GeneralValueType]", result.value)

    def run_mypy_stub_hints(
        self,
        project_path: Path,
        venv_bin: Path,
        *,
        timeout: int = 300,
    ) -> FlextResult[tuple[list[str], list[str]]]:
        mypy_bin = venv_bin / "mypy"
        if not mypy_bin.exists():
            return r[tuple[list[str], list[str]]].ok(([], []))

        cmd: list[str] = [
            str(mypy_bin),
            "src",
            "--config-file",
            "pyproject.toml",
            "--no-error-summary",
        ]
        env = {
            **os.environ,
            "VIRTUAL_ENV": str(venv_bin.parent),
            "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
        }
        result = self._runner.run_raw(
            cmd,
            cwd=project_path,
            timeout=timeout,
            env=env,
        )
        if result.is_failure:
            return r[tuple[list[str], list[str]]].fail(
                result.error or "mypy execution failed"
            )

        output = f"{result.value.stdout}\n{result.value.stderr}"
        hinted = {
            match.group(1).strip()
            for match in InfraPatterns.MYPY_HINT_RE.finditer(output)
            if match.group(1).strip()
        }
        missing = {
            match.group(1).strip()
            for match in InfraPatterns.MYPY_STUB_RE.finditer(output)
            if match.group(1).strip()
        }
        return r[tuple[list[str], list[str]]].ok((sorted(hinted), sorted(missing)))

    def module_to_types_package(
        self,
        module_name: str,
        limits: dict[str, t.GeneralValueType],
    ) -> str | None:
        root = module_name.split(".", 1)[0]
        if root.startswith(InfraPatterns.INTERNAL_PREFIXES):
            return None

        typing_libraries = limits.get("typing_libraries")
        if isinstance(typing_libraries, Mapping):
            module_to_package = typing_libraries.get("module_to_package")
            if isinstance(module_to_package, Mapping) and root in module_to_package:
                value = module_to_package[root]
                return str(value)

        return self.DEFAULT_MODULE_TO_TYPES_PACKAGE.get(root.lower())

    def get_current_typings_from_pyproject(self, project_path: Path) -> list[str]:
        pyproject = project_path / ic.Files.PYPROJECT_FILENAME
        read_result = self._toml.read(pyproject)
        if read_result.is_failure:
            return []
        data = read_result.value
        if not data:
            return []

        names: set[str] = set()

        tool = data.get("tool")
        if isinstance(tool, Mapping):
            poetry = tool.get("poetry")
            if isinstance(poetry, Mapping):
                group = poetry.get("group")
                if isinstance(group, Mapping):
                    typings_group = group.get("typings")
                    if isinstance(typings_group, Mapping):
                        deps = typings_group.get("dependencies")
                        if isinstance(deps, Mapping):
                            names.update(str(key) for key in deps)

        project = data.get("project")
        if isinstance(project, Mapping):
            optional = project.get("optional-dependencies")
            if isinstance(optional, Mapping):
                typings = optional.get("typings")
                if isinstance(typings, list):
                    for spec in typings:
                        if isinstance(spec, str):
                            names.add(
                                spec.split("[")[0].split(">=")[0].split("==")[0].strip()
                            )
                elif isinstance(typings, Mapping):
                    names.update(str(key) for key in typings)

        return sorted(names)

    def get_required_typings(
        self,
        project_path: Path,
        venv_bin: Path,
        limits_path: Path | None = None,
        *,
        include_mypy: bool = True,
    ) -> FlextResult[dm.TypingsReport]:
        limits = self.load_dependency_limits(limits_path)
        exclude_set: set[str] = set()

        typing_libraries = limits.get("typing_libraries")
        if isinstance(typing_libraries, Mapping):
            excluded = typing_libraries.get("exclude")
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
            hinted, missing_modules = hints_result.value

        required_set: set[str] = set(hinted)
        for module_name in missing_modules:
            package = self.module_to_types_package(module_name, limits)
            if package:
                required_set.add(package)
        required_set -= exclude_set

        current = self.get_current_typings_from_pyproject(project_path)
        current_set = set(current)
        python_cfg = limits.get("python")
        python_version = (
            str(python_cfg.get("version"))
            if isinstance(python_cfg, Mapping) and python_cfg.get("version") is not None
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


_service = DependencyDetectionService()


def discover_projects(
    workspace_root: Path,
    projects_filter: list[str] | None = None,
) -> FlextResult[list[Path]]:
    return _service.discover_projects(workspace_root, projects_filter=projects_filter)


def run_deptry(
    project_path: Path,
    venv_bin: Path,
    *,
    config_path: Path | None = None,
    json_output_path: Path | None = None,
    extend_exclude: list[str] | None = None,
) -> FlextResult[tuple[list[IssueMap], int]]:
    return _service.run_deptry(
        project_path,
        venv_bin,
        config_path=config_path,
        json_output_path=json_output_path,
        extend_exclude=extend_exclude,
    )


def run_pip_check(
    workspace_root: Path,
    venv_bin: Path,
) -> FlextResult[tuple[list[str], int]]:
    return _service.run_pip_check(workspace_root, venv_bin)


def classify_issues(issues: list[IssueMap]) -> dm.DeptryIssueGroups:
    return _service.classify_issues(issues)


def build_project_report(
    project_name: str,
    deptry_issues: list[IssueMap],
) -> dm.ProjectDependencyReport:
    return _service.build_project_report(project_name, deptry_issues)


def load_dependency_limits(
    limits_path: Path | None = None,
) -> dict[str, t.GeneralValueType]:
    return _service.load_dependency_limits(limits_path)


def run_mypy_stub_hints(
    project_path: Path,
    venv_bin: Path,
    *,
    timeout: int = 300,
) -> FlextResult[tuple[list[str], list[str]]]:
    return _service.run_mypy_stub_hints(project_path, venv_bin, timeout=timeout)


def module_to_types_package(
    module_name: str,
    limits: dict[str, t.GeneralValueType],
) -> str | None:
    return _service.module_to_types_package(module_name, limits)


def get_current_typings_from_pyproject(project_path: Path) -> list[str]:
    return _service.get_current_typings_from_pyproject(project_path)


def get_required_typings(
    project_path: Path,
    venv_bin: Path,
    limits_path: Path | None = None,
    *,
    include_mypy: bool = True,
) -> FlextResult[dm.TypingsReport]:
    return _service.get_required_typings(
        project_path,
        venv_bin,
        limits_path=limits_path,
        include_mypy=include_mypy,
    )


__all__ = [
    "DependencyDetectionModels",
    "DependencyDetectionService",
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
