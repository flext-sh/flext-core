"""Runtime vs dev dependency detector CLI with deptry, pip-check, and typing analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import tomlkit
from pydantic import Field, ValidationError
from tomlkit.items import Table

from flext_core import FlextLogger, r
from flext_infra import (
    FlextInfraCommandRunner,
    FlextInfraJsonService,
    FlextInfraPathResolver,
    FlextInfraReportingService,
    c,
    m,
    p,
    t,
)
from flext_infra.deps.detection import FlextInfraDependencyDetectionService
from flext_infra.toml_io import (
    CONTAINER_LIST_ADAPTER,
    array,
    as_string_list,
    dedupe_specs,
    discover_first_party_namespaces,
    ensure_pyright_execution_envs,
    ensure_table,
    find_ruff_shared_path,
    project_dev_groups,
    toml_get,
    unwrap_item,
)

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
        projects: dict[str, dict[str, t.ContainerValue]] = Field(default_factory=dict)
        pip_check: FlextInfraDependencyDetectorModels.PipCheckReport | None = None
        dependency_limits: (
            FlextInfraDependencyDetectorModels.DependencyLimitsInfo | None
        ) = None


ddm = FlextInfraDependencyDetectorModels


class ConsolidateGroupsPhase:
    """Consolidate optional-dependencies and Poetry groups into single dev group."""

    def apply(self, doc: tomlkit.TOMLDocument, canonical_dev: list[str]) -> list[str]:
        changes: list[str] = []
        project: object | None = None
        if c.Infra.Toml.PROJECT in doc:
            project = doc[c.Infra.Toml.PROJECT]
        if not isinstance(project, Table):
            project = tomlkit.table()
            doc[c.Infra.Toml.PROJECT] = project

        optional: object | None = None
        if c.Infra.Toml.OPTIONAL_DEPENDENCIES in project:
            optional = project[c.Infra.Toml.OPTIONAL_DEPENDENCIES]
        if not isinstance(optional, Table):
            optional = tomlkit.table()
            project[c.Infra.Toml.OPTIONAL_DEPENDENCIES] = optional
        existing = project_dev_groups(doc)
        merged_dev = dedupe_specs([
            *canonical_dev,
            *existing.get(c.Infra.Toml.DEV, []),
            *existing.get(c.Infra.Directories.DOCS, []),
            *existing.get(c.Infra.Gates.SECURITY, []),
            *existing.get(c.Infra.Toml.TEST, []),
            *existing.get(c.Infra.Directories.TYPINGS, []),
        ])
        current_dev = as_string_list(toml_get(optional, c.Infra.Toml.DEV))
        if current_dev != merged_dev:
            optional[c.Infra.Toml.DEV] = array(merged_dev)
            changes.append("project.optional-dependencies.dev consolidated")
        for old_key in (
            c.Infra.Toml.DOCS,
            c.Infra.Toml.SECURITY,
            c.Infra.Toml.TEST,
            c.Infra.Directories.TYPINGS,
        ):
            if old_key in optional:
                del optional[old_key]
                changes.append(f"project.optional-dependencies.{old_key} removed")
        tool: object | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        poetry = ensure_table(tool, c.Infra.Toml.POETRY)
        poetry_group_raw: object | None = None
        if c.Infra.Toml.GROUP in poetry:
            poetry_group_raw = poetry[c.Infra.Toml.GROUP]
        poetry_group = poetry_group_raw if isinstance(poetry_group_raw, Table) else None
        poetry_dev_table: Table | None = None
        for old_group in (
            c.Infra.Toml.DOCS,
            c.Infra.Toml.SECURITY,
            c.Infra.Toml.TEST,
            c.Infra.Directories.TYPINGS,
        ):
            if poetry_group is None:
                continue
            old_group_table: object | None = None
            if old_group in poetry_group:
                old_group_table = poetry_group[old_group]
            if not isinstance(old_group_table, Table):
                continue
            old_deps: object | None = None
            if c.Infra.Toml.DEPENDENCIES in old_group_table:
                old_deps = old_group_table[c.Infra.Toml.DEPENDENCIES]
            if isinstance(old_deps, Table):
                if poetry_dev_table is None:
                    poetry_dev_table = ensure_table(
                        ensure_table(poetry_group, c.Infra.Toml.DEV),
                        c.Infra.Toml.DEPENDENCIES,
                    )
                for dep_name_raw in old_deps:
                    dep_name = dep_name_raw
                    dep_value = old_deps[dep_name_raw]
                    if dep_name not in poetry_dev_table:
                        poetry_dev_table[dep_name] = dep_value
            del poetry_group[old_group]
            changes.append(f"tool.poetry.group.{old_group} removed")
        deptry = ensure_table(tool, c.Infra.Toml.DEPTRY)
        current_groups = as_string_list(
            toml_get(deptry, "pep621_dev_dependency_groups"),
        )
        if current_groups != [c.Infra.Toml.DEV]:
            deptry["pep621_dev_dependency_groups"] = array([c.Infra.Toml.DEV])
            changes.append("tool.deptry.pep621_dev_dependency_groups set to ['dev']")
        return changes


class EnsurePytestConfigPhase:
    """Ensure standard pytest configuration without removing project-specific entries."""

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        changes: list[str] = []
        tool = toml_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        pytest_tbl = ensure_table(tool, c.Infra.Toml.PYTEST)
        ini = ensure_table(pytest_tbl, c.Infra.Toml.INI_OPTIONS)
        if unwrap_item(toml_get(ini, c.Infra.Toml.MINVERSION)) != "8.0":
            ini[c.Infra.Toml.MINVERSION] = "8.0"
            changes.append("tool.pytest.ini_options.minversion set to 8.0")
        current_classes = as_string_list(toml_get(ini, c.Infra.Toml.PYTHON_CLASSES))
        if "Test*" not in current_classes:
            ini[c.Infra.Toml.PYTHON_CLASSES] = array(
                sorted({*current_classes, "Test*"}),
            )
            changes.append("tool.pytest.ini_options.python_classes updated")
        standard_files = {"*_test.py", "*_tests.py", "test_*.py"}
        current_files = set(as_string_list(toml_get(ini, c.Infra.Toml.PYTHON_FILES)))
        if not standard_files.issubset(current_files):
            ini[c.Infra.Toml.PYTHON_FILES] = array(
                sorted(current_files | standard_files),
            )
            changes.append("tool.pytest.ini_options.python_files updated")
        current_addopts = set(as_string_list(toml_get(ini, c.Infra.Toml.ADDOPTS)))
        needed_addopts = set(c.Infra.Deps.PYTEST_STANDARD_ADDOPTS)
        if not needed_addopts.issubset(current_addopts):
            ini[c.Infra.Toml.ADDOPTS] = array(sorted(current_addopts | needed_addopts))
            changes.append("tool.pytest.ini_options.addopts updated")
        current_markers = as_string_list(toml_get(ini, c.Infra.Toml.MARKERS))
        current_names = {m.split(":")[0].strip() for m in current_markers}
        added: list[str] = []
        for marker in c.Infra.Deps.PYTEST_STANDARD_MARKERS:
            name = marker.split(":")[0].strip()
            if name not in current_names:
                added.append(marker)
        if added:
            ini[c.Infra.Toml.MARKERS] = array([*current_markers, *added])
            names = ", ".join(m.split(":")[0].strip() for m in added)
            changes.append(f"tool.pytest.ini_options.markers: added {names}")
        return changes


class EnsureMypyConfigPhase:
    """Ensure standard mypy configuration with pydantic plugin across all projects."""

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        changes: list[str] = []
        tool = toml_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        mypy = ensure_table(tool, c.Infra.Toml.MYPY)
        if (
            unwrap_item(toml_get(mypy, c.Infra.Toml.PYTHON_VERSION_UNDERSCORE))
            != "3.13"
        ):
            mypy[c.Infra.Toml.PYTHON_VERSION_UNDERSCORE] = "3.13"
            changes.append("tool.mypy.python_version set to 3.13")
        current_plugins = as_string_list(toml_get(mypy, c.Infra.Toml.PLUGINS))
        needed_plugins = [
            p for p in c.Infra.Deps.MYPY_PLUGINS if p not in current_plugins
        ]
        if needed_plugins:
            mypy[c.Infra.Toml.PLUGINS] = array(
                sorted(set(current_plugins) | set(c.Infra.Deps.MYPY_PLUGINS)),
            )
            changes.append(f"tool.mypy.plugins added {', '.join(needed_plugins)}")
        current_disabled = as_string_list(
            toml_get(mypy, c.Infra.Toml.DISABLE_ERROR_CODE),
        )
        needed_disabled = [
            ec
            for ec in c.Infra.Deps.MYPY_DISABLED_ERROR_CODES
            if ec not in current_disabled
        ]
        if needed_disabled:
            mypy[c.Infra.Toml.DISABLE_ERROR_CODE] = array(
                sorted(
                    set(current_disabled) | set(c.Infra.Deps.MYPY_DISABLED_ERROR_CODES),
                ),
            )
            changes.append(
                f"tool.mypy.disable_error_code added {', '.join(needed_disabled)}",
            )
        for key, value in c.Infra.Deps.MYPY_BOOLEAN_SETTINGS:
            if unwrap_item(toml_get(mypy, key)) is not value:
                mypy[key] = value
                changes.append(f"tool.mypy.{key} set to {value}")
        return changes


class EnsurePydanticMypyConfigPhase:
    """Ensure standard pydantic-mypy configuration for strict model typing."""

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        changes: list[str] = []
        tool = toml_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        pydantic_mypy = ensure_table(tool, "pydantic-mypy")
        for key, value in c.Infra.Deps.PYDANTIC_MYPY_SETTINGS:
            if unwrap_item(toml_get(pydantic_mypy, key)) is not value:
                pydantic_mypy[key] = value
                changes.append(f"tool.pydantic-mypy.{key} set to {value}")
        return changes


class EnsurePyrightConfigPhase:
    """Ensure standard Pyright configuration for strict type checking."""

    def apply(self, doc: tomlkit.TOMLDocument, *, is_root: bool) -> list[str]:
        changes: list[str] = []
        tool = toml_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        pyright = ensure_table(tool, c.Infra.Toml.PYRIGHT)
        expected_envs: list[dict[str, str]] = [
            {"root": c.Infra.Paths.DEFAULT_SRC_DIR, "reportPrivateUsage": "error"},
            {"root": c.Infra.Directories.TESTS, "reportPrivateUsage": "none"},
        ]
        if is_root:
            if (
                unwrap_item(toml_get(pyright, "typeCheckingMode"))
                != c.Infra.Modes.STRICT
            ):
                pyright["typeCheckingMode"] = c.Infra.Modes.STRICT
                changes.append("tool.pyright.typeCheckingMode set to strict")
            ensure_pyright_execution_envs(pyright, expected_envs, changes)
            return changes
        for key, value in c.Infra.Deps.PYRIGHT_STRICT_SETTINGS:
            if unwrap_item(toml_get(pyright, key)) != value:
                pyright[key] = value
                changes.append(f"tool.pyright.{key} set to {value}")
        ensure_pyright_execution_envs(pyright, expected_envs, changes)
        return changes


class EnsureRuffConfigPhase:
    """Ensure standard Ruff configuration with extend and known-first-party."""

    def apply(
        self,
        doc: tomlkit.TOMLDocument,
        *,
        path: Path,
        workspace_root: Path,
    ) -> list[str]:
        changes: list[str] = []
        tool = toml_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        ruff = ensure_table(tool, c.Infra.Toml.RUFF)
        _target_shared, expected_extend = find_ruff_shared_path(
            path.parent,
            workspace_root,
        )
        if unwrap_item(toml_get(ruff, c.Infra.Toml.EXTEND)) != expected_extend:
            ruff[c.Infra.Toml.EXTEND] = expected_extend
            changes.append(f"tool.ruff.extend set to {expected_extend}")
        detected_packages = discover_first_party_namespaces(path.parent)
        if detected_packages:
            lint = ensure_table(ruff, c.Infra.Toml.LINT_SECTION)
            isort = ensure_table(lint, c.Infra.Toml.ISORT)
            current_kfp = as_string_list(
                toml_get(isort, c.Infra.Toml.KNOWN_FIRST_PARTY_HYPHEN),
            )
            if current_kfp != detected_packages:
                isort[c.Infra.Toml.KNOWN_FIRST_PARTY_HYPHEN] = array(detected_packages)
                changes.append(
                    f"tool.ruff.lint.isort.known-first-party set to {detected_packages}",
                )
        return changes


class EnsurePyreflyConfigPhase:
    """Ensure standard Pyrefly configuration for max-strict typing."""

    def apply(self, doc: tomlkit.TOMLDocument, *, is_root: bool) -> list[str]:
        changes: list[str] = []
        tool = toml_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        pyrefly = ensure_table(tool, c.Infra.Toml.PYREFLY)
        if unwrap_item(toml_get(pyrefly, c.Infra.Toml.PYTHON_VERSION_HYPHEN)) != "3.13":
            pyrefly[c.Infra.Toml.PYTHON_VERSION_HYPHEN] = "3.13"
            changes.append("tool.pyrefly.python-version set to 3.13")
        if (
            unwrap_item(toml_get(pyrefly, c.Infra.Toml.IGNORE_ERRORS_IN_GENERATED))
            is not True
        ):
            pyrefly[c.Infra.Toml.IGNORE_ERRORS_IN_GENERATED] = True
            changes.append("tool.pyrefly.ignore-errors-in-generated-code enabled")
        expected_search = ["."]
        current_search = as_string_list(toml_get(pyrefly, c.Infra.Toml.SEARCH_PATH))
        if current_search != expected_search:
            pyrefly[c.Infra.Toml.SEARCH_PATH] = array(expected_search)
            changes.append(f"tool.pyrefly.search-path set to {expected_search}")
        errors = ensure_table(pyrefly, "errors")
        for error_rule in c.Infra.Deps.PYREFLY_STRICT_ERRORS:
            if unwrap_item(toml_get(errors, error_rule)) is not True:
                errors[error_rule] = True
                changes.append(f"tool.pyrefly.errors.{error_rule} enabled")
        for error_rule in c.Infra.Deps.PYREFLY_DISABLED_ERRORS:
            if unwrap_item(toml_get(errors, error_rule)) is not False:
                errors[error_rule] = False
                changes.append(f"tool.pyrefly.errors.{error_rule} disabled")
        current_excludes = as_string_list(
            toml_get(pyrefly, c.Infra.Toml.PROJECT_EXCLUDES),
        )
        pb2_globs = ["**/*_pb2*.py", "**/*_pb2_grpc*.py"]
        needed = set(pb2_globs) - set(current_excludes)
        if needed and (
            is_root or any(glob in current_excludes for glob in pb2_globs) or True
        ):
            pyrefly[c.Infra.Toml.PROJECT_EXCLUDES] = array(
                sorted(set(current_excludes) | set(pb2_globs)),
            )
            changes.append(f"tool.pyrefly.project-excludes added {', '.join(needed)}")
        return changes


class EnsureNamespaceToolingPhase:
    """Ensure namespace discovery is reflected across project tooling tables."""

    def apply(self, doc: tomlkit.TOMLDocument, *, path: Path) -> list[str]:
        changes: list[str] = []
        detected = discover_first_party_namespaces(path.parent)
        if not detected:
            return changes
        tool = toml_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        deptry = ensure_table(tool, c.Infra.Toml.DEPTRY)
        current_deptry = as_string_list(
            toml_get(deptry, c.Infra.Toml.KNOWN_FIRST_PARTY_UNDERSCORE),
        )
        if current_deptry != detected:
            deptry[c.Infra.Toml.KNOWN_FIRST_PARTY_UNDERSCORE] = array(detected)
            changes.append(f"tool.deptry.known_first_party set to {detected}")
        pyright = ensure_table(tool, c.Infra.Toml.PYRIGHT)
        extra_paths = as_string_list(toml_get(pyright, "extraPaths"))
        if c.Infra.Paths.DEFAULT_SRC_DIR not in extra_paths:
            pyright["extraPaths"] = array(
                sorted({*extra_paths, c.Infra.Paths.DEFAULT_SRC_DIR}),
            )
            changes.append("tool.pyright.extraPaths includes src")
        return changes


class EnsureFormattingToolingPhase:
    """Ensure safe default config for TOML/YAML formatting tools."""

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        changes: list[str] = []
        tool = toml_get(doc, c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        tomlsort = ensure_table(tool, "tomlsort")
        for key, value in c.Infra.Deps.TOMLSORT_DEFAULTS:
            current = unwrap_item(toml_get(tomlsort, key))
            if current != value:
                if isinstance(value, list):
                    try:
                        validated_items = CONTAINER_LIST_ADAPTER.validate_python(value)
                        tomlsort[key] = array([str(item) for item in validated_items])
                    except ValidationError:
                        tomlsort[key] = array([])
                else:
                    tomlsort[key] = value
                changes.append(f"tool.tomlsort.{key} set")
        yamlfix = ensure_table(tool, "yamlfix")
        for key, value in c.Infra.Deps.YAMLFIX_DEFAULTS:
            if unwrap_item(toml_get(yamlfix, key)) != value:
                yamlfix[key] = value
                changes.append(f"tool.yamlfix.{key} set to {value}")
        return changes


class InjectCommentsPhase:
    """Inject managed/custom/auto markers into pyproject.toml."""

    def apply(self, rendered: str) -> tuple[str, list[str]]:
        changes: list[str] = []
        lines = rendered.splitlines()
        existing_text = rendered
        out: list[str] = []
        has_banner = bool(
            lines and "[MANAGED] FLEXT pyproject standardization" in lines[0],
        )
        if not has_banner:
            out.extend(c.Infra.Deps.BANNER.splitlines())
            changes.append("managed banner injected")
        marker_map = dict(c.Infra.Deps.COMMENT_MARKERS)
        skip_broken_group_section = False
        for line in lines:
            if line.strip() == "[group.dev.dependencies]":
                skip_broken_group_section = True
                changes.append("broken [group.dev.dependencies] section removed")
                continue
            if skip_broken_group_section and (not line.strip()):
                continue
            if skip_broken_group_section and line.strip():
                skip_broken_group_section = False
            marker = marker_map.get(line.strip())
            if marker:
                recent = (
                    out[-c.Infra.Deps.RECENT_LINES_FOR_MARKER :]
                    if len(out) >= c.Infra.Deps.RECENT_LINES_FOR_MARKER
                    else out
                )
                if marker not in recent and marker not in existing_text:
                    out.append(marker)
                    changes.append(f"marker injected for {line.strip()}")
            if line.strip().startswith("optional-dependencies.dev"):
                recent = (
                    out[-c.Infra.Deps.RECENT_LINES_FOR_DEV_DEP :]
                    if len(out) >= c.Infra.Deps.RECENT_LINES_FOR_DEV_DEP
                    else out
                )
                marker = "# [MANAGED] consolidated development dependencies"
                auto = "# [AUTO] merged from dev/docs/security/test/typings"
                if marker not in recent and marker not in existing_text:
                    out.append(marker)
                    changes.append("marker injected for optional-dependencies.dev")
                if auto not in recent and auto not in existing_text:
                    out.append(auto)
                    changes.append("auto marker injected for optional-dependencies.dev")
            out.append(line)
        return ("\n".join(out).rstrip() + "\n", changes)


class FlextInfraRuntimeDevDependencyDetector:
    """CLI tool for detecting runtime vs dev dependencies across workspace."""

    def __init__(self) -> None:
        """Initialize the detector with path resolver, reporting, JSON, deps, and runner services."""
        super().__init__()
        self._paths = FlextInfraPathResolver()
        self._reporting = FlextInfraReportingService()
        self._json = FlextInfraJsonService()
        self._deps = FlextInfraDependencyDetectionService()
        self._runner: p.Infra.CommandRunner = FlextInfraCommandRunner()

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
            help="Write report to this path (default: .reports/dependencies/detect-runtime-dev-latest.json).",
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

    def run(self, argv: list[str] | None = None) -> r[int]:
        """Execute dependency detection and generate workspace report."""
        root_result = self._paths.workspace_root_from_file(__file__)
        if root_result.is_failure:
            return r[int].fail(root_result.error or "workspace root resolution failed")
        root: Path = root_result.value
        venv_bin = root / c.Infra.Paths.VENV_BIN_REL
        limits_default = Path(__file__).resolve().parent / "dependency_limits.toml"
        parser = self._parser(limits_default)
        args = parser.parse_args(argv)
        projects_result = self._deps.discover_projects(
            root,
            projects_filter=self._project_filter(args),
        )
        if projects_result.is_failure:
            return r[int].fail(projects_result.error or "project discovery failed")
        projects: list[Path] = projects_result.value
        if not projects:
            logger.error("deps_no_projects_found")
            return r[int].ok(2)
        if not (venv_bin / c.Infra.Toml.DEPTRY).exists():
            logger.error(
                "deps_deptry_missing",
                path=str(venv_bin / c.Infra.Toml.DEPTRY),
            )
            return r[int].ok(3)
        apply_typings = bool(args.apply_typings)
        do_typings = bool(args.typings) or apply_typings
        limits_path = Path(args.limits) if args.limits else limits_default
        projects_report: dict[str, dict[str, t.ContainerValue]] = {}
        report_model = ddm.WorkspaceDependencyReport(
            workspace=str(root),
            projects=projects_report,
            pip_check=None,
            dependency_limits=None,
        )
        if do_typings:
            limits_data = self._deps.load_dependency_limits(limits_path)
            if limits_data:
                python_cfg = limits_data.get(c.Infra.Toml.PYTHON)
                python_version = (
                    str(python_cfg.get(c.Infra.Toml.VERSION))
                    if isinstance(python_cfg, dict)
                    and python_cfg.get(c.Infra.Toml.VERSION) is not None
                    else None
                )
                report_model.dependency_limits = ddm.DependencyLimitsInfo(
                    python_version=python_version,
                    limits_path=str(limits_path),
                )
        for project_path in projects:
            project_name = project_path.name
            if not args.quiet:
                logger.info("deps_deptry_running", project=project_name)
            deptry_result = self._deps.run_deptry(project_path, venv_bin)
            if deptry_result.is_failure:
                return r[int].fail(deptry_result.error or "deptry run failed")
            deptry_value: tuple[list[t.Infra.IssueMap], int] = deptry_result.value
            issues, _ = deptry_value
            project_payload = self._deps.build_project_report(project_name, issues)
            project_dict = project_payload.model_dump()
            projects_report[project_name] = project_dict
            if do_typings and (project_path / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir():
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
                typings_report = typings_result.value
                typing_dict = typings_report.model_dump()
                projects_report[project_name][c.Infra.Directories.TYPINGS] = typing_dict
                to_add: list[str] = typings_report.to_add
                if apply_typings and to_add and (not args.dry_run):
                    env = {
                        **os.environ,
                        "VIRTUAL_ENV": str(venv_bin.parent),
                        "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
                    }
                    for package in to_add:
                        run = self._runner.run_raw(
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
                            logger.warning(
                                "deps_typings_add_failed",
                                project=project_name,
                                package=package,
                            )
                        else:
                            run_output: m.Infra.Core.CommandOutput = run.value
                            if run_output.exit_code != 0:
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
            pip_value: tuple[list[str], int] = pip_result.value
            pip_lines, pip_exit = pip_value
            report_model.pip_check = ddm.PipCheckReport(
                ok=pip_exit == 0,
                lines=pip_lines,
            )
        report_payload = report_model.model_dump()
        if args.json_stdout:
            return r[int].ok(0)
        out_path: Path | None = None
        if args.output:
            out_path = Path(args.output)
        elif not args.dry_run:
            report_dir = self._reporting.get_report_dir(
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
            write_result = self._json.write(out_path, report_payload)
            if write_result.is_failure:
                return r[int].fail(write_result.error or "failed to write report")
            if not args.quiet:
                logger.info("deps_report_written", path=str(out_path))
        total_issues = 0
        for payload in projects_report.values():
            deptry_obj = payload.get(c.Infra.Toml.DEPTRY)
            if isinstance(deptry_obj, dict):
                raw_count = deptry_obj.get("raw_count", 0)
                if isinstance(raw_count, int):
                    total_issues += raw_count
        pip_ok = (
            report_model.pip_check.ok if report_model.pip_check is not None else True
        )
        if not args.quiet:
            logger.info(
                "deps_summary",
                projects=len(projects),
                deptry_issues=total_issues,
                pip_check=c.Infra.ReportKeys.OK if pip_ok else "FAIL",
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
