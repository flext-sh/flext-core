"""Modernize workspace pyproject.toml files to standardized format."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path

import tomlkit
from pydantic import TypeAdapter, ValidationError
from tomlkit.items import Array, Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_core import FlextLogger, t
from flext_infra import FlextInfraCommandRunner, c, p

_logger = FlextLogger(__name__)

_CONTAINER_DICT_ADAPTER = TypeAdapter(dict[str, t.ContainerValue])
_CONTAINER_LIST_ADAPTER = TypeAdapter(list[t.ContainerValue])


def _normalize_container_value(
    value: t.ContainerValue
    | Item
    | TOMLDocument
    | Mapping[str, t.ContainerValue]
    | None,
) -> t.ContainerValue | None:
    """Normalize TOML items/documents to a concrete container value."""
    normalized: t.ContainerValue | Item | Mapping[str, t.ContainerValue] | None = value
    if isinstance(value, (TOMLDocument, Item)):
        normalized = value.unwrap()
    if isinstance(normalized, Item):
        return None
    return normalized


def _as_container_dict(
    value: t.ContainerValue
    | Item
    | TOMLDocument
    | Mapping[str, t.ContainerValue]
    | None,
) -> dict[str, t.ContainerValue]:
    """Validate and normalize mapping-like values to typed container dict."""
    normalized = _normalize_container_value(value)
    if normalized is None:
        return {}
    try:
        return _CONTAINER_DICT_ADAPTER.validate_python(normalized)
    except ValidationError:
        return {}


def _as_container_list(value: t.ContainerValue | Item | None) -> list[t.ContainerValue]:
    """Validate and normalize list-like values to typed container list."""
    normalized = _normalize_container_value(value)
    if normalized is None:
        return []
    try:
        return _CONTAINER_LIST_ADAPTER.validate_python(normalized)
    except ValidationError:
        return []


def _workspace_root(start: Path) -> Path:
    """Detect workspace root by searching for .gitmodules or .git with pyproject.toml."""
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / c.Infra.Files.GITMODULES).exists() and (
            parent / c.Infra.Files.PYPROJECT_FILENAME
        ).exists():
            return parent
    for parent in (current, *current.parents):
        if (parent / c.Infra.Git.DIR).exists() and (
            parent / c.Infra.Files.PYPROJECT_FILENAME
        ).exists():
            return parent
    return start.resolve().parents[4]


ROOT = _workspace_root(Path(__file__))


def _find_ruff_shared_path(project_dir: Path, workspace_root: Path) -> tuple[Path, str]:
    """Return target ruff-shared file path and relative extend value."""
    workspace_candidate = workspace_root / "ruff-shared.toml"
    relative = os.path.relpath(workspace_candidate, start=project_dir)
    return workspace_candidate, Path(relative).as_posix()


def _ensure_ruff_shared_template(
    project_dir: Path, workspace_root: Path
) -> tuple[Path, bool]:
    """Create managed ruff-shared.toml in workspace root when missing."""
    target, _ = _find_ruff_shared_path(project_dir, workspace_root)
    if target.exists():
        return target, False

    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(
        c.Infra.Deps.RUFF_SHARED_TEMPLATE.rstrip() + "\n",
        encoding=c.Infra.Encoding.DEFAULT,
    )
    return target, True


def _dep_name(spec: str) -> str:
    """Extract normalized dependency name from requirement specification."""
    base = spec.strip().split("@", 1)[0].strip()
    match = c.Infra.Deps.DEP_NAME_RE.match(base)
    if match:
        return match.group(1).lower().replace("_", "-")
    return base.lower().replace("_", "-")


def _dedupe_specs(specs: Iterable[str]) -> list[str]:
    """Deduplicate dependency specifications by normalized name."""
    seen: MutableMapping[str, str] = {}
    for spec in specs:
        key = _dep_name(spec)
        if key and key not in seen:
            seen[key] = spec
    return [seen[k] for k in sorted(seen)]


def _unwrap_item(value: t.ContainerValue | Item | None) -> t.ContainerValue | None:
    """Unwrap a tomlkit Item to get the underlying value."""
    return _normalize_container_value(value)


def _as_string_list(value: t.ContainerValue | Item | None) -> list[str]:
    """Convert TOML value to list of strings."""
    normalized = _normalize_container_value(value)
    if normalized is None or isinstance(normalized, str):
        return []
    if isinstance(normalized, Sequence) and not isinstance(normalized, Mapping):
        return [str(raw) for raw in normalized]
    return [str(raw) for raw in _as_container_list(normalized)]


def _array(items: list[str]) -> Array:
    """Create multiline TOML array from string items."""
    arr = tomlkit.array()
    for item in items:
        arr.append(item)
    return arr.multiline(True)


def _ensure_table(parent: Table, key: str) -> Table:
    """Get or create a TOML table in parent."""
    existing = parent.get(key)
    if existing is not None and isinstance(existing, Table):
        out: Table = existing
        return out
    table = tomlkit.table()
    parent[key] = table
    return table


def _read_doc(path: Path) -> tomlkit.TOMLDocument | None:
    """Read and parse TOML document from file."""
    if not path.exists():
        return None
    try:
        return tomlkit.parse(path.read_text(encoding=c.Infra.Encoding.DEFAULT))
    except (OSError, ValueError) as exc:
        _logger.warning(
            "Failed to read or parse TOML document",
            path=str(path),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None


def _discover_first_party_namespaces(project_dir: Path) -> list[str]:
    """Discover first-party namespace packages from src/ for tool configuration."""
    src_dir = project_dir / c.Infra.Paths.DEFAULT_SRC_DIR
    if not src_dir.is_dir():
        return []

    namespaces: list[str] = []
    for entry in sorted(src_dir.iterdir()):
        if not entry.is_dir() or entry.name == "__pycache__":
            continue
        if not entry.name.isidentifier() or "-" in entry.name:
            continue
        namespaces.append(entry.name)
    return namespaces


def _project_dev_groups(doc: tomlkit.TOMLDocument) -> Mapping[str, list[str]]:
    """Extract optional-dependencies groups from project table."""
    project = doc.get(c.Infra.Toml.PROJECT)
    if project is None or not isinstance(project, Table):
        return {}
    optional = project.get(c.Infra.Toml.OPTIONAL_DEPENDENCIES)
    if optional is None or not isinstance(optional, Table):
        return {}
    return {
        c.Infra.Toml.DEV: _as_string_list(optional.get(c.Infra.Toml.DEV)),
        c.Infra.Directories.DOCS: _as_string_list(optional.get(c.Infra.Toml.DOCS)),
        c.Infra.Gates.SECURITY: _as_string_list(optional.get(c.Infra.Toml.SECURITY)),
        c.Infra.Toml.TEST: _as_string_list(optional.get(c.Infra.Toml.TEST)),
        c.Infra.Directories.TYPINGS: _as_string_list(
            optional.get(c.Infra.Directories.TYPINGS)
        ),
    }


def _canonical_dev_dependencies(root_doc: tomlkit.TOMLDocument) -> list[str]:
    """Merge all dev dependency groups from root pyproject."""
    groups = _project_dev_groups(root_doc)
    merged = [
        *groups.get(c.Infra.Toml.DEV, []),
        *groups.get(c.Infra.Directories.DOCS, []),
        *groups.get(c.Infra.Gates.SECURITY, []),
        *groups.get(c.Infra.Toml.TEST, []),
        *groups.get(c.Infra.Directories.TYPINGS, []),
    ]
    return _dedupe_specs(merged)


class ConsolidateGroupsPhase:
    """Consolidate optional-dependencies and Poetry groups into single dev group."""

    def apply(
        self,
        doc: tomlkit.TOMLDocument,
        canonical_dev: list[str],
    ) -> list[str]:
        """Apply consolidation phase to pyproject document."""
        changes: list[str] = []

        project = doc.get(c.Infra.Toml.PROJECT)
        if not isinstance(project, Table):
            project = tomlkit.table()
            doc[c.Infra.Toml.PROJECT] = project

        optional = project.get(c.Infra.Toml.OPTIONAL_DEPENDENCIES)
        if not isinstance(optional, Table):
            optional = tomlkit.table()
            project[c.Infra.Toml.OPTIONAL_DEPENDENCIES] = optional

        existing = _project_dev_groups(doc)
        merged_dev = _dedupe_specs([
            *canonical_dev,
            *existing.get(c.Infra.Toml.DEV, []),
            *existing.get(c.Infra.Directories.DOCS, []),
            *existing.get(c.Infra.Gates.SECURITY, []),
            *existing.get(c.Infra.Toml.TEST, []),
            *existing.get(c.Infra.Directories.TYPINGS, []),
        ])
        current_dev = _as_string_list(optional.get(c.Infra.Toml.DEV))
        if current_dev != merged_dev:
            optional[c.Infra.Toml.DEV] = _array(merged_dev)
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

        tool = doc.get(c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        poetry = _ensure_table(tool, c.Infra.Toml.POETRY)
        poetry_group = poetry.get(c.Infra.Toml.GROUP)
        if not isinstance(poetry_group, Table):
            poetry_group = None

        poetry_dev_table: Table | None = None
        for old_group in (
            c.Infra.Toml.DOCS,
            c.Infra.Toml.SECURITY,
            c.Infra.Toml.TEST,
            c.Infra.Directories.TYPINGS,
        ):
            if poetry_group is None:
                continue
            old_group_table = poetry_group.get(old_group)
            if not isinstance(old_group_table, Table):
                continue
            old_deps = old_group_table.get(c.Infra.Toml.DEPENDENCIES)
            if isinstance(old_deps, Table):
                if poetry_dev_table is None:
                    poetry_dev_table = _ensure_table(
                        _ensure_table(poetry_group, c.Infra.Toml.DEV),
                        c.Infra.Toml.DEPENDENCIES,
                    )
                for dep_name, dep_value in old_deps.items():
                    if dep_name not in poetry_dev_table:
                        poetry_dev_table[dep_name] = dep_value
            del poetry_group[old_group]
            changes.append(f"tool.poetry.group.{old_group} removed")

        deptry = _ensure_table(tool, c.Infra.Toml.DEPTRY)
        current_groups = _as_string_list(deptry.get("pep621_dev_dependency_groups"))
        if current_groups != [c.Infra.Toml.DEV]:
            deptry["pep621_dev_dependency_groups"] = _array([c.Infra.Toml.DEV])
            changes.append("tool.deptry.pep621_dev_dependency_groups set to ['dev']")

        return changes


class EnsurePytestConfigPhase:
    """Ensure standard pytest configuration without removing project-specific entries."""

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        """Merge standard pytest config into existing, preserving project-specific entries."""
        changes: list[str] = []

        tool = doc.get(c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        pytest_tbl = _ensure_table(tool, c.Infra.Toml.PYTEST)
        ini = _ensure_table(pytest_tbl, c.Infra.Toml.INI_OPTIONS)

        if _unwrap_item(ini.get(c.Infra.Toml.MINVERSION)) != "8.0":
            ini[c.Infra.Toml.MINVERSION] = "8.0"
            changes.append("tool.pytest.ini_options.minversion set to 8.0")

        current_classes = _as_string_list(ini.get(c.Infra.Toml.PYTHON_CLASSES))
        if "Test*" not in current_classes:
            ini[c.Infra.Toml.PYTHON_CLASSES] = _array(
                sorted({*current_classes, "Test*"})
            )
            changes.append("tool.pytest.ini_options.python_classes updated")

        standard_files = {"*_test.py", "*_tests.py", "test_*.py"}
        current_files = set(_as_string_list(ini.get(c.Infra.Toml.PYTHON_FILES)))
        if not standard_files.issubset(current_files):
            ini[c.Infra.Toml.PYTHON_FILES] = _array(
                sorted(current_files | standard_files)
            )
            changes.append("tool.pytest.ini_options.python_files updated")

        current_addopts = set(_as_string_list(ini.get(c.Infra.Toml.ADDOPTS)))
        needed_addopts = set(c.Infra.Deps.PYTEST_STANDARD_ADDOPTS)
        if not needed_addopts.issubset(current_addopts):
            ini[c.Infra.Toml.ADDOPTS] = _array(sorted(current_addopts | needed_addopts))
            changes.append("tool.pytest.ini_options.addopts updated")

        current_markers = _as_string_list(ini.get(c.Infra.Toml.MARKERS))
        current_names = {m.split(":")[0].strip() for m in current_markers}
        added: list[str] = []
        for marker in c.Infra.Deps.PYTEST_STANDARD_MARKERS:
            name = marker.split(":")[0].strip()
            if name not in current_names:
                added.append(marker)
        if added:
            ini[c.Infra.Toml.MARKERS] = _array([*current_markers, *added])
            names = ", ".join(m.split(":")[0].strip() for m in added)
            changes.append(f"tool.pytest.ini_options.markers: added {names}")

        return changes


class EnsureMypyConfigPhase:
    """Ensure standard mypy configuration with pydantic plugin across all projects."""

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        """Merge standard mypy config into existing, preserving project-specific entries."""
        changes: list[str] = []

        tool = doc.get(c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        mypy = _ensure_table(tool, c.Infra.Toml.MYPY)

        # Ensure Python version
        if _unwrap_item(mypy.get(c.Infra.Toml.PYTHON_VERSION_UNDERSCORE)) != "3.13":
            mypy[c.Infra.Toml.PYTHON_VERSION_UNDERSCORE] = "3.13"
            changes.append("tool.mypy.python_version set to 3.13")

        # Ensure pydantic plugin is always active
        current_plugins = _as_string_list(mypy.get(c.Infra.Toml.PLUGINS))
        needed_plugins = [
            p for p in c.Infra.Deps.MYPY_PLUGINS if p not in current_plugins
        ]
        if needed_plugins:
            mypy[c.Infra.Toml.PLUGINS] = _array(
                sorted(set(current_plugins) | set(c.Infra.Deps.MYPY_PLUGINS))
            )
            changes.append(f"tool.mypy.plugins added {', '.join(needed_plugins)}")

        # Ensure disabled error codes
        current_disabled = _as_string_list(mypy.get(c.Infra.Toml.DISABLE_ERROR_CODE))
        needed_disabled = [
            ec
            for ec in c.Infra.Deps.MYPY_DISABLED_ERROR_CODES
            if ec not in current_disabled
        ]
        if needed_disabled:
            mypy[c.Infra.Toml.DISABLE_ERROR_CODE] = _array(
                sorted(
                    set(current_disabled) | set(c.Infra.Deps.MYPY_DISABLED_ERROR_CODES)
                ),
            )
            changes.append(
                f"tool.mypy.disable_error_code added {', '.join(needed_disabled)}"
            )

        for key, value in c.Infra.Deps.MYPY_BOOLEAN_SETTINGS:
            if _unwrap_item(mypy.get(key)) is not value:
                mypy[key] = value
                changes.append(f"tool.mypy.{key} set to {value}")

        return changes


class EnsurePydanticMypyConfigPhase:
    """Ensure standard pydantic-mypy configuration for strict model typing."""

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        """Merge standard pydantic-mypy config into existing."""
        changes: list[str] = []

        tool = doc.get(c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        pydantic_mypy = _ensure_table(tool, "pydantic-mypy")

        for key, value in c.Infra.Deps.PYDANTIC_MYPY_SETTINGS:
            if _unwrap_item(pydantic_mypy.get(key)) is not value:
                pydantic_mypy[key] = value
                changes.append(f"tool.pydantic-mypy.{key} set to {value}")

        return changes


class EnsurePyrightConfigPhase:
    """Ensure standard Pyright configuration for strict type checking."""

    def apply(self, doc: tomlkit.TOMLDocument, *, is_root: bool) -> list[str]:
        """Merge standard Pyright config into existing, preserving project-specific entries."""
        changes: list[str] = []

        tool = doc.get(c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        pyright = _ensure_table(tool, c.Infra.Toml.PYRIGHT)
        expected_envs: list[dict[str, str]] = [
            {
                "root": c.Infra.Paths.DEFAULT_SRC_DIR,
                "reportPrivateUsage": "error",
            },
            {
                "root": c.Infra.Directories.TESTS,
                "reportPrivateUsage": "none",
            },
        ]

        if is_root:
            # Root has extensive config (140+ keys); keep it intact and enforce key guardrails
            if _unwrap_item(pyright.get("typeCheckingMode")) != c.Infra.Modes.STRICT:
                pyright["typeCheckingMode"] = c.Infra.Modes.STRICT
                changes.append("tool.pyright.typeCheckingMode set to strict")
            current_envs_raw = _unwrap_item(pyright.get("executionEnvironments"))
            current_envs = (
                current_envs_raw if isinstance(current_envs_raw, list) else []
            )
            if current_envs != expected_envs:
                pyright["executionEnvironments"] = expected_envs
                changes.append(
                    "tool.pyright.executionEnvironments set with tests reportPrivateUsage=none"
                )
            return changes

        # Sub-projects: ensure minimal strict settings
        for key, value in c.Infra.Deps.PYRIGHT_STRICT_SETTINGS:
            if _unwrap_item(pyright.get(key)) != value:
                pyright[key] = value
                changes.append(f"tool.pyright.{key} set to {value}")

        current_envs_raw = _unwrap_item(pyright.get("executionEnvironments"))
        current_envs = current_envs_raw if isinstance(current_envs_raw, list) else []
        if current_envs != expected_envs:
            pyright["executionEnvironments"] = expected_envs
            changes.append(
                "tool.pyright.executionEnvironments set with tests reportPrivateUsage=none"
            )

        return changes


class EnsureRuffConfigPhase:
    """Ensure standard Ruff configuration with extend and known-first-party."""

    def apply(
        self,
        doc: tomlkit.TOMLDocument,
        *,
        path: Path,
    ) -> list[str]:
        """Merge standard Ruff config into existing, preserving project-specific entries."""
        changes: list[str] = []

        tool = doc.get(c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        ruff = _ensure_table(tool, c.Infra.Toml.RUFF)

        # Ensure extend points to managed shared config in workspace root
        _target_shared, expected_extend = _find_ruff_shared_path(path.parent, ROOT)
        if _unwrap_item(ruff.get(c.Infra.Toml.EXTEND)) != expected_extend:
            ruff[c.Infra.Toml.EXTEND] = expected_extend
            changes.append(f"tool.ruff.extend set to {expected_extend}")

        detected_packages = _discover_first_party_namespaces(path.parent)
        if detected_packages:
            # Create nested tables: ruff.lint.isort
            lint = _ensure_table(ruff, c.Infra.Toml.LINT_SECTION)
            isort = _ensure_table(lint, c.Infra.Toml.ISORT)

            current_kfp = _as_string_list(
                isort.get(c.Infra.Toml.KNOWN_FIRST_PARTY_HYPHEN)
            )
            if current_kfp != detected_packages:
                isort[c.Infra.Toml.KNOWN_FIRST_PARTY_HYPHEN] = _array(detected_packages)
                changes.append(
                    f"tool.ruff.lint.isort.known-first-party set to {detected_packages}"
                )

        return changes


class EnsurePyreflyConfigPhase:
    """Ensure standard Pyrefly configuration for max-strict typing."""

    # The 13 opt-in strict rules (from Task 1 research + root pyproject.toml cleanup)

    def apply(self, doc: tomlkit.TOMLDocument, *, is_root: bool) -> list[str]:
        """Merge standard Pyrefly config into existing, preserving project-specific entries."""
        changes: list[str] = []

        tool = doc.get(c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        pyrefly = _ensure_table(tool, c.Infra.Toml.PYREFLY)

        if _unwrap_item(pyrefly.get(c.Infra.Toml.PYTHON_VERSION_HYPHEN)) != "3.13":
            pyrefly[c.Infra.Toml.PYTHON_VERSION_HYPHEN] = "3.13"
            changes.append("tool.pyrefly.python-version set to 3.13")

        if (
            _unwrap_item(pyrefly.get(c.Infra.Toml.IGNORE_ERRORS_IN_GENERATED))
            is not True
        ):
            pyrefly[c.Infra.Toml.IGNORE_ERRORS_IN_GENERATED] = True
            changes.append("tool.pyrefly.ignore-errors-in-generated-code enabled")

        expected_search = ["."]
        current_search = _as_string_list(pyrefly.get(c.Infra.Toml.SEARCH_PATH))
        if current_search != expected_search:
            pyrefly[c.Infra.Toml.SEARCH_PATH] = _array(expected_search)
            changes.append(f"tool.pyrefly.search-path set to {expected_search}")

        errors = _ensure_table(pyrefly, "errors")
        for error_rule in c.Infra.Deps.PYREFLY_STRICT_ERRORS:
            if _unwrap_item(errors.get(error_rule)) is not True:
                errors[error_rule] = True
                changes.append(f"tool.pyrefly.errors.{error_rule} enabled")

        for error_rule in c.Infra.Deps.PYREFLY_DISABLED_ERRORS:
            if _unwrap_item(errors.get(error_rule)) is not False:
                errors[error_rule] = False
                changes.append(f"tool.pyrefly.errors.{error_rule} disabled")

        current_excludes = _as_string_list(pyrefly.get(c.Infra.Toml.PROJECT_EXCLUDES))
        pb2_globs = ["**/*_pb2*.py", "**/*_pb2_grpc*.py"]
        needed = set(pb2_globs) - set(current_excludes)
        if needed and (
            is_root or any(glob in current_excludes for glob in pb2_globs) or True
        ):
            pyrefly[c.Infra.Toml.PROJECT_EXCLUDES] = _array(
                sorted(set(current_excludes) | set(pb2_globs)),
            )
            changes.append(f"tool.pyrefly.project-excludes added {', '.join(needed)}")

        return changes


class EnsureNamespaceToolingPhase:
    """Ensure namespace discovery is reflected across project tooling tables."""

    def apply(self, doc: tomlkit.TOMLDocument, *, path: Path) -> list[str]:
        """Apply namespace tooling phase to TOML document."""
        changes: list[str] = []
        detected = _discover_first_party_namespaces(path.parent)
        if not detected:
            return changes

        tool = doc.get(c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        deptry = _ensure_table(tool, c.Infra.Toml.DEPTRY)
        current_deptry = _as_string_list(
            deptry.get(c.Infra.Toml.KNOWN_FIRST_PARTY_UNDERSCORE)
        )
        if current_deptry != detected:
            deptry[c.Infra.Toml.KNOWN_FIRST_PARTY_UNDERSCORE] = _array(detected)
            changes.append(f"tool.deptry.known_first_party set to {detected}")

        pyright = _ensure_table(tool, c.Infra.Toml.PYRIGHT)
        extra_paths = _as_string_list(pyright.get("extraPaths"))
        if c.Infra.Paths.DEFAULT_SRC_DIR not in extra_paths:
            pyright["extraPaths"] = _array(
                sorted({*extra_paths, c.Infra.Paths.DEFAULT_SRC_DIR})
            )
            changes.append("tool.pyright.extraPaths includes src")

        return changes


class EnsureFormattingToolingPhase:
    """Ensure safe default config for TOML/YAML formatting tools."""

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        """Apply formatting tooling phase to TOML document."""
        changes: list[str] = []

        tool = doc.get(c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        tomlsort = _ensure_table(tool, "tomlsort")
        tomlsort_defaults = c.Infra.Deps.TOMLSORT_DEFAULTS
        for key, value in tomlsort_defaults:
            current = _unwrap_item(tomlsort.get(key))
            if current != value:
                tomlsort[key] = _array(value) if isinstance(value, list) else value
                changes.append(f"tool.tomlsort.{key} set")

        yamlfix = _ensure_table(tool, "yamlfix")
        yamlfix_defaults = c.Infra.Deps.YAMLFIX_DEFAULTS
        for key, value in yamlfix_defaults:
            if _unwrap_item(yamlfix.get(key)) != value:
                yamlfix[key] = value
                changes.append(f"tool.yamlfix.{key} set to {value}")

        return changes


class InjectCommentsPhase:
    """Inject managed/custom/auto markers into pyproject.toml."""

    def apply(self, rendered: str) -> tuple[str, list[str]]:
        """Inject markers and banner into rendered TOML content."""
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

            if skip_broken_group_section and not line.strip():
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

        return "\n".join(out).rstrip() + "\n", changes


class FlextInfraPyprojectModernizer:
    """Modernize all workspace pyproject.toml files."""

    def __init__(self, root: Path | None = None) -> None:
        """Initialize modernizer with workspace root."""
        super().__init__()
        self.root = root or ROOT
        self._runner: p.Infra.CommandRunner = FlextInfraCommandRunner()

    def find_pyproject_files(self) -> list[Path]:
        """Find all pyproject.toml files in workspace."""
        files: list[Path] = []
        for path in self.root.rglob(c.Infra.Files.PYPROJECT_FILENAME):
            if any(part in c.Infra.Deps.SKIP_DIRS for part in path.parts):
                continue
            files.append(path)
        return sorted(files)

    def process_file(
        self,
        path: Path,
        *,
        canonical_dev: list[str],
        dry_run: bool,
        skip_comments: bool,
    ) -> list[str]:
        """Process single pyproject.toml file."""
        doc = _read_doc(path)
        if doc is None:
            return ["invalid TOML"]

        is_root = path.parent.resolve() == self.root.resolve()

        shared_path, shared_written = _ensure_ruff_shared_template(
            path.parent, self.root
        )
        changes: list[str] = []
        if shared_written:
            changes.append(f"generated {shared_path.relative_to(path.parent)}")

        changes.extend(ConsolidateGroupsPhase().apply(doc, canonical_dev))
        changes.extend(EnsurePytestConfigPhase().apply(doc))
        changes.extend(EnsurePyreflyConfigPhase().apply(doc, is_root=is_root))
        changes.extend(EnsureMypyConfigPhase().apply(doc))
        changes.extend(EnsurePydanticMypyConfigPhase().apply(doc))
        changes.extend(EnsureFormattingToolingPhase().apply(doc))
        changes.extend(EnsureNamespaceToolingPhase().apply(doc, path=path))
        changes.extend(EnsureRuffConfigPhase().apply(doc, path=path))
        changes.extend(EnsurePyrightConfigPhase().apply(doc, is_root=is_root))

        tool = doc.get(c.Infra.Toml.TOOL)
        if isinstance(tool, Table):
            poetry = tool.get(c.Infra.Toml.POETRY)
            if isinstance(poetry, Table):
                group = poetry.get(c.Infra.Toml.GROUP)
                if isinstance(group, Table):
                    empty_groups: list[str] = []
                    for name in list(group.keys()):
                        group_item = group.get(name)
                        if isinstance(group_item, Table):
                            deps = group_item.get(c.Infra.Toml.DEPENDENCIES)
                            if isinstance(deps, Table) and len(deps) == 0:
                                empty_groups.append(name)
                    for name in empty_groups:
                        del group[name]
                        changes.append(f"removed empty poetry group '{name}'")
                    if len(group) == 0:
                        del poetry[c.Infra.Toml.GROUP]
                        changes.append("removed empty poetry group container")

        rendered = tomlkit.dumps(doc)
        if not skip_comments:
            rendered, comment_changes = InjectCommentsPhase().apply(rendered)
            changes.extend(comment_changes)

        if changes and not dry_run:
            _ = path.write_text(rendered, encoding=c.Infra.Encoding.DEFAULT)

        return changes

    def run(self, args: argparse.Namespace) -> int:
        """Execute modernization with command-line arguments."""
        dry_run = bool(args.dry_run or args.audit)
        files = self.find_pyproject_files()

        root_doc = _read_doc(self.root / c.Infra.Files.PYPROJECT_FILENAME)
        if root_doc is None:
            return 2
        canonical_dev = _canonical_dev_dependencies(root_doc)

        violations: MutableMapping[str, list[str]] = {}
        total = 0
        for file_path in files:
            changes = self.process_file(
                file_path,
                canonical_dev=canonical_dev,
                dry_run=dry_run,
                skip_comments=bool(args.skip_comments),
            )
            if not changes:
                continue
            rel = str(file_path.relative_to(self.root))
            violations[rel] = changes
            total += len(changes)

        if violations:
            for changes in violations.values():
                for _item in changes:
                    pass

        if args.audit and total > 0:
            return 1

        if not dry_run and not args.skip_check:
            return self._run_poetry_check(files)
        return 0

    def _run_poetry_check(self, files: list[Path]) -> int:
        """Run poetry check on each project directory."""
        has_warning = False
        for path in files:
            project_dir = path.parent
            result = self._runner.run_raw(
                [c.Infra.Cli.POETRY, c.Infra.Cli.PoetryCmd.CHECK],
                cwd=project_dir,
            )
            if result.is_failure:
                has_warning = True
                continue
            if result.value.exit_code != 0:
                has_warning = True
        return 1 if has_warning else 0


def _parser() -> argparse.ArgumentParser:
    """Create argument parser for modernizer CLI."""
    parser = argparse.ArgumentParser(description="Modernize workspace pyproject files")
    _ = parser.add_argument("--audit", action="store_true")
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument("--skip-comments", action="store_true")
    _ = parser.add_argument("--skip-check", action="store_true")
    return parser


def main() -> int:
    """Execute pyproject modernization from command line."""
    parser = _parser()
    args = parser.parse_args()
    return FlextInfraPyprojectModernizer().run(args)


if __name__ == "__main__":
    raise SystemExit(main())
