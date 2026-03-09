"""TOML I/O service for reading and writing TOML files.

Wraps TOML operations with r error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import override

import tomlkit
import tomlkit.exceptions
from pydantic import BaseModel, TypeAdapter, ValidationError
from tomlkit.items import Array, Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_core import FlextLogger, r, s
from flext_infra import c, t, u

_logger = FlextLogger(__name__)
_CONTAINER_DICT_ADAPTER = TypeAdapter(dict[str, t.ContainerValue])
_CONTAINER_LIST_ADAPTER = TypeAdapter(list[t.ContainerValue])


def _normalize_container_value(
    value: t.ContainerValue | Item | TOMLDocument | dict[str, t.ContainerValue] | None,
) -> t.ContainerValue | None:
    """Normalize TOML items/documents to a concrete container value."""
    normalized: t.ContainerValue | Item | dict[str, t.ContainerValue] | None = value
    if isinstance(value, (TOMLDocument, Item)):
        normalized = value.unwrap()
    if isinstance(normalized, Item):
        return None
    return normalized


def _as_container_list(value: t.ContainerValue | Item | None) -> list[t.ContainerValue]:
    """Validate and normalize list-like values to typed container list."""
    normalized = _normalize_container_value(value)
    if normalized is None:
        return []
    try:
        return _CONTAINER_LIST_ADAPTER.validate_python(normalized)
    except ValidationError:
        return []


def _find_ruff_shared_path(project_dir: Path, workspace_root: Path) -> tuple[Path, str]:
    """Return target ruff-shared file path and relative extend value."""
    workspace_candidate = workspace_root / "ruff-shared.toml"
    relative = os.path.relpath(workspace_candidate, start=project_dir)
    return (workspace_candidate, Path(relative).as_posix())


def ensure_ruff_shared_template(
    project_dir: Path,
    workspace_root: Path,
) -> tuple[Path, bool]:
    """Create managed ruff-shared.toml in workspace root when missing."""
    target, _ = _find_ruff_shared_path(project_dir, workspace_root)
    if target.exists():
        return (target, False)
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(
        c.Infra.Deps.RUFF_SHARED_TEMPLATE.rstrip() + "\n",
        encoding=c.Infra.Encoding.DEFAULT,
    )
    return (target, True)


def _dep_name(spec: str) -> str:
    """Extract normalized dependency name from requirement specification."""
    base = spec.strip().split("@", 1)[0].strip()
    match = c.Infra.Deps.DEP_NAME_RE.match(base)
    if match:
        return match.group(1).lower().replace("_", "-")
    return base.lower().replace("_", "-")


def _dedupe_specs(specs: list[str]) -> list[str]:
    """Deduplicate dependency specifications by normalized name."""
    seen: dict[str, str] = {}
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
    if isinstance(normalized, list):
        return [str(raw) for raw in normalized]
    return [str(raw) for raw in _as_container_list(normalized)]


def array(items: list[str]) -> Array:
    """Create multiline TOML array from string items."""
    arr: Array = tomlkit.array()
    for item in items:
        arr.add_line(item)
    return arr.multiline(True)


def ensure_table(parent: Table, key: str) -> Table:
    """Get or create a TOML table in parent."""
    existing: object | None = None
    if key in parent:
        existing = parent[key]
    if isinstance(existing, Table):
        return existing
    table = tomlkit.table()
    parent[key] = table
    return table


def _toml_get(
    container: TOMLDocument | Table,
    key: object,
) -> t.ContainerValue | Item | None:
    if not isinstance(key, str):
        return None
    raw_value: object | None = None
    if key in container:
        raw_value = container[key]
    if raw_value is None:
        return None
    if isinstance(raw_value, Item):
        return raw_value
    if isinstance(raw_value, (str, int, float, bool, type(None), BaseModel, Path)):
        return raw_value
    normalized_mapping = _normalize_container_value(raw_value)
    if isinstance(normalized_mapping, dict):
        try:
            return _CONTAINER_DICT_ADAPTER.validate_python(normalized_mapping)
        except ValidationError:
            return None
    if isinstance(normalized_mapping, list):
        try:
            return _CONTAINER_LIST_ADAPTER.validate_python(normalized_mapping)
        except ValidationError:
            return None
    if isinstance(
        normalized_mapping,
        (str, int, float, bool, type(None), BaseModel, Path),
    ):
        return normalized_mapping
    return None


def table_string_keys(table: Table) -> list[str]:
    """Return table keys as strings."""
    return list(table)


def ensure_pyright_execution_envs(
    pyright: Table,
    expected: list[dict[str, str]],
    changes: list[str],
) -> None:
    """Ensure pyright executionEnvironments matches expected; append to changes if updated."""
    raw = _unwrap_item(_toml_get(pyright, "executionEnvironments"))
    current: list[t.ContainerValue] = raw if isinstance(raw, list) else []
    if list(current) != expected:
        pyright["executionEnvironments"] = expected
        changes.append(
            "tool.pyright.executionEnvironments set with tests reportPrivateUsage=none",
        )


def read_doc(path: Path) -> tomlkit.TOMLDocument | None:
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


def discover_first_party_namespaces(project_dir: Path) -> list[str]:
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


def _project_dev_groups(doc: tomlkit.TOMLDocument) -> dict[str, list[str]]:
    """Extract optional-dependencies groups from project table."""
    project = _toml_get(doc, c.Infra.Toml.PROJECT)
    if project is None or not isinstance(project, Table):
        return {}
    optional = _toml_get(project, c.Infra.Toml.OPTIONAL_DEPENDENCIES)
    if optional is None or not isinstance(optional, Table):
        return {}
    return {
        c.Infra.Toml.DEV: _as_string_list(_toml_get(optional, c.Infra.Toml.DEV)),
        c.Infra.Directories.DOCS: _as_string_list(
            _toml_get(optional, c.Infra.Toml.DOCS),
        ),
        c.Infra.Gates.SECURITY: _as_string_list(
            _toml_get(optional, c.Infra.Toml.SECURITY),
        ),
        c.Infra.Toml.TEST: _as_string_list(_toml_get(optional, c.Infra.Toml.TEST)),
        c.Infra.Directories.TYPINGS: _as_string_list(
            _toml_get(optional, c.Infra.Directories.TYPINGS),
        ),
    }


def canonical_dev_dependencies(root_doc: tomlkit.TOMLDocument) -> list[str]:
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


CONTAINER_LIST_ADAPTER = _CONTAINER_LIST_ADAPTER
as_string_list = _as_string_list
dep_name = _dep_name
dedupe_specs = _dedupe_specs
find_ruff_shared_path = _find_ruff_shared_path
project_dev_groups = _project_dev_groups
toml_get = _toml_get
unwrap_item = _unwrap_item

_array = array
_canonical_dev_dependencies = canonical_dev_dependencies
_discover_first_party_namespaces = discover_first_party_namespaces
_ensure_pyright_execution_envs = ensure_pyright_execution_envs
_ensure_ruff_shared_template = ensure_ruff_shared_template
_ensure_table = ensure_table
_read_doc = read_doc
_table_string_keys = table_string_keys


class FlextInfraTomlService(s[bool]):
    """Infrastructure service for TOML file I/O.

    Provides r-wrapped TOML read/write operations, replacing
    the bare functions from ``scripts/libs/toml_io.py``.
    """

    def __init__(self) -> None:
        """Initialize the TOML service."""
        super().__init__()

    @staticmethod
    def build_table(data: t.Infra.ContainerDict) -> Table:
        """Build a tomlkit Table from a nested dict."""
        table = tomlkit.table()
        for key, value in data.items():
            nested_mapping = u.Infra.Toml.as_toml_mapping(value)
            if nested_mapping is not None:
                table[key] = FlextInfraTomlService.build_table(nested_mapping)
            else:
                table[key] = value
        return table

    @staticmethod
    def value_differs(current: object, expected: object) -> bool:
        """Return True if current and expected differ.

        Compares as strings for lists.
        """
        return str(current) != str(expected)

    @override
    def execute(self) -> r[bool]:
        """Execute the service (required by s base class)."""
        return r[bool].ok(True)

    def read(self, path: Path) -> r[t.Infra.ContainerDict]:
        """Read and parse a TOML file as a plain dict.

        Args:
            path: Path to the TOML file.

        Returns:
            r with parsed TOML data, or failure on error.

        """
        if not path.exists():
            return r[t.Infra.ContainerDict].ok({})
        try:
            data_raw = tomllib.loads(path.read_text(encoding=c.Infra.Encoding.DEFAULT))
            data: t.Infra.ContainerDict = data_raw
            return r[t.Infra.ContainerDict].ok(data)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            return r[t.Infra.ContainerDict].fail(f"TOML read error: {exc}")

    def read_document(self, path: Path) -> r[tomlkit.TOMLDocument]:
        """Read and parse a TOML file as a tomlkit document.

        Preserves formatting and comments for round-trip editing.

        Args:
            path: Path to the TOML file.

        Returns:
            r with TOMLDocument, or failure on error.

        """
        if not path.exists():
            return r[tomlkit.TOMLDocument].fail(f"file not found: {path}")
        try:
            doc = tomlkit.parse(path.read_text(encoding=c.Infra.Encoding.DEFAULT))
            return r[tomlkit.TOMLDocument].ok(doc)
        except (tomlkit.exceptions.ParseError, OSError) as exc:
            return r[tomlkit.TOMLDocument].fail(f"TOML document read error: {exc}")

    def sync_mapping(
        self,
        target: t.Infra.ContainerDict,
        canonical: t.Infra.ContainerDict,
        *,
        prune_extras: bool,
        prefix: str,
        added: list[str],
        updated: list[str],
        removed: list[str],
    ) -> None:
        """Update target mapping to match canonical; record changes."""
        for key, expected in canonical.items():
            current = target.get(key)
            path = f"{prefix}.{key}" if prefix else key
            expected_mapping = u.Infra.Toml.as_toml_mapping(expected)
            if expected_mapping is not None:
                current_mapping = (
                    u.Infra.Toml.as_toml_mapping(current)
                    if current is not None
                    else None
                )
                if current_mapping is None:
                    target[key] = expected_mapping
                    added.append(path)
                    continue
                self.sync_mapping(
                    current_mapping,
                    expected_mapping,
                    prune_extras=prune_extras,
                    prefix=path,
                    added=added,
                    updated=updated,
                    removed=removed,
                )
                continue
            if current is None:
                target[key] = expected
                added.append(path)
                continue
            if self.value_differs(current, expected):
                target[key] = expected
                updated.append(path)
        if not prune_extras:
            return
        for key in list(target.keys()):
            if key in canonical:
                continue
            path = f"{prefix}.{key}" if prefix else key
            del target[key]
            removed.append(path)

    def write(
        self,
        path: Path,
        payload: tomlkit.TOMLDocument | t.Infra.ContainerDict,
    ) -> r[bool]:
        """Write a TOML payload to a file.

        Creates parent directories as needed.

        Args:
            path: Destination file path.
            payload: Data to serialize as TOML (dict or TOMLDocument).

        Returns:
            r[bool] with True on success.

        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, tomlkit.TOMLDocument):
                content = payload.as_string()
            else:
                doc = tomlkit.document()
                for key, value in payload.items():
                    nested_mapping = u.Infra.Toml.as_toml_mapping(value)
                    if nested_mapping is not None:
                        doc[key] = self.build_table(nested_mapping)
                    else:
                        doc[key] = value
                content = doc.as_string()
            _ = path.write_text(content, encoding=c.Infra.Encoding.DEFAULT)
            return r[bool].ok(True)
        except (OSError, TypeError) as exc:
            return r[bool].fail(f"TOML write error: {exc}")

    def write_document(self, path: Path, doc: tomlkit.TOMLDocument) -> r[bool]:
        """Write a tomlkit document to a TOML file.

        Args:
            path: Destination file path.
            doc: TOMLDocument to write.

        Returns:
            r[bool] with True on success.

        """
        try:
            _ = path.write_text(doc.as_string(), encoding=c.Infra.Encoding.DEFAULT)
        except OSError as exc:
            return r[bool].fail(f"TOML write error: {exc}")
        return r[bool].ok(True)


__all__ = [
    "CONTAINER_LIST_ADAPTER",
    "FlextInfraTomlService",
    "array",
    "as_string_list",
    "canonical_dev_dependencies",
    "dedupe_specs",
    "dep_name",
    "discover_first_party_namespaces",
    "ensure_pyright_execution_envs",
    "ensure_ruff_shared_template",
    "ensure_table",
    "find_ruff_shared_path",
    "project_dev_groups",
    "read_doc",
    "table_string_keys",
    "toml_get",
    "unwrap_item",
]
