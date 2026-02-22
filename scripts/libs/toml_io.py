"""Unified pyproject.toml read utilities."""

from __future__ import annotations

import tomllib
from collections.abc import MutableMapping
from pathlib import Path
from typing import cast

import tomlkit
from tomlkit.items import Table

from .config import DEFAULT_ENCODING, PYPROJECT_FILENAME

_TableLike = Table | MutableMapping[str, object]


def read_pyproject(path: Path) -> dict[str, object]:
    """Read and parse a ``pyproject.toml`` as a plain dict.

    Args:
        path: Directory containing ``pyproject.toml``, or direct file path.

    Returns:
        Parsed TOML as dict. Returns empty dict if file doesn't exist.

    """
    target = path / PYPROJECT_FILENAME if path.is_dir() else path
    return read_toml_file(target)


def read_toml_file(path: Path) -> dict[str, object]:
    """Read and parse a TOML file as a plain dict; return {} if missing or invalid."""
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding=DEFAULT_ENCODING))
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def read_toml_document(path: Path) -> tomlkit.TOMLDocument | None:
    """Read and parse a TOML file as tomlkit document; return None if missing or invalid."""
    if not path.exists():
        return None
    try:
        return tomlkit.parse(path.read_text(encoding=DEFAULT_ENCODING))
    except (tomllib.TOMLDecodeError, OSError):
        return None


def write_toml_document(path: Path, doc: tomlkit.TOMLDocument) -> None:
    """Write a tomlkit document to a TOML file at path."""
    _ = path.write_text(tomlkit.dumps(doc), encoding=DEFAULT_ENCODING)


def value_differs(current: object, expected: object) -> bool:
    """Return True if current and expected differ (compare as strings for lists)."""
    if isinstance(current, list) and isinstance(expected, list):
        return [str(x) for x in current] != [str(x) for x in expected]
    return str(current) != str(expected)


def build_table(data: dict[str, object]) -> Table:
    """Build a tomlkit Table from a nested dict."""
    table = tomlkit.table()
    for key, value in data.items():
        if isinstance(value, dict):
            table[key] = build_table(value)
        else:
            table[key] = value
    return table


def sync_mapping(
    target: MutableMapping[str, object],
    canonical: dict[str, object],
    *,
    prune_extras: bool,
    prefix: str,
    added: list[str],
    updated: list[str],
    removed: list[str],
) -> None:
    """Update target mapping to match canonical; record changes in added/updated/removed."""
    for key, expected in canonical.items():
        current = target.get(key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(expected, dict):
            if current is None or not hasattr(current, "keys"):
                target[key] = build_table(expected)
                added.append(path)
                continue
            sync_mapping(
                cast("MutableMapping[str, object]", current),
                expected,
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
        if value_differs(current, expected):
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


def sync_section(
    target: _TableLike,
    canonical: dict[str, object],
    *,
    prune_extras: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Sync a TOML section to canonical; return (added, updated, removed) paths."""
    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    sync_mapping(
        cast("MutableMapping[str, object]", target),
        canonical,
        prune_extras=prune_extras,
        prefix="",
        added=added,
        updated=updated,
        removed=removed,
    )
    return added, updated, removed


__all__ = [
    "build_table",
    "read_pyproject",
    "read_toml_document",
    "read_toml_file",
    "sync_mapping",
    "sync_section",
    "value_differs",
    "write_toml_document",
]
