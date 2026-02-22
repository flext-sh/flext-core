"""Sync TOML structures from a canonical dict into a mutable target."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

import tomlkit
from tomlkit.items import Table

_TableLike = Table | MutableMapping[str, object]


def value_differs(current: object, expected: object) -> bool:
    """Return True if current and expected differ when stringified or as lists."""
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
    """Update target from canonical, appending paths to added/updated/removed."""
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
    """Sync a section; return (added, updated, removed) paths."""
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
