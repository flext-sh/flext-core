"""TOML I/O service for reading and writing TOML files.

Wraps TOML operations with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tomllib
from collections.abc import MutableMapping
from pathlib import Path

import tomlkit
from flext_core.result import FlextResult, r
from flext_core.typings import t
from tomlkit.items import Table

from flext_infra.constants import ic

type TomlScalar = str | int | float | bool | None
type TomlValue = TomlScalar | list[TomlScalar] | list[TomlValue] | MutableMapping[str, TomlValue]
type TomlMap = MutableMapping[str, TomlValue]
type TomlMutableMap = MutableMapping[str, TomlValue]
_TableLike = Table | TomlMutableMap


def _as_toml_mapping(value: t.ConfigMapValue) -> TomlMutableMap | None:
    if MutableMapping in type(value).__mro__:
        return value
    if Table in type(value).__mro__:
        return value
    return None


class TomlService:
    """Infrastructure service for TOML file I/O.

    Provides FlextResult-wrapped TOML read/write operations, replacing
    the bare functions from ``scripts/libs/toml_io.py``.
    """

    def read(self, path: Path) -> FlextResult[TomlMap]:
        """Read and parse a TOML file as a plain dict.

        Args:
            path: Path to the TOML file.

        Returns:
            FlextResult with parsed TOML data, or failure on error.

        """
        if not path.exists():
            return r[TomlMap].ok({})
        try:
            data = tomllib.loads(
                path.read_text(encoding=ic.Encoding.DEFAULT),
            )
            return r[TomlMap].ok(data)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            return r[TomlMap].fail(f"TOML read error: {exc}")

    def read_pyproject(self, path: Path) -> FlextResult[TomlMap]:
        """Read and parse a pyproject.toml from a directory or file.

        Args:
            path: Directory containing pyproject.toml, or direct file path.

        Returns:
            FlextResult with parsed TOML data.

        """
        target = path / ic.Files.PYPROJECT_FILENAME if path.is_dir() else path
        return self.read(target)

    def read_document(self, path: Path) -> FlextResult[tomlkit.TOMLDocument]:
        """Read and parse a TOML file as a tomlkit document.

        Preserves formatting and comments for round-trip editing.

        Args:
            path: Path to the TOML file.

        Returns:
            FlextResult with TOMLDocument, or failure on error.

        """
        if not path.exists():
            return r[tomlkit.TOMLDocument].fail(f"file not found: {path}")
        try:
            doc = tomlkit.parse(
                path.read_text(encoding=ic.Encoding.DEFAULT),
            )
            return r[tomlkit.TOMLDocument].ok(doc)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            return r[tomlkit.TOMLDocument].fail(
                f"TOML document read error: {exc}",
            )

    def write_document(
        self,
        path: Path,
        doc: tomlkit.TOMLDocument,
    ) -> FlextResult[bool]:
        """Write a tomlkit document to a TOML file.

        Args:
            path: Destination file path.
            doc: TOMLDocument to write.

        Returns:
            FlextResult[bool] with True on success.

        """
        try:
            _ = path.write_text(
                tomlkit.dumps(doc),
                encoding=ic.Encoding.DEFAULT,
            )
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"TOML write error: {exc}")

    @staticmethod
    def value_differs(current: TomlValue, expected: TomlValue) -> bool:
        """Return True if current and expected differ.

        Compares as strings for lists.
        """
        if type(current) is list and type(expected) is list:
            return [str(x) for x in current] != [str(x) for x in expected]
        return str(current) != str(expected)

    @staticmethod
    def build_table(data: TomlMap) -> Table:
        """Build a tomlkit Table from a nested dict."""
        table = tomlkit.table()
        for key, value in data.items():
            if type(value) is dict or Table in type(value).__mro__:
                table[key] = TomlService.build_table(value)
            else:
                table[key] = value
        return table

    def sync_mapping(
        self,
        target: TomlMutableMap,
        canonical: TomlMap,
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
            if type(expected) is dict or Table in type(expected).__mro__:
                current_mapping = _as_toml_mapping(current)
                if current_mapping is None:
                    target[key] = self.build_table(expected)
                    added.append(path)
                    continue
                self.sync_mapping(
                    current_mapping,
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

    def sync_section(
        self,
        target: _TableLike,
        canonical: TomlMap,
        *,
        prune_extras: bool = True,
    ) -> tuple[list[str], list[str], list[str]]:
        """Sync a TOML section to canonical values.

        Returns:
            Tuple of (added, updated, removed) key paths.

        """
        added: list[str] = []
        updated: list[str] = []
        removed: list[str] = []
        target_mapping = _as_toml_mapping(target)
        if target_mapping is None:
            return added, updated, removed
        self.sync_mapping(
            target_mapping,
            canonical,
            prune_extras=prune_extras,
            prefix="",
            added=added,
            updated=updated,
            removed=removed,
        )
        return added, updated, removed


__all__ = ["TomlService"]
