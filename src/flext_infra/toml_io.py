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
from flext_core import FlextResult, FlextService, r, t
from tomlkit.items import Table

from flext_infra import c

type TomlScalar = str | int | float | bool | None
type TomlValue = (
    TomlScalar | list[TomlScalar] | list[TomlValue] | MutableMapping[str, TomlValue]
)
type TomlMap = MutableMapping[str, t.ConfigMapValue]
type TomlMutableMap = MutableMapping[str, t.ConfigMapValue]


def _as_toml_mapping(value: t.ConfigMapValue) -> TomlMutableMap | None:
    if isinstance(value, MutableMapping) and all(isinstance(key, str) for key in value):
        return value
    return None


class FlextInfraTomlService(FlextService[FlextResult[bool]]):
    """Infrastructure service for TOML file I/O.

    Provides FlextResult-wrapped TOML read/write operations, replacing
    the bare functions from ``scripts/libs/toml_io.py``.
    """

    def __init__(self) -> None:
        """Initialize the TOML service."""
        super().__init__()

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
                path.read_text(encoding=c.Encoding.DEFAULT),
            )
            return r[TomlMap].ok(data)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            return r[TomlMap].fail(f"TOML read error: {exc}")

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
                path.read_text(encoding=c.Encoding.DEFAULT),
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
                encoding=c.Encoding.DEFAULT,
            )
        except OSError as exc:
            return r[bool].fail(f"TOML write error: {exc}")

    @staticmethod
    def value_differs(current: t.ConfigMapValue, expected: t.ConfigMapValue) -> bool:
        """Return True if current and expected differ.

        Compares as strings for lists.
        """
        if isinstance(current, list) and isinstance(expected, list):
            return [str(x) for x in current] != [str(x) for x in expected]
        return str(current) != str(expected)

    @staticmethod
    def build_table(data: TomlMap) -> Table:
        """Build a tomlkit Table from a nested dict."""
        table = tomlkit.table()
        for key, value in data.items():
            nested_mapping = _as_toml_mapping(value)
            if nested_mapping is not None:
                table[key] = TomlService.build_table(nested_mapping)
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
            expected_mapping = _as_toml_mapping(expected)
            if expected_mapping is not None:
                current_mapping = _as_toml_mapping(current)
                if current_mapping is None:
                    target[key] = self.build_table(expected_mapping)
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

    def execute(self) -> FlextResult[bool]:
        """Execute the service (required by FlextService base class)."""
        return r[bool].ok(True)


__all__ = ["FlextInfraTomlService"]

        """Execute the service (required by FlextService base class)."""
