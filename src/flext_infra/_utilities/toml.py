"""TOML utility helpers for flext-infra.

Provides type-safe TOML operations: normalization, reading, table manipulation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import tomlkit
from pydantic import BaseModel, TypeAdapter, ValidationError
from tomlkit.items import Array, Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_core import FlextLogger, u
from flext_infra.constants import FlextInfraConstants as c
from flext_infra.typings import FlextInfraTypes as t


class FlextInfraUtilitiesToml:
    """TOML utility helpers — normalization, reading, table manipulation.

    Usage::

        from flext_infra import u

        result = u.Infra.Toml.as_toml_mapping(value)
        doc = u.Infra.Toml.read(some_path)
    """

    logger = FlextLogger(__name__)

    _CONTAINER_DICT_ADAPTER: TypeAdapter[dict[str, t.ContainerValue]] | None = None
    _CONTAINER_LIST_ADAPTER: TypeAdapter[list[t.ContainerValue]] | None = None

    @staticmethod
    def _get_container_dict_adapter() -> TypeAdapter[dict[str, t.ContainerValue]]:
        """Get or create TypeAdapter for dict[str, ContainerValue]."""
        if FlextInfraUtilitiesToml._CONTAINER_DICT_ADAPTER is None:
            FlextInfraUtilitiesToml._CONTAINER_DICT_ADAPTER = TypeAdapter(
                dict[str, t.ContainerValue],
            )
        return FlextInfraUtilitiesToml._CONTAINER_DICT_ADAPTER

    @staticmethod
    def _get_container_list_adapter() -> TypeAdapter[list[t.ContainerValue]]:
        """Get or create TypeAdapter for list[ContainerValue]."""
        if FlextInfraUtilitiesToml._CONTAINER_LIST_ADAPTER is None:
            FlextInfraUtilitiesToml._CONTAINER_LIST_ADAPTER = TypeAdapter(
                list[t.ContainerValue],
            )
        return FlextInfraUtilitiesToml._CONTAINER_LIST_ADAPTER

    @staticmethod
    def as_toml_mapping(value: t.ContainerValue) -> t.Infra.ContainerDict | None:
        """Check if value is a MutableMapping and return it typed, otherwise None."""
        if not isinstance(value, dict):
            return None
        for item in value.values():
            if not u.is_general_value_type(item):
                return None
        result: t.Infra.ContainerDict = value
        return result

    @staticmethod
    def normalize_container_value(
        value: t.ContainerValue
        | Item
        | TOMLDocument
        | dict[str, t.ContainerValue]
        | None,
    ) -> t.ContainerValue | None:
        """Normalize TOML items/documents to a concrete container value."""
        normalized: t.ContainerValue | Item | dict[str, t.ContainerValue] | None = value
        if isinstance(value, (TOMLDocument, Item)):
            normalized = value.unwrap()
        if isinstance(normalized, Item):
            return None
        return normalized

    @staticmethod
    def as_container_list(
        value: t.ContainerValue | Item | None,
    ) -> list[t.ContainerValue]:
        """Validate and normalize list-like values to typed container list."""
        normalized = FlextInfraUtilitiesToml.normalize_container_value(value)
        if normalized is None:
            return []
        try:
            return (
                FlextInfraUtilitiesToml._get_container_list_adapter().validate_python(
                    normalized
                )
            )
        except ValidationError:
            return []

    @staticmethod
    def unwrap_item(
        value: t.ContainerValue | Item | None,
    ) -> t.ContainerValue | None:
        """Unwrap a tomlkit Item to get the underlying value."""
        return FlextInfraUtilitiesToml.normalize_container_value(value)

    @staticmethod
    def as_string_list(value: t.ContainerValue | Item | None) -> list[str]:
        """Convert TOML value to list of strings."""
        normalized = FlextInfraUtilitiesToml.normalize_container_value(value)
        if normalized is None or isinstance(normalized, str):
            return []
        if isinstance(normalized, list):
            return [str(raw) for raw in normalized]
        return [
            str(raw) for raw in FlextInfraUtilitiesToml.as_container_list(normalized)
        ]

    @staticmethod
    def array(items: list[str]) -> Array:
        """Create multiline TOML array from string items."""
        arr: Array = tomlkit.array()
        for item in items:
            arr.add_line(item)
        return arr.multiline(True)

    @staticmethod
    def ensure_table(parent: Table, key: str) -> Table:
        """Get or create a TOML table in parent.

        When the key already exists as a dotted-key implicit ("super") table,
        promote it to an explicit table so that tomlkit serializes sub-tables
        under the correct parent path instead of creating bare top-level sections.
        """
        existing: object | None = None
        if key in parent:
            existing = parent[key]
        if isinstance(existing, Table):
            if not existing.is_super_table():
                return existing
            del parent[key]
            table = tomlkit.table()
            for k in FlextInfraUtilitiesToml.table_string_keys(existing):
                table[k] = existing[k]
            parent[key] = table
            return table
        table = tomlkit.table()
        parent[key] = table
        return table

    @staticmethod
    def get(
        container: TOMLDocument | Table,
        key: t.ContainerValue,
    ) -> t.ContainerValue | None:
        """Retrieve and normalize a value from a TOML container by key."""
        if not isinstance(key, str):
            return None
        raw_value: t.ContainerValue | None = None
        if key in container:
            raw_value = FlextInfraUtilitiesToml.normalize_container_value(
                container[key],
            )
        if raw_value is None:
            return None
        if isinstance(
            raw_value,
            (str, int, float, bool, type(None), BaseModel, Path),
        ):
            return raw_value
        if not isinstance(raw_value, (dict, list)):
            return None
        normalized = FlextInfraUtilitiesToml.normalize_container_value(raw_value)
        if isinstance(normalized, dict):
            try:
                return FlextInfraUtilitiesToml._get_container_dict_adapter().validate_python(
                    normalized
                )
            except ValidationError:
                return None
        if isinstance(normalized, list):
            try:
                return FlextInfraUtilitiesToml._get_container_list_adapter().validate_python(
                    normalized
                )
            except ValidationError:
                return None
        if isinstance(
            normalized,
            (str, int, float, bool, type(None), BaseModel, Path),
        ):
            return normalized
        return None

    @staticmethod
    def table_string_keys(table: Table) -> list[str]:
        """Return table keys as strings."""
        return list(table)

    @staticmethod
    def read(path: Path) -> TOMLDocument | None:
        """Read and parse TOML document from file."""
        if not path.exists():
            return None
        try:
            return tomlkit.parse(
                path.read_text(encoding=c.Infra.Encoding.DEFAULT),
            )
        except (OSError, ValueError) as exc:
            FlextInfraUtilitiesToml.logger.warning(
                "Failed to read or parse TOML document",
                path=str(path),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None


# Module-level aliases for backward compatibility and direct import
unwrap_item = FlextInfraUtilitiesToml.unwrap_item
as_string_list = FlextInfraUtilitiesToml.as_string_list
array = FlextInfraUtilitiesToml.array
ensure_table = FlextInfraUtilitiesToml.ensure_table
toml_get = FlextInfraUtilitiesToml.get
table_string_keys = FlextInfraUtilitiesToml.table_string_keys
read_doc = FlextInfraUtilitiesToml.read
as_toml_mapping = FlextInfraUtilitiesToml.as_toml_mapping
normalize_container_value = FlextInfraUtilitiesToml.normalize_container_value
as_container_list = FlextInfraUtilitiesToml.as_container_list

__all__ = [
    "FlextInfraUtilitiesToml",
    "array",
    "as_container_list",
    "as_string_list",
    "as_toml_mapping",
    "ensure_table",
    "normalize_container_value",
    "read_doc",
    "table_string_keys",
    "toml_get",
    "unwrap_item",
]
