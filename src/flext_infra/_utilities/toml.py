"""TOML utility helpers for flext-infra.

Provides type-safe TOML operations: normalization, reading, table manipulation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import tomlkit
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from tomlkit.items import Array, Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_core import FlextLogger, r
from flext_infra import c, t


class FlextInfraUtilitiesToml:
    """TOML utility helpers — normalization, reading, table manipulation.

    Usage::

        from flext_infra import u

        result = u.Infra.as_toml_mapping(value)
        doc = u.Infra.read(some_path)
    """

    logger = FlextLogger(__name__)

    _CONTAINER_DICT_ADAPTER: TypeAdapter[dict[str, t.Infra.InfraValue]] | None = None
    _CONTAINER_LIST_ADAPTER: TypeAdapter[list[JsonValue]] | None = None

    @staticmethod
    def _get_container_dict_adapter() -> TypeAdapter[dict[str, t.Infra.InfraValue]]:
        """Get or create TypeAdapter for dict[str, t.Infra.InfraValue]."""
        if FlextInfraUtilitiesToml._CONTAINER_DICT_ADAPTER is None:
            FlextInfraUtilitiesToml._CONTAINER_DICT_ADAPTER = TypeAdapter(
                dict[str, t.Infra.InfraValue],
            )
        return FlextInfraUtilitiesToml._CONTAINER_DICT_ADAPTER

    @staticmethod
    def _get_container_list_adapter() -> TypeAdapter[list[JsonValue]]:
        """Get or create TypeAdapter for list[JsonValue]."""
        if FlextInfraUtilitiesToml._CONTAINER_LIST_ADAPTER is None:
            FlextInfraUtilitiesToml._CONTAINER_LIST_ADAPTER = TypeAdapter(
                list[JsonValue],
            )
        return FlextInfraUtilitiesToml._CONTAINER_LIST_ADAPTER

    @staticmethod
    def as_toml_mapping(value: t.Infra.InfraValue) -> t.Infra.ContainerDict | None:
        """Check if value is a MutableMapping and return it typed, otherwise None."""
        if not isinstance(value, dict):
            return None
        try:
            normalized_value = (
                FlextInfraUtilitiesToml._get_container_dict_adapter().validate_python(
                    value
                )
            )
        except ValidationError:
            return None
        result: t.Infra.ContainerDict = {
            str(key): normalized_value[key] for key in normalized_value
        }
        return result

    @staticmethod
    def normalize_container_value(
        value: t.Infra.InfraValue
        | Item
        | TOMLDocument
        | dict[str, t.Infra.InfraValue]
        | None,
    ) -> t.Infra.InfraValue | None:
        """Normalize TOML items/documents to a concrete container value."""
        normalized: t.Infra.InfraValue | Item | dict[str, t.Infra.InfraValue] | None = (
            value
        )
        if isinstance(value, (TOMLDocument, Item)):
            normalized = value.unwrap()
        if isinstance(normalized, Item):
            return None
        return normalized

    @staticmethod
    def as_container_list(
        value: t.Infra.InfraValue | Item | None,
    ) -> list[t.Infra.InfraValue]:
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
        value: t.Infra.InfraValue | Item | None,
    ) -> t.Infra.InfraValue | None:
        """Unwrap a tomlkit Item to get the underlying value."""
        return FlextInfraUtilitiesToml.normalize_container_value(value)

    @staticmethod
    def as_string_list(value: t.Infra.InfraValue | Item | None) -> list[str]:
        """Convert TOML value to list of strings."""
        normalized = FlextInfraUtilitiesToml.normalize_container_value(value)
        if normalized is None or isinstance(normalized, str):
            return []
        if isinstance(normalized, list):
            try:
                typed_items = FlextInfraUtilitiesToml._get_container_list_adapter().validate_python(
                    normalized
                )
            except ValidationError:
                return []
            return [str(raw) for raw in typed_items]
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
        existing: t.Infra.InfraValue | Item | None = None
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
        key: t.Infra.InfraValue,
    ) -> t.Infra.InfraValue | None:
        """Retrieve and normalize a value from a TOML container by key."""
        if not isinstance(key, str):
            return None
        raw_value: t.Infra.InfraValue | None = None
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
        if isinstance(raw_value, dict):
            try:
                return FlextInfraUtilitiesToml._get_container_dict_adapter().validate_python(
                    raw_value
                )
            except ValidationError:
                return None
        if isinstance(raw_value, list):
            try:
                return FlextInfraUtilitiesToml._get_container_list_adapter().validate_python(
                    raw_value
                )
            except ValidationError:
                return None
        if not isinstance(raw_value, (dict, list)):
            return None
        return None

    @staticmethod
    def table_string_keys(table: Table) -> list[str]:
        """Return table keys as strings."""
        return list(table)

    @staticmethod
    def read(path: Path) -> TOMLDocument | None:
        """Read and parse TOML document from file.

        Returns None when the file does not exist or is invalid TOML.
        Prefer ``read_document`` for r-wrapped semantics.
        """
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

    @staticmethod
    def read_document(path: Path) -> r[TOMLDocument]:
        """Read and parse a TOML document, returning r.

        Args:
            path: Path to the TOML file.

        Returns:
            r[TOMLDocument] with parsed document on success,
            or failure with descriptive error.

        """
        doc = FlextInfraUtilitiesToml.read(path)
        if doc is None:
            return r[TOMLDocument].fail(f"failed to read TOML: {path}")
        return r[TOMLDocument].ok(doc)

    @staticmethod
    def write_document(path: Path, doc: TOMLDocument) -> r[bool]:
        """Write a TOML document to file.

        Creates parent directories as needed.

        Args:
            path: Destination file path.
            doc: TOML document to serialize.

        Returns:
            r[bool] with True on success.

        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(
                doc.as_string(),
                encoding=c.Infra.Encoding.DEFAULT,
            )
        except OSError as exc:
            return r[bool].fail(f"TOML write error: {exc}")
        return r[bool].ok(True)


__all__ = ["FlextInfraUtilitiesToml"]
