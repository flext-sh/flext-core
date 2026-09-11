"""Generic TOML helpers shared through ``u.Cli.toml_*``."""

from __future__ import annotations

from collections.abc import MutableMapping

from tomlkit.items import Table
from tomlkit.toml_document import TOMLDocument

from flext_cli import t
from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_01 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart01,
)
from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_02 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart02,
)
from flext_core import u


class FlextCliUtilitiesToml:
    """Implementation part for FlextCliUtilitiesToml."""

    @staticmethod
    def toml_table_path(
        parent: TOMLDocument | Table, path: t.StrSequence
    ) -> Table | None:
        """Return a nested table path without creating missing tables."""
        current: TOMLDocument | Table = parent
        for segment in path:
            table = FlextCliUtilitiesTomlPart02.toml_table_child(current, segment)
            if table is None:
                return None
            current = table
        return current if FlextCliUtilitiesTomlPart02.toml_is_table(current) else None

    @staticmethod
    def toml_ensure_tool_table(doc: TOMLDocument) -> Table:
        """Return the top-level ``[tool]`` table."""
        return FlextCliUtilitiesTomlPart02.toml_ensure_table(doc, "tool")

    @staticmethod
    def toml_value(container: TOMLDocument | Table, key: str) -> t.JsonValue | None:
        """Return a normalized TOML value from a container."""
        if key not in container:
            return None
        raw_value = FlextCliUtilitiesTomlPart01.toml_unwrap_item(container[key])
        if raw_value is None:
            return None
        return raw_value

    @staticmethod
    def toml_mapping_child(container: t.JsonMapping, key: str) -> t.JsonMapping | None:
        """Return a plain mapping child from one normalized TOML mapping."""
        value = container.get(key, None)
        if not u.mapping(value):
            return None
        return t.Cli.JSON_MAPPING_ADAPTER.validate_python(value)

    @staticmethod
    def toml_mapping_ensure_table(parent: t.MutableJsonMapping, key: str) -> t.JsonDict:
        """Return one mutable plain mapping child, creating it when missing."""
        existing = parent.get(key, None)
        if isinstance(existing, dict):
            return existing
        normalized_mapping = FlextCliUtilitiesTomlPart01.toml_as_mapping(existing)
        if normalized_mapping is not None:
            normalized_table: t.JsonDict = dict(normalized_mapping)
            parent[key] = normalized_table
            return normalized_table
        # NOTE (multi-agent): TOML mutation requires this explicitly typed table.
        table: t.JsonDict = {}
        parent[key] = table
        return table

    @staticmethod
    def toml_mapping_ensure_path(
        parent: t.MutableJsonMapping, path: t.StrSequence
    ) -> t.MutableJsonMapping:
        """Return one nested mutable mapping path, creating tables as needed."""
        current = parent
        for segment in path:
            current = FlextCliUtilitiesToml.toml_mapping_ensure_table(current, segment)
        return current

    @staticmethod
    def toml_mapping_path(
        parent: t.JsonMapping, path: t.StrSequence
    ) -> t.MutableJsonMapping | None:
        """Return one nested mutable mapping path without creating missing tables."""
        if not isinstance(parent, MutableMapping):
            return None
        current: t.MutableJsonMapping = parent
        for segment in path:
            value = current.get(segment, None)
            if not isinstance(value, MutableMapping):
                return None
            current = value
        return current

    @staticmethod
    def toml_remove_key_if_present(container: TOMLDocument | Table, key: str) -> bool:
        """Remove a TOML key when it exists; return True if removed."""
        if key not in container:
            return False
        del container[key]
        return True


__all__: list[str] = ["FlextCliUtilitiesToml"]
