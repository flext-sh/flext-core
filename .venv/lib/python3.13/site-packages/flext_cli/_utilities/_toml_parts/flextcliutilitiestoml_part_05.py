"""Generic TOML helpers shared through ``u.Cli.toml_*``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from tomlkit.items import Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_01 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart01,
)
from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_02 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart02,
)
from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_03 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart03,
)
from flext_core import u

if TYPE_CHECKING:
    from flext_cli import t


class FlextCliUtilitiesToml:
    """Implementation part for FlextCliUtilitiesToml."""

    @staticmethod
    def toml_sync_mapping_table(
        container: TOMLDocument | Table,
        key: str,
        expected: t.JsonMapping,
        *,
        sort_keys: bool = False,
    ) -> bool:
        """Synchronize a TOML table mapping in place; return True if mutated."""
        existing = container.get(key, None)
        current = FlextCliUtilitiesTomlPart01.toml_as_mapping(
            existing
            if existing is not None
            and (u.mapping(existing) or isinstance(existing, TOMLDocument | Item))
            else None
        )
        normalized_expected = {
            item_key: expected[item_key]
            for item_key in (sorted(expected) if sort_keys else tuple(expected))
        }
        if current == normalized_expected:
            return False
        table = FlextCliUtilitiesTomlPart02.toml_ensure_table(container, key)
        for existing_key in list(table):
            if existing_key not in normalized_expected:
                del table[existing_key]
        for item_key, item_value in normalized_expected.items():
            table[item_key] = item_value
        return True

    @staticmethod
    def toml_mapping_sync_string_list(
        container: t.MutableJsonMapping,
        key: str,
        expected: t.StrSequence,
        *,
        sort_values: bool = False,
    ) -> bool:
        """Synchronize a plain string-list field; return True if mutated."""
        current = FlextCliUtilitiesTomlPart01.toml_as_string_list(
            container.get(key, None)
        )
        normalized_expected = sorted(expected) if sort_values else [*expected]
        normalized_current = sorted(current) if sort_values else [*current]
        if normalized_current == normalized_expected:
            return False
        normalized_list: t.JsonValueList = list(normalized_expected)
        container[key] = normalized_list
        return True

    @staticmethod
    def toml_mapping_sync_mapping_table(
        container: t.MutableJsonMapping,
        key: str,
        expected: t.JsonMapping,
        *,
        sort_keys: bool = False,
    ) -> bool:
        """Synchronize a plain mapping-table field; return True if mutated."""
        existing = container.get(key, None)
        current = FlextCliUtilitiesTomlPart01.toml_as_mapping(
            existing if isinstance(existing, Mapping) else None
        )
        normalized_expected = {
            item_key: expected[item_key]
            for item_key in (sorted(expected) if sort_keys else tuple(expected))
        }
        if current == normalized_expected:
            return False
        table = FlextCliUtilitiesTomlPart03.toml_mapping_ensure_table(container, key)
        stale_keys = [
            existing_key
            for existing_key in table
            if existing_key not in normalized_expected
        ]
        for existing_key in stale_keys:
            del table[existing_key]
        for item_key, item_value in normalized_expected.items():
            table[item_key] = item_value
        return True

    @staticmethod
    def toml_navigate_path(doc: TOMLDocument, path: t.StrSequence) -> Table:
        """Navigate to a nested TOML table by path segments.

        Always roots at [tool]. Skips "tool" in path if present.
        Creates intermediate tables as needed.
        """
        return FlextCliUtilitiesTomlPart02.toml_ensure_path(
            FlextCliUtilitiesTomlPart03.toml_ensure_tool_table(doc),
            [segment for segment in path if segment != "tool"],
        )

    @staticmethod
    def toml_dot_path(*parts: str) -> str:
        """Build one dotted TOML path from non-empty segments."""
        return ".".join(part for part in parts if part)

    @staticmethod
    def toml_table_prefix(path: t.StrSequence) -> str:
        """Build a dotted prefix string from table path (e.g. "tool.ruff.lint")."""
        return FlextCliUtilitiesToml.toml_dot_path(
            "tool", *(segment for segment in path if segment != "tool")
        )


__all__: list[str] = ["FlextCliUtilitiesToml"]
