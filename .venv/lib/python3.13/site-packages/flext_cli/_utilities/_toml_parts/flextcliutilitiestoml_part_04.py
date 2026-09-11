"""Generic TOML helpers shared through ``u.Cli.toml_*``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_01 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart01,
)
from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_03 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart03,
)
from flext_core import u

if TYPE_CHECKING:
    from tomlkit.items import Table
    from tomlkit.toml_document import TOMLDocument

    from flext_cli import t


class FlextCliUtilitiesToml:
    """Implementation part for FlextCliUtilitiesToml."""

    @staticmethod
    def toml_sync_value(
        container: TOMLDocument | Table, key: str, expected: t.JsonValue
    ) -> bool:
        """Synchronize a scalar TOML value; return True if mutated."""
        current = FlextCliUtilitiesTomlPart03.toml_value(container, key)
        if current == expected:
            return False
        container[key] = expected
        return True

    @staticmethod
    def toml_sync_string_list(
        container: TOMLDocument | Table,
        key: str,
        expected: t.StrSequence,
        *,
        sort_values: bool = False,
    ) -> bool:
        """Synchronize a TOML string-array value; return True if mutated."""
        current = FlextCliUtilitiesTomlPart01.toml_as_string_list(
            FlextCliUtilitiesTomlPart03.toml_value(container, key)
        )
        normalized_expected = sorted(expected) if sort_values else [*expected]
        normalized_current = sorted(current) if sort_values else [*current]
        if normalized_current == normalized_expected:
            return False
        container[key] = FlextCliUtilitiesTomlPart01.toml_array(normalized_expected)
        return True

    @staticmethod
    def toml_merge_string_list(
        container: TOMLDocument | Table, key: str, required: t.StrSequence
    ) -> bool:
        """Merge required values into a TOML string-array; return True if mutated."""
        current = FlextCliUtilitiesTomlPart01.toml_as_string_list(
            FlextCliUtilitiesTomlPart03.toml_value(container, key)
        )
        merged = sorted({*current, *required})
        if list(current) == merged:
            return False
        container[key] = FlextCliUtilitiesTomlPart01.toml_array(merged)
        return True

    @staticmethod
    def toml_mapping_remove_key_if_present(
        container: t.MutableJsonMapping, key: str
    ) -> bool:
        """Remove one plain mapping key when it exists; return True if removed."""
        if key not in container:
            return False
        del container[key]
        return True

    @staticmethod
    def toml_mapping_sync_value(
        container: t.MutableJsonMapping, key: str, expected: t.JsonValue
    ) -> bool:
        """Synchronize a scalar/structured plain TOML value; return True if mutated."""
        current: t.JsonValue = u.normalize_to_json_value(container.get(key, None))
        normalized_expected: t.JsonValue = u.normalize_to_json_value(expected)
        if current == normalized_expected:
            return False
        container[key] = normalized_expected
        return True

    @staticmethod
    def toml_mapping_merge_string_list(
        container: t.MutableJsonMapping, key: str, required: t.StrSequence
    ) -> bool:
        """Merge required values into a plain string-list; return True if mutated."""
        current = FlextCliUtilitiesTomlPart01.toml_as_string_list(
            container.get(key, None)
        )
        merged = sorted({*current, *required})
        if list(current) == merged:
            return False
        normalized_list: t.JsonValueList = list(merged)
        container[key] = normalized_list
        return True


__all__: list[str] = ["FlextCliUtilitiesToml"]
