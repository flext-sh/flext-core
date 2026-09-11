"""Generic TOML helpers shared through ``u.Cli.toml_*``."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from typing import ClassVar

import tomlkit
from tomlkit.items import AoT, Array, Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_cli import c, p, t
from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_02 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart02,
)
from flext_core import u


class FlextCliUtilitiesToml:
    """Implementation part for FlextCliUtilitiesToml."""

    _module_logger: ClassVar[p.Logger] = u.fetch_logger(__name__)

    @staticmethod
    def toml_as_mapping(value: t.Cli.TomlMappingSource | None) -> t.JsonMapping | None:
        """Normalize a TOML mapping into a typed plain mapping."""
        normalized = FlextCliUtilitiesToml.toml_unwrap_item(value)
        if normalized is None or not u.mapping(normalized):
            return None
        return t.Cli.JSON_MAPPING_ADAPTER.validate_python(normalized)

    @staticmethod
    def toml_unwrap_item(
        value: t.Cli.TomlMappingSource | t.JsonValue | None,
    ) -> t.JsonValue | None:
        """Unwrap TOML items and documents to plain Python values."""
        if value is None:
            return None
        if isinstance(value, Mapping) and not isinstance(value, TOMLDocument | Item):
            return u.normalize_to_json_value(value)
        normalized = value.unwrap() if isinstance(value, TOMLDocument | Item) else value
        if isinstance(normalized, Item):
            return None
        return u.normalize_to_json_value(normalized)

    @staticmethod
    def toml_as_string_list(value: t.Cli.TomlStringListSource | None) -> t.StrSequence:
        """Normalize a TOML array into a string sequence."""
        normalized: t.Cli.TomlStringListSource | None = (
            value.unwrap() if isinstance(value, TOMLDocument | Item) else value
        )
        if normalized is None or isinstance(normalized, str | bytes):
            return ()
        if not isinstance(normalized, Sequence):
            return ()
        # NOTE (multi-agent): String sequences are immutable at this boundary.
        return tuple(str(item) for item in normalized)

    @staticmethod
    def toml_array(items: t.StrSequence) -> Array:
        """Create a multiline TOML array."""
        array = tomlkit.array()
        for item in items:
            array.add_line(item)
        return array.multiline(True)

    @staticmethod
    def toml_document() -> TOMLDocument:
        """Create a new TOML document."""
        return tomlkit.document()

    @staticmethod
    def toml_table(*, super_table: bool = False) -> Table:
        """Create a new TOML table (``super_table`` for a dotted parent table)."""
        return tomlkit.table(is_super_table=True) if super_table else tomlkit.table()

    @staticmethod
    def toml_aot() -> AoT:
        """Create a new TOML array-of-tables."""
        return tomlkit.aot()

    @staticmethod
    def toml_parse_text(text: str) -> TOMLDocument | None:
        """Parse TOML text, returning ``None`` on invalid input."""
        try:
            return tomlkit.parse(text)
        except c.EXC_TYPE_VALIDATION:
            return None

    @staticmethod
    def toml_dumps(doc: TOMLDocument) -> str:
        """Serialize a TOML document to text (round-trip preserving)."""
        return tomlkit.dumps(doc)

    @staticmethod
    def toml_mapping_from_text(text: str) -> t.JsonMapping | None:
        """Parse TOML text into one validated plain mapping."""
        try:
            loaded = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return None
        try:
            return t.Cli.JSON_MAPPING_ADAPTER.validate_python(loaded)
        except c.ValidationError:
            return None

    @staticmethod
    def toml_document_from_mapping(mapping: t.JsonMapping) -> TOMLDocument:
        """Build one TOML document from a validated plain mapping."""
        document = FlextCliUtilitiesToml.toml_document()
        for key, value in mapping.items():
            document[key] = FlextCliUtilitiesTomlPart02.toml_item_from_json_value(value)
        return document


__all__: list[str] = ["FlextCliUtilitiesToml"]
