"""CLI base type aliases and adapters."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from types import GenericAlias, UnionType
from typing import ClassVar, TypeAliasType

from jinja2.sandbox import SandboxedEnvironment
from tomlkit.container import Container
from tomlkit.items import AoT, Array, Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_core import t


class FlextCliTypesBase:
    """Base CLI aliases shared across services and models."""

    type TableMappingRow = t.MappingKV[str, t.JsonPayload]
    type TableSequenceRow = t.SequenceOf[t.JsonPayload]
    type DefaultMapping = t.MappingKV[str, t.Scalar | t.StrSequence]
    type TableRow = TableMappingRow | TableSequenceRow
    type TableConfigValue = (
        t.JsonValue | t.StrSequence | t.SequenceOf[int] | t.SequenceOf[str | int] | None
    )
    type TabularData = TableMappingRow | t.SequenceOf[TableRow]
    type TableRows = t.SequenceOf[TableRow]
    type TableIndexValue = str | int
    type TableIndexSelection = t.SequenceOf[TableIndexValue]
    type TableShowIndex = bool | TableIndexSelection
    type TableDisableNumparse = bool | t.SequenceOf[int]
    type TableColAlign = t.StrSequence | None
    type CliValue = t.Scalar | t.StrSequence | DefaultMapping
    type CliDefaultSource = CliValue | t.SequenceOf[str | int] | Path
    type CliAnnotations = MutableMapping[str, type | GenericAlias]
    type TomlDocument = TOMLDocument
    type TomlTable = Table
    type TomlItem = Item
    type TomlArray = Array
    type TomlAoT = AoT
    type TomlContainer = Container
    type TomlParent = TOMLDocument | Table
    type TomlValue = TOMLDocument | Table | Item | Array | AoT | Container
    type RuntimeAnnotation = type | GenericAlias | UnionType | TypeAliasType
    type TemplateEnvironmentCache = MutableMapping[str, SandboxedEnvironment]

    PRIMITIVE_TYPES: ClassVar[tuple[type[str], type[int], type[float], type[bool]]] = (
        t.PRIMITIVES_TYPES
    )
    SCALAR_TYPES: ClassVar[tuple[type, ...]] = t.SCALAR_TYPES

    STR_SEQUENCE_ADAPTER: ClassVar[t.ValueAdapter[t.StrSequence]] = (
        t.str_sequence_adapter()
    )
    JSON_VALUE_ADAPTER: ClassVar[t.ValueAdapter[t.JsonValue]] = t.json_value_adapter()
    JSON_MAPPING_ADAPTER: ClassVar[t.ValueAdapter[t.JsonMapping]] = (
        t.json_mapping_adapter()
    )
    JSON_LIST_ADAPTER: ClassVar[t.ValueAdapter[t.JsonList]] = t.json_list_adapter()
    YAML_DICT_ADAPTER: ClassVar[t.ValueAdapter[t.JsonMapping]] = (
        t.json_mapping_adapter()
    )
    YAML_SEQ_ADAPTER: ClassVar[t.ValueAdapter[t.JsonList]] = t.json_list_adapter()
    CLI_DEFAULT_SOURCE_ADAPTER: ClassVar[t.ValueAdapter[CliDefaultSource]] = (
        t.TypeAdapter(CliDefaultSource)
    )


__all__: list[str] = ["FlextCliTypesBase"]
