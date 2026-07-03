"""Centralized tp.TypeAdapter cache for FLEXT typing aliases.

Each adapter is a `@classmethod @cache` that returns the canonical
``tp.TypeAdapter[...]``. The ``functools.cache`` wrapper memoizes per-cls,
so each adapter is constructed exactly once across the process — the
same caching contract the previous ClassVar pattern provided, with
~6 LOC eliminated per adapter (no per-adapter ``ClassVar`` slot, no
``if cls._x is None: cls._x = …`` boilerplate).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
)
from enum import StrEnum
from functools import cache

from .annotateds import FlextTypesAnnotateds as ta
from .base import FlextTypingBase as t
from .core import FlextTypesCore as tc
from .pydantic import FlextTypesPydantic as tp
from .services import FlextTypesServices as ts


class FlextTypesTypeAdapters:
    """Cached tp.TypeAdapter factories shared through the ``t`` facade."""

    @classmethod
    @cache
    def metadata_map_adapter(cls) -> tp.TypeAdapter[Mapping[str, tp.JsonValue]]:
        return tp.TypeAdapter(Mapping[str, tp.JsonValue])

    @classmethod
    @cache
    def json_value_adapter(cls) -> tp.TypeAdapter[tp.JsonValue]:
        return tp.TypeAdapter(tp.JsonValue)

    @classmethod
    @cache
    def json_mapping_adapter(cls) -> tp.TypeAdapter[t.JsonMapping]:
        return tp.TypeAdapter(t.JsonMapping)

    @classmethod
    @cache
    def strict_json_mapping_adapter(cls) -> tp.TypeAdapter[t.JsonMapping]:
        return tp.TypeAdapter(t.JsonMapping, config=tp.ConfigDict(strict=True))

    @classmethod
    @cache
    def json_dict_adapter(cls) -> tp.TypeAdapter[t.JsonDict]:
        return tp.TypeAdapter(t.JsonDict)

    @classmethod
    @cache
    def json_dict_sequence_adapter(cls) -> tp.TypeAdapter[t.SequenceOf[t.JsonDict]]:
        return tp.TypeAdapter(t.SequenceOf[t.JsonDict])

    @classmethod
    @cache
    def json_mapping_sequence_adapter(
        cls,
    ) -> tp.TypeAdapter[t.SequenceOf[t.JsonMapping]]:
        return tp.TypeAdapter(t.SequenceOf[t.JsonMapping])

    @classmethod
    @cache
    def json_mapping_by_str_adapter(
        cls,
    ) -> tp.TypeAdapter[t.MappingKV[str, t.JsonMapping]]:
        return tp.TypeAdapter(t.MappingKV[str, t.JsonMapping])

    @classmethod
    @cache
    def json_list_adapter(cls) -> tp.TypeAdapter[t.JsonList]:
        return tp.TypeAdapter(t.JsonList)

    @classmethod
    @cache
    def strict_json_list_adapter(cls) -> tp.TypeAdapter[t.JsonList]:
        return tp.TypeAdapter(t.JsonList, config=tp.ConfigDict(strict=True))

    @classmethod
    @cache
    def primitives_adapter(cls) -> tp.TypeAdapter[t.Primitives]:
        return tp.TypeAdapter(t.Primitives)

    @classmethod
    @cache
    def container_set_adapter(cls) -> tp.TypeAdapter[set[tp.JsonValue]]:
        return tp.TypeAdapter(set[tp.JsonValue])

    @classmethod
    @cache
    def string_set_adapter(cls) -> tp.TypeAdapter[set[str]]:
        return tp.TypeAdapter(set[str])

    @classmethod
    @cache
    def scalar_set_adapter(cls) -> tp.TypeAdapter[set[t.Scalar]]:
        return tp.TypeAdapter(set[t.Scalar])

    @classmethod
    @cache
    def sortable_dict_adapter(
        cls,
    ) -> tp.TypeAdapter[Mapping[ts.SortableObjectType, tp.JsonValue | None]]:
        return tp.TypeAdapter(Mapping[ts.SortableObjectType, tp.JsonValue | None])

    @classmethod
    @cache
    def bool_adapter(cls) -> tp.TypeAdapter[bool]:
        return tp.TypeAdapter(bool)

    @classmethod
    @cache
    def int_adapter(cls) -> tp.TypeAdapter[tp.StrictInt]:
        return tp.TypeAdapter(tp.StrictInt)

    @classmethod
    @cache
    def scalar_adapter(cls) -> tp.TypeAdapter[t.Scalar]:
        return tp.TypeAdapter(t.Scalar)

    @classmethod
    @cache
    def scalar_mapping_adapter(cls) -> tp.TypeAdapter[t.ScalarMapping]:
        return tp.TypeAdapter(t.ScalarMapping)

    @classmethod
    @cache
    def float_adapter(cls) -> tp.TypeAdapter[tp.StrictFloat]:
        return tp.TypeAdapter(tp.StrictFloat)

    @classmethod
    @cache
    def str_adapter(cls) -> tp.TypeAdapter[tp.StrictStr]:
        return tp.TypeAdapter(tp.StrictStr)

    @classmethod
    @cache
    def binary_content_adapter(cls) -> tp.TypeAdapter[tp.StrictBytes]:
        return tp.TypeAdapter(tp.StrictBytes)

    @classmethod
    @cache
    def str_mapping_adapter(cls) -> tp.TypeAdapter[t.StrMapping]:
        return tp.TypeAdapter(t.StrMapping)

    @classmethod
    @cache
    def header_mapping_adapter(cls) -> tp.TypeAdapter[t.HeaderMapping]:
        return tp.TypeAdapter(t.HeaderMapping)

    @classmethod
    @cache
    def str_dict_adapter(cls) -> tp.TypeAdapter[t.StrDict]:
        return tp.TypeAdapter(t.StrDict)

    @classmethod
    @cache
    def int_dict_adapter(cls) -> tp.TypeAdapter[t.IntDict]:
        return tp.TypeAdapter(t.IntDict)

    @classmethod
    @cache
    def hostname_str_adapter(cls) -> tp.TypeAdapter[ta.HostnameStr]:
        return tp.TypeAdapter(ta.HostnameStr)

    @classmethod
    @cache
    def port_number_adapter(cls) -> tp.TypeAdapter[ta.PortNumber]:
        return tp.TypeAdapter(ta.PortNumber)

    @classmethod
    @cache
    def str_sequence_adapter(cls) -> tp.TypeAdapter[t.StrSequence]:
        return tp.TypeAdapter(t.StrSequence)

    @classmethod
    @cache
    def strict_str_sequence_adapter(cls) -> tp.TypeAdapter[t.StrSequence]:
        return tp.TypeAdapter(t.StrSequence, config=tp.ConfigDict(strict=True))

    @classmethod
    @cache
    def str_or_bytes_adapter(cls) -> tp.TypeAdapter[tc.TextOrBinaryContent]:
        return tp.TypeAdapter(tc.TextOrBinaryContent)

    @classmethod
    @cache
    def enum_type_adapter(cls) -> tp.TypeAdapter[type[StrEnum]]:
        return tp.TypeAdapter(type[StrEnum])

    @classmethod
    @cache
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> tp.TypeAdapter[Mapping[str, t.Primitives]]:
        return tp.TypeAdapter(Mapping[str, t.Primitives])

    @classmethod
    @cache
    def structlog_processor_adapter(
        cls,
    ) -> tp.TypeAdapter[Callable[..., tp.JsonValue]]:
        return tp.TypeAdapter(Callable[..., tp.JsonValue])


__all__: list[str] = ["FlextTypesTypeAdapters"]
