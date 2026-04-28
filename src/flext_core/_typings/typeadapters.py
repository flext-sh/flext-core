"""Centralized mp.TypeAdapter cache for FLEXT typing aliases.

Each adapter is a `@classmethod @cache` that returns the canonical
``mp.TypeAdapter[...]``. The ``functools.cache`` wrapper memoizes per-cls,
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
    Sequence,
)
from enum import StrEnum
from functools import cache
from typing import no_type_check

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.annotateds import FlextTypesAnnotateds as ta
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.core import FlextTypesCore as tc
from flext_core._typings.pydantic import FlextTypesPydantic as tp

type SortableObjectType = str | int | float
type ConfigurationMapping = Mapping[str, t.Scalar]
StrictValue = (
    t.Scalar | ConfigurationMapping | t.JsonList | tuple[tp.JsonValue | t.Scalar, ...]
)


@no_type_check
class FlextTypesTypeAdapters:
    """Cached mp.TypeAdapter factories shared through the ``t`` facade."""

    @classmethod
    @cache
    def metadata_map_adapter(cls) -> mp.TypeAdapter[Mapping[str, tp.JsonValue]]:
        return mp.TypeAdapter(Mapping[str, tp.JsonValue])

    @classmethod
    @cache
    def json_value_adapter(cls) -> mp.TypeAdapter[tp.JsonValue]:
        return mp.TypeAdapter(tp.JsonValue)

    @classmethod
    @cache
    def json_mapping_adapter(cls) -> mp.TypeAdapter[t.JsonMapping]:
        return mp.TypeAdapter(t.JsonMapping)

    @classmethod
    @cache
    def json_list_adapter(cls) -> mp.TypeAdapter[t.JsonList]:
        return mp.TypeAdapter(t.JsonList)

    @classmethod
    @cache
    def primitives_adapter(cls) -> mp.TypeAdapter[t.Primitives]:
        return mp.TypeAdapter(t.Primitives)

    @classmethod
    @cache
    def container_set_adapter(cls) -> mp.TypeAdapter[set[tp.JsonValue]]:
        return mp.TypeAdapter(set[tp.JsonValue])

    @classmethod
    @cache
    def string_set_adapter(cls) -> mp.TypeAdapter[set[str]]:
        return mp.TypeAdapter(set[str])

    @classmethod
    @cache
    def scalar_set_adapter(cls) -> mp.TypeAdapter[set[t.Scalar]]:
        return mp.TypeAdapter(set[t.Scalar])

    @classmethod
    @cache
    def sortable_dict_adapter(
        cls,
    ) -> mp.TypeAdapter[Mapping[SortableObjectType, tp.JsonValue | None]]:
        return mp.TypeAdapter(Mapping[SortableObjectType, tp.JsonValue | None])

    @classmethod
    @cache
    def strict_json_list_adapter(cls) -> mp.TypeAdapter[Sequence[StrictValue]]:
        return mp.TypeAdapter(Sequence[StrictValue])

    @classmethod
    @cache
    def bool_adapter(cls) -> mp.TypeAdapter[bool]:
        return mp.TypeAdapter(bool)

    @classmethod
    @cache
    def int_adapter(cls) -> mp.TypeAdapter[tp.StrictInt]:
        return mp.TypeAdapter(tp.StrictInt)

    @classmethod
    @cache
    def scalar_adapter(cls) -> mp.TypeAdapter[t.Scalar]:
        return mp.TypeAdapter(t.Scalar)

    @classmethod
    @cache
    def float_adapter(cls) -> mp.TypeAdapter[tp.StrictFloat]:
        return mp.TypeAdapter(tp.StrictFloat)

    @classmethod
    @cache
    def str_adapter(cls) -> mp.TypeAdapter[tp.StrictStr]:
        return mp.TypeAdapter(tp.StrictStr)

    @classmethod
    @cache
    def binary_content_adapter(cls) -> mp.TypeAdapter[tp.StrictBytes]:
        return mp.TypeAdapter(tp.StrictBytes)

    @classmethod
    @cache
    def str_mapping_adapter(cls) -> mp.TypeAdapter[t.StrMapping]:
        return mp.TypeAdapter(t.StrMapping)

    @classmethod
    @cache
    def hostname_str_adapter(cls) -> mp.TypeAdapter[ta.HostnameStr]:
        return mp.TypeAdapter(ta.HostnameStr)

    @classmethod
    @cache
    def port_number_adapter(cls) -> mp.TypeAdapter[ta.PortNumber]:
        return mp.TypeAdapter(ta.PortNumber)

    @classmethod
    @cache
    def str_sequence_adapter(cls) -> mp.TypeAdapter[t.StrSequence]:
        return mp.TypeAdapter(t.StrSequence)

    @classmethod
    @cache
    def str_or_bytes_adapter(cls) -> mp.TypeAdapter[tc.TextOrBinaryContent]:
        return mp.TypeAdapter(tc.TextOrBinaryContent)

    @classmethod
    @cache
    def enum_type_adapter(cls) -> mp.TypeAdapter[type[StrEnum]]:
        return mp.TypeAdapter(type[StrEnum])

    @classmethod
    @cache
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> mp.TypeAdapter[Mapping[str, t.Primitives]]:
        return mp.TypeAdapter(Mapping[str, t.Primitives])

    @classmethod
    @cache
    def structlog_processor_adapter(
        cls,
    ) -> mp.TypeAdapter[Callable[..., tp.JsonValue]]:
        return mp.TypeAdapter(Callable[..., tp.JsonValue])


__all__: list[str] = ["FlextTypesTypeAdapters"]
