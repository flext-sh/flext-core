"""JSON-shape cached TypeAdapter factories for FLEXT typing aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache

from typing import Annotated

from pydantic import ConfigDict, TypeAdapter

from flext_core._typings.base import FlextTypingBase
from flext_core._typings.pydantic import FlextTypesPydantic
from flext_core._typings.services import FlextTypesServices


class FlextTypesTypeAdapterJson:
    """JSON-shape cached TypeAdapter factories.

    Shared through the ``FlextTypingBase`` facade.
    """

    @classmethod
    @cache
    def metadata_map_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[Mapping[str, FlextTypesPydantic.JsonValue]]:
        return TypeAdapter(Mapping[str, FlextTypesPydantic.JsonValue])

    @classmethod
    @cache
    def json_value_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.JsonValue]:
        return TypeAdapter(FlextTypesPydantic.JsonValue)

    @classmethod
    @cache
    def json_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonMapping]:
        return TypeAdapter(Annotated[FlextTypingBase.JsonMapping, None])

    @classmethod
    @cache
    def strict_json_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonMapping]:
        return TypeAdapter(Annotated[FlextTypingBase.JsonMapping, None], config=ConfigDict(strict=True))

    @classmethod
    @cache
    def json_dict_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonDict]:
        return TypeAdapter(Annotated[FlextTypingBase.JsonDict, None])

    @classmethod
    @cache
    def json_dict_sequence_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[
        FlextTypingBase.SequenceOf[FlextTypingBase.JsonDict]
    ]:
        return TypeAdapter(Annotated[FlextTypingBase.SequenceOf[FlextTypingBase.JsonDict], None])

    @classmethod
    @cache
    def json_mapping_sequence_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[
        FlextTypingBase.SequenceOf[FlextTypingBase.JsonMapping]
    ]:
        return TypeAdapter(Annotated[FlextTypingBase.SequenceOf[FlextTypingBase.JsonMapping], None])

    @classmethod
    @cache
    def json_mapping_by_str_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[
        FlextTypingBase.MappingKV[str, FlextTypingBase.JsonMapping]
    ]:
        return TypeAdapter(Annotated[FlextTypingBase.MappingKV[str, FlextTypingBase.JsonMapping], None])

    @classmethod
    @cache
    def json_list_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonList]:
        return TypeAdapter(Annotated[FlextTypingBase.JsonList, None])

    @classmethod
    @cache
    def strict_json_list_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonList]:
        return TypeAdapter(Annotated[FlextTypingBase.JsonList, None], config=ConfigDict(strict=True))

    @classmethod
    @cache
    def primitives_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.Primitives]:
        return TypeAdapter(Annotated[FlextTypingBase.Primitives, None])

    @classmethod
    @cache
    def container_set_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[set[FlextTypesPydantic.JsonValue]]:
        return TypeAdapter(set[FlextTypesPydantic.JsonValue])

    @classmethod
    @cache
    def string_set_adapter(cls) -> FlextTypesPydantic.TypeAdapter[set[str]]:
        return TypeAdapter(set[str])

    @classmethod
    @cache
    def scalar_set_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[set[FlextTypingBase.Scalar]]:
        return TypeAdapter(set[FlextTypingBase.Scalar])

    @classmethod
    @cache
    def sortable_dict_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[
        Mapping[
            FlextTypesServices.SortableObjectType,
            FlextTypesPydantic.JsonValue | None,
        ]
    ]:
        return TypeAdapter(
            Mapping[
                FlextTypesServices.SortableObjectType,
                FlextTypesPydantic.JsonValue | None,
            ],
        )


__all__: list[str] = ["FlextTypesTypeAdapterJson"]
