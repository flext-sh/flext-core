"""JSON-shape cached TypeAdapter factories for the FlextUtilities facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache

from pydantic import ConfigDict
from pydantic import TypeAdapter as PydanticTypeAdapter
from typing_extensions import TypeForm

from flext_core._typings.base import FlextTypingBase
from flext_core._typings.pydantic import FlextTypesPydantic
from flext_core._typings.services import FlextTypesServices



class FlextUtilitiesTypingAdapterJson:
    """JSON-shape cached TypeAdapter factories.

    Shared through the ``FlextUtilities`` facade.
    """

    @classmethod
    @cache
    def metadata_map_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[Mapping[str, FlextTypesPydantic.JsonValue]]:
        return cls.type_adapter(Mapping[str, FlextTypesPydantic.JsonValue])

    @classmethod
    @cache
    def json_value_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypesPydantic.JsonValue]:
        return cls.type_adapter(TypeForm(FlextTypesPydantic.JsonValue))

    @classmethod
    @cache
    def json_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonMapping]:
        return cls.type_adapter(TypeForm(FlextTypingBase.JsonMapping))

    @classmethod
    @cache
    def strict_json_mapping_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonMapping]:
        return cls.type_adapter(TypeForm(
            TypeForm(FlextTypingBase.JsonMapping)), config=ConfigDict(strict=True)
        )

    @classmethod
    @cache
    def json_dict_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonDict]:
        return cls.type_adapter(TypeForm(FlextTypingBase.JsonDict))

    @classmethod
    @cache
    def json_dict_sequence_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[
        FlextTypingBase.SequenceOf[FlextTypingBase.JsonDict]
    ]:
        return cls.type_adapter(TypeForm(
            TypeForm(FlextTypingBase.SequenceOf[FlextTypingBase.JsonDict]))
        )

    @classmethod
    @cache
    def json_mapping_sequence_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[
        FlextTypingBase.SequenceOf[FlextTypingBase.JsonMapping]
    ]:
        return cls.type_adapter(TypeForm(
            TypeForm(FlextTypingBase.SequenceOf[FlextTypingBase.JsonMapping]))
        )

    @classmethod
    @cache
    def json_mapping_by_str_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[
        FlextTypingBase.MappingKV[str, FlextTypingBase.JsonMapping]
    ]:
        return cls.type_adapter(TypeForm(
            TypeForm(FlextTypingBase.MappingKV[str, FlextTypingBase.JsonMapping]))
        )

    @classmethod
    @cache
    def json_list_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonList]:
        return cls.type_adapter(TypeForm(FlextTypingBase.JsonList))

    @classmethod
    @cache
    def strict_json_list_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.JsonList]:
        return cls.type_adapter(TypeForm(
            TypeForm(FlextTypingBase.JsonList)), config=ConfigDict(strict=True)
        )

    @classmethod
    @cache
    def primitives_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[FlextTypingBase.Primitives]:
        return cls.type_adapter(TypeForm(FlextTypingBase.Primitives))

    @classmethod
    @cache
    def container_set_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[set[FlextTypesPydantic.JsonValue]]:
        return cls.type_adapter(set[FlextTypesPydantic.JsonValue])

    @classmethod
    @cache
    def string_set_adapter(cls) -> FlextTypesPydantic.TypeAdapter[set[str]]:
        return cls.type_adapter(set[str])

    @classmethod
    @cache
    def scalar_set_adapter(
        cls,
    ) -> FlextTypesPydantic.TypeAdapter[set[FlextTypingBase.Scalar]]:
        return cls.type_adapter(TypeForm(set[FlextTypingBase.Scalar]))

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
        return cls.type_adapter(TypeForm(
            Mapping[
                FlextTypesServices.SortableObjectType,
                FlextTypesPydantic.JsonValue | None,
            ],
        ))


__all__: list[str] = ["FlextUtilitiesTypingAdapterJson"]
