"""Centralized FlextModelsPydantic.TypeAdapter cache for FLEXT typing aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, ClassVar

from flext_core import FlextTypesCore, FlextTypesServices, FlextTypingBase
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities.pydantic import FlextUtilitiesPydantic


class FlextTypesTypeAdapters:
    """Cached FlextModelsPydantic.TypeAdapter factories shared through the ``t`` facade."""

    FlextModelsPydantic.TypeAdapter = FlextModelsPydantic.TypeAdapter

    _str_sequence_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypingBase.StrSequence] | None
    ] = None
    _flat_container_list_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypingBase.FlatContainerList] | None
    ] = None
    _strict_string_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[
            Annotated[str, FlextUtilitiesPydantic.Field(strict=True)]
        ]
        | None
    ] = None
    _metadata_map_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[Mapping[str, FlextTypesServices.MetadataValue]]
        | None
    ] = None
    _flat_container_mapping_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypesServices.FlatContainerMapping] | None
    ] = None
    _tuple_container_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[tuple[FlextTypingBase.Container, ...]] | None
    ] = None
    _primitives_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypingBase.Primitives] | None
    ] = None
    _dict_str_metadata_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypingBase.ContainerMapping] | None
    ] = None
    _list_serializable_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[Sequence[FlextTypesCore.Serializable]] | None
    ] = None
    _tuple_serializable_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[tuple[FlextTypesCore.Serializable, ...]] | None
    ] = None
    _set_container_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[set[FlextTypingBase.Container]] | None
    ] = None
    _set_str_adapter: ClassVar[FlextModelsPydantic.TypeAdapter[set[str]] | None] = None
    _set_scalar_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[set[FlextTypingBase.Scalar]] | None
    ] = None
    _sortable_dict_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[
            Mapping[
                FlextTypesServices.SortableObjectType,
                FlextTypesCore.Serializable | None,
            ]
        ]
        | None
    ] = None
    _strict_json_list_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[Sequence[FlextTypesServices.StrictValue]] | None
    ] = None
    _bool_adapter: ClassVar[FlextModelsPydantic.TypeAdapter[bool] | None] = None
    _int_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypesCore.IntegerValue] | None
    ] = None
    _scalar_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypingBase.Scalar] | None
    ] = None
    _float_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypesCore.FloatValue] | None
    ] = None
    _str_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypesCore.TextValue] | None
    ] = None
    _str_or_bytes_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypesCore.TextOrBinaryContent] | None
    ] = None
    _enum_type_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[type[StrEnum]] | None
    ] = None
    _serializable_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[FlextTypesCore.Serializable] | None
    ] = None
    _primitive_metadata_mapping_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[Mapping[str, FlextTypingBase.Primitives]] | None
    ] = None
    _structlog_processor_adapter: ClassVar[
        FlextModelsPydantic.TypeAdapter[Callable[..., FlextTypingBase.Container]] | None
    ] = None

    @classmethod
    def metadata_map_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[
        Mapping[str, FlextTypesServices.MetadataValue]
    ]:
        if cls._metadata_map_adapter is None:
            cls._metadata_map_adapter = FlextModelsPydantic.TypeAdapter(
                Mapping[str, FlextTypesServices.MetadataValue],
            )
        return cls._metadata_map_adapter

    @classmethod
    def strict_string_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[
        Annotated[str, FlextUtilitiesPydantic.Field(strict=True)]
    ]:
        if cls._strict_string_adapter is None:
            cls._strict_string_adapter = FlextModelsPydantic.TypeAdapter(
                Annotated[str, FlextUtilitiesPydantic.Field(strict=True)]
            )
        return cls._strict_string_adapter

    @classmethod
    def flat_container_mapping_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[FlextTypesServices.FlatContainerMapping]:
        if cls._flat_container_mapping_adapter is None:
            cls._flat_container_mapping_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypesServices.FlatContainerMapping,
            )
        return cls._flat_container_mapping_adapter

    @classmethod
    def flat_container_list_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[FlextTypingBase.FlatContainerList]:
        if cls._flat_container_list_adapter is None:
            cls._flat_container_list_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypingBase.FlatContainerList,
            )
        return cls._flat_container_list_adapter

    @classmethod
    def tuple_container_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[tuple[FlextTypingBase.Container, ...]]:
        if cls._tuple_container_adapter is None:
            cls._tuple_container_adapter = FlextModelsPydantic.TypeAdapter(
                tuple[FlextTypingBase.Container, ...],
            )
        return cls._tuple_container_adapter

    @classmethod
    def primitives_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[FlextTypingBase.Primitives]:
        if cls._primitives_adapter is None:
            cls._primitives_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypingBase.Primitives
            )
        return cls._primitives_adapter

    @classmethod
    def dict_str_metadata_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[FlextTypingBase.ContainerMapping]:
        if cls._dict_str_metadata_adapter is None:
            cls._dict_str_metadata_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypingBase.ContainerMapping,
            )
        return cls._dict_str_metadata_adapter

    @classmethod
    def list_serializable_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[Sequence[FlextTypesCore.Serializable]]:
        if cls._list_serializable_adapter is None:
            cls._list_serializable_adapter = FlextModelsPydantic.TypeAdapter(
                Sequence[FlextTypesCore.Serializable],
            )
        return cls._list_serializable_adapter

    @classmethod
    def tuple_serializable_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[tuple[FlextTypesCore.Serializable, ...]]:
        if cls._tuple_serializable_adapter is None:
            cls._tuple_serializable_adapter = FlextModelsPydantic.TypeAdapter(
                tuple[FlextTypesCore.Serializable, ...],
            )
        return cls._tuple_serializable_adapter

    @classmethod
    def container_set_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[set[FlextTypingBase.Container]]:
        if cls._set_container_adapter is None:
            cls._set_container_adapter = FlextModelsPydantic.TypeAdapter(
                set[FlextTypingBase.Container]
            )
        return cls._set_container_adapter

    @classmethod
    def string_set_adapter(cls) -> FlextModelsPydantic.TypeAdapter[set[str]]:
        if cls._set_str_adapter is None:
            cls._set_str_adapter = FlextModelsPydantic.TypeAdapter(set[str])
        return cls._set_str_adapter

    @classmethod
    def scalar_set_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[set[FlextTypingBase.Scalar]]:
        if cls._set_scalar_adapter is None:
            cls._set_scalar_adapter = FlextModelsPydantic.TypeAdapter(
                set[FlextTypingBase.Scalar]
            )
        return cls._set_scalar_adapter

    @classmethod
    def sortable_dict_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[
        Mapping[
            FlextTypesServices.SortableObjectType,
            FlextTypesCore.Serializable | None,
        ]
    ]:
        if cls._sortable_dict_adapter is None:
            cls._sortable_dict_adapter = FlextModelsPydantic.TypeAdapter(
                Mapping[
                    FlextTypesServices.SortableObjectType,
                    FlextTypesCore.Serializable | None,
                ],
            )
        return cls._sortable_dict_adapter

    @classmethod
    def strict_json_list_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[Sequence[FlextTypesServices.StrictValue]]:
        if cls._strict_json_list_adapter is None:
            cls._strict_json_list_adapter = FlextModelsPydantic.TypeAdapter(
                Sequence[FlextTypesServices.StrictValue],
            )
        return cls._strict_json_list_adapter

    @classmethod
    def bool_adapter(cls) -> FlextModelsPydantic.TypeAdapter[bool]:
        if cls._bool_adapter is None:
            cls._bool_adapter = FlextModelsPydantic.TypeAdapter(bool)
        return cls._bool_adapter

    @classmethod
    def int_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[FlextTypesCore.IntegerValue]:
        if cls._int_adapter is None:
            cls._int_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypesCore.IntegerValue
            )
        return cls._int_adapter

    @classmethod
    def scalar_adapter(cls) -> FlextModelsPydantic.TypeAdapter[FlextTypingBase.Scalar]:
        if cls._scalar_adapter is None:
            cls._scalar_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypingBase.Scalar
            )
        return cls._scalar_adapter

    @classmethod
    def float_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[FlextTypesCore.FloatValue]:
        if cls._float_adapter is None:
            cls._float_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypesCore.FloatValue
            )
        return cls._float_adapter

    @classmethod
    def str_adapter(cls) -> FlextModelsPydantic.TypeAdapter[FlextTypesCore.TextValue]:
        if cls._str_adapter is None:
            cls._str_adapter = FlextModelsPydantic.TypeAdapter(FlextTypesCore.TextValue)
        return cls._str_adapter

    @classmethod
    def str_sequence_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[FlextTypingBase.StrSequence]:
        if cls._str_sequence_adapter is None:
            cls._str_sequence_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypingBase.StrSequence
            )
        return cls._str_sequence_adapter

    @classmethod
    def str_or_bytes_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[FlextTypesCore.TextOrBinaryContent]:
        if cls._str_or_bytes_adapter is None:
            cls._str_or_bytes_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypesCore.TextOrBinaryContent
            )
        return cls._str_or_bytes_adapter

    @classmethod
    def enum_type_adapter(cls) -> FlextModelsPydantic.TypeAdapter[type[StrEnum]]:
        if cls._enum_type_adapter is None:
            cls._enum_type_adapter = FlextModelsPydantic.TypeAdapter(type[StrEnum])
        return cls._enum_type_adapter

    @classmethod
    def serializable_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[FlextTypesCore.Serializable]:
        if cls._serializable_adapter is None:
            cls._serializable_adapter = FlextModelsPydantic.TypeAdapter(
                FlextTypesCore.Serializable
            )
        return cls._serializable_adapter

    @classmethod
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[Mapping[str, FlextTypingBase.Primitives]]:
        if cls._primitive_metadata_mapping_adapter is None:
            cls._primitive_metadata_mapping_adapter = FlextModelsPydantic.TypeAdapter(
                Mapping[str, FlextTypingBase.Primitives],
            )
        return cls._primitive_metadata_mapping_adapter

    @classmethod
    def structlog_processor_adapter(
        cls,
    ) -> FlextModelsPydantic.TypeAdapter[Callable[..., FlextTypingBase.Container]]:
        if cls._structlog_processor_adapter is None:
            cls._structlog_processor_adapter = FlextModelsPydantic.TypeAdapter(
                Callable[..., FlextTypingBase.Container],
            )
        return cls._structlog_processor_adapter


__all__: list[str] = ["FlextTypesTypeAdapters"]
