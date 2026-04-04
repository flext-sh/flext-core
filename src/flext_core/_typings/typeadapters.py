"""Centralized TypeAdapter cache for FLEXT typing aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, ClassVar

from pydantic import Field, TypeAdapter

from flext_core import FlextTypesCore, FlextTypesServices, FlextTypingBase


class FlextTypesTypeAdapters:
    """Cached TypeAdapter factories shared through the ``t`` facade."""

    _str_sequence_adapter: ClassVar[TypeAdapter[FlextTypingBase.StrSequence] | None] = (
        None
    )
    _flat_container_list_adapter: ClassVar[
        TypeAdapter[FlextTypingBase.FlatContainerList] | None
    ] = None
    _strict_string_adapter: ClassVar[
        TypeAdapter[Annotated[str, Field(strict=True)]] | None
    ] = None
    _metadata_map_adapter: ClassVar[
        TypeAdapter[Mapping[str, FlextTypesServices.MetadataValue]] | None
    ] = None
    _flat_container_mapping_adapter: ClassVar[
        TypeAdapter[FlextTypesServices.FlatContainerMapping] | None
    ] = None
    _tuple_container_adapter: ClassVar[
        TypeAdapter[tuple[FlextTypingBase.Container, ...]] | None
    ] = None
    _primitives_adapter: ClassVar[TypeAdapter[FlextTypingBase.Primitives] | None] = None
    _dict_str_metadata_adapter: ClassVar[
        TypeAdapter[FlextTypingBase.ContainerMapping] | None
    ] = None
    _list_serializable_adapter: ClassVar[
        TypeAdapter[Sequence[FlextTypesCore.Serializable]] | None
    ] = None
    _tuple_serializable_adapter: ClassVar[
        TypeAdapter[tuple[FlextTypesCore.Serializable, ...]] | None
    ] = None
    _set_container_adapter: ClassVar[
        TypeAdapter[set[FlextTypingBase.Container]] | None
    ] = None
    _set_str_adapter: ClassVar[TypeAdapter[set[str]] | None] = None
    _set_scalar_adapter: ClassVar[TypeAdapter[set[FlextTypingBase.Scalar]] | None] = (
        None
    )
    _sortable_dict_adapter: ClassVar[
        TypeAdapter[
            Mapping[
                FlextTypesServices.SortableObjectType,
                FlextTypesCore.Serializable | None,
            ]
        ]
        | None
    ] = None
    _strict_json_list_adapter: ClassVar[
        TypeAdapter[Sequence[FlextTypesServices.StrictValue]] | None
    ] = None
    _bool_adapter: ClassVar[TypeAdapter[bool] | None] = None
    _int_adapter: ClassVar[TypeAdapter[FlextTypesCore.IntegerValue] | None] = None
    _scalar_adapter: ClassVar[TypeAdapter[FlextTypingBase.Scalar] | None] = None
    _float_adapter: ClassVar[TypeAdapter[FlextTypesCore.FloatValue] | None] = None
    _str_adapter: ClassVar[TypeAdapter[FlextTypesCore.TextValue] | None] = None
    _str_or_bytes_adapter: ClassVar[
        TypeAdapter[FlextTypesCore.TextOrBinaryContent] | None
    ] = None
    _enum_type_adapter: ClassVar[TypeAdapter[type[StrEnum]] | None] = None
    _serializable_adapter: ClassVar[TypeAdapter[FlextTypesCore.Serializable] | None] = (
        None
    )
    _primitive_metadata_mapping_adapter: ClassVar[
        TypeAdapter[Mapping[str, FlextTypingBase.Primitives]] | None
    ] = None
    _structlog_processor_adapter: ClassVar[
        TypeAdapter[Callable[..., FlextTypingBase.Container]] | None
    ] = None

    @classmethod
    def metadata_map_adapter(
        cls,
    ) -> TypeAdapter[Mapping[str, FlextTypesServices.MetadataValue]]:
        if cls._metadata_map_adapter is None:
            cls._metadata_map_adapter = TypeAdapter(
                Mapping[str, FlextTypesServices.MetadataValue],
            )
        return cls._metadata_map_adapter

    @classmethod
    def strict_string_adapter(
        cls,
    ) -> TypeAdapter[Annotated[str, Field(strict=True)]]:
        if cls._strict_string_adapter is None:
            cls._strict_string_adapter = TypeAdapter(Annotated[str, Field(strict=True)])
        return cls._strict_string_adapter

    @classmethod
    def flat_container_mapping_adapter(
        cls,
    ) -> TypeAdapter[FlextTypesServices.FlatContainerMapping]:
        if cls._flat_container_mapping_adapter is None:
            cls._flat_container_mapping_adapter = TypeAdapter(
                FlextTypesServices.FlatContainerMapping
            )
        return cls._flat_container_mapping_adapter

    @classmethod
    def flat_container_list_adapter(
        cls,
    ) -> TypeAdapter[FlextTypingBase.FlatContainerList]:
        if cls._flat_container_list_adapter is None:
            cls._flat_container_list_adapter = TypeAdapter(
                FlextTypingBase.FlatContainerList
            )
        return cls._flat_container_list_adapter

    @classmethod
    def tuple_container_adapter(
        cls,
    ) -> TypeAdapter[tuple[FlextTypingBase.Container, ...]]:
        if cls._tuple_container_adapter is None:
            cls._tuple_container_adapter = TypeAdapter(
                tuple[FlextTypingBase.Container, ...]
            )
        return cls._tuple_container_adapter

    @classmethod
    def primitives_adapter(cls) -> TypeAdapter[FlextTypingBase.Primitives]:
        if cls._primitives_adapter is None:
            cls._primitives_adapter = TypeAdapter(FlextTypingBase.Primitives)
        return cls._primitives_adapter

    @classmethod
    def dict_str_metadata_adapter(
        cls,
    ) -> TypeAdapter[FlextTypingBase.ContainerMapping]:
        if cls._dict_str_metadata_adapter is None:
            cls._dict_str_metadata_adapter = TypeAdapter(
                FlextTypingBase.ContainerMapping
            )
        return cls._dict_str_metadata_adapter

    @classmethod
    def list_serializable_adapter(
        cls,
    ) -> TypeAdapter[Sequence[FlextTypesCore.Serializable]]:
        if cls._list_serializable_adapter is None:
            cls._list_serializable_adapter = TypeAdapter(
                Sequence[FlextTypesCore.Serializable]
            )
        return cls._list_serializable_adapter

    @classmethod
    def tuple_serializable_adapter(
        cls,
    ) -> TypeAdapter[tuple[FlextTypesCore.Serializable, ...]]:
        if cls._tuple_serializable_adapter is None:
            cls._tuple_serializable_adapter = TypeAdapter(
                tuple[FlextTypesCore.Serializable, ...]
            )
        return cls._tuple_serializable_adapter

    @classmethod
    def set_container_adapter(
        cls,
    ) -> TypeAdapter[set[FlextTypingBase.Container]]:
        if cls._set_container_adapter is None:
            cls._set_container_adapter = TypeAdapter(set[FlextTypingBase.Container])
        return cls._set_container_adapter

    @classmethod
    def set_str_adapter(cls) -> TypeAdapter[set[str]]:
        if cls._set_str_adapter is None:
            cls._set_str_adapter = TypeAdapter(set[str])
        return cls._set_str_adapter

    @classmethod
    def set_scalar_adapter(
        cls,
    ) -> TypeAdapter[set[FlextTypingBase.Scalar]]:
        if cls._set_scalar_adapter is None:
            cls._set_scalar_adapter = TypeAdapter(set[FlextTypingBase.Scalar])
        return cls._set_scalar_adapter

    @classmethod
    def sortable_dict_adapter(
        cls,
    ) -> TypeAdapter[
        Mapping[
            FlextTypesServices.SortableObjectType, FlextTypesCore.Serializable | None
        ]
    ]:
        if cls._sortable_dict_adapter is None:
            cls._sortable_dict_adapter = TypeAdapter(
                Mapping[
                    FlextTypesServices.SortableObjectType,
                    FlextTypesCore.Serializable | None,
                ],
            )
        return cls._sortable_dict_adapter

    @classmethod
    def strict_json_list_adapter(
        cls,
    ) -> TypeAdapter[Sequence[FlextTypesServices.StrictValue]]:
        if cls._strict_json_list_adapter is None:
            cls._strict_json_list_adapter = TypeAdapter(
                Sequence[FlextTypesServices.StrictValue]
            )
        return cls._strict_json_list_adapter

    @classmethod
    def bool_adapter(cls) -> TypeAdapter[bool]:
        if cls._bool_adapter is None:
            cls._bool_adapter = TypeAdapter(bool)
        return cls._bool_adapter

    @classmethod
    def int_adapter(cls) -> TypeAdapter[FlextTypesCore.IntegerValue]:
        if cls._int_adapter is None:
            cls._int_adapter = TypeAdapter(FlextTypesCore.IntegerValue)
        return cls._int_adapter

    @classmethod
    def scalar_adapter(cls) -> TypeAdapter[FlextTypingBase.Scalar]:
        if cls._scalar_adapter is None:
            cls._scalar_adapter = TypeAdapter(FlextTypingBase.Scalar)
        return cls._scalar_adapter

    @classmethod
    def float_adapter(cls) -> TypeAdapter[FlextTypesCore.FloatValue]:
        if cls._float_adapter is None:
            cls._float_adapter = TypeAdapter(FlextTypesCore.FloatValue)
        return cls._float_adapter

    @classmethod
    def str_adapter(cls) -> TypeAdapter[FlextTypesCore.TextValue]:
        if cls._str_adapter is None:
            cls._str_adapter = TypeAdapter(FlextTypesCore.TextValue)
        return cls._str_adapter

    @classmethod
    def str_sequence_adapter(cls) -> TypeAdapter[FlextTypingBase.StrSequence]:
        if cls._str_sequence_adapter is None:
            cls._str_sequence_adapter = TypeAdapter(FlextTypingBase.StrSequence)
        return cls._str_sequence_adapter

    @classmethod
    def str_or_bytes_adapter(
        cls,
    ) -> TypeAdapter[FlextTypesCore.TextOrBinaryContent]:
        if cls._str_or_bytes_adapter is None:
            cls._str_or_bytes_adapter = TypeAdapter(FlextTypesCore.TextOrBinaryContent)
        return cls._str_or_bytes_adapter

    @classmethod
    def enum_type_adapter(cls) -> TypeAdapter[type[StrEnum]]:
        if cls._enum_type_adapter is None:
            cls._enum_type_adapter = TypeAdapter(type[StrEnum])
        return cls._enum_type_adapter

    @classmethod
    def serializable_adapter(cls) -> TypeAdapter[FlextTypesCore.Serializable]:
        if cls._serializable_adapter is None:
            cls._serializable_adapter = TypeAdapter(FlextTypesCore.Serializable)
        return cls._serializable_adapter

    @classmethod
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> TypeAdapter[Mapping[str, FlextTypingBase.Primitives]]:
        if cls._primitive_metadata_mapping_adapter is None:
            cls._primitive_metadata_mapping_adapter = TypeAdapter(
                Mapping[str, FlextTypingBase.Primitives],
            )
        return cls._primitive_metadata_mapping_adapter

    @classmethod
    def structlog_processor_adapter(
        cls,
    ) -> TypeAdapter[Callable[..., FlextTypingBase.Container]]:
        if cls._structlog_processor_adapter is None:
            cls._structlog_processor_adapter = TypeAdapter(
                Callable[..., FlextTypingBase.Container],
            )
        return cls._structlog_processor_adapter


__all__ = ["FlextTypesTypeAdapters"]
