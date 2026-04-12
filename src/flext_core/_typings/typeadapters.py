"""Centralized m.TypeAdapter cache for FLEXT typing aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, ClassVar

from flext_core import FlextTypesCore, FlextTypesServices, FlextTypingBase

if TYPE_CHECKING:
    from flext_core import m, u


class FlextTypesTypeAdapters:
    """Cached m.TypeAdapter factories shared through the ``t`` facade."""

    _str_sequence_adapter: ClassVar[
        m.TypeAdapter[FlextTypingBase.StrSequence] | None
    ] = None
    _flat_container_list_adapter: ClassVar[
        m.TypeAdapter[FlextTypingBase.FlatContainerList] | None
    ] = None
    _strict_string_adapter: ClassVar[
        m.TypeAdapter[Annotated[str, u.Field(strict=True)]] | None
    ] = None
    _metadata_map_adapter: ClassVar[
        m.TypeAdapter[Mapping[str, FlextTypesServices.MetadataValue]] | None
    ] = None
    _flat_container_mapping_adapter: ClassVar[
        m.TypeAdapter[FlextTypesServices.FlatContainerMapping] | None
    ] = None
    _tuple_container_adapter: ClassVar[
        m.TypeAdapter[tuple[FlextTypingBase.Container, ...]] | None
    ] = None
    _primitives_adapter: ClassVar[m.TypeAdapter[FlextTypingBase.Primitives] | None] = (
        None
    )
    _dict_str_metadata_adapter: ClassVar[
        m.TypeAdapter[FlextTypingBase.ContainerMapping] | None
    ] = None
    _list_serializable_adapter: ClassVar[
        m.TypeAdapter[Sequence[FlextTypesCore.Serializable]] | None
    ] = None
    _tuple_serializable_adapter: ClassVar[
        m.TypeAdapter[tuple[FlextTypesCore.Serializable, ...]] | None
    ] = None
    _set_container_adapter: ClassVar[
        m.TypeAdapter[set[FlextTypingBase.Container]] | None
    ] = None
    _set_str_adapter: ClassVar[m.TypeAdapter[set[str]] | None] = None
    _set_scalar_adapter: ClassVar[m.TypeAdapter[set[FlextTypingBase.Scalar]] | None] = (
        None
    )
    _sortable_dict_adapter: ClassVar[
        m.TypeAdapter[
            Mapping[
                FlextTypesServices.SortableObjectType,
                FlextTypesCore.Serializable | None,
            ]
        ]
        | None
    ] = None
    _strict_json_list_adapter: ClassVar[
        m.TypeAdapter[Sequence[FlextTypesServices.StrictValue]] | None
    ] = None
    _bool_adapter: ClassVar[m.TypeAdapter[bool] | None] = None
    _int_adapter: ClassVar[m.TypeAdapter[FlextTypesCore.IntegerValue] | None] = None
    _scalar_adapter: ClassVar[m.TypeAdapter[FlextTypingBase.Scalar] | None] = None
    _float_adapter: ClassVar[m.TypeAdapter[FlextTypesCore.FloatValue] | None] = None
    _str_adapter: ClassVar[m.TypeAdapter[FlextTypesCore.TextValue] | None] = None
    _str_or_bytes_adapter: ClassVar[
        m.TypeAdapter[FlextTypesCore.TextOrBinaryContent] | None
    ] = None
    _enum_type_adapter: ClassVar[m.TypeAdapter[type[StrEnum]] | None] = None
    _serializable_adapter: ClassVar[
        m.TypeAdapter[FlextTypesCore.Serializable] | None
    ] = None
    _primitive_metadata_mapping_adapter: ClassVar[
        m.TypeAdapter[Mapping[str, FlextTypingBase.Primitives]] | None
    ] = None
    _structlog_processor_adapter: ClassVar[
        m.TypeAdapter[Callable[..., FlextTypingBase.Container]] | None
    ] = None

    @classmethod
    def metadata_map_adapter(
        cls,
    ) -> m.TypeAdapter[Mapping[str, FlextTypesServices.MetadataValue]]:
        if cls._metadata_map_adapter is None:
            cls._metadata_map_adapter = m.TypeAdapter(
                Mapping[str, FlextTypesServices.MetadataValue],
            )
        return cls._metadata_map_adapter

    @classmethod
    def strict_string_adapter(
        cls,
    ) -> m.TypeAdapter[Annotated[str, u.Field(strict=True)]]:
        if cls._strict_string_adapter is None:
            cls._strict_string_adapter = m.TypeAdapter(
                Annotated[str, u.Field(strict=True)]
            )
        return cls._strict_string_adapter

    @classmethod
    def flat_container_mapping_adapter(
        cls,
    ) -> m.TypeAdapter[FlextTypesServices.FlatContainerMapping]:
        if cls._flat_container_mapping_adapter is None:
            cls._flat_container_mapping_adapter = m.TypeAdapter(
                FlextTypesServices.FlatContainerMapping,
            )
        return cls._flat_container_mapping_adapter

    @classmethod
    def flat_container_list_adapter(
        cls,
    ) -> m.TypeAdapter[FlextTypingBase.FlatContainerList]:
        if cls._flat_container_list_adapter is None:
            cls._flat_container_list_adapter = m.TypeAdapter(
                FlextTypingBase.FlatContainerList,
            )
        return cls._flat_container_list_adapter

    @classmethod
    def tuple_container_adapter(
        cls,
    ) -> m.TypeAdapter[tuple[FlextTypingBase.Container, ...]]:
        if cls._tuple_container_adapter is None:
            cls._tuple_container_adapter = m.TypeAdapter(
                tuple[FlextTypingBase.Container, ...],
            )
        return cls._tuple_container_adapter

    @classmethod
    def primitives_adapter(
        cls,
    ) -> m.TypeAdapter[FlextTypingBase.Primitives]:
        if cls._primitives_adapter is None:
            cls._primitives_adapter = m.TypeAdapter(FlextTypingBase.Primitives)
        return cls._primitives_adapter

    @classmethod
    def dict_str_metadata_adapter(
        cls,
    ) -> m.TypeAdapter[FlextTypingBase.ContainerMapping]:
        if cls._dict_str_metadata_adapter is None:
            cls._dict_str_metadata_adapter = m.TypeAdapter(
                FlextTypingBase.ContainerMapping,
            )
        return cls._dict_str_metadata_adapter

    @classmethod
    def list_serializable_adapter(
        cls,
    ) -> m.TypeAdapter[Sequence[FlextTypesCore.Serializable]]:
        if cls._list_serializable_adapter is None:
            cls._list_serializable_adapter = m.TypeAdapter(
                Sequence[FlextTypesCore.Serializable],
            )
        return cls._list_serializable_adapter

    @classmethod
    def tuple_serializable_adapter(
        cls,
    ) -> m.TypeAdapter[tuple[FlextTypesCore.Serializable, ...]]:
        if cls._tuple_serializable_adapter is None:
            cls._tuple_serializable_adapter = m.TypeAdapter(
                tuple[FlextTypesCore.Serializable, ...],
            )
        return cls._tuple_serializable_adapter

    @classmethod
    def container_set_adapter(
        cls,
    ) -> m.TypeAdapter[set[FlextTypingBase.Container]]:
        if cls._set_container_adapter is None:
            cls._set_container_adapter = m.TypeAdapter(set[FlextTypingBase.Container])
        return cls._set_container_adapter

    @classmethod
    def string_set_adapter(cls) -> m.TypeAdapter[set[str]]:
        if cls._set_str_adapter is None:
            cls._set_str_adapter = m.TypeAdapter(set[str])
        return cls._set_str_adapter

    @classmethod
    def scalar_set_adapter(
        cls,
    ) -> m.TypeAdapter[set[FlextTypingBase.Scalar]]:
        if cls._set_scalar_adapter is None:
            cls._set_scalar_adapter = m.TypeAdapter(set[FlextTypingBase.Scalar])
        return cls._set_scalar_adapter

    @classmethod
    def sortable_dict_adapter(
        cls,
    ) -> m.TypeAdapter[
        Mapping[
            FlextTypesServices.SortableObjectType,
            FlextTypesCore.Serializable | None,
        ]
    ]:
        if cls._sortable_dict_adapter is None:
            cls._sortable_dict_adapter = m.TypeAdapter(
                Mapping[
                    FlextTypesServices.SortableObjectType,
                    FlextTypesCore.Serializable | None,
                ],
            )
        return cls._sortable_dict_adapter

    @classmethod
    def strict_json_list_adapter(
        cls,
    ) -> m.TypeAdapter[Sequence[FlextTypesServices.StrictValue]]:
        if cls._strict_json_list_adapter is None:
            cls._strict_json_list_adapter = m.TypeAdapter(
                Sequence[FlextTypesServices.StrictValue],
            )
        return cls._strict_json_list_adapter

    @classmethod
    def bool_adapter(cls) -> m.TypeAdapter[bool]:
        if cls._bool_adapter is None:
            cls._bool_adapter = m.TypeAdapter(bool)
        return cls._bool_adapter

    @classmethod
    def int_adapter(
        cls,
    ) -> m.TypeAdapter[FlextTypesCore.IntegerValue]:
        if cls._int_adapter is None:
            cls._int_adapter = m.TypeAdapter(FlextTypesCore.IntegerValue)
        return cls._int_adapter

    @classmethod
    def scalar_adapter(cls) -> m.TypeAdapter[FlextTypingBase.Scalar]:
        if cls._scalar_adapter is None:
            cls._scalar_adapter = m.TypeAdapter(FlextTypingBase.Scalar)
        return cls._scalar_adapter

    @classmethod
    def float_adapter(
        cls,
    ) -> m.TypeAdapter[FlextTypesCore.FloatValue]:
        if cls._float_adapter is None:
            cls._float_adapter = m.TypeAdapter(FlextTypesCore.FloatValue)
        return cls._float_adapter

    @classmethod
    def str_adapter(cls) -> m.TypeAdapter[FlextTypesCore.TextValue]:
        if cls._str_adapter is None:
            cls._str_adapter = m.TypeAdapter(FlextTypesCore.TextValue)
        return cls._str_adapter

    @classmethod
    def str_sequence_adapter(
        cls,
    ) -> m.TypeAdapter[FlextTypingBase.StrSequence]:
        if cls._str_sequence_adapter is None:
            cls._str_sequence_adapter = m.TypeAdapter(FlextTypingBase.StrSequence)
        return cls._str_sequence_adapter

    @classmethod
    def str_or_bytes_adapter(
        cls,
    ) -> m.TypeAdapter[FlextTypesCore.TextOrBinaryContent]:
        if cls._str_or_bytes_adapter is None:
            cls._str_or_bytes_adapter = m.TypeAdapter(
                FlextTypesCore.TextOrBinaryContent
            )
        return cls._str_or_bytes_adapter

    @classmethod
    def enum_type_adapter(cls) -> m.TypeAdapter[type[StrEnum]]:
        if cls._enum_type_adapter is None:
            cls._enum_type_adapter = m.TypeAdapter(type[StrEnum])
        return cls._enum_type_adapter

    @classmethod
    def serializable_adapter(
        cls,
    ) -> m.TypeAdapter[FlextTypesCore.Serializable]:
        if cls._serializable_adapter is None:
            cls._serializable_adapter = m.TypeAdapter(FlextTypesCore.Serializable)
        return cls._serializable_adapter

    @classmethod
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> m.TypeAdapter[Mapping[str, FlextTypingBase.Primitives]]:
        if cls._primitive_metadata_mapping_adapter is None:
            cls._primitive_metadata_mapping_adapter = m.TypeAdapter(
                Mapping[str, FlextTypingBase.Primitives],
            )
        return cls._primitive_metadata_mapping_adapter

    @classmethod
    def structlog_processor_adapter(
        cls,
    ) -> m.TypeAdapter[Callable[..., FlextTypingBase.Container]]:
        if cls._structlog_processor_adapter is None:
            cls._structlog_processor_adapter = m.TypeAdapter(
                Callable[..., FlextTypingBase.Container],
            )
        return cls._structlog_processor_adapter


__all__: list[str] = ["FlextTypesTypeAdapters"]
