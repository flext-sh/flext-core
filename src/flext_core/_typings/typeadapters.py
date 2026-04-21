"""Centralized mp.TypeAdapter cache for FLEXT typing aliases.

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
from typing import ClassVar

from flext_core import (
    FlextModelsPydantic as mp,
    FlextTypesAnnotateds as ta,
    FlextTypesCore as tc,
    FlextTypesServices as ts,
    FlextTypingBase as t,
)


class FlextTypesTypeAdapters:
    """Cached mp.TypeAdapter factories shared through the ``t`` facade."""

    _str_sequence_adapter: ClassVar[mp.TypeAdapter[t.StrSequence] | None] = None
    _flat_container_list_adapter: ClassVar[
        mp.TypeAdapter[t.FlatContainerList] | None
    ] = None
    _metadata_map_adapter: ClassVar[
        mp.TypeAdapter[Mapping[str, ts.MetadataValue]] | None
    ] = None
    _container_mapping_adapter: ClassVar[
        mp.TypeAdapter[Mapping[str, t.Container]] | None
    ] = None
    _container_mapping_sequence_adapter: ClassVar[
        mp.TypeAdapter[Sequence[Mapping[str, t.Container]]] | None
    ] = None
    _flat_container_mapping_adapter: ClassVar[
        mp.TypeAdapter[t.FlatContainerMapping] | None
    ] = None
    _tuple_container_adapter: ClassVar[
        mp.TypeAdapter[tuple[t.Container, ...]] | None
    ] = None
    _primitives_adapter: ClassVar[mp.TypeAdapter[t.Primitives] | None] = None
    _list_serializable_adapter: ClassVar[
        mp.TypeAdapter[Sequence[tc.Serializable]] | None
    ] = None
    _tuple_serializable_adapter: ClassVar[
        mp.TypeAdapter[tuple[tc.Serializable, ...]] | None
    ] = None
    _set_container_adapter: ClassVar[mp.TypeAdapter[set[t.Container]] | None] = None
    _set_str_adapter: ClassVar[mp.TypeAdapter[set[str]] | None] = None
    _set_scalar_adapter: ClassVar[mp.TypeAdapter[set[t.Scalar]] | None] = None
    _sortable_dict_adapter: ClassVar[
        mp.TypeAdapter[
            Mapping[
                ts.SortableObjectType,
                tc.Serializable | None,
            ]
        ]
        | None
    ] = None
    _strict_json_list_adapter: ClassVar[
        mp.TypeAdapter[Sequence[ts.StrictValue]] | None
    ] = None
    _bool_adapter: ClassVar[mp.TypeAdapter[bool] | None] = None
    _int_adapter: ClassVar[mp.TypeAdapter[tc.IntegerValue] | None] = None
    _scalar_adapter: ClassVar[mp.TypeAdapter[t.Scalar] | None] = None
    _float_adapter: ClassVar[mp.TypeAdapter[tc.FloatValue] | None] = None
    _str_adapter: ClassVar[mp.TypeAdapter[tc.TextValue] | None] = None
    _binary_content_adapter: ClassVar[mp.TypeAdapter[tc.BinaryContent] | None] = None
    _str_mapping_adapter: ClassVar[mp.TypeAdapter[t.StrMapping] | None] = None
    _hostname_str_adapter: ClassVar[mp.TypeAdapter[ta.HostnameStr] | None] = None
    _port_number_adapter: ClassVar[mp.TypeAdapter[ta.PortNumber] | None] = None
    _str_or_bytes_adapter: ClassVar[mp.TypeAdapter[tc.TextOrBinaryContent] | None] = (
        None
    )
    _enum_type_adapter: ClassVar[mp.TypeAdapter[type[StrEnum]] | None] = None
    _serializable_adapter: ClassVar[mp.TypeAdapter[tc.Serializable] | None] = None
    _primitive_metadata_mapping_adapter: ClassVar[
        mp.TypeAdapter[Mapping[str, t.Primitives]] | None
    ] = None
    _structlog_processor_adapter: ClassVar[
        mp.TypeAdapter[Callable[..., t.Container]] | None
    ] = None
    _json_value_adapter: ClassVar[mp.TypeAdapter[t.JsonValue] | None] = None
    _json_mapping_adapter: ClassVar[mp.TypeAdapter[t.JsonMapping] | None] = None
    _json_list_adapter: ClassVar[mp.TypeAdapter[t.JsonList] | None] = None
    _container_adapter: ClassVar[mp.TypeAdapter[t.Container] | None] = None

    @classmethod
    def metadata_map_adapter(
        cls,
    ) -> mp.TypeAdapter[Mapping[str, ts.MetadataValue]]:
        if cls._metadata_map_adapter is None:
            cls._metadata_map_adapter = mp.TypeAdapter(
                Mapping[str, ts.MetadataValue],
            )
        return cls._metadata_map_adapter

    @classmethod
    def container_mapping_adapter(
        cls,
    ) -> mp.TypeAdapter[Mapping[str, t.Container]]:
        if cls._container_mapping_adapter is None:
            cls._container_mapping_adapter = mp.TypeAdapter(
                Mapping[str, t.Container],
            )
        return cls._container_mapping_adapter

    @classmethod
    def container_mapping_sequence_adapter(
        cls,
    ) -> mp.TypeAdapter[Sequence[Mapping[str, t.Container]]]:
        if cls._container_mapping_sequence_adapter is None:
            cls._container_mapping_sequence_adapter = mp.TypeAdapter(
                Sequence[Mapping[str, t.Container]],
            )
        return cls._container_mapping_sequence_adapter

    @classmethod
    def json_value_adapter(cls) -> mp.TypeAdapter[t.JsonValue]:
        if cls._json_value_adapter is None:
            cls._json_value_adapter = mp.TypeAdapter(t.JsonValue)
        return cls._json_value_adapter

    @classmethod
    def json_mapping_adapter(cls) -> mp.TypeAdapter[t.JsonMapping]:
        if cls._json_mapping_adapter is None:
            cls._json_mapping_adapter = mp.TypeAdapter(t.JsonMapping)
        return cls._json_mapping_adapter

    @classmethod
    def json_list_adapter(cls) -> mp.TypeAdapter[t.JsonList]:
        if cls._json_list_adapter is None:
            cls._json_list_adapter = mp.TypeAdapter(t.JsonList)
        return cls._json_list_adapter

    @classmethod
    def container_adapter(cls) -> mp.TypeAdapter[t.Container]:
        if cls._container_adapter is None:
            cls._container_adapter = mp.TypeAdapter(t.Container)
        return cls._container_adapter

    @classmethod
    def flat_container_mapping_adapter(
        cls,
    ) -> mp.TypeAdapter[t.FlatContainerMapping]:
        if cls._flat_container_mapping_adapter is None:
            cls._flat_container_mapping_adapter = mp.TypeAdapter(
                t.FlatContainerMapping,
            )
        return cls._flat_container_mapping_adapter

    @classmethod
    def flat_container_list_adapter(
        cls,
    ) -> mp.TypeAdapter[t.FlatContainerList]:
        if cls._flat_container_list_adapter is None:
            cls._flat_container_list_adapter = mp.TypeAdapter(
                t.FlatContainerList,
            )
        return cls._flat_container_list_adapter

    @classmethod
    def tuple_container_adapter(
        cls,
    ) -> mp.TypeAdapter[tuple[t.Container, ...]]:
        if cls._tuple_container_adapter is None:
            cls._tuple_container_adapter = mp.TypeAdapter(
                tuple[t.Container, ...],
            )
        return cls._tuple_container_adapter

    @classmethod
    def primitives_adapter(
        cls,
    ) -> mp.TypeAdapter[t.Primitives]:
        if cls._primitives_adapter is None:
            cls._primitives_adapter = mp.TypeAdapter(t.Primitives)
        return cls._primitives_adapter

    @classmethod
    def list_serializable_adapter(
        cls,
    ) -> mp.TypeAdapter[Sequence[tc.Serializable]]:
        if cls._list_serializable_adapter is None:
            cls._list_serializable_adapter = mp.TypeAdapter(
                Sequence[tc.Serializable],
            )
        return cls._list_serializable_adapter

    @classmethod
    def tuple_serializable_adapter(
        cls,
    ) -> mp.TypeAdapter[tuple[tc.Serializable, ...]]:
        if cls._tuple_serializable_adapter is None:
            cls._tuple_serializable_adapter = mp.TypeAdapter(
                tuple[tc.Serializable, ...],
            )
        return cls._tuple_serializable_adapter

    @classmethod
    def container_set_adapter(
        cls,
    ) -> mp.TypeAdapter[set[t.Container]]:
        if cls._set_container_adapter is None:
            cls._set_container_adapter = mp.TypeAdapter(set[t.Container])
        return cls._set_container_adapter

    @classmethod
    def string_set_adapter(cls) -> mp.TypeAdapter[set[str]]:
        if cls._set_str_adapter is None:
            cls._set_str_adapter = mp.TypeAdapter(set[str])
        return cls._set_str_adapter

    @classmethod
    def scalar_set_adapter(
        cls,
    ) -> mp.TypeAdapter[set[t.Scalar]]:
        if cls._set_scalar_adapter is None:
            cls._set_scalar_adapter = mp.TypeAdapter(set[t.Scalar])
        return cls._set_scalar_adapter

    @classmethod
    def sortable_dict_adapter(
        cls,
    ) -> mp.TypeAdapter[
        Mapping[
            ts.SortableObjectType,
            tc.Serializable | None,
        ]
    ]:
        if cls._sortable_dict_adapter is None:
            cls._sortable_dict_adapter = mp.TypeAdapter(
                Mapping[
                    ts.SortableObjectType,
                    tc.Serializable | None,
                ],
            )
        return cls._sortable_dict_adapter

    @classmethod
    def strict_json_list_adapter(
        cls,
    ) -> mp.TypeAdapter[Sequence[ts.StrictValue]]:
        if cls._strict_json_list_adapter is None:
            cls._strict_json_list_adapter = mp.TypeAdapter(
                Sequence[ts.StrictValue],
            )
        return cls._strict_json_list_adapter

    @classmethod
    def bool_adapter(cls) -> mp.TypeAdapter[bool]:
        if cls._bool_adapter is None:
            cls._bool_adapter = mp.TypeAdapter(bool)
        return cls._bool_adapter

    @classmethod
    def int_adapter(
        cls,
    ) -> mp.TypeAdapter[tc.IntegerValue]:
        if cls._int_adapter is None:
            cls._int_adapter = mp.TypeAdapter(tc.IntegerValue)
        return cls._int_adapter

    @classmethod
    def scalar_adapter(cls) -> mp.TypeAdapter[t.Scalar]:
        if cls._scalar_adapter is None:
            cls._scalar_adapter = mp.TypeAdapter(t.Scalar)
        return cls._scalar_adapter

    @classmethod
    def float_adapter(
        cls,
    ) -> mp.TypeAdapter[tc.FloatValue]:
        if cls._float_adapter is None:
            cls._float_adapter = mp.TypeAdapter(tc.FloatValue)
        return cls._float_adapter

    @classmethod
    def str_adapter(cls) -> mp.TypeAdapter[tc.TextValue]:
        if cls._str_adapter is None:
            cls._str_adapter = mp.TypeAdapter(tc.TextValue)
        return cls._str_adapter

    @classmethod
    def binary_content_adapter(
        cls,
    ) -> mp.TypeAdapter[tc.BinaryContent]:
        if cls._binary_content_adapter is None:
            cls._binary_content_adapter = mp.TypeAdapter(tc.BinaryContent)
        return cls._binary_content_adapter

    @classmethod
    def str_mapping_adapter(
        cls,
    ) -> mp.TypeAdapter[t.StrMapping]:
        if cls._str_mapping_adapter is None:
            cls._str_mapping_adapter = mp.TypeAdapter(t.StrMapping)
        return cls._str_mapping_adapter

    @classmethod
    def hostname_str_adapter(
        cls,
    ) -> mp.TypeAdapter[ta.HostnameStr]:
        if cls._hostname_str_adapter is None:
            cls._hostname_str_adapter = mp.TypeAdapter(ta.HostnameStr)
        return cls._hostname_str_adapter

    @classmethod
    def port_number_adapter(
        cls,
    ) -> mp.TypeAdapter[ta.PortNumber]:
        if cls._port_number_adapter is None:
            cls._port_number_adapter = mp.TypeAdapter(ta.PortNumber)
        return cls._port_number_adapter

    @classmethod
    def str_sequence_adapter(
        cls,
    ) -> mp.TypeAdapter[t.StrSequence]:
        if cls._str_sequence_adapter is None:
            cls._str_sequence_adapter = mp.TypeAdapter(t.StrSequence)
        return cls._str_sequence_adapter

    @classmethod
    def str_or_bytes_adapter(
        cls,
    ) -> mp.TypeAdapter[tc.TextOrBinaryContent]:
        if cls._str_or_bytes_adapter is None:
            cls._str_or_bytes_adapter = mp.TypeAdapter(tc.TextOrBinaryContent)
        return cls._str_or_bytes_adapter

    @classmethod
    def enum_type_adapter(cls) -> mp.TypeAdapter[type[StrEnum]]:
        if cls._enum_type_adapter is None:
            cls._enum_type_adapter = mp.TypeAdapter(type[StrEnum])
        return cls._enum_type_adapter

    @classmethod
    def serializable_adapter(
        cls,
    ) -> mp.TypeAdapter[tc.Serializable]:
        if cls._serializable_adapter is None:
            cls._serializable_adapter = mp.TypeAdapter(tc.Serializable)
        return cls._serializable_adapter

    @classmethod
    def primitive_metadata_mapping_adapter(
        cls,
    ) -> mp.TypeAdapter[Mapping[str, t.Primitives]]:
        if cls._primitive_metadata_mapping_adapter is None:
            cls._primitive_metadata_mapping_adapter = mp.TypeAdapter(
                Mapping[str, t.Primitives],
            )
        return cls._primitive_metadata_mapping_adapter

    @classmethod
    def structlog_processor_adapter(
        cls,
    ) -> mp.TypeAdapter[Callable[..., t.Container]]:
        if cls._structlog_processor_adapter is None:
            cls._structlog_processor_adapter = mp.TypeAdapter(
                Callable[..., t.Container],
            )
        return cls._structlog_processor_adapter


__all__: list[str] = ["FlextTypesTypeAdapters"]
