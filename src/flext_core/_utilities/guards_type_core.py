"""Core scalar and container type guards."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeGuard, TypeIs

from flext_core import t


class FlextUtilitiesGuardsTypeCore:
    @staticmethod
    def _is_object_sequence(value: object) -> TypeIs[Sequence[object]]:
        return isinstance(value, (list, tuple))

    @staticmethod
    def _is_object_mapping(value: object) -> TypeIs[Mapping[str, object]]:
        return isinstance(value, Mapping)

    @staticmethod
    def _all_container_sequence(value: Sequence[object]) -> bool:
        for sequence_item in value:
            if not FlextUtilitiesGuardsTypeCore.is_container(sequence_item):
                return False
        return True

    @staticmethod
    def _all_container_mapping_values(value: Mapping[str, object]) -> bool:
        for mapped_value in value.values():
            if not FlextUtilitiesGuardsTypeCore.is_container(mapped_value):
                return False
        return True

    @staticmethod
    def all_container_mapping_values(value: Mapping[str, object]) -> bool:
        return FlextUtilitiesGuardsTypeCore._all_container_mapping_values(value)

    @staticmethod
    def is_dict_non_empty(value: t.NormalizedValue) -> bool:
        return isinstance(value, Mapping) and bool(value)

    @staticmethod
    def is_flexible_value(value: t.NormalizedValue) -> TypeIs[t.NormalizedValue]:
        if value is None or FlextUtilitiesGuardsTypeCore.is_scalar(value):
            return True
        if isinstance(value, (list, tuple)):
            return all(
                item is None or FlextUtilitiesGuardsTypeCore.is_scalar(item)
                for item in value
            )
        if isinstance(value, Mapping):
            return all(
                item is None or FlextUtilitiesGuardsTypeCore.is_scalar(item)
                for item in value.values()
            )
        return False

    @staticmethod
    def is_container(
        value: object,
    ) -> TypeGuard[str | int | float | bool | datetime | Path]:
        if value is None or isinstance(value, (str, int, float, bool, datetime)):
            return True
        if FlextUtilitiesGuardsTypeCore._is_object_sequence(value):
            return FlextUtilitiesGuardsTypeCore._all_container_sequence(value)
        if FlextUtilitiesGuardsTypeCore._is_object_mapping(value):
            return FlextUtilitiesGuardsTypeCore._all_container_mapping_values(value)
        return isinstance(value, Path)

    @staticmethod
    def is_general_value_type(value: t.NormalizedValue) -> bool:
        warnings.warn(
            "is_general_value_type is deprecated; use is_container. Planned removal: v0.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return callable(value) or FlextUtilitiesGuardsTypeCore.is_container(value)

    @staticmethod
    def is_list(value: t.NormalizedValue) -> TypeIs[list[t.NormalizedValue]]:
        return isinstance(value, list)

    @staticmethod
    def is_list_non_empty(value: t.NormalizedValue) -> bool:
        return (
            isinstance(value, Sequence)
            and (not isinstance(value, (str, bytes)))
            and len(value) > 0
        )

    @staticmethod
    def is_mapping(
        value: object,
    ) -> TypeIs[Mapping[str, t.NormalizedValue]]:
        return isinstance(value, Mapping)

    @staticmethod
    def is_primitive(
        value: object,
    ) -> TypeIs[t.Primitives]:
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def is_scalar(
        value: object,
    ) -> TypeIs[t.Scalar]:
        return isinstance(value, (str, int, float, bool, datetime))

    @staticmethod
    def is_string_non_empty(value: t.NormalizedValue) -> TypeIs[str]:
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_instance_of[T](value: object, type_cls: type[T]) -> TypeIs[T]:
        return isinstance(value, getattr(type_cls, "__origin__", None) or type_cls)

    @staticmethod
    def require_initialized[T](value: T | None, name: str) -> T:
        if value is None:
            msg = f"{name} is not initialized"
            raise AttributeError(msg)
        return value

    @staticmethod
    def has(obj: t.NormalizedValue, key: str) -> bool:
        return key in obj if isinstance(obj, dict) else hasattr(obj, key)

    @staticmethod
    def in_(value: t.NormalizedValue, container: t.NormalizedValue) -> bool:
        if isinstance(container, (list, tuple, set, dict)):
            try:
                return value in container
            except TypeError:
                return False
        return False

    @staticmethod
    def none_(*values: t.NormalizedValue) -> bool:
        return all(v is None for v in values)


__all__ = ["FlextUtilitiesGuardsTypeCore"]
