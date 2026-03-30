"""Cache utilities for deterministic normalization and key management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from pydantic import BaseModel

from flext_core import FlextRuntime, FlextUtilitiesGuardsTypeCore, c, m, p, r, t


class FlextUtilitiesCache:
    """Cache utilities for deterministic normalization and key management."""

    @property
    def logger(self) -> p.Logger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def clear_object_cache(
        obj: BaseModel | p.HasModelDump | t.NormalizedValue,
    ) -> r[bool]:
        """Clear cache-like attributes on an t.NormalizedValue.

        Args:
            obj: Object with potential cache attributes

        Returns:
            p.Result[bool]: ok(True) on success, fail(msg) on error

        """
        try:
            cache_attributes = c.CACHE_ATTRIBUTE_NAMES
            cleared_count = 0
            for attr_name in cache_attributes:
                if hasattr(obj, attr_name):
                    cache_attr = getattr(obj, attr_name)
                    if cache_attr is not None:
                        if hasattr(cache_attr, "clear") and callable(cache_attr.clear):
                            _ = cache_attr.clear()
                            cleared_count += 1
                        else:
                            setattr(obj, attr_name, None)
                            cleared_count += 1
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Failed to clear caches: {e}")

    @staticmethod
    def generate_cache_key(*args: t.NormalizedValue, **kwargs: t.Scalar) -> str:
        """Generate a deterministic cache key from arguments.

        Args:
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key

        Returns:
            64-character hexadecimal SHA-256 hash

        """
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()

    @staticmethod
    def generate_cache_key_for_command(
        command: BaseModel | t.ContainerMapping | t.NormalizedValue,
        command_type: type,
    ) -> str:
        if isinstance(command, Mapping):
            command_map = m.Validators.dict_str_metadata_adapter().validate_python(
                command,
            )
            sorted_data = FlextUtilitiesCache.sort_dict_keys(command_map)
            return f"{command_type.__name__}_{hash(str(sorted_data))}"
        command_str = "None" if command is None else str(command)
        try:
            return f"{command_type.__name__}_{hash(command_str)}"
        except TypeError:
            encoded = command_str.encode(c.DEFAULT_ENCODING)
            return f"{command_type.__name__}_{abs(hash(encoded))}"

    @staticmethod
    def has_cache_attributes(
        obj: BaseModel | p.HasModelDump | t.NormalizedValue,
    ) -> bool:
        """Check if an t.NormalizedValue exposes any known cache-related attributes.

        Args:
            obj: target instance

        Returns:
            True if any known cache attribute exists, False otherwise

        """
        cache_attributes = c.CACHE_ATTRIBUTE_NAMES
        return any(hasattr(obj, attr) for attr in cache_attributes)

    @staticmethod
    def normalize_component(
        component: t.ValueOrModel | set[t.NormalizedValue],
    ) -> t.NormalizedValue:
        """Normalize a component recursively for consistent representation."""
        if isinstance(component, BaseModel):
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in component.model_dump().items()
            }
        if isinstance(component, Mapping):
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in component.items()
            }
        if isinstance(component, set):
            normalized_set_items: t.ContainerList = [
                FlextUtilitiesCache.normalize_component(item) for item in component
            ]
            return tuple(normalized_set_items)
        if FlextUtilitiesGuardsTypeCore.is_primitive(component) or component is None:
            return component
        if isinstance(component, (list, tuple)):
            return [FlextUtilitiesCache.normalize_component(item) for item in component]
        return str(component)

    @staticmethod
    def sort_dict_keys(
        data: t.NormalizedValue | t.Serializable | BaseModel,
    ) -> t.NormalizedValue:
        """Sort dictionary keys recursively for consistent representations.

        Args:
            data: input value

        Returns:
            Sorted dict if input is dict-like, unchanged otherwise

        """
        if isinstance(data, BaseModel):
            return FlextUtilitiesCache.sort_dict_keys(data.model_dump())
        if isinstance(data, Mapping):
            data_map = m.Validators.sortable_dict_adapter().validate_python(data)
            result: t.MutableContainerMapping = {}
            for k in sorted(data_map.keys(), key=FlextUtilitiesCache.sort_key):
                value = data_map[k]
                if value is None:
                    empty_val: t.ContainerMapping = {}
                    result[str(k)] = empty_val
                else:
                    sorted_value = FlextUtilitiesCache.sort_dict_keys(value)
                    result[str(k)] = sorted_value
            return result
        if isinstance(data, list):
            return [FlextUtilitiesCache.sort_dict_keys(item) for item in data]
        if isinstance(data, tuple):
            return tuple(FlextUtilitiesCache.sort_dict_keys(item) for item in data)
        return data

    @staticmethod
    def sort_key(key: t.SortableObjectType) -> tuple[int, str]:
        """Generate a sort key for deterministic ordering across types.

        Args:
            key: Sortable t.NormalizedValue (str, int, tuple, etc. - usually dict key)

        Returns:
            Tuple for sorted() key function

        """
        if isinstance(key, str):
            return (0, key.lower())
        return (1, str(key))


__all__ = ["FlextUtilitiesCache"]
