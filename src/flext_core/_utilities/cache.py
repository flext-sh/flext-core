"""Cache utilities extracted for dispatcher-safe reuse.

This module exposes normalization, sorting, and cache lifecycle helpers used by
handlers and services without pulling in higher-level application surfaces.
Each helper is designed to keep deterministic ordering and safe cache clearing
while returning ``FlextResult`` where appropriate.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime, StructlogLogger
from flext_core.typings import FlextTypes


class FlextUtilitiesCache:
    """Cache utilities for deterministic normalization and key management."""

    @property
    def logger(self) -> StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance with all logging methods (debug, info, warning, error, etc).
        Uses same structure/config as FlextLogger but without circular import.
        """
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def normalize_component(
        component: FlextTypes.GeneralValueType | BaseModel,
    ) -> FlextTypes.GeneralValueType:
        """Normalize a component recursively for consistent representation."""
        if FlextRuntime.is_dict_like(component):
            component_dict = component
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in component_dict.items()
            }
        # Check str before Sequence since str is a Sequence
        if isinstance(component, str):
            return component
        if isinstance(component, Sequence):
            return [FlextUtilitiesCache.normalize_component(item) for item in component]
        if isinstance(component, set):
            return tuple(
                FlextUtilitiesCache.normalize_component(item) for item in component
            )
        # Return primitives and other types directly
        # Type narrowing: primitives are valid GeneralValueType
        if isinstance(component, (str, int, float, bool, type(None))):
            return component
        # For other types, convert to string as fallback
        return str(component)

    @staticmethod
    def sort_key(key: FlextTypes.SortableObjectType) -> tuple[int, str]:
        """Generate a sort key for deterministic ordering across types."""
        if isinstance(key, str):
            return (0, key.lower())
        if isinstance(key, (int, float)):
            return (1, str(key))
        return (2, str(key))

    @staticmethod
    def sort_dict_keys(
        data: FlextTypes.SortableObjectType,
    ) -> FlextTypes.SortableObjectType:
        """Sort dictionary keys recursively for consistent representations."""
        if FlextRuntime.is_dict_like(data):
            data_dict = data
            result: dict[str, FlextTypes.SortableObjectType] = {}
            for k in sorted(data_dict.keys(), key=FlextUtilitiesCache.sort_key):
                value = data_dict[k]
                # Handle None values - convert to empty dict for consistency
                if value is None:
                    result[k] = {}
                else:
                    # Recursively sort nested structures
                    sorted_value = FlextUtilitiesCache.sort_dict_keys(value)
                    result[k] = sorted_value
            return result
        return data

    @staticmethod
    def clear_object_cache(
        obj: FlextTypes.GeneralValueType | BaseModel,
    ) -> FlextResult[bool]:
        """Clear cache-like attributes on an object and surface any failure."""
        try:
            # Common cache attribute names to check and clear
            cache_attributes = FlextConstants.Utilities.CACHE_ATTRIBUTE_NAMES

            cleared_count = 0
            for attr_name in cache_attributes:
                if hasattr(obj, attr_name):
                    cache_attr = getattr(obj, attr_name, None)
                    if cache_attr is not None:
                        # Clear dict[str, FlextTypes.GeneralValueType]-like caches
                        if hasattr(cache_attr, "clear") and callable(
                            cache_attr.clear,
                        ):
                            cache_attr.clear()
                            cleared_count += 1
                        # Reset to None for simple cached values
                        else:
                            setattr(obj, attr_name, None)
                            cleared_count += 1

            return FlextResult[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[bool].fail(f"Failed to clear caches: {e}")

    @staticmethod
    def has_cache_attributes(obj: FlextTypes.GeneralValueType) -> bool:
        """Check if an object exposes any known cache-related attributes."""
        cache_attributes = FlextConstants.Utilities.CACHE_ATTRIBUTE_NAMES
        return any(hasattr(obj, attr) for attr in cache_attributes)

    @staticmethod
    def generate_cache_key(
        *args: FlextTypes.GeneralValueType,
        **kwargs: FlextTypes.GeneralValueType,
    ) -> str:
        """Generate a deterministic cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()


__all__ = ["FlextUtilitiesCache"]
