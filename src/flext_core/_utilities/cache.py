"""Utilities module - FlextUtilitiesCache.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib
import logging

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

# Module constants
MAX_PORT_NUMBER: int = 65535
MIN_PORT_NUMBER: int = 1
_logger = logging.getLogger(__name__)


class FlextUtilitiesCache:
    """Cache utility functions for data normalization and sorting."""

    @staticmethod
    def normalize_component(
        component: FlextTypes.GenericDetailsType,
    ) -> object:
        """Normalize a component for consistent representation."""
        if FlextRuntime.is_dict_like(component):
            component_dict = component
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in component_dict.items()
            }
        if isinstance(component, (list, tuple)):
            return [FlextUtilitiesCache.normalize_component(item) for item in component]
        if isinstance(component, set):
            return {FlextUtilitiesCache.normalize_component(item) for item in component}
        # Return primitives and other types directly
        return component

    @staticmethod
    def sort_key(key: FlextTypes.SortableObjectType) -> tuple[int, str]:
        """Generate a sort key for consistent ordering."""
        if isinstance(key, str):
            return (0, key.lower())
        if isinstance(key, (int, float)):
            return (1, str(key))
        return (2, str(key))

    @staticmethod
    def sort_dict_keys(
        data: FlextTypes.SortableObjectType,
    ) -> FlextTypes.SortableObjectType:
        """Sort dictionary keys for consistent representation."""
        if FlextRuntime.is_dict_like(data):
            data_dict = data
            return {
                k: FlextUtilitiesCache.sort_dict_keys(data_dict[k])
                for k in sorted(data_dict.keys(), key=FlextUtilitiesCache.sort_key)
            }
        return data

    @staticmethod
    def clear_object_cache(obj: FlextTypes.CachedObjectType) -> FlextResult[None]:
        """Clear any caches on an object."""
        try:
            # Common cache attribute names to check and clear
            cache_attributes = FlextConstants.Utilities.CACHE_ATTRIBUTE_NAMES

            cleared_count = 0
            for attr_name in cache_attributes:
                if hasattr(obj, attr_name):
                    cache_attr = getattr(obj, attr_name, None)
                    if cache_attr is not None:
                        # Clear dict[str, object]-like caches
                        if hasattr(cache_attr, "clear") and callable(
                            cache_attr.clear,
                        ):
                            cache_attr.clear()
                            cleared_count += 1
                        # Reset to None for simple cached values
                        else:
                            setattr(obj, attr_name, None)
                            cleared_count += 1

            return FlextResult[None].ok(None)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[None].fail(f"Failed to clear caches: {e}")

    @staticmethod
    def has_cache_attributes(obj: FlextTypes.CachedObjectType) -> bool:
        """Check if object has any cache-related attributes."""
        cache_attributes = FlextConstants.Utilities.CACHE_ATTRIBUTE_NAMES
        return any(hasattr(obj, attr) for attr in cache_attributes)

    @staticmethod
    def generate_cache_key(*args: object, **kwargs: object) -> str:
        """Generate a cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()


__all__ = ["FlextUtilitiesCache"]
