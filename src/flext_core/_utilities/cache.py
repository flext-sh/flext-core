"""Cache utilities for deterministic normalization and key management.

Business Rules & Architecture:
=============================

1. **Deterministic Cache Key Generation**:
   - Components normalized to canonical form before hashing
   - Dict keys sorted alphabetically for consistent representation
   - Sets converted to tuples for hashability
   - SHA-256 for collision-resistant cache keys

2. **Normalization Rules** (normalize_component method):
   - BaseModel → dict (via model_dump())
   - Mapping/dict-like → dict with normalized values
   - Sequences (non-str) → list with normalized items
   - Sets → sorted tuple (for hashability)
   - Primitives (str, int, float, bool, None) → unchanged
   - Unknown types → str representation (fallback)

3. **Sort Key Strategy** (sort_key method):
   - Type-aware sorting: strings first, numbers second, others last
   - Case-insensitive string comparison
   - Deterministic ordering across Python runs

4. **Cache Clearing Strategy** (clear_object_cache method):
   - Clears common cache attribute names (_cache, _cached, cache, etc.)
   - Supports dict.clear() for mapping caches
   - Falls back to None assignment for simple cached values
   - Returns r for graceful error handling

Validation Context:
- Python 3.13+: Uses collections.abc.Sequence/Mapping
- Pydantic v2: Uses model_dump() for BaseModel serialization
- r: Cache operations return r for error handling

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_core import FlextRuntime, c, p, r, t

_CACHE_DICT_STR_OBJECT_ADAPTER = TypeAdapter(dict[str, object])
_CACHE_SET_OBJECT_ADAPTER = TypeAdapter(set[object])
_CACHE_SET_STR_ADAPTER = TypeAdapter(set[str])
_CACHE_LIST_OBJECT_ADAPTER = TypeAdapter(list[object])
_CACHE_SORTABLE_DICT_ADAPTER = TypeAdapter(dict[t.SortableObjectType, object])


class FlextUtilitiesCache:
    """Cache utilities for deterministic normalization and key management.

    Business Rules:
    ==============
    This class provides cache-related utilities that ensure:

    1. **Deterministic Behavior**:
       - Same input always produces same cache key
       - No dependency on dict iteration order (Python 3.7+ guarantee)
       - Type-aware sorting for cross-type comparisons

    2. **Type Safety**:
       - Handles all object variants
       - BaseModel special handling with model_dump()
       - Graceful fallback to string representation

    3. **Error Handling**:
       - clear_object_cache returns r (railway pattern)
       - Other methods are pure functions (no side effects)
       - No exceptions propagated to callers
    """

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def clear_object_cache(obj: object) -> r[bool]:
        """Clear cache-like attributes on an object.

        Business Rule: Safe Cache Invalidation
        =====================================
        This method provides safe cache clearing for objects that may
        have cached data that needs invalidation.

        Attribute Detection:
        - Checks FlextConstants.Utilities.CACHE_ATTRIBUTE_NAMES
        - Common names: _cache, _cached, cache, cached_data, etc.
        - Configurable via constants module

        Clearing Strategy:
        1. Dict-like caches (has .clear() method) → call clear()
        2. Simple cached values → set to None
        3. Missing attributes → skip silently

        Error Handling:
        - Returns r (railway pattern)
        - ok(True) on success (even if no caches found)
        - fail(error_msg) on any exception

        Thread Safety:
        - NOT thread-safe (cache clearing is a mutation)
        - Caller responsible for synchronization if needed

        Args:
            obj: Object with potential cache attributes

        Returns:
            r[bool]: ok(True) on success, fail(msg) on error

        """
        try:
            cache_attributes = c.Utilities.CACHE_ATTRIBUTE_NAMES
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
    def generate_cache_key(*args: object, **kwargs: t.Scalar) -> str:
        """Generate a deterministic cache key from arguments.

        Business Rule: SHA-256 Cache Key Generation
        ==========================================
        Generates a unique, deterministic cache key from function arguments.

        Algorithm:
        1. Convert args to string representation
        2. Sort kwargs items (deterministic order)
        3. Concatenate both representations
        4. SHA-256 hash for fixed-length, collision-resistant key

        Properties:
        - Deterministic: Same args/kwargs → same key
        - Fixed length: 64 hex characters (256 bits)
        - Collision-resistant: SHA-256 provides cryptographic security
        - URL-safe: Hexadecimal characters only

        Limitations:
        - Objects with non-deterministic __str__ may produce different keys
        - For complex objects, consider normalizing first

        Args:
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key

        Returns:
            64-character hexadecimal SHA-256 hash

        """
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()

    @staticmethod
    def generate_cache_key_for_command(command: object, command_type: type) -> str:
        if isinstance(command, Mapping):
            command_map = _CACHE_DICT_STR_OBJECT_ADAPTER.validate_python(command)
            sorted_data = FlextUtilitiesCache.sort_dict_keys(command_map)
            return f"{command_type.__name__}_{hash(str(sorted_data))}"
        command_str = "None" if command is None else str(command)
        try:
            return f"{command_type.__name__}_{hash(command_str)}"
        except TypeError:
            encoded = command_str.encode(c.Utilities.DEFAULT_ENCODING)
            return f"{command_type.__name__}_{abs(hash(encoded))}"

    @staticmethod
    def has_cache_attributes(obj: object) -> bool:
        """Check if an object exposes any known cache-related attributes.

        Business Rule: Cache Detection
        ==============================
        Quick check to determine if an object might have cached data.
        Useful for deciding whether to attempt cache clearing.

        Args:
            obj: object object

        Returns:
            True if any known cache attribute exists, False otherwise

        """
        cache_attributes = c.Utilities.CACHE_ATTRIBUTE_NAMES
        return any(hasattr(obj, attr) for attr in cache_attributes)

    @staticmethod
    def normalize_component(component: object) -> object:
        """Normalize a component recursively for consistent representation.

        Business Rule: Recursive Component Normalization
        ================================================
        Components are normalized to ensure deterministic cache keys
        and consistent comparison across different representation formats.

        Type Handling Priority (order matters for correct behavior):
        1. BaseModel → dict (via model_dump(), includes computed fields)
        2. Mapping/dict-like → dict with normalized values
        3. Primitives (str, int, float, bool, None) → unchanged
           Note: str is Sequence, so check primitives BEFORE sequences
        4. Sets → tuple (for hashability, order may vary)
        5. Sequences (list, tuple) → list with normalized items
        6. Other types → str representation (fallback)

        Why Recursion?
        - Nested structures (dict in dict, list in dict, etc.)
        - Pydantic models with nested models
        - Ensures deep normalization for cache key consistency

        Thread Safety:
        - Pure function (no shared state)
        - Safe for concurrent calls
        """
        if isinstance(component, BaseModel):
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in component.model_dump().items()
            }
        if isinstance(component, Mapping):
            dict_component = _CACHE_DICT_STR_OBJECT_ADAPTER.validate_python(component)
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in dict_component.items()
            }
        if isinstance(component, t.PRIMITIVES_TYPES) or component is None:
            return component
        if isinstance(component, set):
            try:
                set_items = _CACHE_SET_OBJECT_ADAPTER.validate_python(
                    component,
                    strict=False,
                )
            except ValidationError:
                fallback_items = _CACHE_SET_STR_ADAPTER.validate_python(
                    component,
                    strict=False,
                )
                return tuple(fallback_items)
            normalized_items: list[object] = [
                FlextUtilitiesCache.normalize_component(item) for item in set_items
            ]
            return tuple(normalized_items)
        if isinstance(component, (list, tuple)):
            sequence = _CACHE_LIST_OBJECT_ADAPTER.validate_python(component)
            return [FlextUtilitiesCache.normalize_component(item) for item in sequence]
        return str(component)

    @staticmethod
    def sort_dict_keys(data: object) -> object:
        """Sort dictionary keys recursively for consistent representations.

        Business Rule: Recursive Key Sorting for Cache Consistency
        =========================================================
        Dict keys are sorted to ensure the same data always produces
        the same cache key, regardless of insertion order.

        None Value Handling:
        - None values converted to empty dict {} for consistency
        - This ensures JSON serialization produces predictable output
        - Empty dict is semantically "no value" in many contexts

        Recursion:
        - Nested dicts are sorted at all levels
        - Non-dict values returned unchanged
        - List items NOT reordered (order may be meaningful)

        Type Safety:
        - Uses FlextRuntime.is_dict_like for Mapping detection
        - Returns input unchanged if not dict-like
        - Preserves object contract

        Args:
            data: object value

        Returns:
            Sorted dict if input is dict-like, unchanged otherwise

        """
        if isinstance(data, BaseModel):
            return FlextUtilitiesCache.sort_dict_keys(data.model_dump())
        if isinstance(data, Mapping):
            data_map = _CACHE_SORTABLE_DICT_ADAPTER.validate_python(data)
            result: dict[str, object] = {}
            for k in sorted(data_map.keys(), key=FlextUtilitiesCache.sort_key):
                value = data_map[k]
                if value is None:
                    result[str(k)] = {}
                else:
                    sorted_value = FlextUtilitiesCache.sort_dict_keys(value)
                    result[str(k)] = sorted_value
            return result
        return data

    @staticmethod
    def sort_key(key: t.SortableObjectType) -> tuple[int, str]:
        """Generate a sort key for deterministic ordering across types.

        Business Rule: Type-Aware Deterministic Sorting
        ===============================================
        Python's default sorting fails when mixing types (str vs int).
        This method provides a deterministic sort key that:

        1. Groups by type (strings first, numbers second, others last)
        2. Sorts within each group using string representation
        3. Case-insensitive for strings (prevents 'Z' < 'a' issues)

        Sort Priority:
        - (0, lower_str) → strings (most common dict keys)
        - (1, str_num) → numbers (int, float)
        - (2, str_repr) → other types (fallback)

        Why tuple[int, str]?
        - First element groups by type (Python compares tuples element-wise)
        - Second element provides within-group ordering
        - Deterministic across Python runs

        Args:
            key: Sortable object (str, int, tuple, etc. - usually dict key)

        Returns:
            Tuple for sorted() key function

        """
        if isinstance(key, str):
            return (0, key.lower())
        return (1, str(key))


__all__ = ["FlextUtilitiesCache"]
