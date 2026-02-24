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
   - Returns FlextResult for graceful error handling

Validation Context:
- Python 3.13+: Uses collections.abc.Sequence/Mapping
- Pydantic v2: Uses model_dump() for BaseModel serialization
- FlextResult: Cache operations return FlextResult for error handling

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel

from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


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
       - Handles all t.GuardInputValue variants
       - BaseModel special handling with model_dump()
       - Graceful fallback to string representation

    3. **Error Handling**:
       - clear_object_cache returns FlextResult (railway pattern)
       - Other methods are pure functions (no side effects)
       - No exceptions propagated to callers
    """

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get logger instance using FlextRuntime.

        Business Rule: Logger access through FlextRuntime avoids circular
        imports between cache utilities and logging modules.

        Returns:
            Structlog logger instance with all logging methods.

        """
        return cast("p.Log.StructlogLogger", FlextRuntime.get_logger(__name__))

    @staticmethod
    def normalize_component(
        component: t.GuardInputValue | BaseModel,
    ) -> t.GuardInputValue:
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
        # Handle BaseModel first - convert to dict
        if BaseModel in component.__class__.__mro__:
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in component.model_dump().items()
            }
        # component is already t.GuardInputValue (not BaseModel)
        # Check if dict-like
        if FlextRuntime.is_dict_like(component):
            # Type narrowing: component is now Mapping[str, t.GuardInputValue]
            # Convert to dict for consistent iteration
            dict_component: Mapping[str, t.GuardInputValue] = dict(component.items())
            # dict_component has mapping semantics for normalized iteration
            # so v is t.GuardInputValue
            return {
                str(k): FlextUtilitiesCache.normalize_component(v)
                for k, v in dict_component.items()
            }
        # Handle primitives first (str is a Sequence, so check early)
        if component.__class__ in {str, int, float, bool, None.__class__}:
            return component
        # Handle collections
        if component.__class__ is set:
            # Type narrowing: component is set[t.GuardInputValue]
            # Explicit type annotation for set items
            items_set: set[t.GuardInputValue] = component
            # Convert set to tuple for hashability - normalize each item
            normalized_items: list[t.GuardInputValue] = [
                FlextUtilitiesCache.normalize_component(item) for item in items_set
            ]
            return tuple(normalized_items)
        if component.__class__ in {list, tuple} or (
            hasattr(component, "__getitem__")
            and component.__class__ not in {str, bytes}
        ):
            # Type narrowing: component is Sequence, so items are t.GuardInputValue
            return [FlextUtilitiesCache.normalize_component(item) for item in component]
        # For other types, convert to string as fallback
        return str(component)

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
        if key.__class__ is str:
            return (0, key.lower())
        if key.__class__ in {int, float}:
            return (1, str(key))
        return (2, str(key))

    @staticmethod
    def sort_dict_keys(
        data: t.GuardInputValue,
    ) -> t.GuardInputValue:
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
        - Preserves t.GuardInputValue contract

        Args:
            data: t.GuardInputValue value

        Returns:
            Sorted dict if input is dict-like, unchanged otherwise

        """
        if FlextRuntime.is_dict_like(data):
            # Type narrowing: data is now Mapping[str, t.GuardInputValue]
            result: dict[str, t.GuardInputValue] = {}
            for k in sorted(data.keys(), key=FlextUtilitiesCache.sort_key):
                value = data[k]
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
        obj: t.GuardInputValue | BaseModel,
    ) -> r[bool]:
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
        - Returns FlextResult (railway pattern)
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
            # Common cache attribute names to check and clear
            cache_attributes = c.Utilities.CACHE_ATTRIBUTE_NAMES

            cleared_count = 0
            for attr_name in cache_attributes:
                if hasattr(obj, attr_name):
                    cache_attr = getattr(obj, attr_name, None)
                    if cache_attr is not None:
                        # Clear mapping-like caches
                        if hasattr(cache_attr, "clear") and callable(
                            cache_attr.clear,
                        ):
                            cache_attr.clear()
                            cleared_count += 1
                        # Reset to None for simple cached values
                        else:
                            setattr(obj, attr_name, None)
                            cleared_count += 1

            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Failed to clear caches: {e}")

    @staticmethod
    def has_cache_attributes(obj: t.GuardInputValue) -> bool:
        """Check if an object exposes any known cache-related attributes.

        Business Rule: Cache Detection
        ==============================
        Quick check to determine if an object might have cached data.
        Useful for deciding whether to attempt cache clearing.

        Args:
            obj: t.GuardInputValue object

        Returns:
            True if any known cache attribute exists, False otherwise

        """
        cache_attributes = c.Utilities.CACHE_ATTRIBUTE_NAMES
        # NOTE: Cannot use u.map() here due to circular import (utilities.py imports cache.py)
        return any(hasattr(obj, attr) for attr in cache_attributes)

    @staticmethod
    def generate_cache_key(
        *args: t.GuardInputValue,
        **kwargs: t.GuardInputValue,
    ) -> str:
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


__all__ = [
    "FlextUtilitiesCache",
]
