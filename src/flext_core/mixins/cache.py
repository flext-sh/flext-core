"""FLEXT Cache - In-memory caching and memoization functionality.

Provides caching capabilities through hierarchical organization
of cache utilities and mixin classes. Built for result memoization and
performance optimization with production-ready patterns.

Module Role in Architecture:
    FlextCache serves as the caching foundation providing in-memory memoization
    patterns for performance optimization. Integrates with all FLEXT ecosystem
    components requiring cached computations and result storage.
"""

from __future__ import annotations

import time
from typing import TypeVar

from flext_core.mixins.identification import FlextIdentification
from flext_core.protocols import FlextProtocols
from flext_core.utilities import FlextUtilities

# Type variables for generic support
T = TypeVar("T")
CacheValueType = object

# =============================================================================
# TIER 1 MODULE PATTERN - SINGLE MAIN EXPORT
# =============================================================================


class FlextCache:
    """Unified caching system implementing single class pattern.

    This class serves as the single main export consolidating ALL caching
    functionality with production-ready patterns. Provides
    in-memory caching capabilities while maintaining clean API.

    Tier 1 Module Pattern: cache.py -> FlextCache
    All caching functionality is accessible through this single interface.
    """

    # =============================================================================
    # CORE CACHE OPERATIONS
    # =============================================================================

    @staticmethod
    def get_cached_value(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        key: str,
    ) -> CacheValueType | None:
        """Get cached value by key.

        Args:
            obj: Object to get cached value from
            key: Cache key

        Returns:
            Cached value if exists, None otherwise

        """
        if not hasattr(obj, "_cache"):
            obj._cache = {}
            obj._cache_stats = {"hits": 0, "misses": 0}

        cache: dict[str, tuple[CacheValueType, float]] = getattr(obj, "_cache", {})
        stats: dict[str, int] = getattr(obj, "_cache_stats", {"hits": 0, "misses": 0})

        if key in cache:
            value, _ = cache[key]
            stats["hits"] += 1
            obj._cache_stats = stats
            return value

        stats["misses"] += 1
        obj._cache_stats = stats
        return None

    @staticmethod
    def set_cached_value(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        key: str,
        value: CacheValueType,
    ) -> None:
        """Set cached value by key.

        Args:
            obj: Object to cache value in
            key: Cache key
            value: Value to cache

        """
        if not hasattr(obj, "_cache"):
            obj._cache = {}
            obj._cache_stats = {"hits": 0, "misses": 0}

        cache: dict[str, tuple[CacheValueType, float]] = getattr(obj, "_cache", {})
        timestamp: float = time.time()
        cache[key] = (value, timestamp)
        obj._cache = cache

    @staticmethod
    def clear_cache(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Clear all cached values.

        Args:
            obj: Object to clear cache from

        """
        obj._cache = {}
        obj._cache_stats = {"hits": 0, "misses": 0}

    @staticmethod
    def has_cached_value(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        key: str,
    ) -> bool:
        """Check if value is cached.

        Args:
            obj: Object to check cache in
            key: Cache key

        Returns:
            True if value is cached, False otherwise

        """
        if not hasattr(obj, "_cache"):
            return False

        cache: dict[str, tuple[CacheValueType, float]] = getattr(obj, "_cache", {})
        return key in cache

    @staticmethod
    def get_cache_key(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> str:
        """Generate cache key for an object.

        Args:
            obj: Object to generate cache key for

        Returns:
            Cache key string

        """
        if FlextIdentification.has_id(obj):
            entity_id = FlextIdentification.get_id(obj)
            if entity_id:
                return f"{obj.__class__.__name__}:{entity_id}"

        obj_hash: str = FlextUtilities.Generators.generate_uuid()
        return f"{obj.__class__.__name__}:{obj_hash}"

    # =============================================================================
    # MIXIN CLASS
    # =============================================================================

    class Cacheable:
        """Mixin class providing caching functionality.

        This mixin adds caching capabilities to any class, including
        value storage, retrieval, and cache management.
        """

        def get_cached_value(self, key: str) -> CacheValueType | None:
            """Get cached value by key."""
            return FlextCache.get_cached_value(self, key)

        def set_cached_value(self, key: str, value: CacheValueType) -> None:
            """Set cached value by key."""
            FlextCache.set_cached_value(self, key, value)

        def clear_cache(self) -> None:
            """Clear all cached values."""
            FlextCache.clear_cache(self)

        def has_cached_value(self, key: str) -> bool:
            """Check if value is cached."""
            return FlextCache.has_cached_value(self, key)

        def get_cache_key(self) -> str:
            """Generate cache key for this object."""
            return FlextCache.get_cache_key(self)


__all__ = [
    "FlextCache",
]
