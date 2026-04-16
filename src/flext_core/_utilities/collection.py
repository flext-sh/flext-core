"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from typing import ClassVar, overload

from flext_core import (
    T,
    U,
    c,
    p,
    r,
    t,
)


class FlextUtilitiesCollection:
    """Utilities for collection operations with full generic type support."""

    @staticmethod
    def _ok_result[TValue](value: TValue) -> p.Result[TValue]:
        """Create a typed success result."""
        return r[TValue].ok(value)

    @staticmethod
    def _is_empty_value(value: t.ValueOrModel) -> bool:
        """Check if value is considered empty (empty string, empty list, etc.)."""
        if value is None:
            return True
        if isinstance(value, str):
            return not value
        if isinstance(value, list):
            return not value
        if isinstance(value, dict):
            return not value
        return False

    @staticmethod
    def _merge_deep_single_key(
        result: t.MutableRecursiveContainerMapping,
        key: str,
        value: t.RecursiveContainer,
    ) -> p.Result[bool]:
        """Merge single key in deep merge strategy."""
        current_val = result.get(key)
        if (
            current_val is not None
            and isinstance(current_val, dict)
            and isinstance(value, dict)
        ):
            merged_result: p.Result[t.RecursiveContainerMapping] = (
                FlextUtilitiesCollection.merge_mappings(
                    value,
                    current_val,
                    strategy="deep",
                )
            )
            if merged_result.success:
                result[key] = merged_result.value
                return FlextUtilitiesCollection._ok_result(True)
            return r[bool].fail(
                f"Failed to merge nested dict for key {key}: {merged_result.error}",
            )
        result[key] = value
        return FlextUtilitiesCollection._ok_result(True)

    @staticmethod
    def count(items: Sequence[T], predicate: Callable[[T], bool] | None = None) -> int:
        """Count items, optionally matching predicate."""
        if predicate is None:
            return len(items)
        return sum(1 for item in items if predicate(item))

    @overload
    @staticmethod
    def filter(
        items: Sequence[T],
        predicate: Callable[[T], bool],
        *,
        mapper: None = None,
    ) -> Sequence[T]: ...

    @overload
    @staticmethod
    def filter(
        items: Sequence[T],
        predicate: Callable[[T], bool],
        *,
        mapper: Callable[[T], U],
    ) -> Sequence[U]: ...

    @overload
    @staticmethod
    def filter(
        items: tuple[T, ...],
        predicate: Callable[[T], bool],
        *,
        mapper: None = None,
    ) -> tuple[T, ...]: ...

    @overload
    @staticmethod
    def filter(
        items: tuple[T, ...],
        predicate: Callable[[T], bool],
        *,
        mapper: Callable[[T], U],
    ) -> tuple[U, ...]: ...

    @overload
    @staticmethod
    def filter(
        items: Mapping[str, T],
        predicate: Callable[[T], bool],
        *,
        mapper: None = None,
    ) -> Mapping[str, T]: ...

    @overload
    @staticmethod
    def filter(
        items: Mapping[str, T],
        predicate: Callable[[T], bool],
        *,
        mapper: Callable[[T], U],
    ) -> Mapping[str, U]: ...

    @staticmethod
    def filter(
        items: Sequence[T] | tuple[T, ...] | Mapping[str, T],
        predicate: Callable[[T], bool],
        *,
        mapper: Callable[[T], U] | None = None,
    ) -> (
        Sequence[T]
        | Sequence[U]
        | tuple[T, ...]
        | tuple[U, ...]
        | Mapping[str, T]
        | Mapping[str, U]
    ):
        """Unified filter function with generic type support.

        Filters elements based on predicate while preserving container type.
        Optionally maps filtered items with mapper function.
        Supports lists, tuples, and dicts.
        """
        if isinstance(items, list):
            if mapper is not None:
                return [mapper(item) for item in items if predicate(item)]
            return [item for item in items if predicate(item)]
        if isinstance(items, tuple):
            if mapper is not None:
                mapped_items = [mapper(item) for item in items if predicate(item)]
                return (*mapped_items,)
            filtered_items = [item for item in items if predicate(item)]
            return (*filtered_items,)
        if isinstance(items, Mapping):
            filtered: MutableMapping[str, T] = {
                k: v for k, v in items.items() if predicate(v)
            }
            if mapper is not None:
                return {k: mapper(v) for k, v in filtered.items()}
            return filtered
        return [item for item in items if predicate(item)]

    @staticmethod
    def find(
        items: Sequence[T] | tuple[T, ...] | Mapping[str, T],
        predicate: Callable[[T], bool],
    ) -> p.Result[T]:
        """Find first item matching predicate with generic type support.

        Returns first item where predicate returns True, or None.
        """
        if isinstance(items, (list, tuple)):
            for item in items:
                if predicate(item):
                    return FlextUtilitiesCollection._ok_result(item)
            return r[T].fail(c.ERR_COLLECTION_NO_MATCHING_ITEM_FOUND)
        if isinstance(items, Mapping):
            for v in items.values():
                if predicate(v):
                    return FlextUtilitiesCollection._ok_result(v)
        return r[T].fail(c.ERR_COLLECTION_NO_MATCHING_ITEM_FOUND)

    @overload
    @staticmethod
    def map(items: Sequence[T], mapper: Callable[[T], U]) -> Sequence[U]: ...

    @overload
    @staticmethod
    def map(items: tuple[T, ...], mapper: Callable[[T], U]) -> tuple[U, ...]: ...

    @overload
    @staticmethod
    def map(items: Mapping[str, T], mapper: Callable[[T], U]) -> Mapping[str, U]: ...

    @overload
    @staticmethod
    def map(items: set[T], mapper: Callable[[T], U]) -> set[U]: ...

    @overload
    @staticmethod
    def map(items: frozenset[T], mapper: Callable[[T], U]) -> frozenset[U]: ...

    @staticmethod
    def map(
        items: Sequence[T] | tuple[T, ...] | Mapping[str, T] | set[T] | frozenset[T],
        mapper: Callable[[T], U],
    ) -> Sequence[U] | tuple[U, ...] | Mapping[str, U] | set[U] | frozenset[U]:
        """Unified map function with generic type support.

        Transforms elements using mapper function while preserving container type.
        Supports lists, tuples, dicts, sets, and frozensets.
        """
        if isinstance(items, list):
            return [mapper(item) for item in items]
        if isinstance(items, tuple):
            return tuple(mapper(item) for item in items)
        if isinstance(items, Mapping):
            return {k: mapper(v) for k, v in items.items()}
        if isinstance(items, set):
            return {mapper(item) for item in items}
        return frozenset(mapper(item) for item in items)

    @staticmethod
    def _merge_replace(
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
    ) -> p.Result[t.RecursiveContainerMapping]:
        """Replace strategy: base values overwrite other."""
        result: t.MutableRecursiveContainerMapping = dict(other)
        result.update(base)
        return FlextUtilitiesCollection._ok_result(result)

    @staticmethod
    def _merge_filter_none(
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
    ) -> p.Result[t.RecursiveContainerMapping]:
        """Filter-none strategy: skip None values from base."""
        result: t.MutableRecursiveContainerMapping = dict(other)
        for key, value in base.items():
            if value is not None:
                result[key] = value
        return FlextUtilitiesCollection._ok_result(result)

    @staticmethod
    def _merge_filter_empty(
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
    ) -> p.Result[t.RecursiveContainerMapping]:
        """Filter-empty strategy: skip empty values from base."""
        result: t.MutableRecursiveContainerMapping = dict(other)
        for key, value in base.items():
            if not FlextUtilitiesCollection._is_empty_value(value):
                result[key] = value
        return FlextUtilitiesCollection._ok_result(result)

    @staticmethod
    def _merge_append(
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
    ) -> p.Result[t.RecursiveContainerMapping]:
        """Append strategy: concatenate lists instead of replacing."""
        result: t.MutableRecursiveContainerMapping = dict(other)
        for key, value in base.items():
            current_val = result.get(key)
            if (
                current_val is not None
                and isinstance(current_val, list)
                and isinstance(value, list)
            ):
                result[key] = current_val + value
            else:
                result[key] = value
        return FlextUtilitiesCollection._ok_result(result)

    @staticmethod
    def _merge_deep(
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
    ) -> p.Result[t.RecursiveContainerMapping]:
        """Deep strategy: recursively merge nested dicts."""
        result: t.MutableRecursiveContainerMapping = dict(other)
        for key, value in base.items():
            merge_result = FlextUtilitiesCollection._merge_deep_single_key(
                result,
                key,
                value,
            )
            if merge_result.failure:
                return r[t.RecursiveContainerMapping].fail(
                    merge_result.error or "Unknown error",
                )
        return FlextUtilitiesCollection._ok_result(result)

    _MergeHandler = Callable[
        [t.RecursiveContainerMapping, t.RecursiveContainerMapping],
        "p.Result[t.RecursiveContainerMapping]",
    ]

    _MERGE_STRATEGIES: ClassVar[Mapping[str, _MergeHandler]] = {
        "replace": _merge_replace,
        "override": _merge_replace,
        "filter_none": _merge_filter_none,
        "filter_empty": _merge_filter_empty,
        "filter_both": _merge_filter_empty,
        "append": _merge_append,
        "deep": _merge_deep,
    }

    @staticmethod
    def merge_mappings(
        other: t.RecursiveContainerMapping | None,
        base: t.RecursiveContainerMapping,
        *,
        strategy: str = "deep",
    ) -> p.Result[t.RecursiveContainerMapping]:
        """Merge two dictionaries with configurable strategy.

        Strategies:
        - "deep": Deep merge nested dicts (default)
        - "replace": Replace all values from base
        - "override": Same as replace (alias)
        - "append": Append lists instead of replacing
        - "filter_none": Skip None values from base
        - "filter_empty": Skip empty values (None, "", [], {}) from base
        - "filter_both": Same as filter_empty (alias)
        """
        if other is None:
            msg = "merge_mappings requires an iterable mapping for 'other', got None"
            raise TypeError(msg)
        handler = FlextUtilitiesCollection._MERGE_STRATEGIES.get(strategy)
        if handler is None:
            return r[t.RecursiveContainerMapping].fail(
                f"Unknown merge strategy: {strategy}",
            )
        return handler(other, base)

    @staticmethod
    def process(
        items: Sequence[T],
        processor: Callable[[T], U],
        *,
        predicate: Callable[[T], bool] | None = None,
        on_error: str = "fail",
    ) -> p.Result[Sequence[U]]:
        """Process items with optional filtering; on_error="skip" to skip failures."""
        results: MutableSequence[U] = []
        for item in items:
            item_typed: T = item
            if predicate is not None and (not predicate(item_typed)):
                continue
            process_result = r[U].create_from_callable(
                lambda current_item=item_typed: processor(current_item),
            )
            if process_result.failure:
                if on_error == "skip":
                    continue
                return r[Sequence[U]].fail(
                    c.ERR_COLLECTION_PROCESSING_FAILED_FOR_ITEM.format(item=item),
                )
            processed_item: U = process_result.unwrap()
            results.append(processed_item)
        return FlextUtilitiesCollection._ok_result(results)


__all__: list[str] = ["FlextUtilitiesCollection"]
