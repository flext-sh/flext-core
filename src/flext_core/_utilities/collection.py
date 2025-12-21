"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum

from flext_core.result import r
from flext_core.typings import T, U, t


class FlextUtilitiesCollection:
    """Utilities for collection operations with full generic type support."""

    @staticmethod
    def map[T, U](
        items: Sequence[T] | Mapping[str, T] | set[T] | frozenset[T],
        mapper: Callable[[T], U],
    ) -> Sequence[U] | Mapping[str, U] | set[U] | frozenset[U]:
        """Unified map function with generic type support.

        Transforms elements using mapper function while preserving container type.
        Supports lists, tuples, dicts, sets, and frozensets.
        """
        if isinstance(items, list):
            return [mapper(item) for item in items]
        if isinstance(items, tuple):
            return tuple(mapper(item) for item in items)
        if isinstance(items, dict):
            return {k: mapper(v) for k, v in items.items()}
        if isinstance(items, set):
            return {mapper(item) for item in items}
        if isinstance(items, frozenset):
            return frozenset(mapper(item) for item in items)
        # Unreachable - all supported types handled above
        msg = f"Unsupported collection type: {type(items)}"
        raise TypeError(msg)

    @staticmethod
    def filter[T](
        items: Sequence[T] | Mapping[str, T],
        predicate: Callable[[T], bool],
    ) -> Sequence[T] | Mapping[str, T]:
        """Unified filter function with generic type support.

        Filters elements based on predicate while preserving container type.
        Supports lists, tuples, and dicts.
        Returns same type as input.
        """
        if isinstance(items, list):
            return [item for item in items if predicate(item)]
        if isinstance(items, tuple):
            return tuple(item for item in items if predicate(item))
        if isinstance(items, Mapping):
            return {k: v for k, v in items.items() if predicate(v)}
        return items

    @staticmethod
    def find(
        items: Sequence[object] | Mapping[str, object],
        predicate: Callable[[object], bool],
    ) -> object | None:
        """Find first item matching predicate with generic type support.

        Returns first item where predicate returns True, or None.
        """
        if isinstance(items, (list, tuple)):
            for item in items:
                if predicate(item):
                    return item
        if isinstance(items, Mapping):
            for v in items.values():
                if predicate(v):
                    return v
        return None

    @staticmethod
    def _merge_deep_single_key(
        result: dict[str, t.GeneralValueType],
        key: str,
        value: t.GeneralValueType,
    ) -> r[None]:
        """Merge single key in deep merge strategy."""
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            current_dict = result[key]
            if isinstance(current_dict, dict):
                merged = FlextUtilitiesCollection.merge(
                    current_dict, value, strategy="deep"
                )
                if merged.is_success:
                    result[key] = merged.value
                    return r[None].ok(None)
                return r[None].fail(
                    f"Failed to merge nested dict for key {key}: {merged.error}"
                )
        result[key] = value
        return r[None].ok(None)

    @staticmethod
    def merge(
        base: dict[str, t.GeneralValueType],
        other: dict[str, t.GeneralValueType],
        *,
        strategy: str = "deep",
    ) -> r[dict[str, t.GeneralValueType]]:
        """Merge two dictionaries with configurable strategy."""
        try:
            if strategy == "replace":
                result = base.copy()
                result.update(other)
                return r[dict[str, t.GeneralValueType]].ok(result)

            if strategy == "deep":
                result = base.copy()
                for key, value in other.items():
                    merge_result = FlextUtilitiesCollection._merge_deep_single_key(
                        result, key, value
                    )
                    if merge_result.is_failure:
                        return r[dict[str, t.GeneralValueType]].fail(
                            merge_result.error or "Unknown error"
                        )
                return r[dict[str, t.GeneralValueType]].ok(result)

            return r[dict[str, t.GeneralValueType]].fail(
                f"Unknown merge strategy: {strategy}"
            )
        except Exception as e:
            return r[dict[str, t.GeneralValueType]].fail(f"Merge failed: {e}")

    @staticmethod
    def batch[T, R](
        items: Sequence[T],
        operation: Callable[[T], R | r[R]],
        *,
        _size: int | None = None,
        _on_error: str | None = None,
        _parallel: bool = False,
        progress: Callable[[int, int], None] | None = None,
        _progress_interval: int = 1,
        pre_validate: Callable[[T], bool] | None = None,
        _flatten: bool = False,
    ) -> r[t.BatchResultDict]:
        """Process items in batches with progress tracking.

        Args:
            items: Items to process
            operation: Function that returns R or r[R]
            _parallel: Enable parallel processing (not implemented, for signature compatibility)
            progress: Callback for progress tracking
            _progress_interval: How often to call progress callback
            pre_validate: Optional validation function

        """
        try:
            results: list[object] = []
            total = len(items)

            for processed, item in enumerate(items, 1):
                # Pre-validate if validator provided
                if pre_validate is not None and not pre_validate(item):
                    results.append(None)
                else:
                    result = operation(item)
                    # Handle both direct returns and FlextResult returns
                    if hasattr(result, "is_success") and hasattr(result, "value"):
                        # It's a FlextResult
                        if result.is_success:
                            results.append(result.value)
                        else:
                            results.append(None)
                    else:
                        # It's a direct return - cast to GeneralValueType
                        results.append(result)

                # Track progress
                if progress is not None and processed % _progress_interval == 0:
                    progress(processed, total)

            result_dict: t.BatchResultDict = {
                "results": results,
                "total": total,
                "success_count": len([r for r in results if r is not None]),
                "error_count": len([r for r in results if r is None]),
                "errors": [],
            }
            return r[t.BatchResultDict].ok(result_dict)
        except Exception as e:
            return r[t.BatchResultDict].fail(f"Batch processing failed: {e}")

    @staticmethod
    def process[T, U](
        items: Sequence[T],
        processor: Callable[[T], U],
        predicate: Callable[[T], bool] | None = None,
        _on_error: str = "fail",
    ) -> r[list[U]]:
        """Process items with optional filtering and error handling.

        Transforms items using processor, optionally filtering with predicate.

        Args:
            items: Items to process
            processor: Function to transform each item
            predicate: Optional filter function (applied before processor)
            _on_error: "fail" to abort on error, "skip" to skip failed items

        Returns:
            FlextResult with list of processed results or error

        """
        try:
            results: list[U] = []
            for item in items:
                if predicate is None or predicate(item):
                    try:
                        result = processor(item)
                        results.append(result)
                    except Exception:
                        if _on_error == "skip":
                            continue
                        return r[list[U]].fail(f"Processing failed for item: {item}")
            return r[list[U]].ok(results)
        except Exception as e:
            return r[list[U]].fail(f"Process failed: {e}")

    @staticmethod
    def parse_sequence(
        enum_cls: type[StrEnum],
        values: Sequence[str | StrEnum],
    ) -> r[tuple[StrEnum, ...]]:
        """Parse sequence of strings to tuple of StrEnum."""
        try:
            parsed: list[StrEnum] = []
            errors: list[str] = []

            for idx, val in enumerate(values):
                if isinstance(val, enum_cls):
                    parsed.append(val)
                else:
                    try:
                        parsed.append(enum_cls(val))
                    except ValueError:
                        errors.append(f"[{idx}]: '{val}'")

            if errors:
                enum_name = getattr(enum_cls, "__name__", "Enum")
                return r[tuple[StrEnum, ...]].fail(
                    f"Invalid {enum_name} values: {', '.join(errors)}",
                )
            return r[tuple[StrEnum, ...]].ok(tuple(parsed))
        except Exception as e:
            return r[tuple[StrEnum, ...]].fail(f"Parse sequence failed: {e}")


__all__ = [
    "FlextUtilitiesCollection",
]
