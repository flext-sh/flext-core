"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import cast

from flext_core.result import r
from flext_core.typings import R, T, U, t


class FlextUtilitiesCollection:
    """Utilities for collection operations with full generic type support."""

    @staticmethod
    def map(
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
    def filter(
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
                    return item  # type: ignore[no-any-return]
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
                    current_dict,
                    value,
                    strategy="deep",
                )
                if merged.is_success:
                    result[key] = merged.value
                    return r.ok(None)
                return r[None].fail(
                    f"Failed to merge nested dict for key {key}: {merged.error}",
                )
        result[key] = value
        return r.ok(None)

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
                        result,
                        key,
                        value,
                    )
                    if merge_result.is_failure:
                        return r[dict[str, t.GeneralValueType]].fail(
                            merge_result.error or "Unknown error",
                        )
                return r[dict[str, t.GeneralValueType]].ok(result)

            return r[dict[str, t.GeneralValueType]].fail(
                f"Unknown merge strategy: {strategy}",
            )
        except Exception as e:
            return r[dict[str, t.GeneralValueType]].fail(f"Merge failed: {e}")

    @staticmethod
    def batch(
        items: Sequence[T],
        operation: Callable[[T], R | r[R]],
        *,
        size: int | None = None,
        on_error: str | None = None,
        parallel: bool = False,
        progress: Callable[[int, int], None] | None = None,
        progress_interval: int = 1,
        pre_validate: Callable[[T], bool] | None = None,
        _flatten: bool = False,
    ) -> r[t.BatchResultDict]:
        """Process items in batches with progress tracking.

        Args:
            items: Items to process
            operation: Function that returns R or r[R]
            parallel: Enable parallel processing (not implemented, for signature compatibility)
            progress: Callback for progress tracking
            progress_interval: How often to call progress callback
            pre_validate: Optional validation function

        """
        _ = size
        _ = on_error
        _ = parallel
        _ = progress_interval
        try:
            results: list[object] = []
            total = len(items)

            for processed, item in enumerate(items, 1):
                # Pre-validate if validator provided
                # Type narrowing: item is T
                item_typed: T = item
                if pre_validate is not None and not pre_validate(item_typed):
                    results.append(None)
                    continue

                result = operation(item)
                # Handle both direct returns and FlextResult returns
                has_result_attrs = hasattr(result, "is_success") and hasattr(
                    result,
                    "value",
                )
                if has_result_attrs:
                    # It's a FlextResult - use getattr for safe access
                    is_success = getattr(result, "is_success", False)
                    if is_success:
                        value = getattr(result, "value", None)
                        results.append(value)
                    else:
                        results.append(None)
                else:
                    # It's a direct return
                    results.append(result)

                # Track progress
                if progress is not None and processed % progress_interval == 0:
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
    def process(
        items: Sequence[T],
        processor: Callable[[T], U],
        predicate: Callable[[T], bool] | None = None,
        on_error: str = "fail",
    ) -> r[list[U]]:
        """Process items with optional filtering and error handling.

        Transforms items using processor, optionally filtering with predicate.

        Args:
            items: Items to process
            processor: Function to transform each item
            predicate: Optional filter function (applied before processor)
            on_error: "fail" to abort on error, "skip" to skip failed items

        Returns:
            FlextResult with list of processed results or error

        """
        try:
            results: list[U] = []
            for item in items:
                # Type narrowing: item is T
                item_typed: T = item
                # Check predicate - item is T from Sequence[T]
                if predicate is not None and not predicate(item_typed):
                    continue

                try:
                    result = processor(item)
                    results.append(result)
                except Exception:
                    if on_error == "skip":
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

    @staticmethod
    def coerce_dict_validator[T](
        target_type: type[T],
    ) -> Callable[[dict[str, object]], dict[str, T]]:
        """Create a validator function for dictionaries with value coercion.

        Args:
            target_type: Type to coerce values to (e.g., StrEnum, int, str)

        Returns:
            Validator function that takes a dict and returns dict with coerced values

        Raises:
            TypeError: If input is not a dict or values cannot be coerced

        """

        def validator(data: dict[str, object]) -> dict[str, T]:

            result: dict[str, T] = {}
            for key, value in data.items():
                try:
                    if isinstance(value, target_type):
                        # Value is already of correct type
                        result[key] = value
                    elif isinstance(target_type, type) and issubclass(
                        target_type, StrEnum
                    ):
                        # Handle StrEnum conversion
                        if not isinstance(value, str):
                            msg = "Expected str"
                            raise TypeError(msg)
                        result[key] = target_type(value)
                    # General type coercion
                    elif target_type is object:
                        # object() doesn't accept arguments, just assign as-is
                        result[key] = cast("T", value)
                    else:
                        result[key] = cast("T", target_type(value))
                except ValueError as e:
                    # For enum validation errors, re-raise as ValueError
                    type_name = getattr(target_type, "__name__", "Unknown")
                    msg = f"Invalid {type_name}"
                    raise ValueError(msg) from e
                except TypeError:
                    # For type errors, re-raise as TypeError
                    raise

            return result

        return validator

    @staticmethod
    def coerce_list_validator[T](
        target_type: type[T],
    ) -> Callable[[object], list[T]]:
        """Create a validator function for lists/sequences with value coercion.

        Args:
            target_type: Type to coerce values to (e.g., StrEnum, int, str)

        Returns:
            Validator function that takes a sequence and returns list with coerced values

        Raises:
            TypeError: If input is not a sequence or values cannot be coerced

        """

        def validator(data: object) -> list[T]:
            if not isinstance(data, (list, tuple, set, frozenset)):
                msg = f"Expected sequence, got {type(data).__name__}"
                raise TypeError(msg)

            result: list[T] = []
            for value in data:
                try:
                    if isinstance(value, target_type):
                        # Value is already of correct type
                        result.append(value)
                    elif isinstance(target_type, type) and issubclass(
                        target_type, StrEnum
                    ):
                        # Handle StrEnum conversion
                        if not isinstance(value, str):
                            msg = "Expected str"
                            raise TypeError(msg)
                        result.append(target_type(value))
                    # General type coercion
                    # Handle special case for object type
                    elif target_type is object:
                        result.append(value)
                    else:
                        result.append(target_type(value))
                except ValueError as e:
                    # For enum validation errors, re-raise as ValueError
                    type_name = getattr(target_type, "__name__", "Unknown")
                    msg = f"Invalid {type_name}"
                    raise ValueError(msg) from e
                except TypeError:
                    # For type errors, re-raise as TypeError
                    raise

            return result

        return validator


__all__ = [
    "FlextUtilitiesCollection",
]
