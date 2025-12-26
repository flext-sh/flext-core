"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import overload

from flext_core.result import r
from flext_core.typings import R, T, U, t


class FlextUtilitiesCollection:
    """Utilities for collection operations with full generic type support."""

    # =========================================================================
    # Public overloaded map function - inlined implementations
    # =========================================================================

    @overload
    @staticmethod
    def map(
        items: list[T],
        mapper: Callable[[T], U],
    ) -> list[U]: ...

    @overload
    @staticmethod
    def map(
        items: tuple[T, ...],
        mapper: Callable[[T], U],
    ) -> tuple[U, ...]: ...

    @overload
    @staticmethod
    def map(
        items: dict[str, T],
        mapper: Callable[[T], U],
    ) -> dict[str, U]: ...

    @overload
    @staticmethod
    def map(
        items: set[T],
        mapper: Callable[[T], U],
    ) -> set[U]: ...

    @overload
    @staticmethod
    def map(
        items: frozenset[T],
        mapper: Callable[[T], U],
    ) -> frozenset[U]: ...

    @staticmethod
    def map(
        items: list[T] | tuple[T, ...] | dict[str, T] | set[T] | frozenset[T],
        mapper: Callable[[T], U],
    ) -> list[U] | tuple[U, ...] | dict[str, U] | set[U] | frozenset[U]:
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
        msg = f"Unsupported collection type: {type(items)}"
        raise TypeError(msg)

    # =========================================================================
    # Public overloaded filter function - inlined implementations
    # =========================================================================

    @overload
    @staticmethod
    def filter(
        items: list[T],
        predicate: Callable[[T], bool],
        *,
        mapper: None = None,
    ) -> list[T]: ...

    @overload
    @staticmethod
    def filter(
        items: list[T],
        predicate: Callable[[T], bool],
        *,
        mapper: Callable[[T], U],
    ) -> list[U]: ...

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
    ) -> dict[str, T]: ...

    @overload
    @staticmethod
    def filter(
        items: Mapping[str, T],
        predicate: Callable[[T], bool],
        *,
        mapper: Callable[[T], U],
    ) -> dict[str, U]: ...

    @staticmethod
    def filter(
        items: list[T] | tuple[T, ...] | Mapping[str, T],
        predicate: Callable[[T], bool],
        *,
        mapper: Callable[[T], U] | None = None,
    ) -> (
        list[T] | list[U] | tuple[T, ...] | tuple[U, ...] | dict[str, T] | dict[str, U]
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
                # Use tuple literal with unpacking for better type inference
                mapped_items = [mapper(item) for item in items if predicate(item)]
                return (*mapped_items,)
            filtered_items = [item for item in items if predicate(item)]
            return (*filtered_items,)
        if isinstance(items, Mapping):
            filtered = {k: v for k, v in items.items() if predicate(v)}
            if mapper is not None:
                return {k: mapper(v) for k, v in filtered.items()}
            return filtered
        msg = f"Unsupported collection type: {type(items)}"
        raise TypeError(msg)

    @staticmethod
    def find(
        items: list[T] | tuple[T, ...] | Mapping[str, T],
        predicate: Callable[[T], bool],
    ) -> T | None:
        """Find first item matching predicate with generic type support.

        Returns first item where predicate returns True, or None.
        """
        if isinstance(items, (list, tuple)):
            for item in items:
                if predicate(item):
                    return item
        elif isinstance(items, Mapping):
            for v in items.values():
                if predicate(v):
                    return v
        return None

    @staticmethod
    def _merge_deep_single_key(
        result: dict[str, t.GeneralValueType],
        key: str,
        value: t.GeneralValueType,
    ) -> r[bool]:
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
                    return r[bool].ok(True)
                return r[bool].fail(
                    f"Failed to merge nested dict for key {key}: {merged.error}",
                )
        result[key] = value
        return r[bool].ok(True)

    @staticmethod
    def _is_empty_value(value: t.GeneralValueType) -> bool:
        """Check if value is considered empty (empty string, empty list, etc.)."""
        if value is None:
            return True
        if isinstance(value, str) and not value:
            return True
        return isinstance(value, (list, dict)) and len(value) == 0

    @staticmethod
    def merge(
        base: dict[str, t.GeneralValueType],
        other: dict[str, t.GeneralValueType],
        *,
        strategy: str = "deep",
    ) -> r[dict[str, t.GeneralValueType]]:
        """Merge two dictionaries with configurable strategy.

        Strategies:
        - "deep": Deep merge nested dicts (default)
        - "replace": Replace all values from other
        - "override": Same as replace (alias)
        - "append": Append lists instead of replacing
        - "filter_none": Skip None values from other
        - "filter_empty": Skip empty values (None, "", [], {}) from other
        - "filter_both": Same as filter_empty (alias)
        """
        try:
            if strategy in {"replace", "override"}:
                result: dict[str, t.GeneralValueType] = dict(base)
                result.update(other)
                return r[dict[str, t.GeneralValueType]].ok(result)

            if strategy == "filter_none":
                result = dict(base)
                for key, value in other.items():
                    if value is not None:
                        result[key] = value
                return r[dict[str, t.GeneralValueType]].ok(result)

            if strategy in {"filter_empty", "filter_both"}:
                result = dict(base)
                for key, value in other.items():
                    if not FlextUtilitiesCollection._is_empty_value(value):
                        result[key] = value
                return r[dict[str, t.GeneralValueType]].ok(result)

            if strategy == "append":
                result = dict(base)
                for key, value in other.items():
                    if (
                        key in result
                        and isinstance(result[key], list)
                        and isinstance(value, list)
                    ):
                        # Append lists
                        base_list = result[key]
                        if isinstance(base_list, list):
                            result[key] = base_list + value
                    else:
                        result[key] = value
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
        _size: int | None = None,  # Alias for size
        on_error: str | None = None,
        parallel: bool = False,
        progress: Callable[[int, int], None] | None = None,
        progress_interval: int = 1,
        pre_validate: Callable[[T], bool] | None = None,
        flatten: bool = False,
        _flatten: bool = False,  # Legacy alias
    ) -> r[t.BatchResultDict]:
        """Process items in batches with progress tracking.

        Args:
            items: Items to process
            operation: Function that returns R or r[R]
            size: Batch size (not used, for signature compatibility)
            on_error: "fail" to stop, "skip" to silently skip, "collect" to collect errors
            parallel: Enable parallel processing (not implemented, for signature compatibility)
            progress: Callback for progress tracking
            progress_interval: How often to call progress callback
            pre_validate: Optional validation function
            flatten: Flatten list results

        """
        _ = size or _size
        _ = parallel
        _ = progress_interval
        do_flatten = flatten or _flatten
        error_mode = on_error or "fail"
        results: list[object] = []
        errors: list[tuple[int, str]] = []
        total = len(items)

        for processed, item in enumerate(items, 1):
            # Pre-validate if validator provided
            item_typed: T = item
            if pre_validate is not None and not pre_validate(item_typed):
                results.append(None)
                continue

            try:
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
                        if do_flatten and isinstance(value, list):
                            results.extend(value)
                        else:
                            results.append(value)
                    else:
                        error_msg = getattr(result, "error", "Unknown error")
                        if error_mode == "fail":
                            return r[t.BatchResultDict].fail(
                                f"Batch processing failed: {error_msg}",
                            )
                        if error_mode == "collect":
                            # Store as (index, error_message) tuple
                            errors.append((processed - 1, str(error_msg)))
                        # skip mode - don't add to errors
                # It's a direct return
                elif do_flatten and isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except Exception as e:
                if error_mode == "fail":
                    return r[t.BatchResultDict].fail(
                        f"Batch processing failed: {e}",
                    )
                if error_mode == "collect":
                    # Store as (index, error_message) tuple
                    errors.append((processed - 1, str(e)))
                # skip mode - silently ignore

            # Track progress
            if progress is not None and processed % progress_interval == 0:
                progress(processed, total)

        result_dict: t.BatchResultDict = {
            "results": results,
            "total": total,
            "success_count": len(results),
            "error_count": len(errors),
            "errors": errors,
        }
        return r[t.BatchResultDict].ok(result_dict)

    @staticmethod
    def process(
        items: Sequence[T],
        processor: Callable[[T], U],
        *,
        predicate: Callable[[T], bool] | None = None,
        on_error: str = "fail",
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> r[list[U]]:
        """Process items with optional filtering and error handling.

        Transforms items using processor, optionally filtering with predicate.

        Args:
            items: Items to process
            processor: Function to transform each item
            predicate: Optional filter function (applied before processor)
            on_error: "fail" to abort on error, "skip" to skip failed items
            filter_keys: Only include items with these keys (for dict items)
            exclude_keys: Exclude items with these keys (for dict items)

        Returns:
            FlextResult with list of processed results or error

        """
        _ = filter_keys  # Documented for dict processing, applied in subclasses
        _ = exclude_keys  # Documented for dict processing, applied in subclasses

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
    def _coerce_value_to_str(value: t.GeneralValueType) -> str:
        """Coerce a value to string."""
        return str(value)

    @staticmethod
    def _coerce_value_to_int(value: t.GeneralValueType) -> int:
        """Coerce a value to int."""
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return int(str(value))

    @staticmethod
    def _coerce_value_to_float(value: t.GeneralValueType) -> float:
        """Coerce a value to float."""
        if isinstance(value, float):
            return value
        return float(str(value))

    @staticmethod
    def _coerce_value_to_bool(value: t.GeneralValueType) -> bool:
        """Coerce a value to bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes"}
        return bool(value)

    @staticmethod
    def coerce_dict_to_str() -> Callable[
        [dict[str, t.GeneralValueType]], dict[str, str]
    ]:
        """Create validator that coerces dict values to str."""

        def validator(data: dict[str, t.GeneralValueType]) -> dict[str, str]:
            return {
                k: FlextUtilitiesCollection._coerce_value_to_str(v)
                for k, v in data.items()
            }

        return validator

    @staticmethod
    def coerce_dict_to_int() -> Callable[
        [dict[str, t.GeneralValueType]], dict[str, int]
    ]:
        """Create validator that coerces dict values to int."""

        def validator(data: dict[str, t.GeneralValueType]) -> dict[str, int]:
            return {
                k: FlextUtilitiesCollection._coerce_value_to_int(v)
                for k, v in data.items()
            }

        return validator

    @staticmethod
    def coerce_dict_to_float() -> Callable[
        [dict[str, t.GeneralValueType]], dict[str, float]
    ]:
        """Create validator that coerces dict values to float."""

        def validator(data: dict[str, t.GeneralValueType]) -> dict[str, float]:
            return {
                k: FlextUtilitiesCollection._coerce_value_to_float(v)
                for k, v in data.items()
            }

        return validator

    @staticmethod
    def coerce_dict_to_bool() -> Callable[
        [dict[str, t.GeneralValueType]], dict[str, bool]
    ]:
        """Create validator that coerces dict values to bool."""

        def validator(data: dict[str, t.GeneralValueType]) -> dict[str, bool]:
            return {
                k: FlextUtilitiesCollection._coerce_value_to_bool(v)
                for k, v in data.items()
            }

        return validator

    @staticmethod
    def coerce_dict_to_enum[E: StrEnum](
        enum_type: type[E],
    ) -> Callable[[dict[str, t.GeneralValueType]], dict[str, E]]:
        """Create validator that coerces dict values to a StrEnum type."""

        def validator(data: dict[str, t.GeneralValueType]) -> dict[str, E]:
            result: dict[str, E] = {}
            for k, v in data.items():
                if isinstance(v, enum_type):
                    result[k] = v
                elif isinstance(v, str):
                    result[k] = enum_type(v)
                else:
                    msg = f"Expected str for enum conversion, got {type(v).__name__}"
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def coerce_list_to_str() -> Callable[[Sequence[t.GeneralValueType]], list[str]]:
        """Create validator that coerces sequence values to str."""

        def validator(data: Sequence[t.GeneralValueType]) -> list[str]:
            return [FlextUtilitiesCollection._coerce_value_to_str(v) for v in data]

        return validator

    @staticmethod
    def coerce_list_to_int() -> Callable[[Sequence[t.GeneralValueType]], list[int]]:
        """Create validator that coerces sequence values to int."""

        def validator(data: Sequence[t.GeneralValueType]) -> list[int]:
            return [FlextUtilitiesCollection._coerce_value_to_int(v) for v in data]

        return validator

    @staticmethod
    def coerce_list_to_float() -> Callable[[Sequence[t.GeneralValueType]], list[float]]:
        """Create validator that coerces sequence values to float."""

        def validator(data: Sequence[t.GeneralValueType]) -> list[float]:
            return [FlextUtilitiesCollection._coerce_value_to_float(v) for v in data]

        return validator

    @staticmethod
    def coerce_list_to_bool() -> Callable[[Sequence[t.GeneralValueType]], list[bool]]:
        """Create validator that coerces sequence values to bool."""

        def validator(data: Sequence[t.GeneralValueType]) -> list[bool]:
            return [FlextUtilitiesCollection._coerce_value_to_bool(v) for v in data]

        return validator

    @staticmethod
    def coerce_list_to_enum[E: StrEnum](
        enum_type: type[E],
    ) -> Callable[[Sequence[t.GeneralValueType]], list[E]]:
        """Create validator that coerces sequence values to a StrEnum type."""

        def validator(data: Sequence[t.GeneralValueType]) -> list[E]:
            result: list[E] = []
            for v in data:
                if isinstance(v, enum_type):
                    result.append(v)
                elif isinstance(v, str):
                    result.append(enum_type(v))
                else:
                    msg = f"Expected str for enum conversion, got {type(v).__name__}"
                    raise TypeError(msg)
            return result

        return validator

    # ========================================================================
    # Additional Collection Convenience Methods
    # ========================================================================

    @staticmethod
    def first(
        items: Sequence[T],
        predicate: Callable[[T], bool] | None = None,
        default: T | None = None,
    ) -> T | None:
        """Get first item (optionally matching predicate).

        Args:
            items: Sequence to search
            predicate: Optional filter function
            default: Value to return if no match found

        Returns:
            First matching item or default

        Example:
            user = u.first(users, lambda u: u.is_active)

        """
        for item in items:
            if predicate is None or predicate(item):
                return item
        return default

    @staticmethod
    def last(
        items: Sequence[T],
        predicate: Callable[[T], bool] | None = None,
        default: T | None = None,
    ) -> T | None:
        """Get last item (optionally matching predicate).

        Args:
            items: Sequence to search
            predicate: Optional filter function
            default: Value to return if no match found

        Returns:
            Last matching item or default

        Example:
            last_error = u.last(logs, lambda l: l.level == "error")

        """
        for item in reversed(items):
            if predicate is None or predicate(item):
                return item
        return default

    @staticmethod
    def group_by(
        items: Sequence[T],
        key_func: Callable[[T], U],
    ) -> dict[U, list[T]]:
        """Group items by key function.

        Args:
            items: Items to group
            key_func: Function to extract group key

        Returns:
            Dict mapping keys to lists of items

        Example:
            by_status = u.group_by(users, lambda u: u.status)
            # {"active": [User1, User2], "inactive": [User3]}

        """
        result: dict[U, list[T]] = {}
        for item in items:
            key = key_func(item)
            if key not in result:
                result[key] = []
            result[key].append(item)
        return result

    @staticmethod
    def unique(
        items: Sequence[T],
        key_func: Callable[[T], object] | None = None,
    ) -> list[T]:
        """Get unique items preserving order.

        Args:
            items: Items to deduplicate
            key_func: Optional function to extract uniqueness key

        Returns:
            List of unique items in order of first appearance

        Example:
            unique_emails = u.unique(users, lambda u: u.email.lower())

        """
        seen: set[object] = set()
        result: list[T] = []
        for item in items:
            key = key_func(item) if key_func else item
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def partition(
        items: Sequence[T],
        predicate: Callable[[T], bool],
    ) -> tuple[list[T], list[T]]:
        """Split items by predicate: (matches, non-matches).

        Args:
            items: Items to partition
            predicate: Function to test each item

        Returns:
            Tuple of (matching_items, non_matching_items)

        Example:
            active, inactive = u.partition(users, lambda u: u.is_active)

        """
        matches: list[T] = []
        non_matches: list[T] = []
        for item in items:
            if predicate(item):
                matches.append(item)
            else:
                non_matches.append(item)
        return matches, non_matches

    @staticmethod
    def flatten(items: Sequence[Sequence[T]]) -> list[T]:
        """Flatten nested sequences into single list.

        Args:
            items: Nested sequences to flatten (one level)

        Returns:
            Flattened list

        Example:
            flat = u.flatten([[1, 2], [3, 4], [5]])
            # [1, 2, 3, 4, 5]

        """
        result: list[T] = []
        for seq in items:
            result.extend(seq)
        return result

    # ========================================================================
    # Generic Coercion Validators (used by tests)
    # ========================================================================

    @staticmethod
    def coerce_dict_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.GeneralValueType], dict[str, E]]:
        """Create validator that coerces dict values to a StrEnum type.

        Raises:
            TypeError: If input is not a dict or value is not str
            ValueError: If string value is not a valid enum member

        """

        def validator(data: t.GeneralValueType) -> dict[str, E]:
            if not isinstance(data, dict):
                msg = f"Expected dict, got {type(data).__name__}"
                raise TypeError(msg)

            result: dict[str, E] = {}
            for k, v in data.items():
                if isinstance(v, enum_cls):
                    result[str(k)] = v
                elif isinstance(v, str):
                    try:
                        result[str(k)] = enum_cls(v)
                    except ValueError:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} value: '{v}'"
                        raise ValueError(msg) from None
                else:
                    msg = f"Expected str for enum conversion, got {type(v).__name__}"
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def coerce_list_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.GeneralValueType], list[E]]:
        """Create validator that coerces list values to a StrEnum type.

        Raises:
            TypeError: If input is not a sequence or item is not str
            ValueError: If string value is not a valid enum member

        """

        def validator(data: t.GeneralValueType) -> list[E]:
            # Check for sequence type (but not str which is also a sequence)
            if isinstance(data, str) or not isinstance(data, Sequence):
                msg = f"Expected sequence, got {type(data).__name__}"
                raise TypeError(msg)

            result: list[E] = []
            for v in data:
                if isinstance(v, enum_cls):
                    result.append(v)
                elif isinstance(v, str):
                    try:
                        result.append(enum_cls(v))
                    except ValueError:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} value: '{v}'"
                        raise ValueError(msg) from None
                else:
                    msg = f"Expected str for enum conversion, got {type(v).__name__}"
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def parse_mapping[E: StrEnum](
        enum_cls: type[E],
        mapping: Mapping[str, str | E],
    ) -> r[dict[str, E]]:
        """Parse dict values from strings to StrEnum.

        Args:
            enum_cls: StrEnum class to parse values to
            mapping: Dict with string or enum values

        Returns:
            FlextResult with parsed dict

        Example:
            result = u.Collection.parse_mapping(Status, {"key": "active"})
            # result.value == {"key": Status.ACTIVE}

        """
        try:
            result: dict[str, E] = {}
            errors: list[str] = []

            for key, value in mapping.items():
                if isinstance(value, enum_cls):
                    result[key] = value
                elif isinstance(value, str):
                    try:
                        result[key] = enum_cls(value)
                    except ValueError:
                        errors.append(f"'{key}': '{value}'")
                else:
                    errors.append(f"'{key}': invalid type {type(value).__name__}")

            if errors:
                enum_name = getattr(enum_cls, "__name__", "Enum")
                return r[dict[str, E]].fail(
                    f"Invalid {enum_name} values: {', '.join(errors)}",
                )
            return r[dict[str, E]].ok(result)
        except Exception as e:
            return r[dict[str, E]].fail(f"Parse mapping failed: {e}")

    @staticmethod
    def count(
        items: Sequence[T],
        predicate: Callable[[T], bool] | None = None,
    ) -> int:
        """Count items, optionally matching predicate.

        Args:
            items: Sequence to count
            predicate: Optional filter function

        Returns:
            Count of matching items

        Example:
            active_count = u.Collection.count(users, lambda u: u.is_active)

        """
        if predicate is None:
            return len(items)
        return sum(1 for item in items if predicate(item))

    @staticmethod
    def group(
        items: Sequence[T],
        key_func: Callable[[T], U],
    ) -> dict[U, list[T]]:
        """Group items by key function.

        This is an alias for group_by for convenience.
        """
        return FlextUtilitiesCollection.group_by(items, key_func)

    @staticmethod
    def chunk(items: Sequence[T], size: int) -> list[list[T]]:
        """Split sequence into chunks of specified size.

        Args:
            items: Sequence to split
            size: Maximum size of each chunk

        Returns:
            List of chunks

        Example:
            batches = u.Collection.chunk(records, 100)
            # [[record1, ..., record100], [record101, ...], ...]

        """
        if size <= 0:
            return [list(items)]
        return [list(items[i : i + size]) for i in range(0, len(items), size)]

    @staticmethod
    def mul(*values: float) -> int | float:
        """Multiply values.

        Args:
            *values: Values to multiply

        Returns:
            Product of all values

        Example:
            total = u.mul(price, quantity, tax_rate)

        """
        result: int | float = 1
        for v in values:
            result *= v
        return result


__all__ = [
    "FlextUtilitiesCollection",
]
