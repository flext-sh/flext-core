"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, MutableMapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Protocol, TypeGuard, TypeVar, cast, overload, runtime_checkable

from flext_core.result import r
from flext_core.typings import R, T, U, t

_PredicateT_contra = TypeVar("_PredicateT_contra", contravariant=True)


@runtime_checkable
class _Predicate(Protocol[_PredicateT_contra]):
    """Protocol for callable predicates that accept a value and return bool."""

    def __call__(self, value: _PredicateT_contra) -> bool: ...


class _BatchResultCompat(t.BatchResultDict):
    def __getitem__(self, key: str) -> t.GuardInputValue:
        values: Mapping[str, t.GuardInputValue] = {
            "results": self.results,
            "errors": self.errors,
            "total": self.total,
            "success_count": self.success_count,
            "error_count": self.error_count,
        }
        return values[key]


class FlextUtilitiesCollection:
    """Utilities for collection operations with full generic type support."""

    # =========================================================================
    # Type Guards for Runtime Type Narrowing
    # =========================================================================

    @staticmethod
    def _is_general_value_dict(
        value: t.GuardInputValue,
    ) -> TypeGuard[dict[str, t.GuardInputValue]]:
        """Type guard to narrow dict to PayloadValue dict."""
        return isinstance(value, dict)

    @staticmethod
    def _is_general_value_list(
        value: t.GuardInputValue,
    ) -> TypeGuard[list[t.GuardInputValue]]:
        """Type guard to narrow list to PayloadValue list."""
        return value.__class__ is list

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
        return frozenset(mapper(item) for item in items)

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
        items: dict[str, T],
        predicate: Callable[[T], bool],
        *,
        mapper: None = None,
    ) -> dict[str, T]: ...

    @overload
    @staticmethod
    def filter(
        items: dict[str, T],
        predicate: Callable[[T], bool],
        *,
        mapper: Callable[[T], U],
    ) -> dict[str, U]: ...

    @staticmethod
    def filter(
        items: list[T] | tuple[T, ...] | dict[str, T],
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
                mapped_items = [mapper(item) for item in items if predicate(item)]
                return (*mapped_items,)
            filtered_items = [item for item in items if predicate(item)]
            return (*filtered_items,)
        filtered: dict[str, T] = {k: v for k, v in items.items() if predicate(v)}
        if mapper is not None:
            return {k: mapper(v) for k, v in filtered.items()}
        return filtered

    @staticmethod
    def find(
        items: list[T] | tuple[T, ...] | dict[str, T],
        predicate: _Predicate[T],
    ) -> T | None:
        """Find first item matching predicate with generic type support.

        Returns first item where predicate returns True, or None.
        """
        if isinstance(items, list | tuple):
            for item in items:
                result: bool = predicate(item)  # Explicit type for result
                if result:
                    return item
            return None
        for v in items.values():
            matched: bool = predicate(v)  # Explicit type for matched
            if matched:
                return v
        return None

    @staticmethod
    def _merge_deep_single_key(
        result: MutableMapping[str, t.GuardInputValue],
        key: str,
        value: t.GuardInputValue,
    ) -> r[bool]:
        """Merge single key in deep merge strategy."""
        current_val = result.get(key)
        if (
            current_val is not None
            and FlextUtilitiesCollection._is_general_value_dict(current_val)
            and FlextUtilitiesCollection._is_general_value_dict(value)
        ):
            merged = FlextUtilitiesCollection.merge(
                current_val,
                value,
                strategy="deep",
            )
            if merged.is_success:
                result[key] = merged.value
                return r[bool].ok(value=True)
            return r[bool].fail(
                f"Failed to merge nested dict for key {key}: {merged.error}",
            )
        result[key] = value
        return r[bool].ok(value=True)

    @staticmethod
    def _is_empty_value(value: t.GuardInputValue) -> bool:
        """Check if value is considered empty (empty string, empty list, etc.)."""
        if value is None:
            return True
        if value.__class__ is str:
            return not value
        if FlextUtilitiesCollection._is_general_value_list(value):
            return len(value) == 0
        if FlextUtilitiesCollection._is_general_value_dict(value):
            return len(value) == 0
        return False

    @staticmethod
    def merge(
        base: dict[str, t.GuardInputValue],
        other: dict[str, t.GuardInputValue],
        *,
        strategy: str = "deep",
    ) -> r[dict[str, t.GuardInputValue]]:
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
                result: dict[str, t.GuardInputValue] = dict(base)
                result.update(other)
                return r[dict[str, t.GuardInputValue]].ok(result)

            if strategy == "filter_none":
                result = dict(base)
                for key, value in other.items():
                    if value is not None:
                        result[key] = value
                return r[dict[str, t.GuardInputValue]].ok(result)

            if strategy in {"filter_empty", "filter_both"}:
                result = dict(base)
                for key, value in other.items():
                    if not FlextUtilitiesCollection._is_empty_value(value):
                        result[key] = value
                return r[dict[str, t.GuardInputValue]].ok(result)

            if strategy == "append":
                result = dict(base)
                for key, value in other.items():
                    current_val = result.get(key)
                    if (
                        current_val is not None
                        and isinstance(current_val, list)
                        and isinstance(value, list)
                    ):
                        # Append lists - both are now properly typed as lists
                        result[key] = current_val + value
                    else:
                        result[key] = value
                return r[dict[str, t.GuardInputValue]].ok(result)

            if strategy == "deep":
                result = base.copy()
                for key, value in other.items():
                    merge_result = FlextUtilitiesCollection._merge_deep_single_key(
                        result,
                        key,
                        value,
                    )
                    if merge_result.is_failure:
                        return r[dict[str, t.GuardInputValue]].fail(
                            merge_result.error or "Unknown error",
                        )
                return r[dict[str, t.GuardInputValue]].ok(result)

            return r[dict[str, t.GuardInputValue]].fail(
                f"Unknown merge strategy: {strategy}",
            )
        except Exception as e:
            return r[dict[str, t.GuardInputValue]].fail(f"Merge failed: {e}")

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
        results: list[t.GuardInputValue] = []
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
                        value_raw: t.GuardInputValue = getattr(result, "value", None)
                        # Convert to PayloadValue using conversion utility
                        value: t.GuardInputValue = value_raw
                        if do_flatten and isinstance(value, list):
                            # Extend results with all items from the list
                            # Convert each item to PayloadValue
                            results.extend(cast("list[t.GuardInputValue]", value))
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
                # It's a direct return - convert to PayloadValue
                elif do_flatten and isinstance(result, list):
                    # Extend results with all items from the list
                    # Convert each item to PayloadValue
                    results.extend(cast("list[t.GuardInputValue]", result))
                else:
                    typed_result: t.GuardInputValue = cast("t.GuardInputValue", result)
                    results.append(typed_result)
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

        result_dict = _BatchResultCompat(
            results=cast("list[str | int | float | bool | datetime | None]", results),
            total=total,
            success_count=len(results),
            error_count=len(errors),
            errors=errors,
        )
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
    def _coerce_value_to_str(value: t.GuardInputValue) -> str:
        """Coerce a value to string."""
        return str(value)

    @staticmethod
    def _coerce_value_to_int(value: t.GuardInputValue) -> int:
        """Coerce a value to int."""
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return int(str(value))

    @staticmethod
    def _coerce_value_to_float(value: t.GuardInputValue) -> float:
        """Coerce a value to float."""
        if isinstance(value, float):
            return value
        return float(str(value))

    @staticmethod
    def _coerce_value_to_bool(value: t.GuardInputValue) -> bool:
        """Coerce a value to bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes"}
        return bool(value)

    @staticmethod
    def coerce_dict_to_str() -> Callable[
        [Mapping[str, t.GuardInputValue]],
        Mapping[str, str],
    ]:
        """Create validator that coerces dict values to str."""

        def validator(data: Mapping[str, t.GuardInputValue]) -> Mapping[str, str]:
            return {
                k: FlextUtilitiesCollection._coerce_value_to_str(v)
                for k, v in data.items()
            }

        return validator

    @staticmethod
    def coerce_dict_to_int() -> Callable[
        [Mapping[str, t.GuardInputValue]],
        Mapping[str, int],
    ]:
        """Create validator that coerces dict values to int."""

        def validator(data: Mapping[str, t.GuardInputValue]) -> Mapping[str, int]:
            return {
                k: FlextUtilitiesCollection._coerce_value_to_int(v)
                for k, v in data.items()
            }

        return validator

    @staticmethod
    def coerce_dict_to_float() -> Callable[
        [Mapping[str, t.GuardInputValue]],
        Mapping[str, float],
    ]:
        """Create validator that coerces dict values to float."""

        def validator(data: Mapping[str, t.GuardInputValue]) -> Mapping[str, float]:
            return {
                k: FlextUtilitiesCollection._coerce_value_to_float(v)
                for k, v in data.items()
            }

        return validator

    @staticmethod
    def coerce_dict_to_bool() -> Callable[
        [Mapping[str, t.GuardInputValue]],
        Mapping[str, bool],
    ]:
        """Create validator that coerces dict values to bool."""

        def validator(data: Mapping[str, t.GuardInputValue]) -> Mapping[str, bool]:
            return {
                k: FlextUtilitiesCollection._coerce_value_to_bool(v)
                for k, v in data.items()
            }

        return validator

    @staticmethod
    def coerce_dict_to_enum[E: StrEnum](
        enum_type: type[E],
    ) -> Callable[[dict[str, t.GuardInputValue]], dict[str, E]]:
        """Create validator that coerces dict values to a StrEnum type."""

        def validator(data: dict[str, t.GuardInputValue]) -> dict[str, E]:
            result: dict[str, E] = {}
            for k, v in data.items():
                if isinstance(v, enum_type):
                    result[k] = v
                elif v.__class__ is str:
                    result[k] = enum_type(v)
                else:
                    msg = (
                        f"Expected str for enum conversion, got {v.__class__.__name__}"
                    )
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def coerce_list_to_str() -> Callable[[Sequence[t.GuardInputValue]], list[str]]:
        """Create validator that coerces sequence values to str."""

        def validator(data: Sequence[t.GuardInputValue]) -> list[str]:
            return [FlextUtilitiesCollection._coerce_value_to_str(v) for v in data]

        return validator

    @staticmethod
    def coerce_list_to_int() -> Callable[[Sequence[t.GuardInputValue]], list[int]]:
        """Create validator that coerces sequence values to int."""

        def validator(data: Sequence[t.GuardInputValue]) -> list[int]:
            return [FlextUtilitiesCollection._coerce_value_to_int(v) for v in data]

        return validator

    @staticmethod
    def coerce_list_to_float() -> Callable[[Sequence[t.GuardInputValue]], list[float]]:
        """Create validator that coerces sequence values to float."""

        def validator(data: Sequence[t.GuardInputValue]) -> list[float]:
            return [FlextUtilitiesCollection._coerce_value_to_float(v) for v in data]

        return validator

    @staticmethod
    def coerce_list_to_bool() -> Callable[[Sequence[t.GuardInputValue]], list[bool]]:
        """Create validator that coerces sequence values to bool."""

        def validator(data: Sequence[t.GuardInputValue]) -> list[bool]:
            return [FlextUtilitiesCollection._coerce_value_to_bool(v) for v in data]

        return validator

    @staticmethod
    def coerce_list_to_enum[E: StrEnum](
        enum_type: type[E],
    ) -> Callable[[Sequence[t.GuardInputValue]], list[E]]:
        """Create validator that coerces sequence values to a StrEnum type."""

        def validator(data: Sequence[t.GuardInputValue]) -> list[E]:
            result: list[E] = []
            for v in data:
                if isinstance(v, enum_type):
                    result.append(v)
                elif v.__class__ is str:
                    result.append(enum_type(v))
                else:
                    msg = (
                        f"Expected str for enum conversion, got {v.__class__.__name__}"
                    )
                    raise TypeError(msg)
            return result

        return validator

    # ========================================================================
    # Additional Collection Convenience Methods
    # ========================================================================

    @staticmethod
    def first(
        items: Sequence[T],
        predicate: _Predicate[T] | None = None,
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
            if predicate is None:
                return item
            result: bool = predicate(item)
            if result:
                return item
        return default

    @staticmethod
    def last(
        items: Sequence[T],
        predicate: _Predicate[T] | None = None,
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
            if predicate is None:
                return item
            result: bool = predicate(item)
            if result:
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
        key_func: Callable[[T], Hashable] | None = None,
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
        seen: set[Hashable] = set()
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
        predicate: _Predicate[T],
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
            result: bool = predicate(item)
            if result:
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
    ) -> Callable[[t.GuardInputValue], dict[str, E]]:
        """Create validator that coerces dict values to a StrEnum type.

        Raises:
            TypeError: If input is not a dict or value is not str
            ValueError: If string value is not a valid enum member

        """

        def validator(data: t.GuardInputValue) -> dict[str, E]:
            if not isinstance(data, dict):
                msg = f"Expected dict, got {data.__class__.__name__}"
                raise TypeError(msg)

            result: dict[str, E] = {}
            # Iterate directly using items() - pyright reports partial unknown for PayloadValue
            # This is acceptable as we validate each value with isinstance checks
            for k_raw in data:
                v_raw = data[k_raw]
                # Convert key to string
                k = str(k_raw)
                # Check value type with isinstance to narrow
                if isinstance(v_raw, enum_cls):
                    result[k] = v_raw
                elif v_raw.__class__ is str:
                    try:
                        result[k] = enum_cls(v_raw)
                    except ValueError:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} value: '{v_raw}'"
                        raise ValueError(msg) from None
                else:
                    msg = (
                        "Expected str for enum conversion, got "
                        f"{v_raw.__class__.__name__}"
                    )
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def coerce_list_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.GuardInputValue], list[E]]:
        """Create validator that coerces list values to a StrEnum type.

        Raises:
            TypeError: If input is not a sequence or item is not str
            ValueError: If string value is not a valid enum member

        """

        def validator(data: t.GuardInputValue) -> list[E]:
            # Check for sequence type (but not str which is also a sequence)
            if isinstance(data, str) or not isinstance(data, Sequence):
                msg = f"Expected sequence, got {data.__class__.__name__}"
                raise TypeError(msg)

            result: list[E] = []
            # Iterate directly - pyright reports partial unknown for PayloadValue
            # This is acceptable as we validate each value with isinstance checks
            for v_raw in data:
                if isinstance(v_raw, enum_cls):
                    result.append(v_raw)
                elif v_raw.__class__ is str:
                    try:
                        result.append(enum_cls(v_raw))
                    except ValueError:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} value: '{v_raw}'"
                        raise ValueError(msg) from None
                else:
                    msg = (
                        "Expected str for enum conversion, got "
                        f"{v_raw.__class__.__name__}"
                    )
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def parse_mapping[E: StrEnum](
        enum_cls: type[E],
        mapping: dict[str, str | E],
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

            for key, value_raw in mapping.items():
                # StrEnum is a subclass of str, so value can be str or StrEnum
                # Try conversion - enum_cls(str_value) creates enum, enum_cls(enum_value) returns same enum
                try:
                    # value_raw is str | StrEnum due to function signature
                    # pyright warns isinstance is unnecessary since StrEnum extends str
                    # We just try the conversion which works for both
                    result[key] = enum_cls(value_raw)
                except (ValueError, TypeError) as e:
                    errors.append(f"'{key}': {e}")

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

    @staticmethod
    def extract_mapping_items[K, V](
        mapping: Mapping[K, V],
    ) -> list[tuple[str, t.GuardInputValue]]:
        """Extract mapping items as typed list for iteration.

        Helper function to properly type narrow Mapping.items() for pyright.
        Converts keys to strings and values to PayloadValue.

        Args:
            mapping: Mapping to extract items from

        Returns:
            List of (key, value) tuples with proper typing

        """
        result: list[tuple[str, t.GuardInputValue]] = []
        items_iter = mapping.items()
        for item_tuple in items_iter:
            key_obj = item_tuple[0]
            value_raw = item_tuple[1]
            key_str: str = str(key_obj)
            value_typed: t.GuardInputValue = cast("t.GuardInputValue", value_raw)
            result.append((key_str, value_typed))
        return result

    @staticmethod
    def extract_callable_mapping[K, V](
        mapping: Mapping[K, V],
    ) -> dict[str, Callable[[], t.GuardInputValue]]:
        """Extract mapping of callables for resources/factories.

        Helper function to properly type narrow callable mappings for pyright.
        Filters to only callable values and converts to proper signature.

        Args:
            mapping: Mapping containing callable values

        Returns:
            Dict mapping string keys to callable functions

        """
        result: dict[str, Callable[[], t.GuardInputValue]] = {}
        items_iter = mapping.items()
        for item_tuple in items_iter:
            key_obj = item_tuple[0]
            value_raw = item_tuple[1]
            if callable(value_raw):
                key_str: str = str(key_obj)
                # Create wrapper function with explicit return type
                # This ensures type safety - container validates at runtime
                # Type narrow: callable() narrows value_raw to a callable value
                captured_fn = cast("Callable[[], object]", value_raw)

                def _wrap_callable(
                    fn: Callable[[], object] = captured_fn,
                ) -> Callable[[], t.GuardInputValue]:
                    """Wrap callable with proper return type signature."""

                    def _wrapped() -> t.GuardInputValue:
                        # fn is callable (checked before capture) - call and convert
                        return cast("t.GuardInputValue", fn())

                    return _wrapped

                result[key_str] = _wrap_callable()
        return result


__all__ = [
    "FlextUtilitiesCollection",
]
