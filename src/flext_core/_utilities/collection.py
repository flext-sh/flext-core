"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import copy
import datetime
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from typing import cast, overload

from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.constants import c
from flext_core.result import FlextResult, r
from flext_core.typings import t


class FlextUtilitiesCollection:
    """Utilities for collection conversion with StrEnums.

    PATTERNS collections.abc:
    ────────────────────────
    - Sequence[E] for immutable lists
    - Mapping[str, E] for immutable dicts
    - Iterable[E] for any iterable
    """

    # =========================================================================
    # TYPE GUARDS AND HELPERS - Replace casts with proper type narrowing
    # =========================================================================

    @staticmethod
    def _narrow_to_mapping_str_t(value: object) -> Mapping[str, object]:
        """Safely narrow object to Mapping[str, object] with runtime validation.

        Python 3.13: Uses isinstance for proper type narrowing with Mapping protocol.
        """
        # Python 3.13: isinstance with Mapping provides proper type narrowing
        if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
            # Type narrowing: value is Mapping[str, object] after isinstance checks
            return value  # Mapping[str, object] compatible with Mapping[str, object]
        error_msg = f"Cannot narrow {type(value)} to Mapping[str, object]"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_list_object(value: object) -> list[object]:
        """Safely narrow object to list[object] with runtime validation.

        Uses TypeGuard instead of isinstance for proper type narrowing.
        """
        # Use isinstance for proper type narrowing
        if isinstance(value, list):
            return value  # Type narrowing via isinstance
        if isinstance(value, (list, tuple)):
            return list(value)  # Convert tuple to list
        error_msg = f"Cannot narrow {type(value)} to list[object]"
        raise TypeError(error_msg)

    # ─────────────────────────────────────────────────────────────
    # LIST CONVERSIONS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def parse_sequence[E: StrEnum](
        enum_cls: type[E],
        values: Iterable[str | E],
    ) -> r[tuple[E, ...]]:
        """Convert sequence of strings to tuple of StrEnum.

        Example:
             result = FlextUtilitiesCollection.parse_sequence(
                 Status, ["active", "pending"]
             )
             if result.is_success:
                 statuses: tuple[Status, ...] = result.value

        """
        parsed: list[E] = []
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
            return r[tuple[E, ...]].fail(
                f"Invalid {enum_name} values: {', '.join(errors)}",
            )
        return r[tuple[E, ...]].ok(tuple(parsed))

    @staticmethod
    def coerce_list_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.FlexibleValue], list[E]]:
        """BeforeValidator for list of StrEnums.

        Example:
             StatusList = Annotated[
                 list[Status],
                 BeforeValidator(FlextUtilitiesCollection.coerce_list_validator(Status))
             ]

             class MyModel(BaseModel):
                 statuses: StatusList  # Accepts ["active", "pending"]

        """

        def _coerce(value: t.FlexibleValue) -> list[E]:
            # Use TypeGuard for proper type narrowing - eliminates need for isinstance
            # Check for list/tuple using TypeGuard
            # Access private methods for TypeGuard return type (needed for type narrowing)
            if isinstance(value, (list, tuple, set, frozenset)):
                pass  # Valid sequence type
            else:
                msg = f"Expected sequence, got {type(value).__name__}"
                raise TypeError(msg)

            result: list[E] = []
            for idx, item in enumerate(value):
                if isinstance(item, enum_cls):
                    result.append(item)
                elif isinstance(item, str):
                    # Python 3.13: isinstance provides proper type narrowing to str
                    try:
                        result.append(enum_cls(item))
                    except ValueError as err:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} at [{idx}]: {item!r}"
                        raise ValueError(msg) from err
                else:
                    msg = f"Expected str at [{idx}], got {type(item).__name__}"
                    raise TypeError(msg)
            return result

        return _coerce

    # ─────────────────────────────────────────────────────────────
    # DICT CONVERSIONS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def parse_mapping[E: StrEnum](
        enum_cls: type[E],
        mapping: Mapping[str, str | E],
    ) -> r[dict[str, E]]:
        """Convert Mapping with string values to dict with StrEnum.

        Example:
             result = FlextUtilitiesCollection.parse_mapping(
                 Status, {"user1": "active", "user2": "pending"}
             )

        """
        parsed: dict[str, E] = {}
        errors: list[str] = []

        for key, val in mapping.items():
            if isinstance(val, enum_cls):
                parsed[key] = val
            else:
                try:
                    parsed[key] = enum_cls(val)
                except ValueError:
                    errors.append(f"'{key}': '{val}'")

        if errors:
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return r[dict[str, E]].fail(
                f"Invalid {enum_name} values: {', '.join(errors)}",
            )
        return r[dict[str, E]].ok(parsed)

    @staticmethod
    def coerce_dict_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.FlexibleValue], dict[str, E]]:
        """BeforeValidator for dict with StrEnum values.

        Example:
             StatusDict = Annotated[
                 dict[str, Status],
                 BeforeValidator(FlextUtilitiesCollection.coerce_dict_validator(Status))
             ]

        """

        def _coerce(value: t.FlexibleValue) -> dict[str, E]:
            # Use TypeGuard for proper type narrowing - eliminates need for isinstance
            # Access private methods for TypeGuard return type (needed for type narrowing)
            if not isinstance(value, dict):
                msg = f"Expected dict, got {type(value).__name__}"
                raise TypeError(msg)

            # Python 3.13: Type narrowing - value is dict[str, t.GeneralValueType] after TypeGuard
            # t.GeneralValueType includes ScalarValue, so we can safely narrow
            # Use type narrowing with runtime validation
            # Use dict comprehension for type narrowing
            value_dict: dict[str, t.ScalarValue] = {
                k: v
                for k, v in value.items()
                if isinstance(v, (str, int, float, bool, type(None), datetime.datetime))
            }
            result: dict[str, E] = {}
            for key, val in value_dict.items():
                if isinstance(val, enum_cls):
                    result[key] = val
                elif isinstance(val, str):
                    try:
                        result[key] = enum_cls(val)
                    except ValueError as err:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} at '{key}': {val!r}"
                        raise ValueError(msg) from err
                else:
                    msg = f"Expected str at '{key}', got {type(val).__name__}"
                    raise TypeError(msg)
            return result

        return _coerce

    # ─────────────────────────────────────────────────────────────
    # MAP METHODS
    # ─────────────────────────────────────────────────────────────

    @overload
    @staticmethod
    def map[T, R](
        items: r[T],
        mapper: Callable[[T], R],
        *,
        default_error: str = "Operation failed",
    ) -> r[R]: ...

    @overload
    @staticmethod
    def map[T, R](
        items: list[T] | tuple[T, ...],
        mapper: Callable[[T], R],
    ) -> list[R]: ...

    @overload
    @staticmethod
    def map[T, R](
        items: set[T] | frozenset[T],
        mapper: Callable[[T], R],
    ) -> set[R] | frozenset[R]: ...

    @overload
    @staticmethod
    def map[T, R](
        items: dict[str, T] | Mapping[str, T],
        mapper: Callable[[str, T], R],
    ) -> dict[str, R]: ...

    @staticmethod
    def map[T, R](
        items: (
            T
            | list[T]
            | tuple[T, ...]
            | set[T]
            | frozenset[T]
            | dict[str, T]
            | Mapping[str, T]
            | r[T]
        ),
        mapper: Callable[[T], R] | Callable[[str, T], R],
        *,
        default_error: str = "Operation failed",
    ) -> list[R] | set[R] | frozenset[R] | dict[str, R] | r[R]:
        """Unified map function that auto-detects input type.

        Generic replacement for: List/dict comprehensions, manual loops, map_or

        Args:
            items: Input items (list, tuple, dict, set, or r[T] result)
            mapper: Function to transform items
            default_error: Default error if mapping result fails (only for r[T])

        Returns:
            Mapped results (list, dict, set, or r[R] based on input type)

        Example:
            mapped = FlextUtilitiesCollection.map([1, 2, 3], lambda x: x * 2)
            # → [2, 4, 6]

        """
        return FlextUtilitiesCollection._map_impl(items, mapper, default_error)

    @staticmethod
    def _map_result_only[T, R](
        items: r[T],
        mapper: Callable[[T], R],
        default_error: str,
    ) -> r[R]:
        """Map over FlextResult only - separate function to avoid union type issues."""
        return FlextUtilitiesCollection._map_result(items, mapper, default_error)

    @staticmethod
    def _map_impl[T, R](
        items: (
            T
            | list[T]
            | tuple[T, ...]
            | set[T]
            | frozenset[T]
            | dict[str, T]
            | Mapping[str, T]
            | r[T]
        ),
        mapper: Callable[[T], R] | Callable[[str, T], R],
        default_error: str,
    ) -> list[R] | set[R] | frozenset[R] | dict[str, R] | r[R]:
        """Internal implementation for map - handles all cases.

        Python 3.13: Uses match/case with type narrowing for precise dispatch.
        Eliminates need for  or type: ignore through proper type narrowing.
        """
        # Result case - isinstance check for FlextResult structure
        # Note: Use isinstance for proper type narrowing without
        if isinstance(items, FlextResult):
            # Type narrowing: isinstance ensures items is r[T]
            # For FlextResult, mapper is always Callable[[T], R] (single argument)
            # Union type limitation: pyrefly cannot resolve Callable[[T], R] from union
            # This is safe because isinstance(items, FlextResult) guarantees single-arg mapper
            return FlextUtilitiesCollection._map_result_only(
                items,
                cast("Callable[[T], R]", mapper),
                default_error,
            )

        # Sequence case - TypeGuard for list/tuple
        if isinstance(items, (list, tuple)):
            # Type narrowing: TypeGuard ensures list[T] | tuple[T, ...]
            # Python 3.13: Structural typing - sequences use Callable[[T], R]
            return FlextUtilitiesCollection._map_sequence(
                items,  # TypeGuard ensures sequence
                cast("Callable[[T], R]", mapper),
            )

        # Set case - isinstance needed for runtime behavior (set vs frozenset)
        if isinstance(items, (set, frozenset)):
            # Type narrowing: isinstance ensures set[T] | frozenset[T]
            # Python 3.13: Structural typing - sets use Callable[[T], R]
            return FlextUtilitiesCollection._map_set(
                items,  # isinstance ensures set
                cast("Callable[[T], R]", mapper),
            )

        # Mapping case - TypeGuard for dict/Mapping
        if isinstance(items, Mapping):
            # Type narrowing: TypeGuard ensures Mapping[str, T]
            # Python 3.13: Structural typing - mappings use Callable[[str, T], R]
            items_dict: dict[str, T] = (
                dict(items) if not isinstance(items, dict) else items
            )
            return FlextUtilitiesCollection._map_dict(
                items_dict,
                cast("Callable[[str, T], R]", mapper),
            )

        # Single item case - mapper is Callable[[T], R] or Callable[[object], R]
        # Python 3.13: Single item uses Callable[[object], R] for maximum compatibility
        return FlextUtilitiesCollection._map_single(
            items,
            cast("Callable[[object], R]", mapper),
        )

    @staticmethod
    def _map_result[T, R](
        items: r[T],
        mapper: Callable[[T], R],
        default_error: str,
    ) -> r[R]:
        """Map over r.

        For r[T], mapper is always Callable[[T], R] (single argument).
        Signature change eliminates need for type: ignore.
        """
        if items.is_success:
            # Type narrowing: items.value is T when is_success is True
            mapped: R = mapper(items.value)
            return r.ok(mapped)
        # When is_failure is True, error is never None (fail() converts None to "")
        # Use error or default_error as fallback
        error_msg = items.error or default_error
        return r.fail(error_msg)

    @staticmethod
    def _map_sequence[T, R](
        items: list[T] | tuple[T, ...],
        mapper: Callable[[T], R],
    ) -> list[R]:
        """Map over list or tuple.

        For sequences, mapper is always Callable[[T], R] (single argument).
        Signature change eliminates need for type: ignore.
        """
        # Type narrowing: items is list[T] | tuple[T, ...], so item is T
        return [mapper(item) for item in items]

    @staticmethod
    def _map_set[T, R](
        items: set[T] | frozenset[T],
        mapper: Callable[[T], R],
    ) -> set[R] | frozenset[R]:
        """Map over set or frozenset.

        For sets, mapper is always Callable[[T], R] (single argument).
        Signature change eliminates need for type: ignore.
        """
        # Type narrowing: items is set[T] | frozenset[T], so item is T
        mapped_set: set[R] = {mapper(item) for item in items}
        # isinstance is necessary here to distinguish frozenset from set
        # This isinstance is required for runtime behavior, not just type narrowing
        if isinstance(items, frozenset):
            return frozenset(mapped_set)
        return mapped_set

    @staticmethod
    def _map_dict[T, R](
        items: dict[str, T] | Mapping[str, T],
        mapper: Callable[[str, T], R],
    ) -> dict[str, R]:
        """Map over dict or Mapping.

        For mappings, mapper is always Callable[[str, T], R] (key-value pair).
        Signature change eliminates need for type: ignore.
        """
        # Type narrowing: items is dict[str, T] | Mapping[str, T], so v is T
        return {k: mapper(k, v) for k, v in items.items()}

    @staticmethod
    def _map_single[R](
        item: object,
        mapper: Callable[[object], R],
    ) -> list[R]:
        """Map over single value."""
        return [mapper(item)]

    # ─────────────────────────────────────────────────────────────
    # FIND METHODS
    # ─────────────────────────────────────────────────────────────

    @overload
    @staticmethod
    def find[T](
        items: list[T] | tuple[T, ...] | set[T] | frozenset[T],
        predicate: Callable[[T], bool],
        *,
        return_key: bool = False,
    ) -> T | None: ...

    @overload
    @staticmethod
    def find[T](
        items: dict[str, T] | Mapping[str, T],
        predicate: Callable[[str, T], bool],
        *,
        return_key: bool = False,
    ) -> T | tuple[str, T] | None: ...

    @staticmethod
    def find[T](
        items: (
            list[T]
            | tuple[T, ...]
            | set[T]
            | frozenset[T]
            | dict[str, T]
            | Mapping[str, T]
        ),
        predicate: Callable[..., bool],
        *,
        return_key: bool = False,
    ) -> T | tuple[str, T] | None:
        """Unified find function that auto-detects input type.

        Args:
            items: Input items (list, tuple, set, frozenset, or dict)
            predicate: Function to test items
            return_key: If True, returns (key, value) tuple for dicts

        Returns:
            Found value or (key, value) tuple or None if not found

        Example:
            value = FlextUtilitiesCollection.find([1, 2, 3], lambda x: x > 1)
            # → 2

        """
        # Check for sequence types using TypeGuard (avoids isinstance when possible)
        if isinstance(items, (list, tuple)):
            items_sequence: list[T] | tuple[T, ...] = items
            for item in items_sequence:
                if predicate(item):
                    return item
            return None

        # Check for set types - isinstance is necessary here for runtime behavior
        # isinstance is required to distinguish set from other types
        if isinstance(items, (set, frozenset)):
            items_set: set[T] | frozenset[T] = items
            for item in items_set:
                if predicate(item):
                    return item
            return None

        items_mapping: Mapping[str, T] = items
        mapping_items: dict[str, T] = (
            dict(items_mapping)
            if not isinstance(items_mapping, dict)
            else items_mapping
        )
        for k, v in mapping_items.items():
            if predicate(k, v):
                result: T | tuple[str, T] = (k, v) if return_key else v
                return result
        return None

    # ─────────────────────────────────────────────────────────────
    # FILTER METHODS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _filter_list_no_mapper[T](
        items_list: list[T],
        predicate: Callable[[T], bool],
    ) -> list[T]:
        """Filter a list without mapping."""
        return [item for item in items_list if predicate(item)]

    @staticmethod
    def _filter_list_with_mapper[T, R](
        items_list: list[T],
        predicate: Callable[..., bool],
        mapper: Callable[..., R],
    ) -> list[R]:
        """Filter a list with mapping - filter original items, then map."""
        filtered_items: list[R] = []
        for item in items_list:
            # Type narrowing: item is T from items_list[T]
            if predicate(item):
                mapped = mapper(item)
                filtered_items.append(mapped)
        return filtered_items

    @staticmethod
    def _filter_dict[T](
        items_dict: dict[str, T] | Mapping[str, T],
        predicate: Callable[[str, T], bool] | Callable[..., bool],
        _mapper: None = None,
    ) -> dict[str, T]:
        """Filter a dictionary without mapping."""
        return {
            key: value for key, value in items_dict.items() if predicate(key, value)
        }

    @staticmethod
    def _filter_dict_with_mapper[T, R](
        items_dict: dict[str, T] | Mapping[str, T],
        predicate: Callable[[str, T], bool] | Callable[..., bool],
        mapper: Callable[[str, T], R] | Callable[..., R],
    ) -> dict[str, R]:
        """Filter a dictionary with mapping - filter original items, then map."""
        filtered_dict = {}
        for key, value in items_dict.items():
            if predicate(key, value):
                mapped = mapper(key, value)
                filtered_dict[key] = mapped
        return filtered_dict

    @staticmethod
    def _filter_single(
        single_item: object,
        predicate: Callable[..., bool],
        mapper: Callable[..., object] | None = None,
    ) -> list[object]:
        """Filter a single value with optional mapping."""
        if mapper is not None:
            mapped_item = mapper(single_item)
            if predicate(mapped_item):
                return [mapped_item]
            return []
        if predicate(single_item):
            return [single_item]
        return []

    @overload
    @staticmethod
    def filter[T](
        items: list[T] | tuple[T, ...],
        predicate: Callable[..., bool],
        *,
        mapper: None = None,
    ) -> list[T]: ...

    @overload
    @staticmethod
    def filter[T, R](
        items: list[T] | tuple[T, ...],
        predicate: Callable[..., bool],
        *,
        mapper: Callable[..., R],
    ) -> list[R]: ...

    @overload
    @staticmethod
    def filter[T](
        items: dict[str, T] | Mapping[str, T],
        predicate: Callable[..., bool],
        *,
        mapper: None = None,
    ) -> dict[str, T]: ...

    @overload
    @staticmethod
    def filter[T, R](
        items: dict[str, T] | Mapping[str, T],
        predicate: Callable[..., bool],
        *,
        mapper: Callable[..., R],
    ) -> dict[str, R]: ...

    @staticmethod
    def filter[T, R](
        items: T | list[T] | tuple[T, ...] | dict[str, T] | Mapping[str, T],
        predicate: Callable[..., bool],
        *,
        mapper: Callable[..., R] | None = None,
    ) -> list[T] | list[R] | dict[str, T] | dict[str, R]:
        """Unified filter function that auto-detects input type.

        Args:
            items: Input items (single value, list, tuple, or dict)
            predicate: Function to filter items
            mapper: Optional function to map items before filtering

        Returns:
            Filtered results (list or dict based on input)

        Example:
            filtered = FlextUtilitiesCollection.filter([1, 2, 3], lambda x: x > 1)
            # → [2, 3]

        """
        # Check for sequence types using isinstance for proper type narrowing
        if isinstance(items, (list, tuple)):
            # Type narrowing: isinstance narrows T to list[T] | tuple[T, ...]
            items_sequence: list[T] | tuple[T, ...] = items
            items_list_typed: list[T] = list(
                items_sequence
            )  # list[T] | tuple[T, ...] -> list[T]
            if mapper is not None:
                return FlextUtilitiesCollection._filter_list_with_mapper(
                    items_list_typed,
                    predicate,
                    cast("Callable[[T], R]", mapper),
                )
            return FlextUtilitiesCollection._filter_list_no_mapper(
                items_list_typed,
                predicate,
            )
        # Check for mapping types using TypeGuard (avoids isinstance when possible)
        if isinstance(items, Mapping):
            # Type narrowing: items is Mapping[str, T] after TypeGuard check
            # Convert to dict[str, T] for processing
            items_mapping: Mapping[str, T] = items
            # isinstance only when necessary for conversion optimization
            items_dict_filtered: dict[str, T]
            if isinstance(items_mapping, dict):
                items_dict_filtered = items_mapping
            else:
                # dict() constructor from Mapping[str, T] returns dict[str, T]
                items_dict_filtered = dict(items_mapping)
            # _filter_dict accepts Mapping[str, T] | dict[str, T]
            # dict[str, T] is compatible with Mapping[str, T] | dict[str, T]
            if mapper is not None:
                return FlextUtilitiesCollection._filter_dict_with_mapper(
                    items_dict_filtered,
                    predicate,
                    cast("Callable[[str, T], R]", mapper),
                )
            return FlextUtilitiesCollection._filter_dict(
                cast("dict[str, T] | Mapping[str, T]", items_dict_filtered),
                predicate,
            )
        # Single item case
        # Python 3.13: _filter_single returns list[object], cast to expected return type
        # Type narrowing: single item filter returns list[T] (wrapped single item)
        single_result = FlextUtilitiesCollection._filter_single(
            items,
            predicate,
            cast("Callable[..., object] | None", mapper),
        )
        return cast("list[T] | list[R]", single_result)

    # ─────────────────────────────────────────────────────────────
    # PROCESS METHODS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _process_list_items[T, R](
        items_list: list[T],
        processor: Callable[[T], R],
        *,
        predicate: Callable[[T], bool] | None = None,
        on_error: str = "collect",
    ) -> r[list[R] | dict[str, R]]:
        """Helper: Process list items."""
        if predicate is not None:
            items_list = FlextUtilitiesCollection.filter(items_list, predicate)
        list_results: list[R] = []
        list_errors: list[str] = []
        for item in items_list:
            try:
                processed = processor(item)
                list_results.append(processed)
            except Exception as e:
                if on_error == "fail":
                    return r[list[R] | dict[str, R]].fail(f"Processing failed: {e}")
                if on_error == "skip":
                    continue
                list_errors.append(str(e))
        if list_errors and on_error == "collect":
            return r[list[R] | dict[str, R]].fail(
                f"Processing errors: {', '.join(list_errors)}",
            )
        return r[list[R] | dict[str, R]].ok(list_results)

    @staticmethod
    def _process_dict_items[T, R](
        items_dict: dict[str, T],
        processor: Callable[[str, T], R],
        *,
        predicate: Callable[[str, T], bool] | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
        on_error: str = "collect",
    ) -> r[list[R] | dict[str, R]]:
        """Helper: Process dict items."""
        if filter_keys is not None or exclude_keys is not None:

            def key_predicate(k: str, _v: T) -> bool:
                """Filter predicate for dict keys."""
                return (filter_keys is None or k in filter_keys) and (
                    exclude_keys is None or k not in exclude_keys
                )

            items_dict = FlextUtilitiesCollection.filter(items_dict, key_predicate)

        if predicate is not None:
            items_dict = FlextUtilitiesCollection.filter(items_dict, predicate)

        dict_result: dict[str, R] = {}
        dict_errors: list[str] = []
        for key, value in items_dict.items():
            try:
                processed = processor(key, value)
                dict_result[key] = processed
            except Exception as e:
                if on_error == "fail":
                    return r[list[R] | dict[str, R]].fail(
                        f"Processing key '{key}' failed: {e}",
                    )
                if on_error == "skip":
                    continue
                dict_errors.append(str(e))
        if dict_errors and on_error == "collect":
            return r[list[R] | dict[str, R]].fail(
                f"Processing errors: {', '.join(dict_errors)}",
            )
        return r.ok(dict_result)

    @staticmethod
    def process[T, R](
        items: T | list[T] | tuple[T, ...] | dict[str, T] | Mapping[str, T],
        processor: Callable[[T], R] | Callable[[str, T], R],
        *,
        on_error: str = "collect",
        predicate: Callable[[T], bool] | Callable[[str, T], bool] | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> r[list[R] | dict[str, R]]:
        """Unified process function that auto-detects input type.

        Args:
            items: Input items (single value, list, tuple, or dict)
            processor: Function to process each item
            on_error: "collect" (continue), "fail" (stop), or "skip" (ignore errors)
            predicate: Optional filter function
            filter_keys: Optional set of keys to process (dict only)
            exclude_keys: Optional set of keys to skip (dict only)

        Returns:
            r containing processed results (list or dict based on input)

        Example:
            result = FlextUtilitiesCollection.process(
                [1, 2, 3], lambda x: x * 2, predicate=lambda x: x > 1
            )

        """
        # Check for sequence types using TypeGuard (avoids isinstance when possible)
        if isinstance(items, (list, tuple)):
            # Type narrowing: items is list[T] | tuple[T, ...] after TypeGuard check
            # For sequences, processor is Callable[[T], R] (narrowed by context)
            items_sequence: list[T] | tuple[T, ...] = items
            items_list: list[T] = list(items_sequence)
            list_processor: Callable[[T], R] = cast("Callable[[T], R]", processor)
            list_predicate: Callable[[T], bool] | None = cast(
                "Callable[[T], bool] | None",
                predicate if predicate is not None else None,
            )
            # Type narrowing: _process_list_items accepts Callable[[T], R] and Callable[[T], bool] | None
            # list_processor is Callable[[T], R] and list_predicate is Callable[[T], bool] | None
            # Direct assignment works - types match exactly
            return FlextUtilitiesCollection._process_list_items(
                items_list,
                list_processor,  # Callable[[T], R] matches signature
                predicate=list_predicate,  # Callable[[T], bool] | None matches signature
                on_error=on_error,
            )

        # Check for mapping types using TypeGuard (avoids isinstance when possible)
        if isinstance(items, Mapping):
            # Type narrowing: items is Mapping[str, T] after TypeGuard check
            # For mappings, processor is Callable[[str, T], R] (narrowed by context)
            items_mapping: Mapping[str, T] = items
            # isinstance only when necessary for conversion optimization
            items_dict_processed: dict[str, T]
            if isinstance(items_mapping, dict):
                items_dict_processed = items_mapping
            else:
                # dict() constructor from Mapping[str, T] returns dict[str, T]
                items_dict_processed = dict(items_mapping)
            # Type narrowing: _process_dict_items accepts Callable[[str, T], R] and Callable[[str, T], bool] | None
            # processor is Callable[[str, T], R] | Callable[[T], R], predicate is Callable[[str, T], bool] | Callable[[T], bool] | None
            # For mappings, processor must be Callable[[str, T], R] by structural contract
            dict_processor: Callable[[str, T], R] = cast(
                "Callable[[str, T], R]", processor
            )
            dict_predicate: Callable[[str, T], bool] | None = cast(
                "Callable[[str, T], bool] | None",
                predicate if predicate is not None else None,
            )
            return FlextUtilitiesCollection._process_dict_items(
                items_dict_processed,
                dict_processor,
                predicate=dict_predicate,
                filter_keys=filter_keys,
                exclude_keys=exclude_keys,
                on_error=on_error,
            )

        # Single item case - wrap in list and process
        # Type narrowing: for single item, processor is Callable[[T], R] by structural contract
        items_wrapped: list[T] = [items]
        single_processor: Callable[[T], R] = cast("Callable[[T], R]", processor)
        single_predicate: Callable[[T], bool] | None = cast(
            "Callable[[T], bool] | None", predicate if predicate is not None else None
        )
        return FlextUtilitiesCollection._process_list_items(
            items_wrapped,
            single_processor,
            predicate=single_predicate,
            on_error=on_error,
        )

    # ─────────────────────────────────────────────────────────────
    # GROUP/CHUNK METHODS
    # ─────────────────────────────────────────────────────────────

    @overload
    @staticmethod
    def group[T, K](
        items: list[T] | tuple[T, ...],
        key: Callable[[T], K],
    ) -> dict[object, list[T]]: ...

    @overload
    @staticmethod
    def group[T](
        items: list[T] | tuple[T, ...],
        key: str,
    ) -> dict[object, list[T]]: ...

    @staticmethod
    def group[T](
        items: list[T] | tuple[T, ...],
        key: str | Callable[[T], object],
    ) -> dict[object, list[T]]:
        """Group items by key (mnemonic: group = group by).

        Args:
            items: Items to group
            key: Field name (str) or key function (callable)

        Returns:
            dict[key, list[items]]

        Example:
            by_len = FlextUtilitiesCollection.group(words, lambda w: len(w))
            # → {3: ["cat", "dog"], 5: ["house"]}

        """
        result: dict[object, list[T]] = {}
        items_list = list(items)
        for item in items_list:
            if callable(key):
                k: object = key(item)
            # Type narrowing: key is str here (not callable)
            # Use TypeGuard for proper type narrowing - eliminates need for isinstance
            # Access private methods for TypeGuard return type (needed for type narrowing)
            else:
                k: object = getattr(item, cast("str", key), None)
            if k not in result:
                result[k] = []
            result[k].append(item)
        return result

    @staticmethod
    def chunk[T](
        items: list[T] | tuple[T, ...],
        size: int,
    ) -> list[list[T]]:
        """Split items into chunks (mnemonic: chunk = batch).

        Args:
            items: Items to chunk
            size: Chunk size

        Returns:
            list[list[T]] of chunks

        Example:
            batches = FlextUtilitiesCollection.chunk(entries, size=100)
            # → [[entry1...entry100], [entry101...entry200]]

        """
        return [list(items[i : i + size]) for i in range(0, len(items), size)]

    @staticmethod
    def _batch_process_single_item[T, R](
        item: T,
        idx: int,
        operation: Callable[[T], R | r[R]],
        errors: list[tuple[int, str]],
        on_error: str,
    ) -> R | r[R] | r[t.BatchResultDict] | None:
        """Helper: Process a single batch item."""
        try:
            # Type narrowing: operation is Callable, result type is unknown
            result_raw = operation(item)
            # Use Protocol check for proper type narrowing - avoids isinstance when possible
            # Check for result-like structure using hasattr (structural typing)
            if (
                hasattr(result_raw, "is_success")
                and hasattr(result_raw, "value")
                and hasattr(result_raw, "error")
            ):
                # Type narrowing: result_raw has result-like structure (r[R])
                result_typed: r[R] = cast("r[R]", result_raw)
                if result_typed.is_failure:
                    # When is_failure is True, error is never None (fail() converts None to "")
                    # Use error or fallback message
                    error_msg = result_typed.error or "Unknown error"
                    error_text = f"Item {idx} failed: {error_msg}"
                    if on_error == "fail":
                        return r[t.BatchResultDict].fail(error_text)
                    if on_error == "collect":
                        errors.append((idx, error_msg))
                    return None
                # Type narrowing: result_raw is r[R] and is_success is True (implicitly, since we access .value)
                # After is_success check, result.value is R
                return result_raw.value
            # Type narrowing: result_raw is R (not r)
            return result_raw
        except Exception as e:
            error_msg = str(e)
            error_text = f"Item {idx} failed: {error_msg}"
            if on_error == "fail":
                return r[t.BatchResultDict].fail(error_text)
            if on_error == "collect":
                errors.append((idx, error_msg))
            return None

    @staticmethod
    def _batch_flatten_results(
        validated_results: list[t.GeneralValueType],
        *,
        flatten: bool,
    ) -> list[t.GeneralValueType]:
        """Helper: Flatten nested lists if requested."""
        if not flatten:
            return validated_results

        def is_list_or_tuple(item: object) -> bool:
            """Check if item is list or tuple."""
            return FlextUtilitiesGuards.is_type(item, "list_or_tuple")

        nested_filtered = FlextUtilitiesCollection.filter(
            validated_results, is_list_or_tuple
        )
        nested: list[list[t.GeneralValueType] | tuple[t.GeneralValueType, ...]] = cast(
            "list[list[t.GeneralValueType] | tuple[t.GeneralValueType, ...]]",
            nested_filtered,
        )
        if not nested:
            return validated_results

        # Flatten nested items
        flattened: list[t.GeneralValueType] = []
        for sublist in nested:
            flattened.extend(sublist)

        def _not_list_or_tuple(x: t.GeneralValueType) -> bool:
            return not is_list_or_tuple(x)

        non_list = FlextUtilitiesCollection.filter(
            validated_results,
            _not_list_or_tuple,
        )
        return flattened + non_list

    @staticmethod
    def batch[T, R](
        items: list[T],
        operation: Callable[[T], R | r[R]],
        *,
        _size: int = c.DEFAULT_BATCH_SIZE,
        on_error: str = "collect",
        _parallel: bool = False,
        progress: Callable[[int, int], None] | None = None,
        _progress_interval: int = 1,
        pre_validate: Callable[[T], bool] | None = None,
        flatten: bool = False,
    ) -> r[t.BatchResultDict]:
        """Process items in batches (mnemonic: batch = chunk + process).

        Args:
            items: Items to process
            operation: Function to process each item (can return Result)
            _size: Batch size
            on_error: "collect" (continue), "fail" (stop)
            _parallel: Run batches in parallel (not implemented yet)
            progress: Progress callback (processed, total)
            _progress_interval: Progress update interval
            pre_validate: Optional pre-validation function
            flatten: If True, flatten nested lists in results

        Returns:
            r[t.BatchResultDict] containing batch results with errors

        """
        total_items = len(items)

        # Pre-filter items if pre_validate provided
        if pre_validate is not None:
            filtered_items = FlextUtilitiesCollection.filter(items, pre_validate)
            # Type narrowing: filter() on list[T] returns list[T], so isinstance is redundant
            # filtered_items is list[T] when items is list[T]
            items_to_process: list[T] = filtered_items
        else:
            items_to_process = items

        # Process items directly to collect errors properly for batch format
        errors: list[tuple[int, str]] = []
        processed_results: list[R] = []

        for idx, item in enumerate(items_to_process):
            # Type narrowing: process_result is R | "r[t.BatchResultDict]" | None
            process_result_raw = FlextUtilitiesCollection._batch_process_single_item(
                item,
                idx,
                operation,
                errors,
                on_error,
            )
            if process_result_raw is None:
                continue  # Item skipped
            # Check is_success first to avoid accessing .value on failed results
            # Use getattr with default to avoid triggering property access
            if hasattr(process_result_raw, "is_success"):
                is_failure = getattr(process_result_raw, "is_failure", False)
                if is_failure:
                    # Type narrowing: process_result_raw is r[t.BatchResultDict]
                    return cast("r[t.BatchResultDict]", process_result_raw)
                # Type narrowing: process_result_raw is r[R] and is_success is True
                # Extract value from result
                extracted_value = getattr(
                    process_result_raw, "value", process_result_raw
                )
                process_result_raw = extracted_value
            # Type narrowing: process_result_raw is R after extraction
            # Cast to R since we've unwrapped the result
            processed_results.append(cast("R", process_result_raw))

            # Call progress callback if provided
            if progress is not None and idx % _progress_interval == 0:
                progress(idx + 1, total_items)

        # Convert to t.GeneralValueType for flattening
        def to_general_value(item: R) -> t.GeneralValueType:
            """Convert item to t.GeneralValueType."""
            # Cast to t.GeneralValueType - R is known to be a valid t.GeneralValueType
            return cast("t.GeneralValueType", item)

        validated_results_raw_list: list[R] = processed_results
        validated_results = FlextUtilitiesCollection.map(
            validated_results_raw_list,
            to_general_value,
        )

        # Flatten nested lists if requested
        flattened_results = FlextUtilitiesCollection._batch_flatten_results(
            validated_results,
            flatten=flatten,
        )

        # Final progress callback
        if progress is not None:
            progress(total_items, total_items)

        batch_result: t.BatchResultDict = {
            "results": flattened_results,
            "errors": errors,
            "total": total_items,
            "success_count": len(flattened_results),
            "error_count": len(errors),
        }
        return r.ok(batch_result)

    @staticmethod
    def count[T](
        items: list[T] | tuple[T, ...] | Iterable[T],
        predicate: Callable[[T], bool] | None = None,
    ) -> int:
        """Count items matching predicate (mnemonic: count = count).

        Generic replacement for: sum(1 for x in items if pred(x))
        or len([x for x in items if pred(x)])
        Uses filter() for unified filtering.

        Args:
            items: Items to count
            predicate: Optional filter function (counts all if None)

        Returns:
            Count of matching items

        Example:
            total = FlextUtilitiesCollection.count(items)
            active = FlextUtilitiesCollection.count(users, lambda u: u.is_active)

        """
        if predicate is None:
            return len(list(items))
        # Use list comprehension for filtering and counting
        items_list = list(items)
        return sum(1 for item in items_list if predicate(item))

    @staticmethod
    def merge(
        base: t.ConfigurationMapping,
        other: t.ConfigurationMapping,
        *,
        strategy: str = "deep",
    ) -> r[t.ConfigurationDict]:
        """Merge two dictionaries using specified strategy.

        Strategies:
        - "deep" (default): Recursive merge, other overrides base.
        - "override": Shallow merge (dict.update).
        - "append": Recursive merge, but appends lists instead of overriding.
        - "filter_none": Recursive merge, None values in other are ignored.
        - "filter_empty": Recursive merge, Empty values ("", [], {}, None) in other are ignored.
        - "filter_both": Recursive merge, None and Empty values are ignored.

        Args:
            base: Base dictionary.
            other: Dictionary to merge into base.
            strategy: Merge strategy.

        Returns:
            FlextResult containing the merged dictionary.

        """
        try:
            # Convert inputs to dicts for manipulation
            # Deep copy base to avoid side effects
            # ConfigurationMapping is always a Mapping, so isinstance is redundant
            merged: t.ConfigurationDict = copy.deepcopy(dict(base))
            other_dict: t.ConfigurationDict = dict(other)

            if strategy == c.Mixins.OPERATION_OVERRIDE:
                merged.update(other_dict)
                return r.ok(merged)

            # Helper to check if value is an empty container
            def _is_empty_container(value: t.GeneralValueType) -> bool:
                """Check if value is an empty container (string, list, dict, tuple, set)."""
                if isinstance(value, str):
                    return len(value) == 0
                if isinstance(value, list):
                    return len(value) == 0
                if isinstance(value, dict):
                    return len(value) == 0
                if isinstance(value, (tuple, set)):
                    return len(value) == 0
                return False

            # Helper for deep merge
            def _deep_merge(
                target: t.ConfigurationDict,
                source: t.ConfigurationDict,
                mode: str,
            ) -> None:
                for key, value_raw in source.items():
                    # Type narrowing: value from ConfigurationDict is t.GeneralValueType
                    value: t.GeneralValueType = value_raw
                    # Filter logic
                    if mode in {"filter_none", "filter_both"} and value is None:
                        continue
                    if mode in {"filter_empty", "filter_both"}:
                        if value is None:
                            continue
                        # Check for empty containers using TypeGuards and len()
                        # Access private methods for TypeGuard return type (needed for type narrowing)
                        # isinstance is necessary here for runtime behavior (different types have different empty checks)
                        is_empty = _is_empty_container(value)
                        if is_empty:
                            continue

                    # Recursive logic
                    if key in target:
                        target_val = target[key]

                        # Both are dicts -> recurse
                        # Use TypeGuard for proper type narrowing - eliminates need for isinstance
                        # Access private methods for TypeGuard return type (needed for type narrowing)
                        if isinstance(target_val, dict) and isinstance(value, dict):
                            # Type narrowing: TypeGuard ensures both are dict
                            # No cast needed - TypeGuard provides type narrowing
                            target_dict: t.ConfigurationDict = target_val
                            value_dict: t.ConfigurationDict = value
                            _deep_merge(
                                target_dict,
                                value_dict,
                                mode,
                            )
                            continue

                        # Both are lists and mode is append -> append
                        # Use TypeGuard for proper type narrowing - eliminates need for isinstance
                        # Access private methods for TypeGuard return type (needed for type narrowing)
                        if (
                            mode == "append"
                            and isinstance(target_val, list)
                            and FlextUtilitiesGuards.is_list(value)
                        ):
                            # Python 3.13: Type narrowing - TypeGuard ensures both are list
                            # list() preserves element types from original list
                            target_list: list[t.GeneralValueType] = list(target_val)
                            value_list: list[t.GeneralValueType] = list(value)  # type: ignore[arg-type]
                            # Append elements from source list to target list
                            # Create new list to avoid mutating original objects if they were refs
                            target[key] = (
                                target_list + value_list
                            )  # list[t.GeneralValueType] compatible with t.GeneralValueType
                            continue

                    # Default: override
                    # If value is a container, deep copy it to avoid shared references
                    # Python 3.13: Use TypeGuard for proper type narrowing
                    # Access private methods for TypeGuard return type (needed for type narrowing)
                    if isinstance(value, (dict, list)):
                        # Type narrowing: value is dict[str, t.GeneralValueType] | list[t.GeneralValueType]
                        # deepcopy preserves type, so result is t.GeneralValueType compatible
                        copied_value = copy.deepcopy(value)
                        target[key] = copied_value
                    else:
                        # Type narrowing: value is t.GeneralValueType (scalar or other types)
                        target[key] = value  # t.GeneralValueType compatible

            _deep_merge(merged, other_dict, strategy)
            return r.ok(merged)
        except Exception as e:
            return r.fail(f"Merge failed: {e}")


__all__ = [
    "FlextUtilitiesCollection",
]
