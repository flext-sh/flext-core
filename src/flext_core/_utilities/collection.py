"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from typing import cast, overload

from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.constants import c
from flext_core.result import r
from flext_core.typings import t


class FlextUtilitiesCollection:
    """Utilities for collection conversion with StrEnums.

    PATTERNS collections.abc:
    ────────────────────────
    - Sequence[E] for immutable lists
    - Mapping[str, E] for immutable dicts
    - Iterable[E] for any iterable
    """

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
                f"Invalid {enum_name} values: {', '.join(errors)}"
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
            if not isinstance(value, (list, tuple, set, frozenset)):
                msg = f"Expected sequence, got {type(value).__name__}"
                raise TypeError(msg)

            result: list[E] = []
            for idx, item in enumerate(value):
                if isinstance(item, enum_cls):
                    result.append(item)
                elif FlextUtilitiesGuards.is_type(item, str):
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
                f"Invalid {enum_name} values: {', '.join(errors)}"
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
            if not FlextUtilitiesGuards.is_type(value, dict):
                msg = f"Expected dict, got {type(value).__name__}"
                raise TypeError(msg)

            value_dict: dict[str, t.ScalarValue] = cast(
                "dict[str, t.ScalarValue]",
                value,
            )
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
        items: T
        | list[T]
        | tuple[T, ...]
        | set[T]
        | frozenset[T]
        | dict[str, T]
        | Mapping[str, T]
        | r[T],
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
    def _map_impl[T, R](
        items: T
        | list[T]
        | tuple[T, ...]
        | set[T]
        | frozenset[T]
        | dict[str, T]
        | Mapping[str, T]
        | r[T],
        mapper: Callable[[T], R] | Callable[[str, T], R],
        default_error: str,
    ) -> list[R] | set[R] | frozenset[R] | dict[str, R] | r[R]:
        """Internal implementation for map - handles all cases."""
        if isinstance(items, r):
            mapper_result = cast("Callable[[object], R]", mapper)
            return FlextUtilitiesCollection._map_result(
                items,
                mapper_result,
                default_error,
            )

        if isinstance(items, (list, tuple)):
            mapper_sequence = cast("Callable[[object], R]", mapper)
            return FlextUtilitiesCollection._map_sequence(items, mapper_sequence)

        if isinstance(items, (set, frozenset)):
            mapper_set = cast("Callable[[object], R]", mapper)
            return FlextUtilitiesCollection._map_set(items, mapper_set)

        if isinstance(items, (dict, Mapping)):
            mapper_dict = cast("Callable[[str, object], R]", mapper)
            return FlextUtilitiesCollection._map_dict(items, mapper_dict)

        mapper_single = cast("Callable[[object], R]", mapper)
        return FlextUtilitiesCollection._map_single(items, mapper_single)

    @staticmethod
    def _map_result[T, R](
        items: r[T],
        mapper: Callable[[T], R],
        default_error: str,
    ) -> r[R]:
        """Map over r."""
        if items.is_success:
            mapper_typed: Callable[[object], R] = cast("Callable[[object], R]", mapper)
            mapped: R = mapper_typed(items.value)
            return r[R].ok(mapped)
        error_msg = str(items.error) if items.error else default_error
        return r[R].fail(error_msg)

    @staticmethod
    def _map_sequence[T, R](
        items: list[T] | tuple[T, ...],
        mapper: Callable[[T], R],
    ) -> list[R]:
        """Map over list or tuple."""
        return [mapper(item) for item in items]

    @staticmethod
    def _map_set[T, R](
        items: set[T] | frozenset[T],
        mapper: Callable[[object], R] | Callable[[str, object], R],
    ) -> set[R] | frozenset[R]:
        """Map over set or frozenset."""
        mapper_typed: Callable[[object], R] = cast("Callable[[object], R]", mapper)
        mapped_set: set[R] = {mapper_typed(item) for item in items}
        if isinstance(items, frozenset):
            return frozenset(mapped_set)
        return mapped_set

    @staticmethod
    def _map_dict[T, R](
        items: dict[str, T] | Mapping[str, T],
        mapper: Callable[[str, object], R],
    ) -> dict[str, R]:
        """Map over dict or Mapping."""
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
        items: list[T]
        | tuple[T, ...]
        | set[T]
        | frozenset[T]
        | dict[str, T]
        | Mapping[str, T],
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
        if isinstance(items, (list, tuple, set, frozenset)):
            for item in items:
                if predicate(item):
                    return item
            return None

        if isinstance(items, (dict, Mapping)):
            # items is dict[str, T] | Mapping[str, T], so isinstance(items, dict) is redundant
            # dict() constructor works for both dict and Mapping
            mapping_items: dict[str, T] = dict(items)
            for k, v in mapping_items.items():
                if predicate(k, v):
                    result: T | tuple[str, T] = (k, v) if return_key else v
                    return result
        return None

    # ─────────────────────────────────────────────────────────────
    # FILTER METHODS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _filter_list(
        items_list: list[object],
        predicate: Callable[..., bool],
        mapper: Callable[..., object] | None = None,
    ) -> list[object]:
        """Filter a list with optional mapping."""
        if mapper is not None:
            mapped_raw = FlextUtilitiesCollection.map(items_list, mapper)
            mapped_list: list[object] = (
                mapped_raw if FlextUtilitiesGuards.is_type(mapped_raw, list) else []
            )
            return [item for item in mapped_list if predicate(item)]
        return [item for item in items_list if predicate(item)]

    @staticmethod
    def _filter_dict(
        items_dict: t.Types.ConfigurationMapping,
        predicate: Callable[..., bool],
        mapper: Callable[..., object] | None = None,
    ) -> t.Types.ConfigurationMapping:
        """Filter a dict with optional mapping."""
        if mapper is not None:
            mapped_dict_raw = FlextUtilitiesCollection.map(items_dict, mapper)
            mapped_dict_raw_dict = (
                mapped_dict_raw
                if FlextUtilitiesGuards.is_type(mapped_dict_raw, dict)
                else {}
            )
            mapped_dict: t.Types.ConfigurationMapping = cast(
                "t.Types.ConfigurationMapping",
                mapped_dict_raw_dict,
            )
            return {k: v for k, v in mapped_dict.items() if predicate(k, v)}
        return {k: v for k, v in items_dict.items() if predicate(k, v)}

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
        if isinstance(items, (list, tuple)):
            list_items: list[object] = list(items)
            list_result = FlextUtilitiesCollection._filter_list(
                list_items,
                predicate,
                mapper,
            )
            return cast("list[T] | list[R]", list_result)
        if isinstance(items, (dict, Mapping)):
            dict_items_raw = (
                dict(items) if isinstance(items, dict) else dict(items.items())
            )
            dict_items: t.Types.ConfigurationMapping = cast(
                "t.Types.ConfigurationMapping",
                dict_items_raw,
            )
            dict_result = FlextUtilitiesCollection._filter_dict(
                dict_items,
                predicate,
                mapper,
            )
            return cast("dict[str, T] | dict[str, R]", dict_result)
        # Single item case
        single_result = FlextUtilitiesCollection._filter_single(
            items,
            predicate,
            mapper,
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
        return r[list[R] | dict[str, R]].ok(dict_result)

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
        if isinstance(items, (list, tuple)):
            list_processor = cast("Callable[[T], R]", processor)
            list_predicate = (
                cast("Callable[[T], bool] | None", predicate)
                if predicate is not None
                else None
            )
            return FlextUtilitiesCollection._process_list_items(
                list(items),
                list_processor,
                predicate=list_predicate,
                on_error=on_error,
            )

        if isinstance(items, (dict, Mapping)):
            dict_processor = cast("Callable[[str, T], R]", processor)
            dict_predicate = (
                cast("Callable[[str, T], bool] | None", predicate)
                if predicate is not None
                else None
            )
            return FlextUtilitiesCollection._process_dict_items(
                dict(items),
                dict_processor,
                predicate=dict_predicate,
                filter_keys=filter_keys,
                exclude_keys=exclude_keys,
                on_error=on_error,
            )

        single_processor = cast("Callable[[T], R]", processor)
        single_predicate = (
            cast("Callable[[T], bool] | None", predicate)
            if predicate is not None
            else None
        )
        return FlextUtilitiesCollection._process_list_items(
            [items],
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
            k: object = key(item) if callable(key) else getattr(item, key, None)
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
    ) -> R | r[t.Types.BatchResultDict] | None:
        """Helper: Process a single batch item."""
        try:
            result = operation(item)
            if isinstance(result, r):
                if result.is_failure:
                    error_msg = str(result.error) if result.error else "Unknown error"
                    error_text = f"Item {idx} failed: {error_msg}"
                    if on_error == "fail":
                        return r[t.Types.BatchResultDict].fail(error_text)
                    if on_error == "collect":
                        errors.append((idx, error_msg))
                    return None
                return result.value
            return result
        except Exception as e:
            error_msg = str(e)
            error_text = f"Item {idx} failed: {error_msg}"
            if on_error == "fail":
                return r[t.Types.BatchResultDict].fail(error_text)
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

        nested = cast(
            "list[list[t.GeneralValueType] | tuple[t.GeneralValueType, ...]]",
            FlextUtilitiesCollection.filter(validated_results, is_list_or_tuple),
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
    ) -> r[t.Types.BatchResultDict]:
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
            r[t.Types.BatchResultDict] containing batch results with errors

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
            process_result = FlextUtilitiesCollection._batch_process_single_item(
                item,
                idx,
                operation,
                errors,
                on_error,
            )
            if process_result is None:
                continue  # Item skipped
            if isinstance(process_result, r):
                return process_result  # Fail mode returned error
            processed_results.append(process_result)

            # Call progress callback if provided
            if progress is not None and idx % _progress_interval == 0:
                progress(idx + 1, total_items)

        # Convert to GeneralValueType for flattening
        def to_general_value(item: object) -> t.GeneralValueType:
            """Convert item to GeneralValueType."""
            return cast("t.GeneralValueType", item)

        validated_results_raw_list = cast("list[object]", processed_results)
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

        batch_result: t.Types.BatchResultDict = {
            "results": flattened_results,
            "errors": errors,
            "total": total_items,
            "success_count": len(flattened_results),
            "error_count": len(errors),
        }
        return r[t.Types.BatchResultDict].ok(batch_result)

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
        base: t.Types.ConfigurationMapping,
        other: t.Types.ConfigurationMapping,
        *,
        strategy: str = "deep",
    ) -> r[t.Types.ConfigurationDict]:
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
            merged: t.Types.ConfigurationDict = copy.deepcopy(dict(base))
            other_dict: t.Types.ConfigurationDict = dict(other)

            if strategy == c.Mixins.OPERATION_OVERRIDE:
                merged.update(other_dict)
                return r[t.Types.ConfigurationDict].ok(merged)

            # Helper for deep merge
            def _deep_merge(
                target: t.Types.ConfigurationDict,
                source: t.Types.ConfigurationDict,
                mode: str,
            ) -> None:
                for key, value in source.items():
                    # Filter logic
                    if mode in {"filter_none", "filter_both"} and value is None:
                        continue
                    if mode in {"filter_empty", "filter_both"}:
                        if value is None:
                            continue
                        if (
                            isinstance(value, (str, list, dict, tuple, set))
                            and len(value) == 0
                        ):
                            continue

                    # Recursive logic
                    if key in target:
                        target_val = target[key]

                        # Both are dicts -> recurse
                        if isinstance(target_val, dict) and isinstance(value, dict):
                            # Type narrowing: isinstance ensures both are dict
                            # No cast needed - isinstance provides type narrowing
                            _deep_merge(
                                target_val,
                                value,
                                mode,
                            )
                            continue

                        # Both are lists and mode is append -> append
                        if (
                            mode == "append"
                            and isinstance(target_val, list)
                            and isinstance(value, list)
                        ):
                            # Append elements from source list to target list
                            # Create new list to avoid mutating original objects if they were refs
                            target[key] = list(target_val) + list(value)
                            continue

                    # Default: override
                    # If value is a container, deep copy it to avoid shared references
                    if isinstance(value, (dict, list)):
                        target[key] = copy.deepcopy(value)
                    else:
                        target[key] = value

            _deep_merge(merged, other_dict, strategy)
            return r[t.Types.ConfigurationDict].ok(merged)
        except Exception as e:
            return r[t.Types.ConfigurationDict].fail(f"Merge failed: {e}")


__all__ = [
    "FlextUtilitiesCollection",
]
