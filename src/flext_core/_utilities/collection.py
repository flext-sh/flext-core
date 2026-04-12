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
from enum import StrEnum
from typing import ClassVar, overload

from flext_core import (
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesResultHelpers,
    T,
    U,
    c,
    r,
    t,
)


class FlextUtilitiesCollection:
    """Utilities for collection operations with full generic type support."""

    @staticmethod
    def _ok_result[TValue](value: TValue) -> r[TValue]:
        """Create a typed success result without leaking generic inference noise."""

        def _return_value() -> TValue:
            return value

        return r[TValue].create_from_callable(_return_value)

    @staticmethod
    def _ok_mapping_result(
        value: t.RecursiveContainerMapping | t.MutableRecursiveContainerMapping,
    ) -> r[t.RecursiveContainerMapping]:
        """Wrap a recursive mapping as a typed success result."""

        def _return_mapping() -> t.RecursiveContainerMapping:
            return value

        return r[t.RecursiveContainerMapping].create_from_callable(_return_mapping)

    @staticmethod
    def _ok_sequence_result[TValue](
        value: Sequence[TValue],
    ) -> r[Sequence[TValue]]:
        """Wrap a sequence as a typed success result."""

        def _return_sequence() -> Sequence[TValue]:
            return value

        return r[Sequence[TValue]].create_from_callable(_return_sequence)

    @staticmethod
    def _parse_enum_member[E: StrEnum](enum_cls: type[E], value: str) -> r[E]:
        """Parse one enum member through the result carrier."""

        def _build_enum() -> E:
            return enum_cls(value)

        return r[E].create_from_callable(_build_enum)

    @staticmethod
    def _safe_validate_serializable[T](
        value: t.ValueOrModel,
    ) -> t.Serializable:
        """Validate via Pydantic adapter, falling back to str on any error."""
        try:
            return t.serializable_adapter().validate_python(value)
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            result: t.Serializable = str(value)
            return result

    @staticmethod
    def _normalize_unknown_value[T](value: t.ValueOrModel) -> t.RecursiveContainer:
        validated = FlextUtilitiesCollection._safe_validate_serializable(value)
        if FlextUtilitiesGuardsTypeCore.scalar(validated):
            return validated
        if isinstance(validated, list):
            normalized_items: t.MutableRecursiveContainerList = [
                FlextUtilitiesCollection._normalize_unknown_value(item)
                for item in validated
            ]
            return normalized_items
        if isinstance(validated, dict):
            normalized_dict: t.MutableRecursiveContainerMapping = {}
            for dict_key, dict_value in validated.items():
                normalized_dict[str(dict_key)] = (
                    FlextUtilitiesCollection._normalize_unknown_value(dict_value)
                )
            return normalized_dict
        return str(validated)

    @staticmethod
    def _normalize_mapping_items[T](
        data: t.ValueOrModel,
    ) -> Sequence[tuple[str, t.RecursiveContainer]]:
        normalized_mapping: t.RecursiveContainerMapping = (
            t.dict_str_metadata_adapter().validate_python(
                data,
            )
        )
        normalized_items: Sequence[tuple[str, t.RecursiveContainer]] = list(
            normalized_mapping.items(),
        )
        return normalized_items

    @staticmethod
    def _is_empty_value[T](value: t.ValueOrModel) -> bool:
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
    def _merge_deep_single_key[T](
        result: t.MutableRecursiveContainerMapping,
        key: str,
        value: t.RecursiveContainer,
    ) -> r[bool]:
        """Merge single key in deep merge strategy."""
        current_val = result.get(key)
        if (
            current_val is not None
            and isinstance(current_val, dict)
            and isinstance(value, dict)
        ):
            merged_result: r[t.RecursiveContainerMapping] = (
                FlextUtilitiesCollection.merge_mappings(
                    value,
                    current_val,
                    strategy="deep",
                )
            )
            if merged_result.success:
                merged_value: t.RecursiveContainerMapping = (
                    FlextUtilitiesResultHelpers.expect_success(merged_result)
                )
                result[key] = merged_value
                return FlextUtilitiesCollection._ok_result(True)
            return r[bool].fail(
                f"Failed to merge nested dict for key {key}: {merged_result.error}",
            )
        result[key] = value
        return FlextUtilitiesCollection._ok_result(True)

    @staticmethod
    def coerce_list_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.ValueOrModel], Sequence[E]]:
        """Create validator that coerces list values to a StrEnum type.

        Raises:
            TypeError: If input is not a sequence or item is not str
            ValueError: If string value is not a valid enum member

        """

        def validator(data: t.ValueOrModel) -> Sequence[E]:
            if isinstance(data, str):
                msg = c.ERR_COLLECTION_EXPECTED_SEQUENCE.format(
                    type_name=data.__class__.__name__,
                )
                raise TypeError(msg)
            normalized_items_result = r[t.FlatContainerList].create_from_callable(
                lambda: t.flat_container_list_adapter().validate_python(data),
            )
            if normalized_items_result.failure:
                msg = c.ERR_COLLECTION_EXPECTED_SEQUENCE.format(
                    type_name=data.__class__.__name__,
                )
                raise TypeError(msg)
            normalized_items = normalized_items_result.value
            result: MutableSequence[E] = []
            for v_raw in normalized_items:
                if isinstance(v_raw, enum_cls):
                    result.append(v_raw)
                elif isinstance(v_raw, str):
                    enum_result = FlextUtilitiesCollection._parse_enum_member(
                        enum_cls,
                        v_raw,
                    )
                    if enum_result.failure:
                        raise ValueError(
                            c.ERR_COLLECTION_INVALID_ENUM_VALUE.format(
                                enum_name=getattr(enum_cls, "__name__", "Enum"),
                                value=v_raw,
                            ),
                        ) from None
                    parsed_member: E = FlextUtilitiesResultHelpers.expect_success(
                        enum_result,
                    )
                    result.append(parsed_member)
                else:
                    raise TypeError(
                        c.ERR_COLLECTION_EXPECTED_STR_FOR_ENUM.format(
                            type_name=v_raw.__class__.__name__,
                        ),
                    )
            return result

        return validator

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
    ) -> r[T]:
        """Find first item matching predicate with generic type support.

        Returns first item where predicate returns True, or None.
        """
        if isinstance(items, (list, tuple)):
            for item in items:
                result: bool = predicate(item)
                if result:
                    return FlextUtilitiesCollection._ok_result(item)
            return r[T].fail(c.ERR_COLLECTION_NO_MATCHING_ITEM_FOUND)
        if isinstance(items, Mapping):
            for v in items.values():
                matched: bool = predicate(v)
                if matched:
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
    ) -> r[t.RecursiveContainerMapping]:
        """Replace strategy: base values overwrite other."""
        result: t.MutableRecursiveContainerMapping = dict(other)
        result.update(base)
        return FlextUtilitiesCollection._ok_mapping_result(result)

    @staticmethod
    def _merge_filter_none(
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
    ) -> r[t.RecursiveContainerMapping]:
        """Filter-none strategy: skip None values from base."""
        result: t.MutableRecursiveContainerMapping = dict(other)
        for key, value in base.items():
            if value is not None:
                result[key] = value
        return FlextUtilitiesCollection._ok_mapping_result(result)

    @staticmethod
    def _merge_filter_empty(
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
    ) -> r[t.RecursiveContainerMapping]:
        """Filter-empty strategy: skip empty values from base."""
        result: t.MutableRecursiveContainerMapping = dict(other)
        for key, value in base.items():
            if not FlextUtilitiesCollection._is_empty_value(value):
                result[key] = value
        return FlextUtilitiesCollection._ok_mapping_result(result)

    @staticmethod
    def _merge_append(
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
    ) -> r[t.RecursiveContainerMapping]:
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
        return FlextUtilitiesCollection._ok_mapping_result(result)

    @staticmethod
    def _merge_deep(
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
    ) -> r[t.RecursiveContainerMapping]:
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
        return FlextUtilitiesCollection._ok_mapping_result(result)

    _MergeHandler = Callable[
        [t.RecursiveContainerMapping, t.RecursiveContainerMapping],
        "r[t.RecursiveContainerMapping]",
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
        other: t.RecursiveContainerMapping,
        base: t.RecursiveContainerMapping,
        *,
        strategy: str = "deep",
    ) -> r[t.RecursiveContainerMapping]:
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
        handler = FlextUtilitiesCollection._MERGE_STRATEGIES.get(strategy)
        if handler is None:
            return r[t.RecursiveContainerMapping].fail(
                f"Unknown merge strategy: {strategy}",
            )
        return handler(other, base)

    @staticmethod
    def parse_sequence(
        enum_cls: type[StrEnum],
        values: Sequence[str | StrEnum],
    ) -> r[tuple[StrEnum, ...]]:
        """Parse sequence of strings to tuple of StrEnum."""
        parsed: MutableSequence[StrEnum] = []
        errors: MutableSequence[str] = []
        enumerate_result = r[Sequence[tuple[int, str | StrEnum]]].create_from_callable(
            lambda: list(enumerate(values)),
        )
        if enumerate_result.failure:
            return r[tuple[StrEnum, ...]].fail(
                f"Parse sequence failed: {enumerate_result.error}",
            )
        for idx, val in enumerate_result.value:
            if isinstance(val, enum_cls):
                parsed.append(val)
                continue
            enum_result = FlextUtilitiesCollection._parse_enum_member(
                enum_cls,
                str(val),
            )
            if enum_result.failure:
                errors.append(f"[{idx}]: '{val}'")
                continue
            parsed_member: StrEnum = FlextUtilitiesResultHelpers.expect_success(
                enum_result,
            )
            parsed.append(parsed_member)
        if errors:
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return r[tuple[StrEnum, ...]].fail(
                f"Invalid {enum_name} values: {', '.join(errors)}",
            )
        return FlextUtilitiesCollection._ok_result(tuple(parsed))

    @staticmethod
    def process(
        items: Sequence[T],
        processor: Callable[[T], U],
        *,
        predicate: Callable[[T], bool] | None = None,
        on_error: str = "fail",
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> r[Sequence[U]]:
        """Process items with optional filtering; on_error="skip" to skip failures."""
        _ = filter_keys
        _ = exclude_keys
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
            processed_item: U = FlextUtilitiesResultHelpers.expect_success(
                process_result,
            )
            results.append(processed_item)
        return FlextUtilitiesCollection._ok_sequence_result(results)


__all__: list[str] = ["FlextUtilitiesCollection"]
