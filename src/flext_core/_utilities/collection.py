"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import (
    Callable,
    Hashable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from datetime import datetime
from enum import StrEnum
from itertools import batched, chain
from typing import ClassVar, overload

from flext_core import FlextUtilitiesGuardsTypeCore, T, U, m, r, t


class FlextUtilitiesCollection:
    """Utilities for collection operations with full generic type support."""

    @staticmethod
    def vals(
        value: Mapping[str, t.NormalizedValue]
        | r[Mapping[str, t.NormalizedValue]]
        | None,
        *,
        default: Sequence[t.NormalizedValue] | None = None,
    ) -> r[t.ContainerList]:
        """Return mapping values with optional fallback for failures or empties."""
        if isinstance(value, r):
            if value.is_failure:
                if default is None:
                    return r[t.ContainerList].fail(value.error or "No values available")
                return r[t.ContainerList].ok(list(default))
            value = value.value
        if value is None:
            if default is None:
                return r[t.ContainerList].fail("No values available")
            return r[t.ContainerList].ok(list(default))
        values = list(value.values())
        if values:
            return r[t.ContainerList].ok(values)
        if default is not None:
            return r[t.ContainerList].ok(list(default))
        return r[t.ContainerList].fail("No values available")

    @staticmethod
    def _safe_validate_serializable(
        value: t.Serializable | t.NormalizedValue,
    ) -> t.Serializable:
        """Validate via Pydantic adapter, falling back to str on any error."""
        try:
            return m.Validators.serializable_adapter().validate_python(value)
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            result: t.Serializable = str(value)
            return result

    @staticmethod
    def _coerce_guard_value(value: t.Serializable) -> t.NormalizedValue:
        validated = FlextUtilitiesCollection._safe_validate_serializable(value)
        if FlextUtilitiesGuardsTypeCore.is_scalar(validated):
            return validated
        if isinstance(validated, list):
            normalized_list: t.ContainerList = [
                item if FlextUtilitiesGuardsTypeCore.is_scalar(item) else str(item)
                for item in validated
            ]
            return normalized_list
        if isinstance(validated, dict):
            normalized_dict: t.MutableContainerMapping = {}
            for dict_key, dict_val in validated.items():
                if FlextUtilitiesGuardsTypeCore.is_scalar(dict_val):
                    normalized_dict[dict_key] = dict_val
                else:
                    normalized_dict[dict_key] = str(dict_val)
            return normalized_dict
        return str(validated)

    @staticmethod
    def _coerce_or_stringify(value: t.Serializable) -> t.NormalizedValue:
        """Coerce a serializable value via _coerce_guard_value, or stringify."""
        if isinstance(value, (dict, list) + t.SCALAR_TYPES):
            return FlextUtilitiesCollection._coerce_guard_value(value)
        fallback: t.NormalizedValue = value
        return str(fallback)

    @staticmethod
    def _normalize_unknown_value(value: t.NormalizedValue) -> t.NormalizedValue:
        validated = FlextUtilitiesCollection._safe_validate_serializable(value)
        if FlextUtilitiesGuardsTypeCore.is_scalar(validated):
            return validated
        if isinstance(validated, list):
            normalized_items: t.ContainerList = [
                FlextUtilitiesCollection._normalize_unknown_value(item)
                for item in validated
            ]
            return normalized_items
        if isinstance(validated, dict):
            normalized_dict: t.MutableContainerMapping = {}
            for dict_key, dict_value in validated.items():
                normalized_dict[str(dict_key)] = (
                    FlextUtilitiesCollection._normalize_unknown_value(dict_value)
                )
            return normalized_dict
        return str(validated)

    @staticmethod
    def _normalize_mapping_items(
        data: t.NormalizedValue,
    ) -> Sequence[tuple[str, t.NormalizedValue]]:
        normalized_mapping = m.Validators.dict_str_metadata_adapter().validate_python(
            data,
        )
        return list(normalized_mapping.items())

    @staticmethod
    def _is_empty_value(value: t.NormalizedValue) -> bool:
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
        result: t.MutableContainerMapping,
        key: str,
        value: t.NormalizedValue,
    ) -> r[bool]:
        """Merge single key in deep merge strategy."""
        current_val = result.get(key)
        if (
            current_val is not None
            and isinstance(current_val, dict)
            and isinstance(value, dict)
        ):
            merged = FlextUtilitiesCollection.merge_mappings(
                value,
                current_val,
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
    def _to_batch_scalar(value: t.NormalizedValue) -> t.Scalar:
        if value is None:
            return ""
        if FlextUtilitiesGuardsTypeCore.is_scalar(value):
            return value
        return str(value)

    @staticmethod
    def _to_batch_scalars(
        values: t.ContainerList,
    ) -> Sequence[t.Scalar | None]:
        return [FlextUtilitiesCollection._to_batch_scalar(value) for value in values]

    @staticmethod
    def _flatten_items(
        items: Sequence[t.Serializable],
        results: t.MutableContainerList,
    ) -> None:
        """Flatten list items into results, coercing each to container type."""
        for inner_item in items:
            if isinstance(
                inner_item,
                (dict, list, str, int, float, bool, datetime),
            ):
                results.append(
                    FlextUtilitiesCollection._coerce_guard_value(inner_item),
                )
            else:
                results.append(str(inner_item))

    @staticmethod
    def _handle_batch_error(
        error_mode: str,
        error_msg: str,
        processed: int,
        errors: MutableSequence[tuple[int, str]],
    ) -> r[m.BatchResultDict] | None:
        """Handle batch error: return fail result or collect and continue."""
        if error_mode == "fail":
            return r[m.BatchResultDict].fail(
                f"Batch processing failed: {error_msg}",
            )
        if error_mode == "collect":
            errors.append((processed - 1, str(error_msg)))
        return None

    @staticmethod
    def _append_or_extend(
        value: t.NormalizedValue,
        results: t.MutableContainerList,
        *,
        do_flatten: bool,
    ) -> None:
        """Append value to results, extending if flatten and value is a list."""
        if do_flatten and isinstance(value, list):
            results.extend(value)
        else:
            results.append(value)

    @staticmethod
    def _process_result_value(
        result_raw: r[t.Serializable],
        results: t.MutableContainerList,
        errors: MutableSequence[tuple[int, str]],
        error_mode: str,
        processed: int,
        *,
        do_flatten: bool,
    ) -> r[m.BatchResultDict] | None:
        """Process a Result-wrapped operation return value."""
        if result_raw.is_success:
            result_value = m.Validators.serializable_adapter().validate_python(
                result_raw.unwrap_or(None),
            )
            if do_flatten and isinstance(result_value, list):
                FlextUtilitiesCollection._flatten_items(result_value, results)
            else:
                FlextUtilitiesCollection._append_or_extend(
                    FlextUtilitiesCollection._coerce_or_stringify(result_value),
                    results,
                    do_flatten=do_flatten,
                )
            return None
        return FlextUtilitiesCollection._handle_batch_error(
            error_mode,
            result_raw.error or "Unknown error",
            processed,
            errors,
        )

    @staticmethod
    def _process_raw_value(
        result_raw: t.Serializable,
        results: t.MutableContainerList,
        *,
        do_flatten: bool,
    ) -> None:
        """Validate and append a non-Result operation return value."""
        try:
            normalized = m.Validators.serializable_adapter().validate_python(
                result_raw,
            )
            if do_flatten and isinstance(normalized, list):
                FlextUtilitiesCollection._flatten_items(normalized, results)
                return
            direct_result: t.NormalizedValue = (
                FlextUtilitiesCollection._coerce_or_stringify(normalized)
            )
        except (TypeError, ValueError):
            direct_result = str(result_raw)
        FlextUtilitiesCollection._append_or_extend(
            direct_result,
            results,
            do_flatten=do_flatten,
        )

    @staticmethod
    def batch(
        items: Sequence[T],
        operation: Callable[[T], t.Serializable | r[t.Serializable]],
        *,
        spec: m.CollectionBatchSpec | None = None,
        **batch_kwargs: t.RuntimeData
        | Callable[[int, int], None]
        | Callable[[T], bool],
    ) -> r[m.BatchResultDict]:
        """Process items in batches with progress tracking.

        Args:
            items: Items to process
            operation: Function that returns R or p.Result[R]
            spec: Batch execution specification.

        """
        base_spec = spec if spec is not None else m.CollectionBatchSpec()
        if batch_kwargs:
            override_spec = m.CollectionBatchSpec.model_validate(batch_kwargs)
            override_data = {
                k: v
                for k, v in override_spec.model_dump().items()
                if batch_kwargs.get(k) is not None
            }
            resolved_spec = base_spec.model_copy(update=override_data)
        else:
            resolved_spec = base_spec
        _ = resolved_spec.size
        _ = resolved_spec.parallel
        progress = resolved_spec.progress
        progress_interval = resolved_spec.progress_interval
        pre_validate = resolved_spec.pre_validate
        do_flatten = resolved_spec.flatten
        error_mode = resolved_spec.on_error or "fail"
        results: t.MutableContainerList = []
        errors: MutableSequence[tuple[int, str]] = []
        total = len(items)
        for processed, item in enumerate(items, 1):
            item_typed: T = item
            if pre_validate is not None:
                normalized_item: t.MetadataValue = str(item_typed)
                if not pre_validate(normalized_item):
                    results.append(None)
                    continue
            try:
                result_raw = operation(item)
                if isinstance(result_raw, r):
                    err = FlextUtilitiesCollection._process_result_value(
                        result_raw,
                        results,
                        errors,
                        error_mode,
                        processed,
                        do_flatten=do_flatten,
                    )
                    if err is not None:
                        return err
                    continue
                FlextUtilitiesCollection._process_raw_value(
                    result_raw,
                    results,
                    do_flatten=do_flatten,
                )
            except (
                TypeError,
                ValueError,
                RuntimeError,
                AttributeError,
                KeyError,
            ) as exc:
                err = FlextUtilitiesCollection._handle_batch_error(
                    error_mode,
                    str(exc),
                    processed,
                    errors,
                )
                if err is not None:
                    return err
            if progress is not None and processed % progress_interval == 0:
                progress(processed, total)
        result_dict = m.BatchResultDict(
            results=FlextUtilitiesCollection._to_batch_scalars(results),
            total=total,
            success_count=len(results),
            error_count=len(errors),
            errors=errors,
        )
        return r[m.BatchResultDict].ok(result_dict)

    @staticmethod
    def chunk(items: Sequence[T], size: int) -> Sequence[Sequence[T]]:
        """Split sequence into chunks of specified size.

        Args:
            items: Sequence to split
            size: Maximum size of each chunk

        Returns:
            List of chunks

        Example:
            batches = u.chunk(records, 100)
            # [[record1, ..., record100], [record101, ...], ...]

        """
        return (
            [list(batch) for batch in batched(items, size, strict=False)]
            if size > 0
            else [list(items)]
        )

    @staticmethod
    def coerce_list_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.NormalizedValue], Sequence[E]]:
        """Create validator that coerces list values to a StrEnum type.

        Raises:
            TypeError: If input is not a sequence or item is not str
            ValueError: If string value is not a valid enum member

        """

        def validator(data: t.NormalizedValue) -> Sequence[E]:
            if isinstance(data, str):
                msg = f"Expected sequence, got {data.__class__.__name__}"
                raise TypeError(msg)
            normalized_items_result = r[t.FlatContainerList].create_from_callable(
                lambda: m.Validators.list_container_adapter().validate_python(data),
            )
            if normalized_items_result.is_failure:
                msg = f"Expected sequence, got {data.__class__.__name__}"
                raise TypeError(msg)
            normalized_items = normalized_items_result.value
            result: MutableSequence[E] = []
            for v_raw in normalized_items:
                if isinstance(v_raw, enum_cls):
                    result.append(v_raw)
                elif isinstance(v_raw, str):
                    enum_result = r[str].ok(v_raw).map(enum_cls)
                    if enum_result.is_failure:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} value: '{v_raw}'"
                        raise ValueError(msg) from None
                    result.append(enum_result.value)
                else:
                    msg = f"Expected str for enum conversion, got {v_raw.__class__.__name__}"
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def count(items: Sequence[T], predicate: Callable[[T], bool] | None = None) -> int:
        """Count items, optionally matching predicate.

        Args:
            items: Sequence to count
            predicate: Optional filter function

        Returns:
            Count of matching items

        Example:
            active_count = u.count(users, lambda u: u.is_active)

        """
        if predicate is None:
            return len(items)
        return sum(1 for item in items if predicate(item))

    @staticmethod
    def extract_callable_mapping[K](
        mapping: Mapping[K, Callable[[], t.NormalizedValue]],
    ) -> Mapping[str, Callable[[], t.NormalizedValue]]:
        """Extract mapping of callables for resources/factories.

        Helper function to properly type narrow callable mappings for pyright.
        Filters to only callable values and converts to proper signature.

        Args:
            mapping: Mapping containing callable values

        Returns:
            Dict mapping string keys to callable functions

        """
        result: MutableMapping[str, Callable[[], t.NormalizedValue]] = {}
        items_iter = mapping.items()
        for item_tuple in items_iter:
            key_obj = item_tuple[0]
            value_raw = item_tuple[1]
            key_str: str = str(key_obj)
            result[key_str] = value_raw
        return result

    @staticmethod
    def extract_mapping_items[K](
        mapping: Mapping[K, t.NormalizedValue],
    ) -> Sequence[tuple[str, t.NormalizedValue]]:
        """Extract mapping items as typed list for iteration.

        Helper function to properly type narrow Mapping.items() for pyright.
        Converts keys to strings and values to t.NormalizedValue.

        Args:
            mapping: Mapping to extract items from

        Returns:
            List of (key, value) tuples with proper typing

        """
        result: Sequence[tuple[str, t.NormalizedValue]] = [
            (str(item_tuple[0]), item_tuple[1]) for item_tuple in mapping.items()
        ]
        return result

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
                    return r[T].ok(item)
            return r[T].fail("No matching item found")
        if isinstance(items, Mapping):
            for v in items.values():
                matched: bool = predicate(v)
                if matched:
                    return r[T].ok(v)
        return r[T].fail("No matching item found")

    @staticmethod
    def first(
        items: Sequence[T],
        predicate: Callable[[T], bool] | None = None,
        default: T | None = None,
    ) -> r[T]:
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
                return r[T].ok(item)
            result: bool = predicate(item)
            if result:
                return r[T].ok(item)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail("No first item found")

    @staticmethod
    def flatten(items: Sequence[Sequence[T]]) -> Sequence[T]:
        """Flatten nested sequences into single list.

        Args:
            items: Nested sequences to flatten (one level)

        Returns:
            Flattened list

        Example:
            flat = u.flatten([[1, 2], [3, 4], [5]])
            # [1, 2, 3, 4, 5]

        """
        return list(chain.from_iterable(items))

    @staticmethod
    def group_by(
        items: Sequence[T],
        key_func: Callable[[T], U],
    ) -> Mapping[U, Sequence[T]]:
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
        result: defaultdict[U, MutableSequence[T]] = defaultdict(list)
        for item in items:
            result[key_func(item)].append(item)
        return dict(result)

    @staticmethod
    def last(
        items: Sequence[T],
        predicate: Callable[[T], bool] | None = None,
        default: T | None = None,
    ) -> r[T]:
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
                return r[T].ok(item)
            result: bool = predicate(item)
            if result:
                return r[T].ok(item)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail("No last item found")

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
        other: t.ContainerMapping,
        base: t.ContainerMapping,
    ) -> r[t.ContainerMapping]:
        """Replace strategy: base values overwrite other."""
        result: t.MutableContainerMapping = dict(other)
        result.update(base)
        return r[t.ContainerMapping].ok(result)

    @staticmethod
    def _merge_filter_none(
        other: t.ContainerMapping,
        base: t.ContainerMapping,
    ) -> r[t.ContainerMapping]:
        """Filter-none strategy: skip None values from base."""
        result: t.MutableContainerMapping = dict(other)
        for key, value in base.items():
            if value is not None:
                result[key] = value
        return r[t.ContainerMapping].ok(result)

    @staticmethod
    def _merge_filter_empty(
        other: t.ContainerMapping,
        base: t.ContainerMapping,
    ) -> r[t.ContainerMapping]:
        """Filter-empty strategy: skip empty values from base."""
        result: t.MutableContainerMapping = dict(other)
        for key, value in base.items():
            if not FlextUtilitiesCollection._is_empty_value(value):
                result[key] = value
        return r[t.ContainerMapping].ok(result)

    @staticmethod
    def _merge_append(
        other: t.ContainerMapping,
        base: t.ContainerMapping,
    ) -> r[t.ContainerMapping]:
        """Append strategy: concatenate lists instead of replacing."""
        result: t.MutableContainerMapping = dict(other)
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
        return r[t.ContainerMapping].ok(result)

    @staticmethod
    def _merge_deep(
        other: t.ContainerMapping,
        base: t.ContainerMapping,
    ) -> r[t.ContainerMapping]:
        """Deep strategy: recursively merge nested dicts."""
        result: t.MutableContainerMapping = dict(other)
        for key, value in base.items():
            merge_result = FlextUtilitiesCollection._merge_deep_single_key(
                result,
                key,
                value,
            )
            if merge_result.is_failure:
                return r[t.ContainerMapping].fail(
                    merge_result.error or "Unknown error",
                )
        return r[t.ContainerMapping].ok(result)

    _MergeHandler = Callable[
        [t.ContainerMapping, t.ContainerMapping],
        "r[t.ContainerMapping]",
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
        other: t.ContainerMapping,
        base: t.ContainerMapping,
        *,
        strategy: str = "deep",
    ) -> r[t.ContainerMapping]:
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
            return r[t.ContainerMapping].fail(
                f"Unknown merge strategy: {strategy}",
            )
        return handler(other, base)

    @staticmethod
    def mul(*values: float) -> t.Numeric:
        """Multiply values.

        Args:
            *values: Values to multiply

        Returns:
            Product of all values

        Example:
            total = u.mul(price, quantity, tax_rate)

        """
        return math.prod(values) if values else 1

    @staticmethod
    def parse_mapping[E: StrEnum](
        enum_cls: type[E],
        mapping: Mapping[str, str | E],
    ) -> r[Mapping[str, E]]:
        """Parse dict values from strings to StrEnum.

        Args:
            enum_cls: StrEnum class to parse values to
            mapping: Dict with string or enum values

        Returns:
            p.Result with parsed dict

        Example:
            result = u.parse_mapping(Status, {"key": "active"})
            # result.value == {"key": Status.ACTIVE}

        """
        result: MutableMapping[str, E] = {}
        errors: MutableSequence[str] = []
        mapping_items_result = (
            r[Sequence[tuple[str, str | E]]].ok([]).map(lambda _: list(mapping.items()))
        )
        if mapping_items_result.is_failure:
            return r[Mapping[str, E]].fail(
                f"Parse mapping failed: {mapping_items_result.error}",
            )
        for key, value_raw in mapping_items_result.value:
            enum_result = r[str].ok(value_raw).map(enum_cls)
            if enum_result.is_failure:
                errors.append(f"'{key}': {enum_result.error}")
                continue
            result[key] = enum_result.value
        if errors:
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return r[Mapping[str, E]].fail(
                f"Invalid {enum_name} values: {', '.join(errors)}",
            )
        return r[Mapping[str, E]].ok(result)

    @staticmethod
    def parse_sequence(
        enum_cls: type[StrEnum],
        values: Sequence[str | StrEnum],
    ) -> r[tuple[StrEnum, ...]]:
        """Parse sequence of strings to tuple of StrEnum."""
        parsed: MutableSequence[StrEnum] = []
        errors: MutableSequence[str] = []
        enumerate_result = (
            r[Sequence[tuple[int, str | StrEnum]]]
            .ok([])
            .map(lambda _: list(enumerate(values)))
        )
        if enumerate_result.is_failure:
            return r[tuple[StrEnum, ...]].fail(
                f"Parse sequence failed: {enumerate_result.error}",
            )
        for idx, val in enumerate_result.value:
            if isinstance(val, enum_cls):
                parsed.append(val)
                continue
            enum_result = r[str].ok(val).map(enum_cls)
            if enum_result.is_failure:
                errors.append(f"[{idx}]: '{val}'")
                continue
            parsed.append(enum_result.value)
        if errors:
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return r[tuple[StrEnum, ...]].fail(
                f"Invalid {enum_name} values: {', '.join(errors)}",
            )
        return r[tuple[StrEnum, ...]].ok(tuple(parsed))

    @staticmethod
    def partition(
        items: Sequence[T],
        predicate: Callable[[T], bool],
    ) -> tuple[Sequence[T], Sequence[T]]:
        """Split items by predicate: (matches, non-matches).

        Args:
            items: Items to partition
            predicate: Function to test each item

        Returns:
            Tuple of (matching_items, non_matching_items)

        Example:
            active, inactive = u.partition(users, lambda u: u.is_active)

        """
        matches: MutableSequence[T] = []
        non_matches: MutableSequence[T] = []
        for item in items:
            result: bool = predicate(item)
            if result:
                matches.append(item)
            else:
                non_matches.append(item)
        return (matches, non_matches)

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
            p.Result with list of processed results or error

        """
        _ = filter_keys
        _ = exclude_keys
        results: MutableSequence[U] = []
        for item in items:
            item_typed: T = item
            if predicate is not None and (not predicate(item_typed)):
                continue
            process_result = r[T].ok(item_typed).map(processor)
            if process_result.is_failure:
                if on_error == "skip":
                    continue
                return r[Sequence[U]].fail(f"Processing failed for item: {item}")
            results.append(process_result.value)
        return r[Sequence[U]].ok(results)

    @staticmethod
    def unique(
        items: Sequence[T],
        key_func: Callable[[T], Hashable] | None = None,
    ) -> Sequence[T]:
        """Get unique items preserving order.

        Args:
            items: Items to deduplicate
            key_func: Optional function to extract uniqueness key

        Returns:
            List of unique items in order of first appearance

        Example:
            unique_emails = u.unique(users, lambda u: u.email.lower())

        """
        if key_func is None:
            return list(dict.fromkeys(items))
        seen: set[Hashable] = set()
        result: MutableSequence[T] = []
        for item in items:
            key = key_func(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


__all__ = ["FlextUtilitiesCollection"]
