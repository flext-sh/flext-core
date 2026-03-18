"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from itertools import batched, chain
from typing import overload

from flext_core import T, U, m, r, t
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes


class FlextUtilitiesCollection:
    """Utilities for collection operations with full generic type support."""

    @staticmethod
    def _coerce_guard_value(value: t.Serializable) -> t.NormalizedValue:
        validated_result = r[t.Serializable].create_from_callable(
            lambda: m.Validators.serializable_adapter().validate_python(value),
        )
        if validated_result.is_failure:
            return str(value)
        validated = validated_result.value
        if FlextUtilitiesGuardsTypeCore.is_scalar(validated):
            return validated
        if isinstance(validated, list):
            normalized_list: list[t.NormalizedValue] = []
            for item in validated:
                if FlextUtilitiesGuardsTypeCore.is_scalar(item):
                    normalized_list.append(item)
                else:
                    normalized_list.append(str(item))
            return normalized_list
        if isinstance(validated, dict):
            normalized_dict: dict[str, t.NormalizedValue] = {}
            for dict_key, dict_val in validated.items():
                if FlextUtilitiesGuardsTypeCore.is_scalar(dict_val):
                    normalized_dict[dict_key] = dict_val
                else:
                    normalized_dict[dict_key] = str(dict_val)
            return normalized_dict
        return str(validated)

    @staticmethod
    def _validate_dict_str_metadata(
        data: t.NormalizedValue,
    ) -> r[dict[str, t.NormalizedValue]]:
        return r[dict[str, t.NormalizedValue]].create_from_callable(
            lambda: m.Validators.dict_str_metadata_adapter().validate_python(data),
        )

    @staticmethod
    def _validate_list_container(data: t.NormalizedValue) -> r[list[t.Container]]:
        return r[list[t.Container]].create_from_callable(
            lambda: m.Validators.list_container_adapter().validate_python(data),
        )

    @staticmethod
    def _coerce_value_to_bool(value: t.NormalizedValue) -> bool:
        """Coerce a value to bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes"}
        return bool(value)

    @staticmethod
    def _coerce_value_to_float(value: t.NormalizedValue) -> float:
        """Coerce a value to float."""
        if isinstance(value, float):
            return value
        return float(str(value))

    @staticmethod
    def _coerce_value_to_int(value: t.NormalizedValue) -> int:
        """Coerce a value to int."""
        if isinstance(value, int) and (not isinstance(value, bool)):
            return value
        return int(str(value))

    @staticmethod
    def _coerce_value_to_str(value: t.NormalizedValue) -> str:
        """Coerce a value to string."""
        return str(value)

    @staticmethod
    def _normalize_unknown_value(value: object) -> t.NormalizedValue:
        validated_result = r[t.Serializable].create_from_callable(
            lambda: m.Validators.serializable_adapter().validate_python(value),
        )
        if validated_result.is_failure:
            return str(value)
        validated = validated_result.value
        if FlextUtilitiesGuardsTypeCore.is_scalar(validated):
            return validated
        if isinstance(validated, list):
            normalized_items: list[t.NormalizedValue] = [
                FlextUtilitiesCollection._normalize_unknown_value(item)
                for item in validated
            ]
            return normalized_items
        if isinstance(validated, dict):
            normalized_dict: dict[str, t.NormalizedValue] = {}
            for dict_key, dict_value in validated.items():
                normalized_dict[str(dict_key)] = (
                    FlextUtilitiesCollection._normalize_unknown_value(dict_value)
                )
            return normalized_dict
        return str(validated)

    @staticmethod
    def _normalize_mapping_items(
        data: t.NormalizedValue,
    ) -> list[tuple[str, t.NormalizedValue]]:
        normalized_mapping = m.Validators.dict_str_metadata_adapter().validate_python(
            data,
        )
        return list(normalized_mapping.items())

    @staticmethod
    def _normalize_sequence_items(data: t.NormalizedValue) -> list[t.Container]:
        return m.Validators.list_container_adapter().validate_python(data)

    @staticmethod
    def _is_empty_value(value: t.NormalizedValue) -> bool:
        """Check if value is considered empty (empty string, empty list, etc.)."""
        if value is None:
            return True
        if isinstance(value, str):
            return not value
        if isinstance(value, list):
            return len(value) == 0
        if isinstance(value, dict):
            return len(value) == 0
        return False

    @staticmethod
    def _merge_deep_single_key(
        result: dict[str, t.NormalizedValue],
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
            merged = FlextUtilitiesCollection.merge(current_val, value, strategy="deep")
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
    def _to_batch_scalars(values: Sequence[t.NormalizedValue]) -> list[t.Scalar | None]:
        return [FlextUtilitiesCollection._to_batch_scalar(value) for value in values]

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
            operation: Function that returns R or r[R]
            spec: Batch execution specification.

        """
        base_spec = spec if spec is not None else m.CollectionBatchSpec()
        if batch_kwargs:
            override_spec = m.CollectionBatchSpec.model_validate(batch_kwargs)
            resolved_spec = m.CollectionBatchSpec(
                size=override_spec.size
                if override_spec.size is not None
                else base_spec.size,
                on_error=(
                    override_spec.on_error
                    if override_spec.on_error is not None
                    else base_spec.on_error
                ),
                parallel=override_spec.parallel
                if batch_kwargs.get("parallel") is not None
                else base_spec.parallel,
                progress=(
                    override_spec.progress
                    if override_spec.progress is not None
                    else base_spec.progress
                ),
                progress_interval=(
                    override_spec.progress_interval
                    if batch_kwargs.get("progress_interval") is not None
                    else base_spec.progress_interval
                ),
                pre_validate=(
                    override_spec.pre_validate
                    if override_spec.pre_validate is not None
                    else base_spec.pre_validate
                ),
                flatten=override_spec.flatten
                if batch_kwargs.get("flatten") is not None
                else base_spec.flatten,
            )
        else:
            resolved_spec = base_spec
        _ = resolved_spec.size
        _ = resolved_spec.parallel
        progress = resolved_spec.progress
        progress_interval = resolved_spec.progress_interval
        pre_validate = resolved_spec.pre_validate
        do_flatten = resolved_spec.flatten
        error_mode = resolved_spec.on_error or "fail"
        results: list[t.NormalizedValue] = []
        errors: list[tuple[int, str]] = []
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
                    if result_raw.is_success:
                        result_value = (
                            m.Validators.serializable_adapter().validate_python(
                                result_raw.unwrap_or(None),
                            )
                        )
                        if do_flatten and isinstance(result_value, list):
                            for inner_item in result_value:
                                if isinstance(
                                    inner_item,
                                    (dict, list, str, int, float, bool, datetime),
                                ):
                                    results.append(
                                        FlextUtilitiesCollection._coerce_guard_value(
                                            inner_item,
                                        ),
                                    )
                                else:
                                    inner_item_obj: object = inner_item
                                    results.append(str(inner_item_obj))
                            continue
                        if isinstance(result_value, dict):
                            dict_value: dict[str, t.Serializable] = result_value
                            value = FlextUtilitiesCollection._coerce_guard_value(
                                dict_value,
                            )
                        elif isinstance(result_value, list):
                            list_value: list[t.Serializable] = result_value
                            value = FlextUtilitiesCollection._coerce_guard_value(
                                list_value,
                            )
                        elif isinstance(result_value, t.SCALAR_TYPES):
                            value = FlextUtilitiesCollection._coerce_guard_value(
                                result_value,
                            )
                        else:
                            result_value_obj: object = result_value
                            value = str(result_value_obj)
                        if do_flatten and isinstance(value, list):
                            results.extend(value)
                        else:
                            results.append(value)
                    else:
                        error_msg = result_raw.error or "Unknown error"
                        if error_mode == "fail":
                            return r[m.BatchResultDict].fail(
                                f"Batch processing failed: {error_msg}",
                            )
                        if error_mode == "collect":
                            errors.append((processed - 1, str(error_msg)))
                    continue
                try:
                    normalized_result_raw = (
                        m.Validators.serializable_adapter().validate_python(result_raw)
                    )
                    if do_flatten and isinstance(normalized_result_raw, list):
                        result_raw_flat: list[t.Serializable] = normalized_result_raw
                        for inner_item in result_raw_flat:
                            if isinstance(
                                inner_item,
                                (dict, list, str, int, float, bool, datetime),
                            ):
                                results.append(
                                    FlextUtilitiesCollection._coerce_guard_value(
                                        inner_item,
                                    ),
                                )
                            else:
                                inner_item_obj_flat: object = inner_item
                                results.append(str(inner_item_obj_flat))
                        continue
                    if isinstance(normalized_result_raw, dict):
                        raw_dict: dict[str, t.Serializable] = normalized_result_raw
                        direct_result = FlextUtilitiesCollection._coerce_guard_value(
                            raw_dict,
                        )
                    elif isinstance(normalized_result_raw, list):
                        raw_list: list[t.Serializable] = normalized_result_raw
                        direct_result = FlextUtilitiesCollection._coerce_guard_value(
                            raw_list,
                        )
                    elif isinstance(normalized_result_raw, t.SCALAR_TYPES):
                        direct_result = FlextUtilitiesCollection._coerce_guard_value(
                            normalized_result_raw,
                        )
                    else:
                        raw_unknown: object = normalized_result_raw
                        direct_result = str(raw_unknown)
                except (TypeError, ValueError):
                    raw_error: object = result_raw
                    direct_result = str(raw_error)
                if do_flatten and isinstance(direct_result, list):
                    results.extend(direct_result)
                else:
                    results.append(direct_result)
            except (TypeError, ValueError, RuntimeError, AttributeError, KeyError) as e:
                if error_mode == "fail":
                    return r[m.BatchResultDict].fail(f"Batch processing failed: {e}")
                if error_mode == "collect":
                    errors.append((processed - 1, str(e)))
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
    def chunk(items: Sequence[T], size: int) -> list[list[T]]:
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
    def _coerce_dict_values[V](
        coerce_fn: Callable[[t.NormalizedValue], V],
    ) -> Callable[[t.NormalizedValue], Mapping[str, V]]:
        """Create validator that validates a dict and coerces each value via *coerce_fn*."""

        def validator(data: t.NormalizedValue) -> Mapping[str, V]:
            normalized_map_result = (
                FlextUtilitiesCollection._validate_dict_str_metadata(data)
            )
            if normalized_map_result.is_failure:
                msg = f"Expected mapping, got {data.__class__.__name__}"
                raise TypeError(msg)
            return {
                key: coerce_fn(val) for key, val in normalized_map_result.value.items()
            }

        return validator

    @staticmethod
    def coerce_dict_to_bool() -> Callable[[t.NormalizedValue], Mapping[str, bool]]:
        """Create validator that coerces dict values to bool."""
        return FlextUtilitiesCollection._coerce_dict_values(
            FlextUtilitiesCollection._coerce_value_to_bool,
        )

    @staticmethod
    def coerce_dict_to_float() -> Callable[[t.NormalizedValue], Mapping[str, float]]:
        """Create validator that coerces dict values to float."""
        return FlextUtilitiesCollection._coerce_dict_values(
            FlextUtilitiesCollection._coerce_value_to_float,
        )

    @staticmethod
    def coerce_dict_to_int() -> Callable[[t.NormalizedValue], Mapping[str, int]]:
        """Create validator that coerces dict values to int."""
        return FlextUtilitiesCollection._coerce_dict_values(
            FlextUtilitiesCollection._coerce_value_to_int,
        )

    @staticmethod
    def coerce_dict_to_str() -> Callable[[t.NormalizedValue], Mapping[str, str]]:
        """Create validator that coerces dict values to str."""
        return FlextUtilitiesCollection._coerce_dict_values(
            FlextUtilitiesCollection._coerce_value_to_str,
        )

    @staticmethod
    def coerce_dict_to_enum[E: StrEnum](
        enum_type: type[E],
    ) -> Callable[[dict[str, t.NormalizedValue]], dict[str, E]]:
        """Create validator that coerces dict values to a StrEnum type."""

        def validator(data: dict[str, t.NormalizedValue]) -> dict[str, E]:
            result: dict[str, E] = {}
            for k, v in data.items():
                if isinstance(v, enum_type):
                    result[k] = v
                elif isinstance(v, str):
                    result[k] = enum_type(v)
                else:
                    msg = (
                        f"Expected str for enum conversion, got {v.__class__.__name__}"
                    )
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def coerce_dict_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.NormalizedValue], dict[str, E]]:
        """Create validator that coerces dict values to a StrEnum type.

        Raises:
            TypeError: If input is not a dict or value is not str
            ValueError: If string value is not a valid enum member

        """

        def validator(data: t.NormalizedValue) -> dict[str, E]:
            normalized_map_result = (
                FlextUtilitiesCollection._validate_dict_str_metadata(data)
            )
            if normalized_map_result.is_failure:
                msg = f"Expected dict, got {data.__class__.__name__}"
                raise TypeError(msg)
            normalized_map = normalized_map_result.value
            result: dict[str, E] = {}
            for k, v_raw in normalized_map.items():
                if isinstance(v_raw, enum_cls):
                    result[k] = v_raw
                elif isinstance(v_raw, str):
                    enum_result = r[E].ok(v_raw).map(enum_cls)
                    if enum_result.is_failure:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} value: '{v_raw}'"
                        raise ValueError(msg) from None
                    result[k] = enum_result.value
                else:
                    msg = f"Expected str for enum conversion, got {v_raw.__class__.__name__}"
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def _coerce_list_values[V](
        coerce_fn: Callable[[t.NormalizedValue], V],
    ) -> Callable[[Sequence[t.NormalizedValue]], list[V]]:
        """Create validator that coerces each sequence element via *coerce_fn*."""

        def validator(data: Sequence[t.NormalizedValue]) -> list[V]:
            return [coerce_fn(v) for v in data]

        return validator

    @staticmethod
    def coerce_list_to_bool() -> Callable[[Sequence[t.NormalizedValue]], list[bool]]:
        """Create validator that coerces sequence values to bool."""
        return FlextUtilitiesCollection._coerce_list_values(
            FlextUtilitiesCollection._coerce_value_to_bool,
        )

    @staticmethod
    def coerce_list_to_enum[E: StrEnum](
        enum_type: type[E],
    ) -> Callable[[Sequence[t.NormalizedValue]], list[E]]:
        """Create validator that coerces sequence values to a StrEnum type."""

        def validator(data: Sequence[t.NormalizedValue]) -> list[E]:
            result: list[E] = []
            for v in data:
                if isinstance(v, enum_type):
                    result.append(v)
                elif isinstance(v, str):
                    result.append(enum_type(v))
                else:
                    msg = (
                        f"Expected str for enum conversion, got {v.__class__.__name__}"
                    )
                    raise TypeError(msg)
            return result

        return validator

    @staticmethod
    def coerce_list_to_float() -> Callable[[Sequence[t.NormalizedValue]], list[float]]:
        """Create validator that coerces sequence values to float."""
        return FlextUtilitiesCollection._coerce_list_values(
            FlextUtilitiesCollection._coerce_value_to_float,
        )

    @staticmethod
    def coerce_list_to_int() -> Callable[[Sequence[t.NormalizedValue]], list[int]]:
        """Create validator that coerces sequence values to int."""
        return FlextUtilitiesCollection._coerce_list_values(
            FlextUtilitiesCollection._coerce_value_to_int,
        )

    @staticmethod
    def coerce_list_to_str() -> Callable[[Sequence[t.NormalizedValue]], list[str]]:
        """Create validator that coerces sequence values to str."""
        return FlextUtilitiesCollection._coerce_list_values(
            FlextUtilitiesCollection._coerce_value_to_str,
        )

    @staticmethod
    def coerce_list_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.NormalizedValue], list[E]]:
        """Create validator that coerces list values to a StrEnum type.

        Raises:
            TypeError: If input is not a sequence or item is not str
            ValueError: If string value is not a valid enum member

        """

        def validator(data: t.NormalizedValue) -> list[E]:
            if isinstance(data, str):
                msg = f"Expected sequence, got {data.__class__.__name__}"
                raise TypeError(msg)
            normalized_items_result = FlextUtilitiesCollection._validate_list_container(
                data,
            )
            if normalized_items_result.is_failure:
                msg = f"Expected sequence, got {data.__class__.__name__}"
                raise TypeError(msg)
            normalized_items = normalized_items_result.value
            result: list[E] = []
            for v_raw in normalized_items:
                if isinstance(v_raw, enum_cls):
                    result.append(v_raw)
                elif isinstance(v_raw, str):
                    enum_result = r[E].ok(v_raw).map(enum_cls)
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
    ) -> dict[str, Callable[[], t.NormalizedValue]]:
        """Extract mapping of callables for resources/factories.

        Helper function to properly type narrow callable mappings for pyright.
        Filters to only callable values and converts to proper signature.

        Args:
            mapping: Mapping containing callable values

        Returns:
            Dict mapping string keys to callable functions

        """
        result: dict[str, Callable[[], t.NormalizedValue]] = {}
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
    ) -> list[tuple[str, t.NormalizedValue]]:
        """Extract mapping items as typed list for iteration.

        Helper function to properly type narrow Mapping.items() for pyright.
        Converts keys to strings and values to object.

        Args:
            mapping: Mapping to extract items from

        Returns:
            List of (key, value) tuples with proper typing

        """
        result: list[tuple[str, t.NormalizedValue]] = []
        items_iter = mapping.items()
        for item_tuple in items_iter:
            key_obj = item_tuple[0]
            value_raw = item_tuple[1]
            key_str: str = str(key_obj)
            result.append((key_str, value_raw))
        return result

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
        return list(chain.from_iterable(items))

    @staticmethod
    def group_by(items: Sequence[T], key_func: Callable[[T], U]) -> dict[U, list[T]]:
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
        result: defaultdict[U, list[T]] = defaultdict(list)
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
    def map(items: list[T], mapper: Callable[[T], U]) -> list[U]: ...

    @overload
    @staticmethod
    def map(items: tuple[T, ...], mapper: Callable[[T], U]) -> tuple[U, ...]: ...

    @overload
    @staticmethod
    def map(items: dict[str, T], mapper: Callable[[T], U]) -> dict[str, U]: ...

    @overload
    @staticmethod
    def map(items: set[T], mapper: Callable[[T], U]) -> set[U]: ...

    @overload
    @staticmethod
    def map(items: frozenset[T], mapper: Callable[[T], U]) -> frozenset[U]: ...

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

    @staticmethod
    def merge(
        base: dict[str, t.NormalizedValue],
        other: dict[str, t.NormalizedValue],
        *,
        strategy: str = "deep",
    ) -> r[dict[str, t.NormalizedValue]]:
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
        if strategy in {"replace", "override"}:
            result: dict[str, t.NormalizedValue] = dict(base)
            result.update(other)
            return r[dict[str, t.NormalizedValue]].ok(result)
        if strategy == "filter_none":
            result = dict(base)
            for key, value in other.items():
                if value is not None:
                    result[key] = value
            return r[dict[str, t.NormalizedValue]].ok(result)
        if strategy in {"filter_empty", "filter_both"}:
            result = dict(base)
            for key, value in other.items():
                if not FlextUtilitiesCollection._is_empty_value(value):
                    result[key] = value
            return r[dict[str, t.NormalizedValue]].ok(result)
        if strategy == "append":
            result = dict(base)
            for key, value in other.items():
                current_val = result.get(key)
                if (
                    current_val is not None
                    and isinstance(current_val, list)
                    and isinstance(value, list)
                ):
                    result[key] = current_val + value
                else:
                    result[key] = value
            return r[dict[str, t.NormalizedValue]].ok(result)
        if strategy == "deep":
            result = base.copy()
            for key, value in other.items():
                merge_result = FlextUtilitiesCollection._merge_deep_single_key(
                    result,
                    key,
                    value,
                )
                if merge_result.is_failure:
                    return r[dict[str, t.NormalizedValue]].fail(
                        merge_result.error or "Unknown error",
                    )
            return r[dict[str, t.NormalizedValue]].ok(result)
        return r[dict[str, t.NormalizedValue]].fail(
            f"Unknown merge strategy: {strategy}",
        )

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
        mapping: dict[str, str | E],
    ) -> r[dict[str, E]]:
        """Parse dict values from strings to StrEnum.

        Args:
            enum_cls: StrEnum class to parse values to
            mapping: Dict with string or enum values

        Returns:
            r with parsed dict

        Example:
            result = u.parse_mapping(Status, {"key": "active"})
            # result.value == {"key": Status.ACTIVE}

        """
        result: dict[str, E] = {}
        errors: list[str] = []
        mapping_items_result = (
            r[list[tuple[str, str | E]]].ok([]).map(lambda _: list(mapping.items()))
        )
        if mapping_items_result.is_failure:
            return r[dict[str, E]].fail(
                f"Parse mapping failed: {mapping_items_result.error}",
            )
        for key, value_raw in mapping_items_result.value:
            enum_result = r[E].ok(value_raw).map(enum_cls)
            if enum_result.is_failure:
                errors.append(f"'{key}': {enum_result.error}")
                continue
            result[key] = enum_result.value
        if errors:
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return r[dict[str, E]].fail(
                f"Invalid {enum_name} values: {', '.join(errors)}",
            )
        return r[dict[str, E]].ok(result)

    @staticmethod
    def parse_sequence(
        enum_cls: type[StrEnum],
        values: Sequence[str | StrEnum],
    ) -> r[tuple[StrEnum, ...]]:
        """Parse sequence of strings to tuple of StrEnum."""
        parsed: list[StrEnum] = []
        errors: list[str] = []
        enumerate_result = (
            r[list[tuple[int, str | StrEnum]]]
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
            enum_result = r[StrEnum].ok(val).map(enum_cls)
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
            r with list of processed results or error

        """
        _ = filter_keys
        _ = exclude_keys
        results: list[U] = []
        for item in items:
            item_typed: T = item
            if predicate is not None and (not predicate(item_typed)):
                continue
            process_result = r[T].ok(item_typed).map(processor)
            if process_result.is_failure:
                if on_error == "skip":
                    continue
                return r[list[U]].fail(f"Processing failed for item: {item}")
            results.append(process_result.value)
        return r[list[U]].ok(results)

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
        if key_func is None:
            return list(dict.fromkeys(items))
        seen: set[Hashable] = set()
        result: list[T] = []
        for item in items:
            key = key_func(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


__all__ = ["FlextUtilitiesCollection"]
