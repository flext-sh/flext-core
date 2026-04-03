"""Utilities module - FlextUtilitiesMapper.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from itertools import starmap
from typing import ClassVar

from pydantic import BaseModel

from flext_core import (
    FlextRuntime,
    FlextUtilitiesCache,
    FlextUtilitiesCollection,
    FlextUtilitiesGuards,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    p,
    r,
    t,
)


class FlextUtilitiesMapper:
    """Data structure mapping and transformation utilities.

    Provides generic methods for mapping between data structures, building
    objects from flags/mappings, and transforming dict/list structures.
    """

    _BUILD_OPS: ClassVar[
        Sequence[
            tuple[
                str,
                str,
            ]
        ]
    ] = [
        ("ensure", "_op_ensure"),
        ("filter", "_op_filter"),
        ("map", "_op_map"),
        ("normalize", "_op_normalize"),
        ("convert", "_op_convert"),
        ("transform", "_op_transform"),
        ("process", "_op_process"),
        ("group", "_op_group"),
        ("sort", "_op_sort"),
        ("unique", "_op_unique"),
        ("slice", "_op_slice"),
        ("chunk", "_op_chunk"),
    ]

    @staticmethod
    def _apply_exclude_keys[V](
        result: Mapping[str, V],
        *,
        exclude_keys: set[str] | None,
    ) -> MutableMapping[str, V] | Mapping[str, V]:
        """Apply exclude keys step."""
        if exclude_keys:
            filtered_result: MutableMapping[str, V] = dict(result)
            for key in exclude_keys:
                _ = filtered_result.pop(key, None)
            return filtered_result
        return result

    @staticmethod
    def _apply_filter_keys[V](
        result: Mapping[str, V],
        *,
        filter_keys: set[str] | None,
    ) -> MutableMapping[str, V] | Mapping[str, V]:
        """Apply filter keys step."""
        if filter_keys:
            filtered_dict: MutableMapping[str, V] = {}
            for key in filter_keys:
                if key in result:
                    filtered_dict[key] = result[key]
            return filtered_dict
        return result

    @staticmethod
    def _apply_map_keys[V](
        result: Mapping[str, V],
        *,
        map_keys: t.StrMapping | None,
    ) -> MutableMapping[str, V] | Mapping[str, V]:
        """Apply map keys step."""
        if map_keys:
            mapped = FlextUtilitiesMapper.map_dict_keys(
                result,
                map_keys,
            )
            if mapped.is_success:
                mapped_value = mapped.value
                return {str(key): value for key, value in mapped_value.items()}
        return result

    @staticmethod
    def _apply_normalize[V](
        result: Mapping[str, V],
        *,
        normalize: bool,
    ) -> MutableMapping[str, V] | Mapping[str, V]:
        """Apply normalize step."""
        if normalize:
            normalized = FlextUtilitiesCache.normalize_component(
                dict(result),
            )
            if FlextUtilitiesGuardsTypeCore.is_mapping(normalized):
                normalized_result: MutableMapping[str, V] = {}
                for key, value in normalized.items():
                    # Cast value back to generic type.
                    # If normalization returned different types this assumes V is loose enough
                    normalized_result[str(key)] = value
                return normalized_result
        return result

    @staticmethod
    def _apply_strip_empty[V](
        result: Mapping[str, V],
        *,
        strip_empty: bool,
    ) -> MutableMapping[str, V] | Mapping[str, V]:
        """Apply strip empty step."""
        if strip_empty:
            return FlextUtilitiesCollection.filter(
                result, lambda _k, v: v not in ("", [], {}, None)
            )
        return result

    @staticmethod
    def _apply_strip_none[V](
        result: Mapping[str, V],
        *,
        strip_none: bool,
    ) -> MutableMapping[str, V] | Mapping[str, V]:
        """Apply strip none step."""
        if strip_none:
            return FlextUtilitiesCollection.filter(result, lambda _k, v: v is not None)
        return result

    @staticmethod
    def _apply_transform_steps[V](
        result: Mapping[str, V],
        *,
        normalize: bool,
        map_keys: t.StrMapping | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
    ) -> MutableMapping[str, V] | Mapping[str, V]:
        """Apply transform steps to result dict."""
        result = FlextUtilitiesMapper._apply_normalize(result, normalize=normalize)
        result = FlextUtilitiesMapper._apply_map_keys(result, map_keys=map_keys)
        result = FlextUtilitiesMapper._apply_filter_keys(
            result,
            filter_keys=filter_keys,
        )
        result = FlextUtilitiesMapper._apply_exclude_keys(
            result,
            exclude_keys=exclude_keys,
        )
        result = FlextUtilitiesMapper._apply_strip_none(result, strip_none=strip_none)
        return FlextUtilitiesMapper._apply_strip_empty(result, strip_empty=strip_empty)

    @staticmethod
    def _narrow_list_items[T](items: Sequence[T]) -> MutableSequence[T] | Sequence[T]:
        """Narrow each item in a sequence generically."""
        return list(items)

    @staticmethod
    def _resolve_field_key_func(
        field_name: str,
    ) -> Callable[[t.RecursiveContainer], str]:
        """Build a key function extracting a named field from Mapping or BaseModel."""

        def _key_func(item: t.RecursiveContainer) -> str:
            if FlextUtilitiesGuardsTypeCore.is_mapping(item):
                return str(item.get(field_name, ""))
            if FlextUtilitiesGuardsTypeModel.is_pydantic_model(item) and hasattr(
                item, field_name
            ):
                return str(getattr(item, field_name))
            return ""

        return _key_func

    @staticmethod
    def _apply_callable_over_collection[T](
        current: Sequence[T] | Mapping[str, T] | T,
        func: Callable[[T], T],
    ) -> Sequence[T] | Mapping[str, T] | T:
        """Apply callable to each element of a collection or to a scalar."""
        if isinstance(current, (list, tuple)):
            seq_current: Sequence[T] = current
            return [func(x) for x in seq_current]
        if FlextUtilitiesGuardsTypeCore.is_mapping(current):
            current_dict: Mapping[str, T] = current
            return {k: func(v) for k, v in current_dict.items()}
        return func(current)

    @staticmethod
    def _op_chunk(
        current: t.RecursiveContainer,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.RecursiveContainer,
        _on_error: str,
    ) -> t.RecursiveContainer:
        """Helper: Apply chunk operation to split into sublists."""
        if not isinstance(current, (list, tuple)):
            return current
        chunk_size = ops.get("chunk")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            return current
        current_list: t.ContainerList = FlextUtilitiesMapper._narrow_list_items(current)
        chunked: t.MutableContainerList = []
        for i in range(0, len(current_list), chunk_size):
            chunk = current_list[i : i + chunk_size]
            chunked.append(chunk)
        return chunked

    @staticmethod
    def _op_convert[T](
        current: T,
        ops: Mapping[str, t.MapperInput],
        _default_val: T,
        _on_error: str,
    ) -> T | list[T] | tuple[T, ...]:
        """Helper: Apply convert operation."""
        convert_func_result = FlextUtilitiesMapper._get_callable_from_dict(
            ops,
            "convert",
        )
        if convert_func_result.is_failure:
            return current
        convert_callable_raw = convert_func_result.value
        convert_default_raw = ops.get("convert_default")
        fallback: t.MapperInput | None = (
            convert_default_raw if not callable(convert_default_raw) else None
        )

        converter_name = (
            convert_callable_raw.__name__
            if hasattr(convert_callable_raw, "__name__")
            else ""
        )
        if fallback is None:
            converter_defaults: Mapping[str, t.MapperInput] = {
                "int": 0,
                "float": 0.0,
                "str": "",
                "bool": False,
                "list": [],
                "dict": {},
                "tuple": (),
                "set": [],
            }
            fallback = converter_defaults.get(converter_name, current)

        def _convert[V](value: V) -> t.MapperInput | T:
            try:
                raw = convert_callable_raw(value)
                return raw if raw is not None else fallback
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                return fallback

        match current:
            case list():
                return [_convert(item) for item in current]
            case tuple():
                return tuple(_convert(item) for item in current)
            case _:
                return _convert(current)

    @staticmethod
    def _ensure_as_list(
        current: t.RecursiveContainer,
        default_val: t.RecursiveContainer,
        *,
        stringify: bool = False,
    ) -> t.RecursiveContainer:
        """Ensure value is a list, optionally stringifying items."""
        if isinstance(current, list):
            items: t.ContainerList = current
            return (
                [str(FlextRuntime.normalize_to_container(x)) for x in items]
                if stringify
                else [FlextRuntime.normalize_to_container(x) for x in items]
            )
        if current is None:
            return default_val
        item = FlextRuntime.normalize_to_container(current)
        if stringify:
            return [str(item)]
        res: t.ContainerList = [item]
        return res

    @staticmethod
    def _op_ensure[T](
        current: T,
        ops: Mapping[str, t.MapperInput],
        _default_val: T,
        _on_error: str,
    ) -> t.ValueOrModel:
        """Helper: Apply ensure operation."""
        ensure_type = FlextUtilitiesMapper._get_str_from_dict(ops, "ensure", "")
        ensure_default_raw = ops.get("ensure_default")
        ensure_default_val: t.RecursiveContainer = (
            FlextRuntime.normalize_to_container(ensure_default_raw)
            if ensure_default_raw is not None and not callable(ensure_default_raw)
            else None
        )
        default_map: t.ContainerMapping = {
            "str_list": list[t.RecursiveContainer](),
            "dict": dict[str, t.RecursiveContainer](),
            "list": list[t.RecursiveContainer](),
            "str": "",
        }
        default_val: t.RecursiveContainer = (
            ensure_default_val
            if ensure_default_val is not None
            else default_map.get(ensure_type, "")
        )
        match ensure_type:
            case "str":
                return str(current) if current is not None else default_val
            case "list":
                return FlextUtilitiesMapper._ensure_as_list(current, default_val)
            case "str_list":
                return FlextUtilitiesMapper._ensure_as_list(
                    current, default_val, stringify=True
                )
            case "dict":
                if FlextUtilitiesGuardsTypeCore.is_mapping(current):
                    return FlextUtilitiesMapper._narrow_to_configuration_dict(current)
                return default_val
            case _:
                return current

    @staticmethod
    def _op_filter[T](
        current: T,
        ops: Mapping[str, t.MapperInput],
        default_val: T,
        _on_error: str,
    ) -> T | list[T] | Mapping[str, T]:
        """Helper: Apply filter operation."""
        filter_pred_result = FlextUtilitiesMapper._get_callable_from_dict(ops, "filter")
        if filter_pred_result.is_failure:
            return current
        filter_pred_callable = filter_pred_result.value

        def filter_pred(value: t.RecursiveContainer) -> bool:
            return bool(filter_pred_callable(value))

        if isinstance(current, (list, tuple)):
            seq_current: t.ContainerList = current
            return [
                FlextRuntime.normalize_to_container(x)
                for x in seq_current
                if filter_pred(FlextRuntime.normalize_to_container(x))
            ]
        if FlextUtilitiesGuardsTypeCore.is_mapping(current):
            current_dict: t.ContainerMapping = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(current)
            )
            return FlextUtilitiesCollection.filter(
                current_dict, lambda _k, v: bool(filter_pred(v))
            )
        return default_val if not bool(filter_pred(current)) else current

    @staticmethod
    def _group_by_field[T](
        field_name: str,
        current_items: Sequence[T],
        current_list: Sequence[T],
    ) -> MutableMapping[str, MutableSequence[T]]:
        """Group mapping items by a named field key."""
        grouped: MutableMapping[str, MutableSequence[T]] = {}
        for orig_item, item in zip(current_items, current_list, strict=False):
            match orig_item:
                case Mapping():
                    key_raw = (
                        item.get(field_name)
                        if FlextUtilitiesGuardsTypeCore.is_mapping(item)
                        else None
                    )
                case _:
                    continue
            key = "" if key_raw is None else str(key_raw)
            grouped.setdefault(key, []).append(item)
        return grouped

    @staticmethod
    def _group_by_callable[T, V](
        group_callable: Callable[[T], V],
        current_list: Sequence[T],
    ) -> MutableMapping[str, MutableSequence[T]]:
        """Group items using a callable key function."""
        grouped: MutableMapping[str, MutableSequence[T]] = {}
        for item in current_list:
            try:
                group_key = group_callable(item)
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                continue
            if group_key is None:
                continue
            grouped.setdefault(str(group_key), []).append(item)
        return grouped

    @staticmethod
    def _op_group[T](
        current: list[T] | tuple[T, ...] | T,
        ops: Mapping[str, t.MapperInput],
        _default_val: T,
        _on_error: str,
    ) -> Mapping[str, Sequence[T]] | list[T] | tuple[T, ...] | T:
        """Helper: Apply group operation."""
        if not isinstance(current, (list, tuple)):
            return current
        group_spec_raw: t.MapperInput = ops["group"]
        current_items: t.ContainerList = current
        current_list: t.ContainerList = FlextUtilitiesMapper._narrow_list_items(
            current_items,
        )
        match group_spec_raw:
            case str() as field_name:
                return FlextUtilitiesMapper._group_by_field(
                    field_name,
                    current_items,
                    current_list,
                )
            case _ if callable(group_spec_raw):
                callable_result = FlextUtilitiesMapper._get_callable_from_dict(
                    ops, "group"
                )
                if callable_result.is_failure:
                    return current
                return FlextUtilitiesMapper._group_by_callable(
                    callable_result.value,
                    current_list,
                )
            case _:
                return current

    @staticmethod
    def _op_map[T](
        current: T,
        ops: Mapping[str, t.MapperInput],
        _default_val: T,
        _on_error: str,
    ) -> t.ValueOrModel:
        """Helper: Apply map operation."""
        map_func_result = FlextUtilitiesMapper._get_callable_from_dict(ops, "map")
        if map_func_result.is_failure:
            return current
        map_callable: t.MapperCallable = map_func_result.value
        return FlextUtilitiesMapper._apply_callable_over_collection(
            current,
            map_callable,
        )

    @staticmethod
    def _op_normalize[T](
        current: T,
        ops: Mapping[str, t.MapperInput],
        _default_val: T,
        _on_error: str,
    ) -> t.ValueOrModel:
        """Helper: Apply normalize operation."""
        normalize_case = FlextUtilitiesMapper._get_str_from_dict(ops, "normalize", "")
        if isinstance(current, str):
            return current.lower() if normalize_case == "lower" else current.upper()
        if isinstance(current, (list, tuple)):
            seq_current: t.ContainerList = current
            result: t.MutableContainerList = []
            for x in seq_current:
                x_general = FlextRuntime.normalize_to_container(x)
                if isinstance(x_general, str):
                    result.append(
                        x_general.lower()
                        if normalize_case == "lower"
                        else x_general.upper(),
                    )
                else:
                    result.append(x_general)
            return result
        return current

    @staticmethod
    def _op_process[T](
        current: T,
        ops: Mapping[str, t.MapperInput],
        default_val: T,
        on_error: str,
    ) -> t.ValueOrModel:
        """Helper: Apply process operation."""
        process_func_result = FlextUtilitiesMapper._get_callable_from_dict(
            ops,
            "process",
        )
        if process_func_result.is_failure:
            return current
        process_callable: t.MapperCallable = process_func_result.value

        def _process_current() -> t.RecursiveContainer:
            return FlextUtilitiesMapper._apply_callable_over_collection(
                current,
                process_callable,
            )

        process_result: r[t.RecursiveContainer] = r[
            t.RecursiveContainer
        ].create_from_callable(_process_current)
        if process_result.is_failure:
            return default_val if on_error == "stop" else current
        process_val: t.RecursiveContainer = process_result.value
        return process_val if process_val is not None else current

    @staticmethod
    def _op_slice[T](
        current: list[T] | tuple[T, ...] | T,
        ops: Mapping[str, t.MapperInput],
        _default_val: T,
        _on_error: str,
    ) -> list[T] | tuple[T, ...] | T:
        """Helper: Apply slice operation."""
        if not isinstance(current, (list, tuple)):
            return current
        current_items: t.ContainerList = current
        slice_spec_raw = ops["slice"]
        slice_spec: t.RecursiveContainer = (
            slice_spec_raw if not callable(slice_spec_raw) else None
        )
        min_slice_length = 2
        if (
            isinstance(slice_spec, (list, tuple))
            and len(slice_spec) >= min_slice_length
        ):
            start_raw: t.RecursiveContainer = slice_spec[0]
            end_raw: t.RecursiveContainer = slice_spec[1]
            start: int | None = start_raw if isinstance(start_raw, int) else None
            end: int | None = end_raw if isinstance(end_raw, int) else None
            if isinstance(current, list):
                sliced_list: t.ContainerList = [
                    FlextRuntime.normalize_to_container(item)
                    for item in current_items[start:end]
                ]
                return sliced_list
            sliced_tuple: tuple[t.RecursiveContainer, ...] = tuple(
                FlextRuntime.normalize_to_container(item)
                for item in current_items[start:end]
            )
            return sliced_tuple
        return current

    @staticmethod
    def _as_original_type(
        sorted_items: t.ContainerList,
        original: t.ContainerList | tuple[t.RecursiveContainer, ...],
    ) -> t.ContainerList | tuple[t.RecursiveContainer, ...]:
        """Preserve the original collection type (list vs tuple) after sorting."""
        return list(sorted_items) if isinstance(original, list) else tuple(sorted_items)

    @staticmethod
    def _op_sort[T](
        current: list[T] | tuple[T, ...] | T,
        ops: Mapping[str, t.MapperInput],
        _default_val: T,
        _on_error: str,
    ) -> list[T] | tuple[T, ...] | T:
        """Helper: Apply sort operation."""
        if not isinstance(current, (list, tuple)):
            return current
        sort_spec_raw: t.MapperInput = ops["sort"]
        current_list: t.ContainerList = FlextUtilitiesMapper._narrow_list_items(current)
        match sort_spec_raw:
            case str() as field_name:
                key_func = FlextUtilitiesMapper._resolve_field_key_func(field_name)
                return FlextUtilitiesMapper._as_original_type(
                    sorted(current_list, key=key_func),
                    current,
                )
            case bool() if sort_spec_raw is True:
                comparable: t.ContainerList = [
                    FlextRuntime.normalize_to_container(
                        item
                        if FlextUtilitiesGuardsTypeCore.is_primitive(item)
                        or item is None
                        else str(item)
                    )
                    for item in current_list
                ]
                return FlextUtilitiesMapper._as_original_type(
                    sorted(comparable, key=str),
                    current,
                )
            case _ if callable(sort_spec_raw):
                callable_result = FlextUtilitiesMapper._get_callable_from_dict(
                    ops, "sort"
                )
                if callable_result.is_failure:
                    return current
                sort_fn = callable_result.value

                def sort_key(item: t.RecursiveContainer) -> tuple[bool, float, str]:
                    key_value = FlextRuntime.normalize_to_container(sort_fn(item))
                    if isinstance(key_value, (int, float)):
                        return False, float(key_value), ""
                    return True, 0.0, str(key_value)

                sorted_result = r[t.ContainerList].create_from_callable(
                    lambda: sorted(current_list, key=sort_key),
                )
                if sorted_result.is_failure:
                    return current
                return FlextUtilitiesMapper._as_original_type(
                    sorted_result.value, current
                )
            case _:
                return current

    @staticmethod
    def _op_transform[T](
        current: T,
        ops: Mapping[str, t.MapperInput],
        default_val: T,
        on_error: str,
    ) -> t.ValueOrModel:
        """Helper: Apply transform operation."""
        if not FlextUtilitiesGuards.is_type(
            current,
            "mapping",
        ):
            return current
        transform_opts_val = ops["transform"]
        transform_opts_raw: t.RecursiveContainer = (
            transform_opts_val if not callable(transform_opts_val) else None
        )
        if not FlextUtilitiesGuardsTypeCore.is_mapping(transform_opts_raw):
            return current
        transform_opts = FlextUtilitiesMapper._narrow_to_configuration_dict(
            FlextRuntime.normalize_to_container(transform_opts_raw),
        )
        (
            normalize_bool,
            strip_none_bool,
            strip_empty_bool,
            map_keys_dict,
            filter_keys_set,
            exclude_keys_set,
        ) = FlextUtilitiesMapper._extract_transform_options(transform_opts)
        current_dict: t.ContainerMapping = (
            FlextUtilitiesMapper._narrow_to_configuration_dict(current)
        )
        transform_result = r[t.RecursiveContainer].create_from_callable(
            lambda: FlextUtilitiesMapper._apply_transform_steps(
                dict(current_dict),
                normalize=normalize_bool,
                map_keys=map_keys_dict,
                filter_keys=filter_keys_set,
                exclude_keys=exclude_keys_set,
                strip_none=strip_none_bool,
                strip_empty=strip_empty_bool,
            ),
        )
        if transform_result.is_failure:
            if on_error == "stop":
                return default_val
            return current
        return transform_result.value

    @staticmethod
    def _op_unique[T](
        current: list[T] | tuple[T, ...] | T,
        ops: Mapping[str, t.MapperInput],
        _default_val: T,
        _on_error: str,
    ) -> list[T] | tuple[T, ...] | T:
        """Helper: Apply unique operation to remove duplicates."""
        if not ops.get("unique"):
            return current
        if not isinstance(current, (list, tuple)):
            return current
        current_list_unique: t.ContainerList = FlextUtilitiesMapper._narrow_list_items(
            current,
        )
        seen: set[t.Scalar | None] = set()
        unique_list: t.MutableContainerList = []
        for item in current_list_unique:
            item_hashable: t.Scalar | None = (
                item
                if FlextUtilitiesGuardsTypeCore.is_primitive(item) or item is None
                else str(item)
            )
            if item_hashable not in seen:
                seen.add(item_hashable)
                unique_list.append(item)
        if isinstance(current, list):
            return unique_list
        return tuple(unique_list)

    @staticmethod
    def _extract_field_value(
        item: t.ValueOrModel | t.ContainerMapping | Mapping[str, t.ValueOrModel],
        field_name: str,
    ) -> t.RecursiveContainer:
        """Extract field value from dict or model for pyrefly type inference."""
        if FlextUtilitiesGuardsTypeCore.is_mapping(item):
            dict_item: t.MutableContainerMapping = {}
            for key, value in item.items():
                coerced_value: t.RecursiveContainer = (
                    value if FlextUtilitiesGuards.is_container(value) else str(value)
                )
                dict_item[str(key)] = coerced_value
            return dict_item.get(field_name)
        if hasattr(item, field_name):
            attr_value = getattr(item, field_name)
            if FlextUtilitiesGuards.is_container(attr_value):
                return attr_value
            return str(attr_value)
        return None

    @staticmethod
    def _resolve_raw_value[T, V](
        raw: V,
        key_part: str,
    ) -> r[T]:
        """Wrap a raw value into a Result: fail on None, narrow containers, stringify rest."""
        if raw is None:
            return r[T].fail(f"found_none:{key_part}")
        if FlextUtilitiesGuards.is_container(raw):
            return r[T].ok(
                raw,
            )
        return r[T].ok(str(raw))

    @staticmethod
    def _extract_get_value[T, V](
        current: V,
        key_part: str,
    ) -> r[T]:
        """Get raw value from dict/object/model, returning found_none or not-found failures."""
        if isinstance(current, Mapping):
            mapping_obj: Mapping[str, t.ValueOrModel] = current  # type: ignore
            if key_part in mapping_obj:
                return FlextUtilitiesMapper._resolve_raw_value(
                    mapping_obj[key_part], key_part
                )
            return r[T].fail(f"Key '{key_part}' not found in Mapping")
        if hasattr(current, key_part):
            return FlextUtilitiesMapper._resolve_raw_value(
                getattr(current, key_part), key_part
            )
        if isinstance(
            current, BaseModel
        ) and FlextUtilitiesGuardsTypeModel.is_pydantic_model(current):
            model_dump_attr = current.model_dump
            if callable(model_dump_attr):
                model_dict = model_dump_attr()
                if key_part in model_dict:
                    val = model_dict[key_part]
                    if val is None:
                        return r[T].fail(f"found_none:{key_part}")
                    return r[T].ok(val)
        return r[T].fail(f"Key '{key_part}' not found")

    @staticmethod
    def _extract_handle_array_index[T, V](
        current: V,
        array_match: str,
    ) -> r[T]:
        """Handle array indexing with negative index support."""
        if not isinstance(current, (list, tuple)):
            return r[T].fail("Not a sequence")
        sequence: Sequence[t.ValueOrModel] = current  # type: ignore
        try:
            idx = int(array_match)
            if idx < 0:
                idx = len(sequence) + idx
            if 0 <= idx < len(sequence):
                item = sequence[idx]
                if item is None:
                    return r[T].fail("found_none:index")
                return r[T].ok(item)
            return r[T].fail(f"Index {int(array_match)} out of range")
        except (ValueError, IndexError):
            return r[T].fail(f"Invalid index {array_match}")

    @staticmethod
    def _extract_parse_array_index(part: str) -> tuple[str, str]:
        """Helper: Parse array index from path part (e.g., "items[0]")."""
        if "[" in part and part.endswith("]"):
            bracket_pos = part.index("[")
            array_match = part[bracket_pos + 1 : -1]
            key_part = part[:bracket_pos]
            return (key_part, array_match)
        return (part, "")

    @staticmethod
    def _extract_transform_options(
        transform_opts: t.ContainerMapping,
    ) -> tuple[
        bool,
        bool,
        bool,
        t.StrMapping | None,
        set[str] | None,
        set[str] | None,
    ]:
        """Extract transform options from dict."""
        normalize_val = transform_opts.get("normalize")
        normalize_bool = normalize_val if isinstance(normalize_val, bool) else False
        strip_none_val = transform_opts.get("strip_none")
        strip_none_bool = strip_none_val if isinstance(strip_none_val, bool) else False
        strip_empty_val = transform_opts.get("strip_empty")
        strip_empty_bool = (
            strip_empty_val if isinstance(strip_empty_val, bool) else False
        )
        map_keys_val = transform_opts.get("map_keys")
        map_keys_dict: t.StrMapping | None = None
        if FlextUtilitiesGuardsTypeCore.is_mapping(map_keys_val) and all(
            isinstance(v, str) for v in map_keys_val.values()
        ):
            map_keys_dict = {str(k): str(v) for k, v in map_keys_val.items()}
        filter_keys_val = transform_opts.get("filter_keys")
        filter_keys_set: set[str] | None = None
        if isinstance(filter_keys_val, set):
            filter_keys_set = set(map(str, filter_keys_val))
        exclude_keys_val = transform_opts.get("exclude_keys")
        exclude_keys_set: set[str] | None = None
        if isinstance(exclude_keys_val, set):
            exclude_keys_set = set(map(str, exclude_keys_val))
        return (
            normalize_bool,
            strip_none_bool,
            strip_empty_bool,
            map_keys_dict,
            filter_keys_set,
            exclude_keys_set,
        )

    @staticmethod
    def _get_callable_from_dict(
        ops: Mapping[str, t.MapperInput],
        key: str,
    ) -> r[t.MapperCallable]:
        value: t.MapperInput = ops.get(key)
        if callable(value):
            return r[t.MapperCallable].ok(value)
        return r[t.MapperCallable].fail(f"Operation '{key}' is not callable")

    @staticmethod
    def _get_raw(
        data: p.AccessibleData,
        key: str,
        *,
        default: t.RecursiveContainer = None,
    ) -> t.RecursiveContainer:
        """Internal helper for raw get without DSL conversion."""
        fallback: t.RecursiveContainer = default if default is not None else ""
        raw_value: t.RecursiveContainer = None
        match data:
            case dict() | Mapping():
                raw_value = FlextRuntime.normalize_to_container(data.get(key))
            case t.ConfigMap() | t.Dict():
                raw_value = FlextRuntime.normalize_to_container(data.root.get(key))
            case _ if hasattr(data, key):
                return FlextRuntime.normalize_to_container(getattr(data, key))
            case _:
                return fallback
        return (
            FlextRuntime.normalize_to_container(raw_value)
            if raw_value is not None
            else fallback
        )

    @staticmethod
    def _get_str_from_dict(
        ops: Mapping[str, t.MapperInput],
        key: str,
        default: str = "",
    ) -> str:
        """Safely extract str value from ConfigurationDict."""
        value = ops.get(key, default)
        if isinstance(value, str):
            return str(value)
        return str(value) if value is not None else default

    @staticmethod
    def _narrow_to_configuration_dict(
        value: t.RecursiveContainer | t.ContainerMapping,
    ) -> t.ContainerMapping:
        """Safely narrow a recursive container to ConfigurationDict."""
        if FlextUtilitiesGuards.is_configuration_dict(value):
            normalized_dict: t.MutableContainerMapping = {}
            for key, item in value.items():
                normalized_dict[str(key)] = FlextRuntime.normalize_to_container(item)
            return normalized_dict
        error_msg = f"Cannot narrow {value.__class__.__name__} to ConfigurationDict"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_sequence(
        value: t.RecursiveContainer | t.ContainerList,
    ) -> t.ContainerList:
        """Safely narrow a recursive container to t.ContainerList."""
        if isinstance(value, (list, tuple)):
            narrowed_items: t.MutableContainerList = []
            for item_raw in value:
                item = FlextRuntime.normalize_to_container(item_raw)
                narrowed_item = FlextRuntime.normalize_to_container(item)
                narrowed_items.append(narrowed_item)
            return narrowed_items
        error_msg = f"Cannot narrow {value.__class__.__name__} to Sequence"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_string_keyed_dict(
        value: t.RecursiveContainer | t.ContainerMapping,
    ) -> t.ContainerMapping:
        """Narrow to ConfigurationDict with string keys and container values."""
        if FlextUtilitiesGuardsTypeCore.is_mapping(value):
            result: t.MutableContainerMapping = {}
            key: str
            val: t.RecursiveContainer
            for key, val in value.items():
                str_key = str(key)
                if FlextUtilitiesGuards.is_container(val):
                    result[str_key] = val
                else:
                    result[str_key] = str(val)
            return result
        error_msg = f"Cannot narrow {value.__class__.__name__} to ConfigurationDict"
        raise TypeError(error_msg)

    @staticmethod
    def _get_numeric_field(
        item: BaseModel | Mapping[str, t.RecursiveContainer],
        field_name: str,
    ) -> t.Numeric | None:
        """Extract a numeric field value from a BaseModel or Mapping-like object."""
        if isinstance(item, BaseModel):
            val_raw = FlextUtilitiesMapper._extract_field_value(item, field_name)
            return val_raw if isinstance(val_raw, (int, float)) else None
        val = item.get(field_name)
        return val if isinstance(val, (int, float)) else None

    @staticmethod
    def agg[T](
        items: Sequence[T] | tuple[T, ...],
        field: str | Callable[[T], t.Numeric],
        *,
        fn: Callable[[Sequence[t.Numeric]], t.Numeric] | None = None,
    ) -> t.Numeric:
        """Aggregate numeric field values from objects using fn (default: sum)."""
        items_list: Sequence[T] = list(items)
        if callable(field):
            numeric_values: MutableSequence[t.Numeric] = [
                field(item) for item in items_list
            ]
        else:
            numeric_values = [
                val
                for item in items_list
                if isinstance(item, (BaseModel, Mapping))
                and (val := FlextUtilitiesMapper._get_numeric_field(item, field))
                is not None
            ]
        agg_fn: Callable[[Sequence[t.Numeric]], t.Numeric] = (
            fn if fn is not None else sum
        )
        return agg_fn(numeric_values) if numeric_values else 0

    @staticmethod
    def build(
        value: p.AccessibleData,
        *,
        ops: Mapping[str, t.MapperInput] | None = None,
        default: t.RecursiveContainer = None,
        on_error: str = "stop",
    ) -> t.RecursiveContainer:
        """Compose operations via DSL dict applied in order: ensure/filter/map/normalize/convert/transform/process/group/sort/unique/slice/chunk."""
        narrowed_value = FlextRuntime.normalize_to_container(value)
        if ops is None:
            return narrowed_value
        current: t.RecursiveContainer = narrowed_value
        default_val: t.RecursiveContainer = (
            default if default is not None else narrowed_value
        )
        for op_key, op_method_name in FlextUtilitiesMapper._BUILD_OPS:
            if op_key in ops:
                handler: Callable[
                    [
                        t.RecursiveContainer,
                        Mapping[str, t.MapperInput],
                        t.RecursiveContainer,
                        str,
                    ],
                    t.RecursiveContainer,
                ] = getattr(FlextUtilitiesMapper, op_method_name)
                current = handler(current, ops, default_val, on_error)
        return current

    @staticmethod
    def _deep_eq_values[T, V](
        val_a: T,
        val_b: V,
    ) -> bool:
        """Recursive deep equality for any two nested items."""
        if val_a is val_b:
            return True
        match (val_a, val_b):
            case (None, None):
                return True
            case (None, _) | (_, None):
                return False
            case (Mapping() as ma, Mapping() as mb):
                return (
                    hasattr(ma, "items")
                    and hasattr(mb, "items")
                    and FlextUtilitiesMapper.deep_eq(ma, mb)
                )  # type: ignore
            case (list() as la, list() as lb):
                if len(la) != len(lb):
                    return False
                return all(
                    starmap(
                        FlextUtilitiesMapper._deep_eq_values, zip(la, lb, strict=True)
                    )
                )
            case _:
                return val_a == val_b

    @staticmethod
    def deep_eq[T, V](a: Mapping[str, T], b: Mapping[str, V]) -> bool:
        """Recursive deep equality for nested dicts/lists/primitives."""
        if a is b:
            return True
        if len(a) != len(b):
            return False
        for key, val_a in a.items():
            if key not in b:
                return False
            val_b = b[key]
            if not FlextUtilitiesMapper._deep_eq_values(val_a, val_b):
                return False
        return True

    # ensure_str removed (use conversion.to_str)

    @staticmethod
    @staticmethod
    def _extract_fail_or_default[T](
        msg: str,
        *,
        default: T | None,
        required: bool,
    ) -> r[T]:
        """Return fail (required) or ok(default) / fail (no default) for extract paths."""
        if required:
            return r[T].fail(msg)
        if default is None:
            return r[T].fail(f"{msg} and default is None")
        return r[T].ok(default)

    @staticmethod
    def _extract_resolve_path_part[T](
        current: t.ValueOrModel | None,
        part: str,
        *,
        path_context: str,
        default: T | None,
        required: bool,
    ) -> tuple[t.ValueOrModel | None, r[T] | None]:
        """Resolve one path segment; returns (next_current, None) or (None, early_result)."""
        found_none_prefix = "found_none:"
        key_part, array_match = FlextUtilitiesMapper._extract_parse_array_index(part)

        get_result = FlextUtilitiesMapper._extract_get_value(current, key_part)
        if get_result.is_failure:
            error_str = get_result.error or ""
            if error_str.startswith(found_none_prefix):
                next_val: t.ValueOrModel = None
            else:
                return None, FlextUtilitiesMapper._extract_fail_or_default(
                    f"Key '{key_part}' not found at '{path_context}'",
                    default=default,
                    required=required,
                )
        else:
            next_val = get_result.value

        if array_match and next_val is not None:
            narrowed_for_index = FlextRuntime.normalize_to_container(next_val)
            index_result = FlextUtilitiesMapper._extract_handle_array_index(
                narrowed_for_index,
                array_match,
            )
            if index_result.is_failure:
                error_str = index_result.error or ""
                if error_str.startswith(found_none_prefix):
                    next_val = None
                else:
                    return None, FlextUtilitiesMapper._extract_fail_or_default(
                        f"Array error at '{key_part}': {index_result.error}",
                        default=default,
                        required=required,
                    )
            else:
                next_val = index_result.value

        return next_val, None

    @staticmethod
    @staticmethod
    def extract[T](
        data: p.AccessibleData,
        path: str,
        *,
        default: T | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> r[T]:
        """Extract nested value via dot-notation path with array index support (e.g. "user.addresses[0].city")."""
        try:
            parts = path.split(separator)
            current: t.ValueOrModel | None = None
            match data:
                case BaseModel():
                    current = data
                case Mapping():
                    current = FlextRuntime.normalize_to_container(data)
                case p.HasModelDump():
                    current = FlextRuntime.normalize_to_container(data.model_dump())
                case p.ValidatorSpec():
                    current = str(data)
                case _:
                    current = data

            for i, part in enumerate(parts):
                if current is None:
                    return FlextUtilitiesMapper._extract_fail_or_default(
                        f"Path '{separator.join(parts[:i])}' is None",
                        default=default,
                        required=required,
                    )
                current, early_return = FlextUtilitiesMapper._extract_resolve_path_part(
                    current,
                    part,
                    path_context=separator.join(parts[:i]),
                    default=default,
                    required=required,
                )
                if early_return is not None:
                    return early_return

            if current is None:
                return FlextUtilitiesMapper._extract_fail_or_default(
                    "Extracted value is None",
                    default=default,
                    required=required,
                )
            if FlextUtilitiesGuards.is_container(current):
                return r[T].ok(current)
            return r[T].ok(str(current))
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
            return r[T].fail(f"Extract failed: {e}")

    # filter_dict removed (use collection.filter)

    @staticmethod
    def map_get(
        data: p.AccessibleData,
        key: str,
        *,
        default: t.RecursiveContainer = None,
    ) -> t.RecursiveContainer:
        """Get value by key from dict/object, returning default if missing."""
        return FlextUtilitiesMapper._get_raw(data, key, default=default)

    @staticmethod
    def map_dict_keys[V](
        source: Mapping[str, V],
        key_mapping: t.StrMapping,
        *,
        keep_unmapped: bool = True,
    ) -> r[MutableMapping[str, V]]:
        """Rename dict keys using old_key->new_key mapping."""

        def _map_keys() -> MutableMapping[str, V]:
            result: MutableMapping[str, V] = {}
            for key, value in source.items():
                new_key = key_mapping.get(key)
                if new_key:
                    result[new_key] = value
                elif keep_unmapped:
                    result[key] = value
            return result

        mapped_result = r[MutableMapping[str, V]].create_from_callable(_map_keys)
        return mapped_result.fold(
            on_failure=lambda e: r[MutableMapping[str, V]].fail(
                f"Failed to map dict keys: {e}",
            ),
            on_success=lambda _: mapped_result,
        )

    # narrow_to_container suite removed (use FlextRuntime.normalize_to_container)

    @staticmethod
    def prop(
        key: str,
    ) -> Callable[[t.ConfigModelInput], t.RecursiveContainer]:
        """Return an accessor function that extracts the named property from an object."""

        def accessor(obj: t.ConfigModelInput) -> t.RecursiveContainer:
            """Access property from object."""
            result = FlextUtilitiesMapper.map_get(obj, key)
            return result if result is not None else ""

        return accessor

    @staticmethod
    def _take_by_key[T](
        data_or_items: Sequence[T] | Mapping[str, T] | T,
        key: str,
        *,
        as_type: type | None,
        default: T | None,
    ) -> T | None:
        """Extract a value by key from a Mapping or BaseModel."""
        fallback: T | None = default
        match data_or_items:
            case Mapping() if FlextUtilitiesGuards.is_configuration_mapping(
                data_or_items
            ):
                data: p.AccessibleData = data_or_items
            case BaseModel():
                data = data_or_items
            case _:
                return fallback
        value = FlextUtilitiesMapper.map_get(data, key, default=default)
        if value is None:
            return fallback
        if as_type is not None and not isinstance(value, as_type):
            return fallback
        return value

    @staticmethod
    def _take_n_items[T](
        data_or_items: Sequence[T] | Mapping[str, T] | T,
        n: int,
        *,
        default: T | None,
        from_start: bool,
    ) -> Sequence[T] | Mapping[str, T] | T | None:
        """Take N items from a Mapping or Sequence."""
        match data_or_items:
            case Mapping():
                keys = list(data_or_items.keys())
                selected = keys[:n] if from_start else keys[-n:]
                return {k: data_or_items[k] for k in selected}
            case list() | tuple():
                items_list: Sequence[T] = data_or_items
                return items_list[:n] if from_start else items_list[-n:]
            case _:
                return default

    @staticmethod
    def take[T](
        data_or_items: Sequence[T] | Mapping[str, T] | T,
        key_or_n: str | int,
        *,
        as_type: type | None = None,
        default: T | None = None,
        from_start: bool = True,
    ) -> Sequence[T] | Mapping[str, T] | T | None:
        """Extract by key (str) or take N items (int) from a dict/list/tuple."""
        match key_or_n:
            case str() as key:
                return FlextUtilitiesMapper._take_by_key(
                    data_or_items,
                    key,
                    as_type=as_type,
                    default=default,
                )
            case int() as n:
                return FlextUtilitiesMapper._take_n_items(
                    data_or_items,
                    n,
                    default=default,
                    from_start=from_start,
                )

    @staticmethod
    def transform[V](
        source: Mapping[str, V] | t.ConfigMap,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: t.StrMapping | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> r[MutableMapping[str, V]]:
        """Apply normalize/strip_none/strip_empty/map_keys/filter_keys/exclude_keys to a dict."""
        transform_result = r[MutableMapping[str, V]].create_from_callable(
            lambda: FlextUtilitiesMapper._apply_transform_steps(
                {str(k): v for k, v in source.root.items()}
                if isinstance(source, t.ConfigMap)
                else dict(source),
                normalize=normalize,
                map_keys=map_keys,
                filter_keys=filter_keys,
                exclude_keys=exclude_keys,
                strip_none=strip_none,
                strip_empty=strip_empty,
            ),
        )
        return transform_result.fold(
            on_failure=lambda e: r[MutableMapping[str, V]].fail(
                f"Transform failed: {e}"
            ),
            on_success=lambda _: transform_result,
        )


__all__ = ["FlextUtilitiesMapper"]
