"""Utilities module - FlextUtilitiesMapper.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import overload

from pydantic import BaseModel

from flext_core import (
    FlextRuntime,
    m,
    p,
    r,
)
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.typings import t


class FlextUtilitiesMapper:
    """Data structure mapping and transformation utilities.

    Provides generic methods for mapping between data structures, building
    objects from flags/mappings, and transforming dict/list structures.

    **Usage** (flat namespace via runtime alias u; no subdivision):
        >>> from flext_core import u
        >>> result = u.get(data, "key", default="")
        >>> new_dict = u.map_dict_keys({"old_key": "value", "foo": "baz"}, mapping)
        >>> perms = u.build_flags_dict(flags, mapping)
    Subprojects use their project u. Aliases follow MRO registration only.
    """

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def _apply_exclude_keys(
        result: t.ContainerMapping, *, exclude_keys: set[str] | None
    ) -> t.ContainerMapping:
        """Apply exclude keys step."""
        if exclude_keys:
            filtered_result: dict[str, t.NormalizedValue] = dict(result)
            for key in exclude_keys:
                _ = filtered_result.pop(key, None)
            return filtered_result
        return result

    @staticmethod
    def _apply_filter_keys(
        result: t.ContainerMapping, *, filter_keys: set[str] | None
    ) -> t.ContainerMapping:
        """Apply filter keys step."""
        if filter_keys:
            filtered_dict: dict[str, t.NormalizedValue] = {}
            for key in filter_keys:
                if key in result:
                    filtered_dict[key] = result[key]
            return filtered_dict
        return result

    @staticmethod
    def _apply_map_keys(
        result: t.ContainerMapping, *, map_keys: Mapping[str, str] | None
    ) -> t.ContainerMapping:
        """Apply map keys step."""
        if map_keys:
            mapped: r[t.ContainerMapping] = FlextUtilitiesMapper.map_dict_keys(
                result, map_keys
            )
            if mapped.is_success:
                mapped_value: t.ContainerMapping = mapped.value
                return {
                    str(key): FlextUtilitiesMapper.narrow_to_container(value)
                    for key, value in mapped_value.items()
                }
        return result

    @staticmethod
    def _apply_normalize(
        result: t.ContainerMapping, *, normalize: bool
    ) -> t.ContainerMapping:
        """Apply normalize step."""
        if normalize:
            normalized: t.NormalizedValue = FlextUtilitiesCache.normalize_component(
                result
            )
            if isinstance(normalized, Mapping):
                normalized_result: dict[str, t.NormalizedValue] = {}
                for key, value in normalized.items():
                    normalized_result[str(key)] = (
                        FlextUtilitiesMapper.narrow_to_container(value)
                    )
                return normalized_result
        return result

    @staticmethod
    def _apply_strip_empty(
        result: t.ContainerMapping, *, strip_empty: bool
    ) -> t.ContainerMapping:
        """Apply strip empty step."""
        if strip_empty:
            return FlextUtilitiesMapper.filter_dict(
                result, lambda _k, v: v not in ("", [], {}, None)
            )
        return result

    @staticmethod
    def _apply_strip_none(
        result: t.ContainerMapping, *, strip_none: bool
    ) -> t.ContainerMapping:
        """Apply strip none step."""
        if strip_none:
            return FlextUtilitiesMapper.filter_dict(result, lambda _k, v: v is not None)
        return result

    @staticmethod
    def _apply_transform_steps(
        result: t.ContainerMapping,
        *,
        normalize: bool,
        map_keys: Mapping[str, str] | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
    ) -> t.ContainerMapping:
        """Apply transform steps to result dict."""
        result = FlextUtilitiesMapper._apply_normalize(result, normalize=normalize)
        result = FlextUtilitiesMapper._apply_map_keys(result, map_keys=map_keys)
        result = FlextUtilitiesMapper._apply_filter_keys(
            result, filter_keys=filter_keys
        )
        result = FlextUtilitiesMapper._apply_exclude_keys(
            result, exclude_keys=exclude_keys
        )
        result = FlextUtilitiesMapper._apply_strip_none(result, strip_none=strip_none)
        return FlextUtilitiesMapper._apply_strip_empty(result, strip_empty=strip_empty)

    @staticmethod
    def _build_apply_chunk(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> t.NormalizedValue:
        """Helper: Apply chunk operation to split into sublists."""
        if "chunk" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        chunk_size = ops["chunk"]
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            return current
        current_items: Sequence[t.NormalizedValue] = current
        current_list: t.ContainerList = [
            FlextUtilitiesMapper.narrow_to_container(item) for item in current_items
        ]
        chunked: list[t.ContainerList] = []
        for i in range(0, len(current_list), chunk_size):
            chunk: t.ContainerList = current_list[i : i + chunk_size]
            chunked.append(chunk)
        return chunked

    @staticmethod
    def _build_apply_convert(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> t.NormalizedValue:
        """Helper: Apply convert operation."""
        if "convert" not in ops:
            return current
        convert_func_result = FlextUtilitiesMapper._get_callable_from_dict(
            ops, "convert"
        )
        if convert_func_result.is_failure:
            return current
        convert_callable_raw: t.MapperCallable = convert_func_result.value
        convert_default = ops.get("convert_default")
        fallback: t.NormalizedValue = (
            convert_default if not callable(convert_default) else current
        )

        def convert_callable(value: t.NormalizedValue) -> t.NormalizedValue:
            return FlextUtilitiesMapper.narrow_to_container(convert_callable_raw(value))

        converter_name = (
            convert_callable_raw.__name__
            if hasattr(convert_callable_raw, "__name__")
            else ""
        )
        if fallback is None:
            converter_defaults: t.ContainerMapping = {
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

        def _convert(value: t.NormalizedValue) -> t.NormalizedValue:
            converted_result: r[t.NormalizedValue] = r[
                t.NormalizedValue
            ].create_from_callable(
                lambda: FlextUtilitiesMapper.narrow_to_container(
                    convert_callable(value)
                )
            )
            if converted_result.is_success:
                result_val: t.NormalizedValue = converted_result.value
                return result_val if result_val is not None else fallback
            return fallback

        if isinstance(current, (list, tuple)):
            current_items: Sequence[t.NormalizedValue] = current
            converted = [
                _convert(FlextUtilitiesMapper.narrow_to_container(item))
                for item in current_items
            ]
            return converted if isinstance(current, list) else tuple(converted)
        return _convert(current)

    @staticmethod
    def _build_apply_ensure(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> t.NormalizedValue:
        """Helper: Apply ensure operation."""
        if "ensure" not in ops:
            return current
        ensure_type = FlextUtilitiesMapper._get_str_from_dict(ops, "ensure", "")
        ensure_default_raw = ops.get("ensure_default")
        ensure_default_val: t.NormalizedValue = (
            FlextUtilitiesMapper.narrow_to_container(ensure_default_raw)
            if ensure_default_raw is not None and not callable(ensure_default_raw)
            else None
        )
        default_map: t.ContainerMapping = {
            "str_list": [],
            "dict": {},
            "list": [],
            "str": "",
        }
        default_val: t.NormalizedValue = (
            ensure_default_val
            if ensure_default_val is not None
            else default_map.get(ensure_type, "")
        )
        match ensure_type:
            case "str":
                return str(current) if current is not None else default_val
            case "list":
                if isinstance(current, list):
                    list_current: t.ContainerList = current
                    return [
                        FlextUtilitiesMapper.narrow_to_container(item)
                        for item in list_current
                    ]
                return (
                    default_val
                    if current is None
                    else [FlextUtilitiesMapper.narrow_to_container(current)]
                )
            case "str_list":
                if isinstance(current, list):
                    list_current_str: t.ContainerList = current
                    return [
                        str(FlextUtilitiesMapper.narrow_to_container(x))
                        for x in list_current_str
                    ]
                return (
                    default_val
                    if current is None
                    else [str(FlextUtilitiesMapper.narrow_to_container(current))]
                )
            case "dict":
                if isinstance(current, Mapping):
                    return FlextUtilitiesMapper._narrow_to_configuration_dict(current)
                return default_val
            case _:
                return current

    @staticmethod
    def _build_apply_filter(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
        default: t.NormalizedValue,
    ) -> t.NormalizedValue:
        """Helper: Apply filter operation."""
        if "filter" not in ops:
            return current
        filter_pred_result = FlextUtilitiesMapper._get_callable_from_dict(ops, "filter")
        if filter_pred_result.is_failure:
            return current
        filter_pred_callable = filter_pred_result.value

        def filter_pred(value: t.NormalizedValue) -> bool:
            return bool(filter_pred_callable(value))

        if isinstance(current, (list, tuple)):
            seq_current: Sequence[t.NormalizedValue] = current
            return [
                FlextUtilitiesMapper.narrow_to_container(x)
                for x in seq_current
                if filter_pred(FlextUtilitiesMapper.narrow_to_container(x))
            ]
        if isinstance(current, Mapping):
            current_dict: t.ContainerMapping = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(current)
            )
            return FlextUtilitiesMapper.filter_dict(
                current_dict, lambda _k, v: bool(filter_pred(v))
            )
        return default if not bool(filter_pred(current)) else current

    @staticmethod
    def _build_apply_group(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> t.NormalizedValue:
        """Helper: Apply group operation."""
        if "group" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        group_spec_raw: t.NormalizedValue | t.MapperCallable = ops["group"]
        current_items: Sequence[t.NormalizedValue] = current
        current_list: t.ContainerList = [
            FlextUtilitiesMapper.narrow_to_container(item) for item in current_items
        ]
        if isinstance(group_spec_raw, str):
            group_spec = group_spec_raw
            grouped: dict[str, t.ContainerList] = {}
            for orig_item, item in zip(current_items, current_list, strict=False):
                if isinstance(orig_item, BaseModel):
                    if not hasattr(orig_item, group_spec):
                        continue
                    key_raw = getattr(orig_item, group_spec)
                elif isinstance(item, Mapping):
                    key_raw = item.get(group_spec)
                else:
                    continue
                key = "" if key_raw is None else str(key_raw)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(item)
            return grouped
        if callable(group_spec_raw):
            group_callable_result = FlextUtilitiesMapper._get_callable_from_dict(
                ops, "group"
            )
            if group_callable_result.is_failure:
                return current
            group_callable = group_callable_result.value
            grouped_callable: dict[str, t.ContainerList] = {}
            for item in current_list:
                key_result = r[t.NormalizedValue].create_from_callable(
                    partial(group_callable, item)
                )
                if key_result.is_failure:
                    continue
                key = str(key_result.value)
                if key not in grouped_callable:
                    grouped_callable[key] = []
                grouped_callable[key].append(item)
            return grouped_callable
        return current

    @staticmethod
    def _build_apply_map(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> t.NormalizedValue:
        """Helper: Apply map operation."""
        if "map" not in ops:
            return current
        map_func_result = FlextUtilitiesMapper._get_callable_from_dict(ops, "map")
        if map_func_result.is_failure:
            return current
        map_callable: t.MapperCallable = map_func_result.value

        def map_func(value: t.NormalizedValue) -> t.NormalizedValue:
            return FlextUtilitiesMapper.narrow_to_container(map_callable(value))

        if isinstance(current, (list, tuple)):
            seq_current: Sequence[t.NormalizedValue] = current
            return [
                map_func(FlextUtilitiesMapper.narrow_to_container(x))
                for x in seq_current
            ]
        if isinstance(current, Mapping):
            current_dict: t.ContainerMapping = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(current)
            )
            return {k: map_func(v) for k, v in current_dict.items()}
        current_general = FlextUtilitiesMapper.narrow_to_container(current)
        return map_func(current_general)

    @staticmethod
    def _build_apply_normalize(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> t.NormalizedValue:
        """Helper: Apply normalize operation."""
        if "normalize" not in ops:
            return current
        normalize_case = FlextUtilitiesMapper._get_str_from_dict(ops, "normalize", "")
        if isinstance(current, str):
            return current.lower() if normalize_case == "lower" else current.upper()
        if isinstance(current, (list, tuple)):
            seq_current: Sequence[t.NormalizedValue] = current
            result: t.ContainerList = []
            for x in seq_current:
                x_general = FlextUtilitiesMapper.narrow_to_container(x)
                if isinstance(x_general, str):
                    result.append(
                        x_general.lower()
                        if normalize_case == "lower"
                        else x_general.upper()
                    )
                else:
                    result.append(x_general)
            return result
        return current

    @staticmethod
    def _build_apply_process(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
        default: t.NormalizedValue,
        on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply process operation."""
        if "process" not in ops:
            return current
        process_func_result = FlextUtilitiesMapper._get_callable_from_dict(
            ops, "process"
        )
        if process_func_result.is_failure:
            return current
        process_callable: t.MapperCallable = process_func_result.value

        def process_func(value: t.NormalizedValue) -> t.NormalizedValue:
            return FlextUtilitiesMapper.narrow_to_container(process_callable(value))

        def _process_current() -> t.NormalizedValue:
            if isinstance(current, (list, tuple)):
                seq_current: Sequence[t.NormalizedValue] = current
                return [
                    process_func(FlextUtilitiesMapper.narrow_to_container(x))
                    for x in seq_current
                ]
            if isinstance(current, Mapping):
                current_dict: t.ContainerMapping = (
                    FlextUtilitiesMapper._narrow_to_configuration_dict(current)
                )
                return {k: process_func(v) for k, v in current_dict.items()}
            current_general = FlextUtilitiesMapper.narrow_to_container(current)
            return process_func(current_general)

        process_result: r[t.NormalizedValue] = r[
            t.NormalizedValue
        ].create_from_callable(_process_current)
        if process_result.is_failure:
            return default if on_error == "stop" else current
        process_val: t.NormalizedValue = process_result.value
        return process_val if process_val is not None else current

    @staticmethod
    def _build_apply_slice(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> t.NormalizedValue:
        """Helper: Apply slice operation."""
        if "slice" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        current_items: Sequence[t.NormalizedValue] = current
        slice_spec = ops["slice"]
        min_slice_length = 2
        if (
            isinstance(slice_spec, (list, tuple))
            and len(slice_spec) >= min_slice_length
        ):
            start_raw = slice_spec[0]
            end_raw = slice_spec[1]
            start: int | None = start_raw if isinstance(start_raw, int) else None
            end: int | None = end_raw if isinstance(end_raw, int) else None
            if isinstance(current, list):
                sliced_list: t.ContainerList = [
                    FlextUtilitiesMapper.narrow_to_container(item)
                    for item in current_items[start:end]
                ]
                return sliced_list
            sliced_tuple: tuple[t.NormalizedValue, ...] = tuple(
                FlextUtilitiesMapper.narrow_to_container(item)
                for item in current_items[start:end]
            )
            return sliced_tuple
        return current

    @staticmethod
    def _build_apply_sort(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> t.NormalizedValue:
        """Helper: Apply sort operation."""
        if "sort" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        sort_spec_raw: t.NormalizedValue | t.MapperCallable = ops["sort"]
        current_items: Sequence[t.NormalizedValue] = current
        current_list: t.ContainerList = [
            FlextUtilitiesMapper.narrow_to_container(item) for item in current_items
        ]
        if isinstance(sort_spec_raw, str):
            field_name: str = sort_spec_raw

            def key_func(item: t.NormalizedValue) -> str:
                if isinstance(item, Mapping):
                    return str(item.get(field_name, ""))
                if isinstance(item, BaseModel):
                    if hasattr(item, field_name):
                        return str(getattr(item, field_name))
                    return ""
                return ""

            sorted_list_key: t.ContainerList = sorted(current_list, key=key_func)
            return (
                list(sorted_list_key)
                if isinstance(current, list)
                else tuple(sorted_list_key)
            )
        if callable(sort_spec_raw):
            sort_callable_result = FlextUtilitiesMapper._get_callable_from_dict(
                ops, "sort"
            )
            if sort_callable_result.is_failure:
                return current
            sort_callable = sort_callable_result.value

            def sort_key(item: t.NormalizedValue) -> str:
                return str(sort_callable(item))

            sorted_result = r[t.ContainerList].create_from_callable(
                lambda: sorted(current_list, key=sort_key)
            )
            if sorted_result.is_failure:
                return current
            sorted_callable = sorted_result.value
            return (
                list(sorted_callable)
                if isinstance(current, list)
                else tuple(sorted_callable)
            )
        if sort_spec_raw is True:
            comparable_items: t.ContainerList = [
                FlextUtilitiesMapper.narrow_to_container(
                    item
                    if isinstance(item, (str, int, float, bool)) or item is None
                    else str(item)
                )
                for item in current_list
            ]
            sorted_comparable: t.ContainerList = sorted(comparable_items, key=str)
            if isinstance(current, list):
                return sorted_comparable
            return tuple(sorted_comparable)
        return current

    @staticmethod
    def _build_apply_transform(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
        default: t.NormalizedValue,
        on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply transform operation."""
        if "transform" not in ops or not FlextUtilitiesGuards.is_type(
            current, "mapping"
        ):
            return current
        transform_opts_raw = ops["transform"]
        if not isinstance(transform_opts_raw, Mapping) or callable(transform_opts_raw):
            return current
        transform_opts = FlextUtilitiesMapper._narrow_to_configuration_dict(
            FlextUtilitiesMapper.narrow_to_container(transform_opts_raw)
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
        transform_result = r[t.NormalizedValue].create_from_callable(
            lambda: FlextUtilitiesMapper.apply_transform_steps(
                dict(current_dict),
                normalize=normalize_bool,
                map_keys=map_keys_dict,
                filter_keys=filter_keys_set,
                exclude_keys=exclude_keys_set,
                strip_none=strip_none_bool,
                strip_empty=strip_empty_bool,
            )
        )
        if transform_result.is_failure:
            if on_error == "stop":
                return default
            return current
        return transform_result.value

    @staticmethod
    def _build_apply_unique(
        current: t.NormalizedValue,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
    ) -> t.NormalizedValue:
        """Helper: Apply unique operation to remove duplicates."""
        if "unique" not in ops or not ops.get("unique"):
            return current
        if not isinstance(current, (list, tuple)):
            return current
        current_items: Sequence[t.NormalizedValue] = current
        current_list_unique: t.ContainerList = [
            FlextUtilitiesMapper.narrow_to_container(item) for item in current_items
        ]
        seen: set[t.NormalizedValue | str] = set()
        unique_list: t.ContainerList = []
        for item in current_list_unique:
            item_hashable: t.NormalizedValue | str = (
                item
                if isinstance(item, (str, int, float, bool)) or item is None
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
        item: t.NormalizedValue, field_name: str
    ) -> t.NormalizedValue:
        """Extract field value from item (dict or object).

        Helper method to improve type inference for pyrefly.
        """
        if isinstance(item, Mapping):
            dict_item: dict[str, t.NormalizedValue] = {}
            for key, value in item.items():
                coerced_value: t.NormalizedValue = (
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
    def _extract_get_value(
        current: t.NormalizedValue | BaseModel | Mapping[str, t.NormalizedValue],
        key_part: str,
    ) -> r[t.NormalizedValue]:
        """Helper: Get raw value from dict/object/model.

        Returns:
            r[t.NormalizedValue]: Ok with value if found (non-None),
                fail("found_none:...") if found but None,
                fail("Key ... not found") if not found.

        """
        if isinstance(current, Mapping):
            if key_part in current:
                raw_val = current[key_part]
                narrowed = FlextUtilitiesMapper.narrow_to_container(raw_val)
                if narrowed is None:
                    return r[t.NormalizedValue].fail(f"found_none:{key_part}")
                return r[t.NormalizedValue].ok(narrowed)
            return r[t.NormalizedValue].fail(f"Key '{key_part}' not found in Mapping")
        if hasattr(current, key_part):
            attr_val = getattr(current, key_part)
            if attr_val is None:
                return r[t.NormalizedValue].fail(f"found_none:{key_part}")
            if FlextUtilitiesGuards.is_container(attr_val):
                return r[t.NormalizedValue].ok(
                    FlextUtilitiesMapper.narrow_to_container(attr_val)
                )
            return r[t.NormalizedValue].ok(str(attr_val))
        if isinstance(current, BaseModel):
            model_dump_attr = current.model_dump
            if callable(model_dump_attr):
                model_dict = model_dump_attr()
                if isinstance(model_dict, Mapping) and key_part in model_dict:
                    val = FlextUtilitiesMapper.narrow_to_container(model_dict[key_part])
                    if val is None:
                        return r[t.NormalizedValue].fail(f"found_none:{key_part}")
                    return r[t.NormalizedValue].ok(val)
        return r[t.NormalizedValue].fail(f"Key '{key_part}' not found")

    @staticmethod
    def _extract_handle_array_index(
        current: t.NormalizedValue, array_match: str
    ) -> r[t.NormalizedValue]:
        """Helper: Handle array indexing with support for negative indices.

        Returns:
            r[t.NormalizedValue]: Ok with indexed value, fail("found_none:index") if None,
            Fail with error message if out of bounds or invalid.

        """
        if not isinstance(current, (list, tuple)):
            return r[t.NormalizedValue].fail("Not a sequence")
        sequence: Sequence[t.NormalizedValue] = (
            FlextUtilitiesMapper._narrow_to_sequence(current)
        )
        try:
            idx = int(array_match)
            if idx < 0:
                idx = len(sequence) + idx
            if 0 <= idx < len(sequence):
                item = sequence[idx]
                if item is None:
                    return r[t.NormalizedValue].fail("found_none:index")
                return r[t.NormalizedValue].ok(item)
            return r[t.NormalizedValue].fail(f"Index {int(array_match)} out of range")
        except (ValueError, IndexError):
            return r[t.NormalizedValue].fail(f"Invalid index {array_match}")

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
        Mapping[str, str] | None,
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
        map_keys_dict: Mapping[str, str] | None = None
        if isinstance(map_keys_val, Mapping) and all(
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
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable], key: str
    ) -> r[t.MapperCallable]:
        value: t.NormalizedValue | t.MapperCallable = ops.get(key)
        if callable(value):
            return r[t.MapperCallable].ok(value)
        return r[t.MapperCallable].fail(f"Operation '{key}' is not callable")

    @staticmethod
    def _get_raw(
        data: p.AccessibleData, key: str, *, default: t.NormalizedValue = None
    ) -> t.NormalizedValue:
        """Internal helper for raw get without DSL conversion."""
        match data:
            case dict() | Mapping():
                raw_value = data.get(key)
                if raw_value is None:
                    return default if default is not None else ""
                return FlextUtilitiesMapper.narrow_to_container(raw_value)
            case m.ConfigMap() | m.Dict():
                raw_value = data.root.get(key)
                if raw_value is None:
                    return default if default is not None else ""
                return FlextUtilitiesMapper.narrow_to_container(raw_value)
            case _:
                if hasattr(data, key):
                    attr_val = getattr(data, key)
                    return FlextUtilitiesMapper.narrow_to_container(attr_val)
                return default if default is not None else ""

    @staticmethod
    def _get_str_from_dict(
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable],
        key: str,
        default: str = "",
    ) -> str:
        """Safely extract str value from ConfigurationDict."""
        value = ops.get(key, default)
        if isinstance(value, str):
            return str(value)
        if callable(value):
            return default
        return str(value) if value is not None else default

    @staticmethod
    def _narrow_to_configuration_dict(
        value: t.NormalizedValue | Mapping[str, t.NormalizedValue],
    ) -> Mapping[str, t.NormalizedValue]:
        """Safely narrow object to ConfigurationDict with runtime validation."""
        if isinstance(value, Mapping):
            if FlextUtilitiesGuards.is_configuration_dict(value):
                normalized_dict: dict[str, t.NormalizedValue] = {}
                for key, item in value.items():
                    normalized_dict[str(key)] = (
                        FlextUtilitiesMapper.narrow_to_container(item)
                    )
                return normalized_dict
        error_msg = f"Cannot narrow {value.__class__.__name__} to ConfigurationDict"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_configuration_mapping(
        value: t.NormalizedValue | Mapping[str, t.NormalizedValue],
    ) -> m.ConfigMap:
        """Safely narrow object to ConfigurationMapping with runtime validation."""
        if isinstance(value, m.ConfigMap):
            return value
        if isinstance(value, Mapping):
            narrowed_dict = FlextUtilitiesMapper._narrow_to_configuration_dict(value)
            coerced_result = r[m.ConfigMap].create_from_callable(
                lambda: m.ConfigMap(root=narrowed_dict)
            )
            if coerced_result.is_success:
                val: m.ConfigMap = coerced_result.value
                return val
            error_msg = f"Cannot coerce {value.__class__.__name__} to m.ConfigMap: {coerced_result.error}"
            raise TypeError(error_msg)
        error_msg = f"Cannot narrow {value.__class__.__name__} to m.ConfigMap"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_sequence(
        value: t.NormalizedValue | Sequence[t.NormalizedValue],
    ) -> Sequence[t.NormalizedValue]:
        """Safely narrow object to Sequence[t.NormalizedValue]."""
        if isinstance(value, (list, tuple)):
            narrowed_items: t.ContainerList = []
            for item_raw in value:
                item = FlextUtilitiesMapper.narrow_to_container(item_raw)
                narrowed_item = FlextUtilitiesMapper.narrow_to_container(item)
                narrowed_items.append(narrowed_item)
            return narrowed_items
        error_msg = f"Cannot narrow {value.__class__.__name__} to Sequence"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_string_keyed_dict(
        value: t.NormalizedValue | Mapping[str, t.NormalizedValue],
    ) -> dict[str, t.NormalizedValue]:
        """Narrow object to ConfigurationDict (for conversion purposes).

        Validates that the value is a dict with string keys and object values.
        Uses TypeGuard pattern for proper type narrowing.
        """
        if isinstance(value, Mapping):
            result: dict[str, t.NormalizedValue] = {}
            key: str
            val: t.NormalizedValue
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
    def _to_general_value_from_object(
        value: t.NormalizedValue,
    ) -> t.NormalizedValue:
        """Deprecated: use narrow_to_container instead. Planned removal: v0.12."""
        warnings.warn(
            "_to_general_value_from_object is deprecated; use narrow_to_container. "
            "Planned removal: v0.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FlextUtilitiesMapper.narrow_to_container(value)

    @staticmethod
    def agg[T](
        items: list[T] | tuple[T, ...],
        field: str | Callable[[T], int | float],
        *,
        fn: Callable[[list[int | float]], int | float] | None = None,
    ) -> int | float:
        """Aggregate field values from objects (mnemonic: agg = aggregate).

        Generic replacement for: sum(getattr(...)), max(getattr(...))

        Args:
            items: List/tuple of objects
            field: Field name (str) or extractor function (callable)
            fn: Aggregation function (default: sum)

        Returns:
            Aggregated value

        Example:
            # Sum field values
            total = FlextUtilitiesMapper.agg(items, "synced")
            # → 15

            # Max with custom extractor
            max_val = FlextUtilitiesMapper.agg(
                items, lambda r: r.total_entries, fn=max
            )
            # → 30

        """
        items_list: list[T] = list(items)
        numeric_values: list[int | float] = []
        if callable(field):
            for item in items_list:
                val = field(item)
                numeric_values.append(val)
        else:
            field_name: str = field
            for item in items_list:
                if not isinstance(item, (Mapping, BaseModel)):
                    continue
                val_raw = FlextUtilitiesMapper._extract_field_value(item, field_name)
                if isinstance(val_raw, int | float):
                    numeric_values.append(val_raw)
        agg_fn: Callable[[list[int | float]], int | float] = (
            fn if fn is not None else sum
        )
        if numeric_values:
            return agg_fn(numeric_values)
        return 0

    @staticmethod
    def apply_transform_steps(
        result: t.ContainerMapping,
        *,
        normalize: bool,
        map_keys: Mapping[str, str] | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
    ) -> t.ContainerMapping:
        return FlextUtilitiesMapper._apply_transform_steps(
            result,
            normalize=normalize,
            map_keys=map_keys,
            filter_keys=filter_keys,
            exclude_keys=exclude_keys,
            strip_none=strip_none,
            strip_empty=strip_empty,
        )

    @staticmethod
    def as_(
        value: t.NormalizedValue,
        target: type,
        *,
        default: t.NormalizedValue = None,
        strict: bool = False,
    ) -> t.NormalizedValue:
        """Type conversion with guard (mnemonic: as_ = convert to type).

         Generic replacement for: type check +  patterns

        Args:
            value: Value to convert
            target: Target type
            default: Default if conversion fails
            strict: If True, only exact type; if False, allow coercion

        Returns:
            Converted value or default

        Example:
            port = FlextUtilitiesMapper.as_(config.get("port"), int, default=c.Platform.DEFAULT_HTTP_PORT)
            name = FlextUtilitiesMapper.as_(value, str, default="")

        """
        try:
            if isinstance(value, target):
                if FlextUtilitiesGuards.is_container(value):
                    return FlextUtilitiesMapper.narrow_to_container(value)
                return str(value)
        except TypeError:
            if strict:
                return default
        if strict:
            return default
        try:
            if target is int and isinstance(value, str | float | bool):
                return int(value)
            if target is float and isinstance(value, str | int | bool):
                return float(value)
            if target is str:
                return str(value)
            if target is bool and isinstance(value, str):
                normalized = value.lower()
                if normalized in {"true", "1", "yes", "on"}:
                    return True
                if normalized in {"false", "0", "no", "off"}:
                    return False
            return default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def at[T](
        items: list[T] | tuple[T, ...] | Mapping[str, T],
        index: int | str,
        *,
        default: T | None = None,
    ) -> r[T]:
        """Get item at index/key (mnemonic: at = get at position).

        Generic replacement for: items[index] with safe access

        Args:
            items: Items to access
            index: Index (int) or key (str)
            default: Default if not found

        Returns:
            Item at index/key or default

        Example:
            user = FlextUtilitiesMapper.at(users, 0)
            value = FlextUtilitiesMapper.at(data_dict, "key")

        """
        try:
            if isinstance(items, Mapping):
                if isinstance(index, str) and index in items:
                    return r[T].ok(items[index])
                if default is not None:
                    return r[T].ok(default)
                return r[T].fail(f"Key '{index}' not found")
            if isinstance(index, int) and 0 <= index < len(items):
                return r[T].ok(items[index])
            if default is not None:
                return r[T].ok(default)
            return r[T].fail(f"Index '{index}' out of bounds")
        except (IndexError, KeyError, TypeError):
            if default is not None:
                return r[T].ok(default)
            return r[T].fail("Failed to access item at index/key")

    @staticmethod
    def build(
        value: p.AccessibleData,
        *,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable] | None = None,
        default: t.NormalizedValue = None,
        on_error: str = "stop",
    ) -> t.NormalizedValue:
        """Builder pattern for fluent operation composition using DSL.

        Uses DSL dict to compose operations:
        {"ensure": "str", "map": lambda x: x*2, ...}

        Operations are applied in order:
        ensure → filter → map → normalize → convert → transform → process →
        group → sort → unique → slice → chunk

        Args:
            value: Initial value to process
            ops: Dict with operation keys:
                - "ensure": str target type ("str", "str_list", "list", "dict")
                - "ensure_default": default value for ensure
                - "filter": predicate function
                - "map": mapper function
                - "normalize": case ("lower" or "upper")
                - "convert": target type class
                - "convert_default": default for convert
                - "transform": dict with transform options
                - "process": processor function
                - "group": str field name or callable for grouping
                - "sort": str field name, callable, or True for natural sort
                - "unique": bool to remove duplicates
                - "slice": tuple[int, int] for slicing (start, end)
                - "chunk": int size for chunking into sublists
            default: Default value on error
            on_error: Error handling ("stop", "skip", "collect")

        Returns:
            Processed value (type depends on operations applied)

        Example:
            result = FlextUtilitiesMapper.build(
                value,
                ops={"ensure": "str", "normalize": "lower"},
            )

        """
        narrowed_value = FlextUtilitiesMapper.narrow_to_container(value)
        if ops is None:
            return narrowed_value
        current: t.NormalizedValue = narrowed_value
        default_val: t.NormalizedValue = (
            default if default is not None else narrowed_value
        )
        current = FlextUtilitiesMapper._build_apply_ensure(current, ops)
        current = FlextUtilitiesMapper._build_apply_filter(current, ops, default_val)
        current = FlextUtilitiesMapper._build_apply_map(current, ops)
        current = FlextUtilitiesMapper._build_apply_normalize(current, ops)
        current = FlextUtilitiesMapper._build_apply_convert(current, ops)
        current = FlextUtilitiesMapper._build_apply_transform(
            current, ops, default_val, on_error
        )
        current = FlextUtilitiesMapper._build_apply_process(
            current, ops, default_val, on_error
        )
        current = FlextUtilitiesMapper._build_apply_group(current, ops)
        current = FlextUtilitiesMapper._build_apply_sort(current, ops)
        current = FlextUtilitiesMapper._build_apply_unique(current, ops)
        current = FlextUtilitiesMapper._build_apply_slice(current, ops)
        return FlextUtilitiesMapper._build_apply_chunk(current, ops)

    @staticmethod
    def build_flags_dict(
        active_flags: list[str],
        flag_mapping: Mapping[str, str],
        *,
        default_value: bool = False,
    ) -> r[Mapping[str, bool]]:
        """Build boolean flags dictionary from list of active flags.

        **Generic replacement for**: Permission building, feature flags

        Args:
            active_flags: List of active flag names
            flag_mapping: Mapping of flag_name → output_key
            default_value: Default value for inactive flags (default: False)

        Returns:
            r with flags dictionary or error

        Example:
            >>> flags = ["read", "write"]
            >>> mapping = {
            ...     "read": "can_read",
            ...     "write": "can_write",
            ...     "delete": "can_delete",
            ... }
            >>> result = FlextUtilitiesMapper.build_flags_dict(flags, mapping)
            >>> flags_dict = result.value
            >>> # {"can_read": True, "can_write": True, "can_delete": False}

        """

        def _build_flags() -> Mapping[str, bool]:
            result: dict[str, bool] = {}
            for output_key in flag_mapping.values():
                result[output_key] = default_value
            for flag in active_flags:
                mapped_key: str | None = flag_mapping.get(flag)
                if mapped_key:
                    result[mapped_key] = True
            return result

        flags_result = r[Mapping[str, bool]].create_from_callable(_build_flags)
        return flags_result.fold(
            on_failure=lambda e: r[Mapping[str, bool]].fail(
                f"Failed to build flags dict: {e}"
            ),
            on_success=lambda _: flags_result,
        )

    @staticmethod
    def cast_generic[T](
        value: t.NormalizedValue,
        target_type: Callable[[t.NormalizedValue | BaseModel], T] | None = None,
        *,
        default: T | None = None,
    ) -> T | t.NormalizedValue:
        """Safe cast with fallback.

        Args:
            value: Value to cast
            target_type: Callable/type that converts input to T (optional)
            default: Default value if cast fails

        Returns:
            Cast value or default

        Example:
            port = u.cast_generic(config.get("port"), int, default=8080)

        """
        if target_type is None:
            return value
        cast_result = r[T].create_from_callable(lambda: target_type(value))
        if cast_result.is_success and cast_result.value is not None:
            return cast_result.value
        if default is not None:
            return default
        return value

    @staticmethod
    def collect_active_keys(
        source: Mapping[str, bool], key_mapping: Mapping[str, str]
    ) -> r[list[str]]:
        """Collect list of output keys where source value is True.

        **Generic replacement for**: Collecting active permissions/flags

        Args:
            source: Dictionary with boolean values
            key_mapping: Mapping of source_key → output_key

        Returns:
            r with list of active output keys or error

        Example:
            >>> source = {"read": True, "write": True, "delete": False}
            >>> mapping = {"read": "r", "write": "w", "delete": "d"}
            >>> result = FlextUtilitiesMapper.collect_active_keys(source, mapping)
            >>> active = result.value  # ["r", "w"]

        """

        def _collect_keys() -> list[str]:
            active_keys: list[str] = []
            for source_key, output_key in key_mapping.items():
                if source.get(source_key):
                    active_keys.append(output_key)
            return active_keys

        active_keys_result = r[list[str]].create_from_callable(_collect_keys)
        return active_keys_result.fold(
            on_failure=lambda e: r[list[str]].fail(
                f"Failed to collect active keys: {e}"
            ),
            on_success=lambda _: active_keys_result,
        )

    @staticmethod
    def construct(
        spec: Mapping[str, t.NormalizedValue | t.MapperCallable],
        source: m.ConfigMap | BaseModel | None = None,
        *,
        on_error: str = "stop",
    ) -> t.ContainerMapping:
        """Construct object using mnemonic specification pattern.

        Builds object from mnemonic spec that maps target keys to source
        fields or DSL operations. Supports field mapping, default values,
        and DSL operations.

        Args:
            spec: Construction specification:
                - Direct: {"target_key": "source_field"}
                - Default: {"target_key": {"field": "source_field", "default": value}}
                - DSL: {"target_key": {"field": "source_field", "ops": {...}}}
                - Literal: {"target_key": {"value": literal}}
            source: Optional source data
            on_error: Error handling ("stop", "skip", "collect")

        Returns:
            Constructed dict with target keys

        Example:
            plugin_info = FlextUtilitiesMapper.construct(
                {
                    "name": "plugin_name",
                    "type": "plugin_type",
                },
                source=plugin_data,
            )

        """
        constructed: dict[str, t.NormalizedValue] = {}
        for target_key, target_spec in spec.items():
            try:
                target_spec_mapping: t.ContainerMapping | None = None
                if isinstance(target_spec, Mapping):
                    target_spec_mapping = target_spec
                    if "value" in target_spec_mapping:
                        constructed[target_key] = target_spec_mapping["value"]
                        continue
                if isinstance(target_spec, str):
                    source_field = target_spec
                    field_default = None
                    field_ops = None
                elif target_spec_mapping is not None:
                    source_field_raw = target_spec_mapping.get("field", target_key)
                    source_field = (
                        str(source_field_raw)
                        if source_field_raw is not None
                        else target_key
                    )
                    field_default = target_spec_mapping.get("default")
                    field_ops = target_spec_mapping.get("ops")
                else:
                    constructed[target_key] = target_spec
                    continue
                if source is None:
                    constructed[target_key] = field_default
                    continue
                extracted_result = FlextUtilitiesMapper.extract(
                    source, source_field, default=field_default, required=False
                )
                extracted_raw = (
                    extracted_result.value
                    if extracted_result.is_success
                    else field_default
                )
                if field_ops is not None and extracted_raw is not None:
                    if isinstance(field_ops, Mapping):
                        extracted = FlextUtilitiesMapper.build(
                            extracted_raw, ops=dict(field_ops)
                        )
                    else:
                        extracted = extracted_raw
                else:
                    extracted = extracted_raw
                final_value: t.NormalizedValue = (
                    FlextUtilitiesMapper.narrow_to_container(
                        extracted if extracted is not None else field_default
                    )
                )
                constructed[target_key] = final_value
            except (TypeError, ValueError, KeyError, AttributeError) as e:
                error_msg = f"Failed to construct '{target_key}': {e}"
                if on_error == "stop":
                    raise ValueError(error_msg) from e
                if on_error == "skip":
                    continue
        return constructed

    @staticmethod
    def deep_eq(a: t.ContainerMapping, b: t.ContainerMapping) -> bool:
        """Deep equality comparison for nested structures.

        Generic replacement for: Manual deep dict comparison.

        Compares nested dictionaries recursively, handling:
        - Nested dicts and lists
        - Different ordering of keys (dict comparison)
        - None values
        - Primitive types (str, int, float, bool)

        Args:
            a: First dictionary to compare.
            b: Second dictionary to compare.

        Returns:
            True if dictionaries are deeply equal, False otherwise.

        Example:
            >>> a = {"nested": {"key": "value"}, "list": [1, 2, 3]}
            >>> b = {"nested": {"key": "value"}, "list": [1, 2, 3]}
            >>> FlextUtilitiesMapper.deep_eq(a, b)
            True

        """
        if a is b:
            return True
        if len(a) != len(b):
            return False
        for key, val_a in a.items():
            if key not in b:
                return False
            val_b = b[key]
            if val_a is None and val_b is None:
                continue
            if val_a is None or val_b is None:
                return False
            if isinstance(val_a, Mapping) and isinstance(val_b, Mapping):
                val_a_dict = FlextUtilitiesMapper._narrow_to_configuration_dict(val_a)
                val_b_dict = FlextUtilitiesMapper._narrow_to_configuration_dict(val_b)
                if not FlextUtilitiesMapper.deep_eq(val_a_dict, val_b_dict):
                    return False
                continue
            if isinstance(val_a, list) and isinstance(val_b, list):
                val_a_list = val_a
                val_b_list = val_b
                if len(val_a_list) != len(val_b_list):
                    return False
                for item_a, item_b in zip(val_a_list, val_b_list, strict=True):
                    if isinstance(item_a, Mapping) and isinstance(item_b, Mapping):
                        item_a_dict = (
                            FlextUtilitiesMapper._narrow_to_configuration_dict(item_a)
                        )
                        item_b_dict = (
                            FlextUtilitiesMapper._narrow_to_configuration_dict(item_b)
                        )
                        if not FlextUtilitiesMapper.deep_eq(item_a_dict, item_b_dict):
                            return False
                    elif item_a != item_b:
                        return False
                continue
            if val_a != val_b:
                return False
        return True

    @staticmethod
    def ensure_str_list(
        value: t.NormalizedValue, default: list[str] | None = None
    ) -> list[str]:
        """Ensure value is a list of strings, converting if needed.

        **Generic replacement for**: [str(item) for item in list] patterns
        **Renamed from**: ensure_str_list

        Args:
            value: Value to convert (list, tuple, set, or single value)
            default: Default value if None (empty list if not specified)

        Returns:
            List of strings

        Example:
            >>> FlextUtilitiesMapper.ensure_str_list(["a", "b"])
            ['a', 'b']
            >>> FlextUtilitiesMapper.ensure_str_list([1, 2, 3])
            ['1', '2', '3']
            >>> FlextUtilitiesMapper.ensure_str_list("single")
            ['single']
            >>> FlextUtilitiesMapper.ensure_str_list(None)
            []

        """
        if default is None:
            default = []
        if value is None:
            return default
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def ensure_str(value: t.NormalizedValue, default: str = "") -> str:
        """Ensure value is a string, converting if needed.

        **Generic replacement for**: Manual str() conversions with isinstance checks

        Args:
            value: Value to convert to string
            default: Default value if None or conversion fails

        Returns:
            String value or default

        Example:
            >>> FlextUtilitiesMapper.ensure_str("hello")
            'hello'
            >>> FlextUtilitiesMapper.ensure_str(123)
            '123'
            >>> FlextUtilitiesMapper.ensure_str(None, "default")
            'default'

        """
        if value is None:
            return default
        if isinstance(value, str):
            return str(value)
        return str(value)

    @staticmethod
    def ensure_str_or_none(value: t.NormalizedValue) -> r[str]:
        """Ensure value is a string result.

        **Generic replacement for**: value if isinstance(value, str) else None

        Args:
            value: Value to check/convert

        Returns:
            r[str]: ok(str) when value is str, fail otherwise

        Example:
            >>> FlextUtilitiesMapper.ensure_str_or_none("hello")
            r[str].ok("hello")
            >>> FlextUtilitiesMapper.ensure_str_or_none(123)
            r[str].fail("Value is not a string")
            >>> FlextUtilitiesMapper.ensure_str_or_none(None)
            r[str].fail("Value is not a string")

        """
        if isinstance(value, str):
            return r[str].ok(value)
        return r[str].fail("Value is not a string")

    @staticmethod
    def extract(
        data: p.AccessibleData,
        path: str,
        *,
        default: t.NormalizedValue = None,
        required: bool = False,
        separator: str = ".",
    ) -> r[t.NormalizedValue]:
        """Safe nested data extraction with dot notation.

        Business Rule: Extracts nested values using dot notation paths.
        Supports dict access, object attributes, and Pydantic model fields.
        Array indexing supported via "key[0]" syntax. Required mode fails
        if path not found, otherwise returns default.

        Args:
            data: Source data (dict, object with attributes, or Pydantic model)
            path: Dot-separated path (e.g., "user.profile.name")
            default: Default value if path not found
            required: Fail if path not found
            separator: Path separator (default: ".")

        Returns:
            r containing extracted value or default

        Example:
            config = {"database": {"host": c.Platform.DEFAULT_HOST, "port": 5432}}
            result = FlextUtilitiesMapper.extract(config, "database.port")
            # → r[t.NormalizedValue].ok(5432)

        """
        try:
            parts = path.split(separator)
            current: t.NormalizedValue | BaseModel = None
            if isinstance(data, BaseModel):
                current = data
            elif isinstance(data, Mapping):
                current = FlextUtilitiesMapper.narrow_to_container(data)
            else:
                current = data
            found_none_prefix = "found_none:"
            for i, part in enumerate(parts):
                if current is None:
                    if required:
                        return r[t.NormalizedValue].fail(
                            f"Path '{separator.join(parts[:i])}' is None"
                        )
                    if default is None:
                        return r[t.NormalizedValue].fail(
                            f"Path '{separator.join(parts[:i])}' is None and default is None"
                        )
                    return r[t.NormalizedValue].ok(default)
                key_part, array_match = FlextUtilitiesMapper._extract_parse_array_index(
                    part
                )
                get_result = FlextUtilitiesMapper._extract_get_value(current, key_part)
                if get_result.is_failure:
                    error_str = get_result.error or ""
                    if error_str.startswith(found_none_prefix):
                        current = None
                    else:
                        path_context = separator.join(parts[:i])
                        if required:
                            return r[t.NormalizedValue].fail(
                                f"Key '{key_part}' not found at '{path_context}'"
                            )
                        if default is None:
                            return r[t.NormalizedValue].fail(
                                f"Key '{key_part}' not found at '{path_context}' and default is None"
                            )
                        return r[t.NormalizedValue].ok(default)
                else:
                    current = get_result.value
                if array_match and current is not None:
                    index_result = FlextUtilitiesMapper._extract_handle_array_index(
                        current, array_match
                    )
                    if index_result.is_failure:
                        error_str = index_result.error or ""
                        if error_str.startswith(found_none_prefix):
                            current = None
                        else:
                            if required:
                                return r[t.NormalizedValue].fail(
                                    f"Array error at '{key_part}': {index_result.error}"
                                )
                            if default is None:
                                return r[t.NormalizedValue].fail(
                                    f"Array error at '{key_part}': {index_result.error} and default is None"
                                )
                            return r[t.NormalizedValue].ok(default)
                    else:
                        current = index_result.value
            if current is None:
                if required:
                    return r[t.NormalizedValue].fail("Extracted value is None")
                if default is None:
                    return r[t.NormalizedValue].fail(
                        "Extracted value is None and default is None"
                    )
                return r[t.NormalizedValue].ok(default)
            if FlextUtilitiesGuards.is_container(current):
                return r[t.NormalizedValue].ok(
                    FlextUtilitiesMapper.narrow_to_container(current)
                )
            return r[t.NormalizedValue].ok(str(current))
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
            return r[t.NormalizedValue].fail(f"Extract failed: {e}")

    @staticmethod
    def field(
        source: p.AccessibleData,
        name: str,
        *,
        default: t.NormalizedValue = None,
        required: bool = False,
        ops: Mapping[str, t.NormalizedValue | t.MapperCallable] | None = None,
    ) -> t.NormalizedValue:
        """Extract single field from source with optional DSL processing.

        FLEXT Pattern: Simplified single-field extraction (split from overloaded fields).

        Args:
            source: Source data (dict or object)
            name: Field name to extract
            default: Default value if field not found
            required: If True, returns None on missing
            ops: Optional DSL operations dict

        Returns:
            Extracted value or default/None

        Example:
            name = FlextUtilitiesMapper.field(payload, "name", default="")
            age = FlextUtilitiesMapper.field(user, "age", default=0)

        """
        has_value = False
        if isinstance(source, Mapping):
            has_value = name in source
        elif hasattr(source, name):
            has_value = True
        if not has_value:
            if required:
                return None
            value: t.NormalizedValue = default if default is not None else ""
        else:
            raw_value: t.NormalizedValue = FlextUtilitiesMapper.get(
                source,
                name,
                default=default,
            )
            value = raw_value if raw_value is not None else default
        if ops is not None:
            value_for_build: t.NormalizedValue = (
                FlextUtilitiesMapper.narrow_to_container(value)
                if value is not None
                else FlextUtilitiesMapper.narrow_to_container("")
            )
            return FlextUtilitiesMapper.build(value_for_build, ops=ops, on_error="stop")
        return value

    @staticmethod
    def fields(
        obj: p.AccessibleData,
        *field_names: str | t.NormalizedValue,
    ) -> t.ContainerMapping:
        """Extract specified fields from object.

        Supports two patterns:
        1. Simple: u.fields(obj, "name", "email", "id")
        2. DSL spec: u.fields(obj, {"name": {"default": ""}, ...})

        Args:
            obj: Object or dict to extract from
            *field_names: Field names (str) or field specs (dict)

        Returns:
            Mapping with extracted fields (string-keyed payload values)

        Example:
            # Simple extraction
            data = u.fields(user, "name", "email", "id")

            # With field specs
            data = u.fields(payload, {
                "name": {"default": ""},
                "count": {"default": 0}
            })

        """
        result: dict[str, t.NormalizedValue] = {}
        spec_item: str | t.NormalizedValue
        for spec_item in field_names:
            if isinstance(spec_item, Mapping):
                name: str
                field_config: t.NormalizedValue
                for name, field_config in spec_item.items():
                    if isinstance(obj, Mapping):
                        if name in obj:
                            result[name] = FlextUtilitiesMapper.narrow_to_container(
                                obj[name]
                            )
                        elif isinstance(field_config, Mapping):
                            default_value = field_config.get("default")
                            if default_value is not None:
                                result[name] = default_value
                        else:
                            result[name] = field_config
                    elif hasattr(obj, name):
                        result[name] = FlextUtilitiesMapper.narrow_to_container(
                            getattr(obj, name)
                        )
                    elif isinstance(field_config, Mapping):
                        default_value = field_config.get("default")
                        if default_value is not None:
                            result[name] = default_value
            else:
                field_name: str = spec_item
                if isinstance(obj, Mapping):
                    if field_name in obj:
                        result[field_name] = FlextUtilitiesMapper.narrow_to_container(
                            obj[field_name]
                        )
                elif hasattr(obj, field_name):
                    result[field_name] = FlextUtilitiesMapper.narrow_to_container(
                        getattr(obj, field_name)
                    )
        return result

    @staticmethod
    def fields_multi(
        source: t.NormalizedValue | BaseModel | Mapping[str, t.NormalizedValue],
        spec: Mapping[str, t.NormalizedValue],
    ) -> t.ContainerMapping:
        """Extract multiple fields using specification dict.

        FLEXT Pattern: Simplified multi-field extraction (split from overloaded fields).

        Args:
            source: Source data (dict or object)
            spec: Field specification dict {field_name: default_value}

        Returns:
            dict with extracted values

        Example:
            data = FlextUtilitiesMapper.fields_multi(
                payload,
                {"name": "", "age": 0, "email": ""},
            )

        """
        result: dict[str, t.NormalizedValue] = {}
        for field_name, field_default in spec.items():
            value: t.NormalizedValue = FlextUtilitiesMapper.get(
                source,
                field_name,
                default=field_default,
            )
            result[field_name] = value if value is not None else field_default
        return result

    @staticmethod
    def filter_dict(
        source: t.ContainerMapping,
        predicate: Callable[[str, t.NormalizedValue], bool],
    ) -> t.ContainerMapping:
        """Filter dict by predicate function on key-value pairs.

        Args:
            source: Source dictionary
            predicate: Function(key, value) returning bool

        Returns:
            Filtered dictionary

        Example:
            >>> source = {"a": 1, "b": 2, "c": 3}
            >>> result = FlextUtilitiesMapper.filter_dict(
            ...     source, predicate=lambda k, v: v > 1
            ... )
            >>> # {"b": 2, "c": 3}

        """
        return {k: v for k, v in source.items() if predicate(k, v)}

    @staticmethod
    def find_callable[T](callables: Mapping[str, p.Predicate[T]], value: T) -> r[str]:
        """Find first matching callable key from dict of predicates.

        Iterates through mapping of named predicates and returns the key of
        the first predicate that returns True for the given value.

        Args:
            callables: Mapping of name → predicate function
            value: Value to test against predicates

        Returns:
            Key of matching predicate, or None if no match found

        Example:
            >>> predicates = {
            ...     "is_empty": lambda v: len(v) == 0,
            ...     "is_single": lambda v: len(v) == 1,
            ...     "is_multiple": lambda v: len(v) > 1,
            ... }
            >>> result = u.find_callable(predicates, [1, 2])
            >>> assert result == "is_multiple"

        """

        def build_predicate_call(predicate_fn: p.Predicate[T]) -> Callable[[], bool]:

            def _call_predicate() -> bool:
                return predicate_fn(value)

            return _call_predicate

        for name, predicate in callables.items():
            predicate_result = r[bool].create_from_callable(
                build_predicate_call(predicate)
            )
            if predicate_result.is_success and predicate_result.value:
                return r[str].ok(name)
        return r[str].fail("No callable predicate matched")

    @staticmethod
    def flat[T](
        items: list[list[T] | tuple[T, ...]]
        | list[list[T]]
        | list[tuple[T, ...]]
        | tuple[list[T], ...],
    ) -> list[T]:
        """Flatten nested lists (mnemonic: flat = flatten).

        Generic replacement for: [item for sublist in items for item in sublist]

        Args:
            items: Nested list/tuple structure

        Returns:
            Flattened list

        Example:
            flat_list = FlextUtilitiesMapper.flat([[1, 2], [3, 4]])
            # → [1, 2, 3, 4]

        """
        return [item for sublist in items for item in sublist]

    @staticmethod
    def get(
        data: p.AccessibleData, key: str, *, default: t.NormalizedValue = None
    ) -> t.NormalizedValue:
        """Unified get function for dict/object access with default.

        Generic replacement for: get(), get_str(), get_list()

        Automatically detects if data is dict or object and extracts value.
        Uses DSL conversion when default type indicates desired return type.

        Args:
            data: Source data (dict or object)
            key: Key/attribute name
            default: Default value if not found
                - str (e.g., "") -> returns str (generalized from get_str)
                - list[T] (e.g., []) -> returns list[T] (generalized from get_list)
                - Other -> returns T

        Returns:
            Extracted value or default (type inferred from default)

        Example:
            # String (generalized from get_str)
            name = FlextUtilitiesMapper.get(data, "name", default="")

            # List (generalized from get_list)
            models = FlextUtilitiesMapper.get(data, "models", default=[])

            # Generic
            port = FlextUtilitiesMapper.get(config, "port", default=c.Platform.DEFAULT_HTTP_PORT)

        """
        return FlextUtilitiesMapper._get_raw(data, key, default=default)

    @staticmethod
    def invert_dict(
        source: Mapping[str, str], *, handle_collisions: str = "last"
    ) -> Mapping[str, str]:
        """Invert dict mapping (values become keys, keys become values).

        **Generic replacement for**: Manual dict inversion

        Args:
            source: Source dictionary
            handle_collisions: How to handle duplicate values:
                - "first": Keep first occurrence
                - "last": Keep last occurrence (default)

        Returns:
            Inverted dictionary

        Example:
            >>> source = {"a": "x", "b": "y", "c": "x"}
            >>> result = FlextUtilitiesMapper.invert_dict(
            ...     source, handle_collisions="first"
            ... )
            >>> # {"x": "a", "y": "b"}  (first "a" kept)

        """
        if handle_collisions == "first":
            result: dict[str, str] = {}
            for k, v in source.items():
                if v not in result:
                    result[v] = k
            return result
        return {v: k for k, v in source.items()}

    @staticmethod
    def key_by[T, K](items: Sequence[T], key_func: Callable[[T], K]) -> Mapping[K, T]:
        """Create dict keyed by function result.

        Generic replacement for: {key_func(item): item for item in items}

        Args:
            items: Items to index
            key_func: Function to extract key from each item

        Returns:
            Dict mapping keys to items (last item wins if duplicate keys)

        Example:
            users_by_id = u.key_by(users, lambda u: u.id)
            # {1: User(id=1, ...), 2: User(id=2, ...)}

            users_by_email = u.key_by(users, lambda u: u.email.lower())

        """
        return {key_func(item): item for item in items}

    @staticmethod
    def map_dict_keys(
        source: t.ContainerMapping,
        key_mapping: Mapping[str, str],
        *,
        keep_unmapped: bool = True,
    ) -> r[t.ContainerMapping]:
        """Map dictionary keys using mapping specification.

        **Generic replacement for**: Key renaming in dicts

        Args:
            source: Source dictionary
            key_mapping: Mapping of old_key → new_key
            keep_unmapped: Keep keys not in mapping (default: True)

        Returns:
            r with remapped dictionary or error

        Example:
            >>> mapping = {"oldName": "newName", "foo": "bar"}
            >>> result = FlextUtilitiesMapper.map_dict_keys(
            ...     {"oldName": "value1", "foo": "value2", "other": "value3"}, mapping
            ... )
            >>> new_dict = result.value
            >>> # {"newName": "value1", "bar": "value2", "other": "value3"}

        """

        def _map_keys() -> t.ContainerMapping:
            result: dict[str, t.NormalizedValue] = {}
            for key, value in source.items():
                new_key = key_mapping.get(key)
                if new_key:
                    result[new_key] = value
                elif keep_unmapped:
                    result[key] = value
            return result

        mapped_result = r[t.ContainerMapping].create_from_callable(_map_keys)
        return mapped_result.fold(
            on_failure=lambda e: r[t.ContainerMapping].fail(
                f"Failed to map dict keys: {e}"
            ),
            on_success=lambda _: mapped_result,
        )

    @staticmethod
    def narrow_to_container(
        value: t.NormalizedValue | BaseModel | Mapping[str, t.NormalizedValue] | None,
    ) -> t.NormalizedValue:
        """Safely narrow any value to t.NormalizedValue (strict container type).

        Uses t.CONTAINER_TYPES for isinstance checks.
        BaseModel instances are converted to dict mapping via model_dump().
        If value is not a valid container, returns string representation.
        """
        if value is None:
            return None
        if isinstance(value, t.CONTAINER_TYPES):
            return value
        if isinstance(value, BaseModel):
            model_dict = value.model_dump()
            return {
                str(k): FlextUtilitiesMapper.narrow_to_container(v)
                for k, v in model_dict.items()
            }
        if isinstance(value, Mapping):
            return {
                str(k): FlextUtilitiesMapper.narrow_to_container(v)
                for k, v in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [FlextUtilitiesMapper.narrow_to_container(item) for item in value]
        return str(value)

    @staticmethod
    def narrow_to_general_value_type(
        value: t.NormalizedValue | BaseModel | Mapping[str, t.NormalizedValue] | None,
    ) -> t.NormalizedValue:
        """Deprecated alias; use narrow_to_container. Planned removal: v0.12."""
        warnings.warn(
            "narrow_to_general_value_type is deprecated; use narrow_to_container. "
            "Planned removal: v0.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FlextUtilitiesMapper.narrow_to_container(value)

    @staticmethod
    def normalize_context_values(
        context: m.ConfigMap | None,
        extra_kwargs: m.ConfigMap,
        **specific_fields: t.MetadataValue,
    ) -> Mapping[str, t.MetadataValue]:
        """Normalize and merge context values for exception handling.

        Convenience method for exception context processing.
        Uses process_context_data with metadata normalization.

        Args:
            context: Optional context mapping to normalize
            extra_kwargs: Additional kwargs to normalize and merge
            **specific_fields: Specific fields to add (field, value, config_key, etc.)

        Returns:
            Normalized metadata attribute dict

        """
        field_overrides_config: t.ContainerMapping = {
            k: FlextRuntime.normalize_to_container(v)
            for k, v in specific_fields.items()
        }
        raw_result: t.ContainerMapping = FlextUtilitiesMapper.process_context_data(
            primary_data=context,
            secondary_data=extra_kwargs,
            transformer=lambda value: FlextRuntime.normalize_to_metadata(value),
            field_overrides=field_overrides_config,
            merge_strategy="merge",
        )
        result: dict[str, t.MetadataValue] = {}
        for k, v in raw_result.items():
            result[k] = FlextRuntime.normalize_to_metadata(v)
        return result

    @staticmethod
    def omit[T](data: Mapping[str, T], *keys: str) -> Mapping[str, T]:
        """Omit specific keys from mapping.

        Generic replacement for: {k: v for k, v in data.items() if k not in keys}

        Args:
            data: Source mapping
            *keys: Keys to omit

        Returns:
            Dict without the specified keys

        Example:
            clean = u.omit(user_data, "password", "secret")
            # {"name": "John", "email": "john@test.com"}

        """
        keys_set = set(keys)
        return {k: v for k, v in data.items() if k not in keys_set}

    @staticmethod
    def or_[T](*values: T | None, default: T | None = None) -> r[T]:
        """Return first non-None value (mnemonic: or_ = fallback chain).

        Generic replacement for: value1 or value2 or default patterns

        Args:
            *values: Values to try in order
            default: Default if all are None

        Returns:
            First non-None value or default

        Example:
            port = FlextUtilitiesMapper.or_(config.get("port"), env.get("PORT"), default=c.Platform.DEFAULT_HTTP_PORT)

        """
        for value in values:
            if value is not None:
                return r[T].ok(value)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail("No non-None value found")

    @staticmethod
    def pick(
        data: p.AccessibleData, *keys: str, as_dict: bool = True
    ) -> t.ContainerMapping | t.ContainerList:
        """Pick multiple fields at once (mnemonic: pick = select fields).

        Generic replacement for: Multiple get() calls

        Args:
            data: Source data (dict or object)
            *keys: Field names to pick
            as_dict: If True, return dict; if False, return list

        Returns:
            Dict with picked fields or list of values

        Example:
            fields = FlextUtilitiesMapper.pick(data, "name", "email", "age")
            values = FlextUtilitiesMapper.pick(data, "x", "y", "z", as_dict=False)

        """
        if as_dict:
            return {k: FlextUtilitiesMapper.get(data, k) for k in keys}
        return [FlextUtilitiesMapper.get(data, k) for k in keys]

    @staticmethod
    def pluck(
        items: Sequence[t.ContainerMapping],
        key: str,
        default: t.NormalizedValue = None,
    ) -> t.ContainerList:
        """Extract single key from sequence of mappings.

        Generic replacement for: [item.get(key) for item in items]

        Args:
            items: Sequence of mappings
            key: Key to extract
            default: Default value if key not found

        Returns:
            List of values for the specified key

        Example:
            names = u.pluck(users, "name")
            # ["Alice", "Bob", "Charlie"]

            ages = u.pluck(users, "age", default=0)
            # [25, 30, 0]

        """
        values: t.ContainerList = []
        for item in items:
            raw_value = item.get(key, default)
            if raw_value is None:
                values.append(None)
            elif FlextUtilitiesGuards.is_container(raw_value):
                values.append(FlextUtilitiesMapper.narrow_to_container(raw_value))
            else:
                values.append(str(raw_value))
        return values

    @staticmethod
    def process_context_data(
        primary_data: m.ConfigMap
        | BaseModel
        | Mapping[str, t.NormalizedValue]
        | None = None,
        secondary_data: m.ConfigMap
        | BaseModel
        | Mapping[str, t.NormalizedValue]
        | None = None,
        *,
        transformer: t.MapperCallable | None = None,
        field_overrides: t.ContainerMapping | None = None,
        merge_strategy: str = "merge",
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> t.ContainerMapping:
        """Process and merge contextual data with flexible transformation options.

        Generic utility for processing context data across the FLEXT ecosystem.
        Handles conversion, transformation, filtering, and merging of contextual information.

        **Usage Examples:**
        ```python
        # Exception context processing
        context = FlextUtilitiesMapper.process_context_data(
            primary_data=user_context,
            secondary_data=extra_kwargs,
            field_overrides={"error_type": "ValidationError"},
            transformer=FlextRuntime.normalize_to_metadata_value,
        )

        # Configuration merging
        config = FlextUtilitiesMapper.process_context_data(
            primary_data=base_config,
            secondary_data=user_overrides,
            merge_strategy="deep_merge",
        )

        # API request processing
        request_data = FlextUtilitiesMapper.process_context_data(
            primary_data=request_body,
            secondary_data=query_params,
            filter_keys={"password", "secret"},
            transformer=str,
        )
        ```

        Args:
            primary_data: Main data source (dict, object, or None)
            secondary_data: Additional data to merge (dict, object, or None)
            transformer: Function to transform all values (default: identity)
            field_overrides: Specific field values to override/add
            merge_strategy: How to merge data ("merge", "primary_only", "secondary_only")
            filter_keys: Only include these keys if specified
            exclude_keys: Exclude these keys from result

        Returns:
            Processed and merged configuration dictionary

        """
        if transformer is None:

            def identity_transformer(
                x: t.NormalizedValue,
            ) -> t.NormalizedValue:
                return x

            transformer = identity_transformer
        result: dict[str, t.NormalizedValue] = {}
        if primary_data is not None:
            primary_source: Mapping[str, t.NormalizedValue | BaseModel] | None = None
            if isinstance(primary_data, m.ConfigMap):
                primary_source = primary_data.root
            else:
                primary_general = FlextUtilitiesMapper.narrow_to_container(primary_data)
                if FlextRuntime.is_dict_like(primary_general) and isinstance(
                    primary_general, Mapping
                ):
                    primary_source = {
                        str(key): item_value
                        for key, item_value in primary_general.items()
                    }
            if primary_source is not None:
                primary_dict = {
                    str(key): FlextUtilitiesMapper.narrow_to_container(value)
                    for key, value in primary_source.items()
                }
                transformed_primary = FlextUtilitiesMapper.transform_values(
                    primary_dict, transformer
                )
                result.update(transformed_primary)
        if secondary_data is not None and merge_strategy != "primary_only":
            secondary_source: Mapping[str, t.NormalizedValue | BaseModel] | None = None
            if isinstance(secondary_data, m.ConfigMap):
                secondary_source = secondary_data.root
            else:
                secondary_general = FlextUtilitiesMapper.narrow_to_container(
                    secondary_data
                )
                if FlextRuntime.is_dict_like(secondary_general) and isinstance(
                    secondary_general, Mapping
                ):
                    secondary_source = {
                        str(key): item_value
                        for key, item_value in secondary_general.items()
                    }
            if secondary_source is not None:
                secondary_dict = {
                    str(key): FlextUtilitiesMapper.narrow_to_container(value)
                    for key, value in secondary_source.items()
                }
                transformed_secondary = FlextUtilitiesMapper.transform_values(
                    secondary_dict, transformer
                )
                if merge_strategy == "secondary_only":
                    result = dict(transformed_secondary)
                elif merge_strategy == "merge":
                    result.update(transformed_secondary)
        if field_overrides:
            for key, value in field_overrides.items():
                transformed_value: t.NormalizedValue = transformer(value)
                result[key] = transformed_value
        if filter_keys:
            result = dict(
                FlextUtilitiesMapper.filter_dict(result, lambda k, _v: k in filter_keys)
            )
        if exclude_keys:
            result = dict(
                FlextUtilitiesMapper.filter_dict(
                    result, lambda k, _v: k not in exclude_keys
                )
            )
        return result

    @staticmethod
    def prop(
        key: str,
    ) -> Callable[[m.ConfigMap | BaseModel], t.NormalizedValue]:
        """Create a property accessor function (functional pattern).

        Returns a function that extracts a property/attribute from an object.
        Useful for functional programming patterns and DSL composition.

        Args:
            key: Property/attribute name to access

        Returns:
            Function that takes an object and returns its property value

        Example:
            >>> get_name = FlextUtilitiesMapper.prop("name")
            >>> name = get_name(user)  # Equivalent to user.name

            >>> # Use in pipelines
            >>> names = [get_name(u) for u in users]

        """

        def accessor(obj: m.ConfigMap | BaseModel) -> t.NormalizedValue:
            """Access property from object."""
            result = FlextUtilitiesMapper.get(obj, key)
            return result if result is not None else ""

        return accessor

    @staticmethod
    @overload
    def take(
        data_or_items: t.ContainerMapping | t.NormalizedValue,
        key_or_n: str,
        *,
        as_type: type | None = None,
        default: t.NormalizedValue = None,
        from_start: bool = True,
    ) -> t.NormalizedValue: ...

    @staticmethod
    @overload
    def take(
        data_or_items: t.ContainerMapping,
        key_or_n: int,
        *,
        as_type: type | None = None,
        default: t.NormalizedValue = None,
        from_start: bool = True,
    ) -> t.ContainerMapping: ...

    @staticmethod
    @overload
    def take(
        data_or_items: t.ContainerList | tuple[t.NormalizedValue, ...],
        key_or_n: int,
        *,
        as_type: type | None = None,
        default: t.NormalizedValue = None,
        from_start: bool = True,
    ) -> t.ContainerList: ...

    @staticmethod
    def take(
        data_or_items: t.ContainerMapping
        | t.NormalizedValue
        | t.ContainerList
        | tuple[t.NormalizedValue, ...],
        key_or_n: str | int,
        *,
        as_type: type | None = None,
        default: t.NormalizedValue = None,
        from_start: bool = True,
    ) -> t.ContainerMapping | t.ContainerList | t.NormalizedValue:
        """Unified take function (generalized from take_n).

        Generic replacement for: list slicing, dict slicing

        Automatically detects operation based on second argument type:
        - If key_or_n is str: extracts value from dict/object with type guard
        - If key_or_n is int: takes first N items from list/dict

        Args:
            data_or_items: Source data (dict/object) or items (list/dict)
            key_or_n: Key name (str) or number of items (int)
            as_type: Optional type to guard against (for extraction mode)
            default: Default value if not found or type mismatch
            from_start: If True, take from start; if False, take from end

        Returns:
            Extracted value with type guard or sliced items

        Example:
            # Extract value (original take behavior)
            port = FlextUtilitiesMapper.take(
                config, "port", as_type=int, default=c.Platform.DEFAULT_HTTP_PORT
            )
            name = FlextUtilitiesMapper.take(
                obj, "name", as_type=str, default="unknown"
            )

            # Take N items (generalized from take_n)
            keys = FlextUtilitiesMapper.take(plugins_dict, 10)
            items = FlextUtilitiesMapper.take(items_list, 5)

        """
        if isinstance(key_or_n, str):
            if isinstance(
                data_or_items, Mapping
            ) and FlextUtilitiesGuards.is_configuration_mapping(data_or_items):
                data: p.AccessibleData = data_or_items
            elif isinstance(data_or_items, BaseModel):
                data = data_or_items
            else:
                return default if default is not None else ""
            key = key_or_n
            value = FlextUtilitiesMapper.get(data, key, default=default)
            if value is None:
                return default if default is not None else ""
            if as_type is not None and (not isinstance(value, as_type)):
                return default if default is not None else ""
            return value
        n = key_or_n
        if isinstance(data_or_items, Mapping):
            keys = list(data_or_items.keys())
            selected_keys = keys[:n] if from_start else keys[-n:]
            return {k: data_or_items[k] for k in selected_keys}
        if isinstance(data_or_items, (list, tuple)):
            items_list: t.ContainerList = [
                FlextUtilitiesMapper.narrow_to_container(
                    FlextUtilitiesMapper.narrow_to_container(item)
                )
                for item in data_or_items
            ]
            return items_list[:n] if from_start else items_list[-n:]
        return default if default is not None else ""

    @staticmethod
    def transform(
        source: t.ContainerMapping | m.ConfigMap,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: Mapping[str, str] | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> r[t.ContainerMapping]:
        """Transform dictionary with multiple options.

        Args:
            source: Source dictionary to transform.
            normalize: Normalize values using cache normalization.
            strip_none: Remove keys with None values.
            strip_empty: Remove keys with empty values (empty strings, lists, dicts).
            map_keys: Dictionary mapping old keys to new keys.
            filter_keys: Set of keys to keep (all others removed).
            exclude_keys: Set of keys to remove.

        Returns:
            r with transformed dictionary.

        Example:
            >>> result = FlextUtilitiesMapper.transform(
            ...     {"old": "value", "remove": None},
            ...     map_keys={"old": "new"},
            ...     strip_none=True,
            ... )
            >>> transformed = result.map_or({})  # {"new": "value"}

        """
        transform_result = r[t.ContainerMapping].create_from_callable(
            lambda: FlextUtilitiesMapper._apply_transform_steps(
                {
                    str(k): FlextUtilitiesMapper.narrow_to_container(v)
                    for k, v in source.root.items()
                }
                if isinstance(source, m.ConfigMap)
                else dict(source),
                normalize=normalize,
                map_keys=map_keys,
                filter_keys=filter_keys,
                exclude_keys=exclude_keys,
                strip_none=strip_none,
                strip_empty=strip_empty,
            )
        )
        return transform_result.fold(
            on_failure=lambda e: r[t.ContainerMapping].fail(f"Transform failed: {e}"),
            on_success=lambda _: transform_result,
        )

    @staticmethod
    def transform_values(
        source: t.ContainerMapping,
        transformer: Callable[[t.NormalizedValue], t.NormalizedValue],
    ) -> t.ContainerMapping:
        """Transform all values in dict using transformer function.

        **Generic replacement for**: Manual dict value transformations

        Args:
            source: Source dictionary
            transformer: Function to apply to each value

        Returns:
            Dictionary with transformed values

        Example:
            >>> source = {"a": "hello", "b": "world"}
            >>> result = FlextUtilitiesMapper.transform_values(
            ...     source, lambda v: str(v).upper()
            ... )
            >>> # {"a": "HELLO", "b": "WORLD"}

        """
        return {k: transformer(v) for k, v in source.items()}


__all__ = ["FlextUtilitiesMapper"]
