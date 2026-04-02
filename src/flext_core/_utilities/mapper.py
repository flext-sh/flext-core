"""Utilities module - FlextUtilitiesMapper.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from itertools import starmap
from typing import ClassVar, overload

from pydantic import BaseModel

from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
from flext_core.protocols import p
from flext_core.result import FlextResult as r
from flext_core.typings import t


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
    def _apply_exclude_keys(
        result: t.ContainerMapping,
        *,
        exclude_keys: set[str] | None,
    ) -> t.ContainerMapping:
        """Apply exclude keys step."""
        if exclude_keys:
            filtered_result: t.MutableContainerMapping = dict(result)
            for key in exclude_keys:
                _ = filtered_result.pop(key, None)
            return filtered_result
        return result

    @staticmethod
    def _apply_filter_keys(
        result: t.ContainerMapping,
        *,
        filter_keys: set[str] | None,
    ) -> t.ContainerMapping:
        """Apply filter keys step."""
        if filter_keys:
            filtered_dict: t.MutableContainerMapping = {}
            for key in filter_keys:
                if key in result:
                    filtered_dict[key] = result[key]
            return filtered_dict
        return result

    @staticmethod
    def _apply_map_keys(
        result: t.ContainerMapping,
        *,
        map_keys: t.StrMapping | None,
    ) -> t.ContainerMapping:
        """Apply map keys step."""
        if map_keys:
            mapped: r[t.ContainerMapping] = FlextUtilitiesMapper.map_dict_keys(
                result,
                map_keys,
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
        result: t.ContainerMapping,
        *,
        normalize: bool,
    ) -> t.ContainerMapping:
        """Apply normalize step."""
        if normalize:
            normalized: t.NormalizedValue = FlextUtilitiesCache.normalize_component(
                result,
            )
            if FlextUtilitiesGuardsTypeCore.is_mapping(normalized):
                normalized_result: t.MutableContainerMapping = {}
                for key, value in normalized.items():
                    normalized_result[str(key)] = (
                        FlextUtilitiesMapper.narrow_to_container(value)
                    )
                return normalized_result
        return result

    @staticmethod
    def _apply_strip_empty(
        result: t.ContainerMapping,
        *,
        strip_empty: bool,
    ) -> t.ContainerMapping:
        """Apply strip empty step."""
        if strip_empty:
            return FlextUtilitiesMapper.filter_dict(
                result,
                lambda _k, v: v not in ("", [], {}, None),
            )
        return result

    @staticmethod
    def _apply_strip_none(
        result: t.ContainerMapping,
        *,
        strip_none: bool,
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
        map_keys: t.StrMapping | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
    ) -> t.ContainerMapping:
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
    def _narrow_list_items(items: t.ContainerList) -> t.ContainerList:
        """Narrow each item in a sequence to a container type."""
        return [FlextUtilitiesMapper.narrow_to_container(item) for item in items]

    @staticmethod
    def _resolve_field_key_func(field_name: str) -> Callable[[t.NormalizedValue], str]:
        """Build a key function extracting a named field from Mapping or BaseModel."""

        def _key_func(item: t.NormalizedValue) -> str:
            if FlextUtilitiesGuardsTypeCore.is_mapping(item):
                return str(item.get(field_name, ""))
            if FlextUtilitiesGuardsTypeModel.is_pydantic_model(item) and hasattr(
                item, field_name
            ):
                return str(getattr(item, field_name))
            return ""

        return _key_func

    @staticmethod
    def _apply_callable_over_collection(
        current: t.NormalizedValue,
        func: t.MapperCallable,
    ) -> t.NormalizedValue:
        """Apply callable to each element of a collection or to a scalar."""
        if isinstance(current, (list, tuple)):
            seq_current: t.ContainerList = current
            return [
                FlextUtilitiesMapper.narrow_to_container(
                    func(FlextUtilitiesMapper.narrow_to_container(x)),
                )
                for x in seq_current
            ]
        if FlextUtilitiesGuardsTypeCore.is_mapping(current):
            current_dict: t.ContainerMapping = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(current)
            )
            return {
                k: FlextUtilitiesMapper.narrow_to_container(func(v))
                for k, v in current_dict.items()
            }
        current_general = FlextUtilitiesMapper.narrow_to_container(current)
        return FlextUtilitiesMapper.narrow_to_container(func(current_general))

    @staticmethod
    def _op_chunk(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply chunk operation to split into sublists."""
        if not isinstance(current, (list, tuple)):
            return current
        chunk_size = ops["chunk"]
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            return current
        current_list: t.ContainerList = FlextUtilitiesMapper._narrow_list_items(current)
        chunked: t.MutableContainerList = []
        for i in range(0, len(current_list), chunk_size):
            chunk: t.ContainerList = current_list[i : i + chunk_size]
            chunked.append(chunk)
        return chunked

    @staticmethod
    def _op_convert(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply convert operation."""
        convert_func_result = FlextUtilitiesMapper._get_callable_from_dict(
            ops,
            "convert",
        )
        if convert_func_result.is_failure:
            return current
        convert_callable_raw: t.MapperCallable = convert_func_result.value
        convert_default_raw = ops.get("convert_default")
        fallback: t.NormalizedValue = (
            convert_default_raw if not callable(convert_default_raw) else None
        )

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
                "list": list[t.NormalizedValue](),
                "dict": dict[str, t.NormalizedValue](),
                "tuple": (),
                "set": list[t.NormalizedValue](),
            }
            fallback = converter_defaults.get(converter_name, current)

        def _convert(value: t.NormalizedValue) -> t.NormalizedValue:
            try:
                raw: t.NormalizedValue = FlextUtilitiesMapper.narrow_to_container(
                    convert_callable_raw(value),
                )
                return raw if raw is not None else fallback
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                return fallback

        match current:
            case list():
                return [
                    _convert(FlextUtilitiesMapper.narrow_to_container(item))
                    for item in current
                ]
            case tuple():
                return tuple(
                    _convert(FlextUtilitiesMapper.narrow_to_container(item))
                    for item in current
                )
            case _:
                return _convert(current)

    @staticmethod
    def _ensure_as_list(
        current: t.NormalizedValue,
        default_val: t.NormalizedValue,
        *,
        stringify: bool = False,
    ) -> t.NormalizedValue:
        """Ensure value is a list, optionally stringifying items."""
        narrow = FlextUtilitiesMapper.narrow_to_container
        if isinstance(current, list):
            items: t.ContainerList = current
            return (
                [str(narrow(x)) for x in items]
                if stringify
                else [narrow(x) for x in items]
            )
        if current is None:
            return default_val
        item = narrow(current)
        return [str(item)] if stringify else [item]

    @staticmethod
    def _op_ensure(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply ensure operation."""
        ensure_type = FlextUtilitiesMapper._get_str_from_dict(ops, "ensure", "")
        ensure_default_raw = ops.get("ensure_default")
        ensure_default_val: t.NormalizedValue = (
            FlextUtilitiesMapper.narrow_to_container(ensure_default_raw)
            if ensure_default_raw is not None and not callable(ensure_default_raw)
            else None
        )
        default_map: t.ContainerMapping = {
            "str_list": list[t.NormalizedValue](),
            "dict": dict[str, t.NormalizedValue](),
            "list": list[t.NormalizedValue](),
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
    def _op_filter(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply filter operation."""
        filter_pred_result = FlextUtilitiesMapper._get_callable_from_dict(ops, "filter")
        if filter_pred_result.is_failure:
            return current
        filter_pred_callable = filter_pred_result.value

        def filter_pred(value: t.NormalizedValue) -> bool:
            return bool(filter_pred_callable(value))

        if isinstance(current, (list, tuple)):
            seq_current: t.ContainerList = current
            return [
                FlextUtilitiesMapper.narrow_to_container(x)
                for x in seq_current
                if filter_pred(FlextUtilitiesMapper.narrow_to_container(x))
            ]
        if FlextUtilitiesGuardsTypeCore.is_mapping(current):
            current_dict: t.ContainerMapping = (
                FlextUtilitiesMapper._narrow_to_configuration_dict(current)
            )
            return FlextUtilitiesMapper.filter_dict(
                current_dict,
                lambda _k, v: bool(filter_pred(v)),
            )
        return default_val if not bool(filter_pred(current)) else current

    @staticmethod
    def _group_by_field(
        field_name: str,
        current_items: t.ContainerList,
        current_list: t.ContainerList,
    ) -> MutableMapping[str, t.MutableContainerList]:
        """Group items by a named field (BaseModel attr or Mapping key)."""
        grouped: MutableMapping[str, t.MutableContainerList] = {}
        for orig_item, item in zip(current_items, current_list, strict=False):
            match orig_item:
                case BaseModel() if hasattr(orig_item, field_name):
                    key_raw = getattr(orig_item, field_name)
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
    def _group_by_callable(
        group_callable: t.MapperCallable,
        current_list: t.ContainerList,
    ) -> MutableMapping[str, t.MutableContainerList]:
        """Group items using a callable key function."""
        grouped: MutableMapping[str, t.MutableContainerList] = {}
        for item in current_list:
            try:
                group_key: t.NormalizedValue | None = group_callable(item)
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
                continue
            if group_key is None:
                continue
            grouped.setdefault(str(group_key), []).append(item)
        return grouped

    @staticmethod
    def _op_group(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
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
    def _op_map(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
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
    def _op_normalize(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply normalize operation."""
        normalize_case = FlextUtilitiesMapper._get_str_from_dict(ops, "normalize", "")
        if isinstance(current, str):
            return current.lower() if normalize_case == "lower" else current.upper()
        if isinstance(current, (list, tuple)):
            seq_current: t.ContainerList = current
            result: t.MutableContainerList = []
            for x in seq_current:
                x_general = FlextUtilitiesMapper.narrow_to_container(x)
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
    def _op_process(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        default_val: t.NormalizedValue,
        on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply process operation."""
        process_func_result = FlextUtilitiesMapper._get_callable_from_dict(
            ops,
            "process",
        )
        if process_func_result.is_failure:
            return current
        process_callable: t.MapperCallable = process_func_result.value

        def _process_current() -> t.NormalizedValue:
            return FlextUtilitiesMapper._apply_callable_over_collection(
                current,
                process_callable,
            )

        process_result: r[t.NormalizedValue] = r[
            t.NormalizedValue
        ].create_from_callable(_process_current)
        if process_result.is_failure:
            return default_val if on_error == "stop" else current
        process_val: t.NormalizedValue = process_result.value
        return process_val if process_val is not None else current

    @staticmethod
    def _op_slice(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply slice operation."""
        if not isinstance(current, (list, tuple)):
            return current
        current_items: t.ContainerList = current
        slice_spec_raw = ops["slice"]
        slice_spec: t.NormalizedValue = (
            slice_spec_raw if not callable(slice_spec_raw) else None
        )
        min_slice_length = 2
        if (
            isinstance(slice_spec, (list, tuple))
            and len(slice_spec) >= min_slice_length
        ):
            start_raw: t.NormalizedValue = slice_spec[0]
            end_raw: t.NormalizedValue = slice_spec[1]
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
    def _as_original_type(
        sorted_items: t.ContainerList,
        original: t.ContainerList | tuple[t.NormalizedValue, ...],
    ) -> t.ContainerList | tuple[t.NormalizedValue, ...]:
        """Preserve the original collection type (list vs tuple) after sorting."""
        return list(sorted_items) if isinstance(original, list) else tuple(sorted_items)

    @staticmethod
    def _op_sort(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
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
                    FlextUtilitiesMapper.narrow_to_container(
                        item
                        if FlextUtilitiesGuardsTypeCore.is_primitive(item)
                        or item is None
                        else str(item),
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

                def sort_key(item: t.NormalizedValue) -> str:
                    return str(sort_fn(item))

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
    def _op_transform(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        default_val: t.NormalizedValue,
        on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply transform operation."""
        if not FlextUtilitiesGuards.is_type(
            current,
            "mapping",
        ):
            return current
        transform_opts_val = ops["transform"]
        transform_opts_raw: t.NormalizedValue = (
            transform_opts_val if not callable(transform_opts_val) else None
        )
        if not FlextUtilitiesGuardsTypeCore.is_mapping(transform_opts_raw):
            return current
        transform_opts = FlextUtilitiesMapper._narrow_to_configuration_dict(
            FlextUtilitiesMapper.narrow_to_container(transform_opts_raw),
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
    def _op_unique(
        current: t.NormalizedValue,
        ops: Mapping[str, t.MapperInput],
        _default_val: t.NormalizedValue,
        _on_error: str,
    ) -> t.NormalizedValue:
        """Helper: Apply unique operation to remove duplicates."""
        if not ops.get("unique"):
            return current
        if not isinstance(current, (list, tuple)):
            return current
        current_list_unique: t.ContainerList = FlextUtilitiesMapper._narrow_list_items(
            current,
        )
        seen: set[t.NormalizedValue | str] = set()
        unique_list: t.MutableContainerList = []
        for item in current_list_unique:
            item_hashable: t.NormalizedValue | str = (
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
    ) -> t.NormalizedValue:
        """Extract field value from dict or model for pyrefly type inference."""
        if FlextUtilitiesGuardsTypeCore.is_mapping(item):
            dict_item: t.MutableContainerMapping = {}
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
    def _resolve_raw_value(
        raw: t.NormalizedValue,
        key_part: str,
    ) -> r[t.NormalizedValue]:
        """Wrap a raw value into a Result: fail on None, narrow containers, stringify rest."""
        if raw is None:
            return r[t.NormalizedValue].fail(f"found_none:{key_part}")
        if FlextUtilitiesGuards.is_container(raw):
            return r[t.NormalizedValue].ok(
                FlextUtilitiesMapper.narrow_to_container(raw),
            )
        return r[t.NormalizedValue].ok(str(raw))

    @staticmethod
    def _extract_get_value(
        current: t.ValueOrModel | p.HasModelDump | p.ValidatorSpec | t.ContainerMapping,
        key_part: str,
    ) -> r[t.NormalizedValue]:
        """Get raw value from dict/object/model, returning found_none or not-found failures."""
        if isinstance(current, Mapping):
            mapping_obj: t.ContainerMapping = current
            if key_part in mapping_obj:
                return FlextUtilitiesMapper._resolve_raw_value(
                    mapping_obj[key_part], key_part
                )
            return r[t.NormalizedValue].fail(f"Key '{key_part}' not found in Mapping")
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
                    val = FlextUtilitiesMapper.narrow_to_container(model_dict[key_part])
                    if val is None:
                        return r[t.NormalizedValue].fail(f"found_none:{key_part}")
                    return r[t.NormalizedValue].ok(val)
        return r[t.NormalizedValue].fail(f"Key '{key_part}' not found")

    @staticmethod
    def _extract_handle_array_index(
        current: t.NormalizedValue,
        array_match: str,
    ) -> r[t.NormalizedValue]:
        """Handle array indexing with negative index support."""
        if not isinstance(current, (list, tuple)):
            return r[t.NormalizedValue].fail("Not a sequence")
        sequence: t.ContainerList = FlextUtilitiesMapper._narrow_to_sequence(current)
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
        default: t.NormalizedValue = None,
    ) -> t.NormalizedValue:
        """Internal helper for raw get without DSL conversion."""
        fallback: t.NormalizedValue = default if default is not None else ""
        raw_value: t.NormalizedValue = None
        match data:
            case dict() | Mapping():
                raw_value = FlextUtilitiesMapper.narrow_to_container(data.get(key))
            case t.ConfigMap() | t.Dict():
                raw_value = FlextUtilitiesMapper.narrow_to_container(data.root.get(key))
            case _ if hasattr(data, key):
                return FlextUtilitiesMapper.narrow_to_container(getattr(data, key))
            case _:
                return fallback
        return (
            FlextUtilitiesMapper.narrow_to_container(raw_value)
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
        value: t.NormalizedValue | t.ContainerMapping,
    ) -> t.ContainerMapping:
        """Safely narrow t.NormalizedValue to ConfigurationDict with runtime validation."""
        if FlextUtilitiesGuards.is_configuration_dict(value):
            normalized_dict: t.MutableContainerMapping = {}
            for key, item in value.items():
                normalized_dict[str(key)] = FlextUtilitiesMapper.narrow_to_container(
                    item,
                )
            return normalized_dict
        error_msg = f"Cannot narrow {value.__class__.__name__} to ConfigurationDict"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_sequence(
        value: t.NormalizedValue | t.ContainerList,
    ) -> t.ContainerList:
        """Safely narrow t.NormalizedValue to t.ContainerList."""
        if isinstance(value, (list, tuple)):
            narrowed_items: t.MutableContainerList = []
            for item_raw in value:
                item = FlextUtilitiesMapper.narrow_to_container(item_raw)
                narrowed_item = FlextUtilitiesMapper.narrow_to_container(item)
                narrowed_items.append(narrowed_item)
            return narrowed_items
        error_msg = f"Cannot narrow {value.__class__.__name__} to Sequence"
        raise TypeError(error_msg)

    @staticmethod
    def _narrow_to_string_keyed_dict(
        value: t.NormalizedValue | t.ContainerMapping,
    ) -> t.ContainerMapping:
        """Narrow to ConfigurationDict with string keys and container values."""
        if FlextUtilitiesGuardsTypeCore.is_mapping(value):
            result: t.MutableContainerMapping = {}
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
    def _get_numeric_field(
        item: BaseModel | Mapping[str, t.NormalizedValue],
        field_name: str,
    ) -> t.Numeric | None:
        """Extract a numeric field value from a BaseModel or Mapping-like object."""
        if isinstance(item, BaseModel):
            val_raw = FlextUtilitiesMapper._extract_field_value(item, field_name)
            return val_raw if isinstance(val_raw, (int, float)) else None
        if isinstance(item, Mapping):
            val = item.get(field_name)
            return val if isinstance(val, (int, float)) else None
        return None

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
        default: t.NormalizedValue = None,
        on_error: str = "stop",
    ) -> t.NormalizedValue:
        """Compose operations via DSL dict applied in order: ensure/filter/map/normalize/convert/transform/process/group/sort/unique/slice/chunk."""
        narrowed_value = FlextUtilitiesMapper.narrow_to_container(value)
        if ops is None:
            return narrowed_value
        current: t.NormalizedValue = narrowed_value
        default_val: t.NormalizedValue = (
            default if default is not None else narrowed_value
        )
        for op_key, op_method_name in FlextUtilitiesMapper._BUILD_OPS:
            if op_key in ops:
                handler: Callable[
                    [
                        t.NormalizedValue,
                        Mapping[str, t.MapperInput],
                        t.NormalizedValue,
                        str,
                    ],
                    t.NormalizedValue,
                ] = getattr(FlextUtilitiesMapper, op_method_name)
                current = handler(current, ops, default_val, on_error)
        return current

    @staticmethod
    def _deep_eq_values(
        val_a: t.NormalizedValue,
        val_b: t.NormalizedValue,
    ) -> bool:
        """Recursive deep equality for any two NormalizedValue items."""
        if val_a is val_b:
            return True
        match (val_a, val_b):
            case (None, None):
                return True
            case (None, _) | (_, None):
                return False
            case (Mapping() as ma, Mapping() as mb):
                dict_a = FlextUtilitiesMapper._narrow_to_configuration_dict(ma)
                dict_b = FlextUtilitiesMapper._narrow_to_configuration_dict(mb)
                return FlextUtilitiesMapper.deep_eq(dict_a, dict_b)
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
    def deep_eq(a: t.ContainerMapping, b: t.ContainerMapping) -> bool:
        """Recursive deep equality for nested dicts/lists/primitives."""
        if a is b:
            return True
        if len(a) != len(b):
            return False
        for key, val_a in a.items():
            if key not in b:
                return False
            if not FlextUtilitiesMapper._deep_eq_values(val_a, b[key]):
                return False
        return True

    @staticmethod
    def ensure_str(value: t.NormalizedValue, default: str = "") -> str:
        """Convert value to str, returning default if None."""
        if value is None:
            return default
        if isinstance(value, str):
            return str(value)
        return str(value)

    @staticmethod
    def _extract_fail_or_default(
        msg: str,
        *,
        default: t.NormalizedValue,
        required: bool,
    ) -> r[t.NormalizedValue]:
        """Return fail (required) or ok(default) / fail (no default) for extract paths."""
        if required:
            return r[t.NormalizedValue].fail(msg)
        if default is None:
            return r[t.NormalizedValue].fail(f"{msg} and default is None")
        return r[t.NormalizedValue].ok(default)

    @staticmethod
    def _extract_resolve_path_part(
        current: t.ValueOrModel,
        part: str,
        *,
        path_context: str,
        default: t.NormalizedValue,
        required: bool,
    ) -> tuple[t.ValueOrModel, r[t.NormalizedValue] | None]:
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
            narrowed_for_index = FlextUtilitiesMapper.narrow_to_container(next_val)
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
    def extract(
        data: p.AccessibleData,
        path: str,
        *,
        default: t.NormalizedValue = None,
        required: bool = False,
        separator: str = ".",
    ) -> r[t.NormalizedValue]:
        """Extract nested value via dot-notation path with array index support (e.g. "user.addresses[0].city")."""
        try:
            parts = path.split(separator)
            current: t.ValueOrModel = None
            match data:
                case BaseModel():
                    current = data
                case Mapping():
                    current = FlextUtilitiesMapper.narrow_to_container(data)
                case p.HasModelDump():
                    current = FlextUtilitiesMapper.narrow_to_container(
                        data.model_dump()
                    )
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
                return r[t.NormalizedValue].ok(
                    FlextUtilitiesMapper.narrow_to_container(current),
                )
            return r[t.NormalizedValue].ok(str(current))
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as e:
            return r[t.NormalizedValue].fail(f"Extract failed: {e}")

    @staticmethod
    def filter_dict(
        source: t.ContainerMapping,
        predicate: Callable[[str, t.NormalizedValue], bool],
    ) -> t.ContainerMapping:
        """Filter dict keeping only entries where predicate(key, value) is True."""
        return {k: v for k, v in source.items() if predicate(k, v)}

    @staticmethod
    def map_get(
        data: p.AccessibleData,
        key: str,
        *,
        default: t.NormalizedValue = None,
    ) -> t.NormalizedValue:
        """Get value by key from dict/object, returning default if missing."""
        return FlextUtilitiesMapper._get_raw(data, key, default=default)

    @staticmethod
    def map_dict_keys(
        source: t.ContainerMapping,
        key_mapping: t.StrMapping,
        *,
        keep_unmapped: bool = True,
    ) -> r[t.ContainerMapping]:
        """Rename dict keys using old_key->new_key mapping."""

        def _map_keys() -> t.ContainerMapping:
            result: t.MutableContainerMapping = {}
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
                f"Failed to map dict keys: {e}",
            ),
            on_success=lambda _: mapped_result,
        )

    @staticmethod
    def _narrow_untyped_dict(
        raw: Mapping[str, t.MetadataOrValue | BaseModel],
    ) -> t.ContainerMapping:
        """Convert heterogeneous dict to NormalizedValue dict."""
        result: t.MutableContainerMapping = {}
        for k in list(raw.keys()):
            v = raw[k]
            if v is None:
                result[str(k)] = None
            elif isinstance(v, (BaseModel, *t.CONTAINER_TYPES, list, dict, tuple)):
                result[str(k)] = FlextUtilitiesMapper.narrow_to_container(v)
            else:
                result[str(k)] = str(v)
        return result

    @staticmethod
    def _narrow_untyped_list(
        raw: Sequence[t.MetadataOrValue | BaseModel],
    ) -> t.ContainerList:
        """Convert heterogeneous list to NormalizedValue list."""
        result: t.MutableContainerList = []
        for item in raw:
            if isinstance(item, (BaseModel, *t.CONTAINER_TYPES, list, dict, tuple)):
                result.append(FlextUtilitiesMapper.narrow_to_container(item))
            elif item is not None:
                result.append(str(item))
        return result

    @staticmethod
    def narrow_to_container(
        value: t.MetadataOrValue
        | t.NormalizedValue
        | BaseModel
        | t.ContainerMapping
        | Mapping[str, t.ValueOrModel]
        | Mapping[str, t.Scalar | t.ScalarList]
        | p.HasModelDump
        | p.ValidatorSpec
        | None,
    ) -> t.NormalizedValue:
        """Narrow any value to t.NormalizedValue; non-containers become str."""
        if value is None:
            return None
        if isinstance(value, t.CONTAINER_TYPES):
            return value
        match value:
            case BaseModel() if FlextUtilitiesGuardsTypeModel.is_pydantic_model(value):
                model_dict = value.model_dump()
                return {
                    str(k): FlextUtilitiesMapper.narrow_to_container(v)
                    for k, v in model_dict.items()
                }
            case dict():
                return FlextUtilitiesMapper._narrow_untyped_dict(value)
            case list():
                return FlextUtilitiesMapper._narrow_untyped_list(value)
            case tuple():
                return FlextUtilitiesMapper._narrow_untyped_list(list(value))
            case Mapping():
                return FlextUtilitiesMapper._narrow_untyped_dict(dict(value.items()))
            case Sequence() if not isinstance(value, (str, bytes)):
                return FlextUtilitiesMapper._narrow_untyped_list(list(value))
            case _:
                return str(value)

    @staticmethod
    def prop(
        key: str,
    ) -> Callable[[t.ConfigModelInput], t.NormalizedValue]:
        """Return an accessor function that extracts the named property from an object."""

        def accessor(obj: t.ConfigModelInput) -> t.NormalizedValue:
            """Access property from object."""
            result = FlextUtilitiesMapper.map_get(obj, key)
            return result if result is not None else ""

        return accessor

    @staticmethod
    def _take_by_key(
        data_or_items: t.ContainerMapping
        | t.NormalizedValue
        | t.ContainerList
        | tuple[t.NormalizedValue, ...],
        key: str,
        *,
        as_type: type | None,
        default: t.NormalizedValue,
    ) -> t.NormalizedValue:
        """Extract a value by key from a Mapping or BaseModel."""
        fallback: t.NormalizedValue = default if default is not None else ""
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
    def _take_n_items(
        data_or_items: t.ContainerMapping
        | t.NormalizedValue
        | t.ContainerList
        | tuple[t.NormalizedValue, ...],
        n: int,
        *,
        default: t.NormalizedValue,
        from_start: bool,
    ) -> t.ContainerMapping | t.ContainerList | t.NormalizedValue:
        """Take N items from a Mapping or Sequence."""
        match data_or_items:
            case Mapping():
                keys = list(data_or_items.keys())
                selected = keys[:n] if from_start else keys[-n:]
                return {k: data_or_items[k] for k in selected}
            case list() | tuple():
                items_list: t.ContainerList = FlextUtilitiesMapper._narrow_list_items(
                    data_or_items,
                )
                return items_list[:n] if from_start else items_list[-n:]
            case _:
                return default if default is not None else ""

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
    def transform(
        source: t.ContainerMapping | t.ConfigMap,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: t.StrMapping | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> r[t.ContainerMapping]:
        """Apply normalize/strip_none/strip_empty/map_keys/filter_keys/exclude_keys to a dict."""
        transform_result = r[t.ContainerMapping].create_from_callable(
            lambda: FlextUtilitiesMapper._apply_transform_steps(
                {
                    str(k): FlextUtilitiesMapper.narrow_to_container(v)
                    for k, v in source.root.items()
                }
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
            on_failure=lambda e: r[t.ContainerMapping].fail(f"Transform failed: {e}"),
            on_success=lambda _: transform_result,
        )


__all__ = ["FlextUtilitiesMapper"]
