"""Utilities module - FlextUtilitiesMapper.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSequence, Sequence
from itertools import starmap
from pathlib import Path
from typing import cast

from flext_core import (
    FlextUtilitiesCache,
    FlextUtilitiesCollection,
    FlextUtilitiesGuards,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    c,
    e,
    m,
    p,
    r,
    t,
)
from flext_core.runtime import FlextRuntime


class FlextUtilitiesMapper:
    """Data structure mapping and transformation utilities.

    Provides generic methods for mapping between data structures, building
    objects from flags/mappings, and transforming dict/list structures.
    """

    @staticmethod
    def _apply_exclude_keys(
        result: t.RecursiveContainerMapping,
        *,
        exclude_keys: set[str] | None,
    ) -> t.MutableRecursiveContainerMapping | t.RecursiveContainerMapping:
        """Apply exclude keys step."""
        if exclude_keys:
            filtered_result: t.MutableRecursiveContainerMapping = dict(result)
            for key in exclude_keys:
                _ = filtered_result.pop(key, None)
            return filtered_result
        return result

    @staticmethod
    def _apply_filter_keys(
        result: t.RecursiveContainerMapping,
        *,
        filter_keys: set[str] | None,
    ) -> t.MutableRecursiveContainerMapping | t.RecursiveContainerMapping:
        """Apply filter keys step."""
        if filter_keys:
            filtered_dict: t.MutableRecursiveContainerMapping = {}
            for key in filter_keys:
                if key in result:
                    filtered_dict[key] = result[key]
            return filtered_dict
        return result

    @staticmethod
    def _apply_map_keys(
        result: t.RecursiveContainerMapping,
        *,
        map_keys: t.StrMapping | None,
    ) -> t.MutableRecursiveContainerMapping | t.RecursiveContainerMapping:
        """Apply map keys step."""
        if map_keys:
            mapped_result: r[t.MutableRecursiveContainerMapping] = (
                FlextUtilitiesMapper.map_dict_keys(
                    result,
                    map_keys,
                )
            )
            if mapped_result.success:
                return mapped_result.unwrap_or(dict(result))
        return result

    @staticmethod
    def _apply_normalize(
        result: t.RecursiveContainerMapping,
        *,
        normalize: bool,
    ) -> t.MutableRecursiveContainerMapping | t.RecursiveContainerMapping:
        """Apply normalize step."""
        if normalize:
            normalized = FlextUtilitiesCache.normalize_component(
                {str(k): v for k, v in result.items()},
            )
            if FlextUtilitiesGuardsTypeCore.mapping(normalized):
                normalized_result: t.MutableRecursiveContainerMapping = {}
                for key, value in normalized.items():
                    normalized_result[str(key)] = value
                return normalized_result
        return result

    @staticmethod
    def _apply_strip_empty(
        result: t.RecursiveContainerMapping,
        *,
        strip_empty: bool,
    ) -> t.RecursiveContainerMapping:
        """Apply strip empty step."""
        if strip_empty:
            return FlextUtilitiesCollection.filter(
                result,
                lambda v: v not in ("", [], {}, None),
            )
        return result

    @staticmethod
    def _apply_strip_none(
        result: t.RecursiveContainerMapping,
        *,
        strip_none: bool,
    ) -> t.RecursiveContainerMapping:
        """Apply strip none step."""
        if strip_none:
            return FlextUtilitiesCollection.filter(result, lambda v: v is not None)
        return result

    @staticmethod
    def _apply_transform_steps(
        result: t.RecursiveContainerMapping,
        *,
        normalize: bool,
        map_keys: t.StrMapping | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
    ) -> t.MutableRecursiveContainerMapping | t.RecursiveContainerMapping:
        """Apply transform steps to result dict."""
        step: t.MutableRecursiveContainerMapping | t.RecursiveContainerMapping = (
            FlextUtilitiesMapper._apply_normalize(result, normalize=normalize)
        )
        step = FlextUtilitiesMapper._apply_map_keys(step, map_keys=map_keys)
        step = FlextUtilitiesMapper._apply_filter_keys(
            step,
            filter_keys=filter_keys,
        )
        step = FlextUtilitiesMapper._apply_exclude_keys(
            step,
            exclude_keys=exclude_keys,
        )
        step = FlextUtilitiesMapper._apply_strip_none(step, strip_none=strip_none)
        return FlextUtilitiesMapper._apply_strip_empty(step, strip_empty=strip_empty)

    @staticmethod
    def _success_value_result(value: t.PresentValueOrModel) -> r[t.ValueOrModel]:
        """Create a successful mapper result with concrete value typing."""

        def _return_value() -> t.ValueOrModel:
            return value

        return r[t.ValueOrModel].create_from_callable(_return_value)

    @staticmethod
    def _normalize_accessible_value(
        value: t.ValueOrModel | p.Model | p.HasModelDump | p.ValidatorSpec,
    ) -> t.RuntimeAtomic | t.RecursiveContainer:
        """Normalize protocol-accessible values to canonical runtime/container shapes."""
        if value is None:
            return ""
        if isinstance(value, m.BaseModel):
            return value
        model_dump_attr = getattr(value, "model_dump", None)
        if callable(model_dump_attr):
            return FlextRuntime.normalize_to_container(
                t.ConfigMap.model_validate(model_dump_attr()),
            )
        if isinstance(value, p.ValidatorSpec):
            return str(value)
        if isinstance(value, (*t.SCALAR_TYPES, Path)):
            return value
        if isinstance(
            value, Mapping
        ) and FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value):
            return value
        if isinstance(value, (list, tuple)) and all(
            FlextUtilitiesGuardsTypeCore.container(item) for item in value
        ):
            return value
        return str(value)

    @staticmethod
    def _extract_field_value(
        item: t.ValueOrModel
        | t.RecursiveContainerMapping
        | Mapping[str, t.ValueOrModel],
        field_name: str,
    ) -> t.RecursiveContainer:
        """Extract field value from dict or model for pyrefly type inference."""
        if FlextUtilitiesGuardsTypeCore.mapping(item):
            dict_item: t.MutableRecursiveContainerMapping = {}
            for key, value in item.items():
                coerced_value: t.RecursiveContainer = (
                    value if FlextUtilitiesGuards.container(value) else str(value)
                )
                dict_item[str(key)] = coerced_value
            return dict_item.get(field_name)
        if hasattr(item, field_name):
            attr_value = getattr(item, field_name)
            if FlextUtilitiesGuards.container(attr_value):
                return attr_value
            return str(attr_value)
        return None

    @staticmethod
    def _resolve_raw_value(
        raw: t.ValueOrModel | None,
        key_part: str,
    ) -> r[t.ValueOrModel]:
        """Wrap a raw value into a Result: fail on None, narrow containers, stringify rest."""
        if raw is None:
            marker = "found_none:"
            return r[t.ValueOrModel].fail_op(
                "resolve extracted value",
                marker + e.render_template(c.ERR_TEMPLATE_FOUND_NONE, key=key_part),
            )
        if FlextUtilitiesGuards.container(raw):
            return FlextUtilitiesMapper._success_value_result(raw)
        return FlextUtilitiesMapper._success_value_result(str(raw))

    @staticmethod
    def _extract_get_value(
        current: t.ValueOrModel | None,
        key_part: str,
    ) -> r[t.ValueOrModel]:
        """Get raw value from dict/object/model, returning found_none or not-found failures."""
        if isinstance(current, Mapping):
            mapping_obj: Mapping[str, t.ValueOrModel] = current
            if key_part in mapping_obj:
                return FlextUtilitiesMapper._resolve_raw_value(
                    mapping_obj[key_part],
                    key_part,
                )
            return r[t.ValueOrModel].fail_op(
                "extract mapping key",
                e.render_template(
                    c.ERR_TEMPLATE_KEY_NOT_FOUND,
                    key=key_part,
                )
                + " in Mapping",
            )
        if isinstance(current, (t.ConfigMap, t.Dict)):
            mapping_obj = current.root
            if key_part in mapping_obj:
                return FlextUtilitiesMapper._resolve_raw_value(
                    mapping_obj[key_part],
                    key_part,
                )
            return r[t.ValueOrModel].fail_op(
                "extract config mapping key",
                e.render_template(
                    c.ERR_TEMPLATE_KEY_NOT_FOUND,
                    key=key_part,
                )
                + " in Mapping",
            )
        if hasattr(current, key_part):
            return FlextUtilitiesMapper._resolve_raw_value(
                getattr(current, key_part),
                key_part,
            )
        if FlextUtilitiesGuardsTypeModel.pydantic_model(current):
            model_dump_attr = current.model_dump
            if callable(model_dump_attr):
                model_dict = model_dump_attr()
                if key_part in model_dict:
                    val = model_dict[key_part]
                    if val is None:
                        marker = "found_none:"
                        return r[t.ValueOrModel].fail_op(
                            "extract model key value",
                            marker
                            + e.render_template(
                                c.ERR_TEMPLATE_FOUND_NONE, key=key_part
                            ),
                        )
                    return FlextUtilitiesMapper._success_value_result(val)
        return r[t.ValueOrModel].fail_op(
            "extract key",
            e.render_template(c.ERR_TEMPLATE_KEY_NOT_FOUND, key=key_part),
        )

    @staticmethod
    def _extract_handle_array_index(
        current: t.ValueOrModel,
        array_match: str,
    ) -> r[t.ValueOrModel]:
        """Handle array indexing with negative index support."""
        raw: t.ValueOrModel = current
        if isinstance(current, t.ObjectList):
            raw = current.root
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return r[t.ValueOrModel].fail_op(
                "extract array index",
                c.ERR_MAPPER_NOT_A_SEQUENCE,
            )
        sequence: Sequence[t.ValueOrModel] = raw
        try:
            idx = int(array_match)
            if idx < 0:
                idx = len(sequence) + idx
            if 0 <= idx < len(sequence):
                item = sequence[idx]
                if item is None:
                    return r[t.ValueOrModel].fail_op(
                        "extract array index value",
                        c.ERR_MAPPER_FOUND_NONE_INDEX,
                    )
                return FlextUtilitiesMapper._success_value_result(item)
            return r[t.ValueOrModel].fail_op(
                "extract array index",
                e.render_template(
                    c.ERR_TEMPLATE_INDEX_OUT_OF_RANGE,
                    index=int(array_match),
                ),
            )
        except (ValueError, IndexError):
            return r[t.ValueOrModel].fail_op(
                "extract array index",
                e.render_template(c.ERR_TEMPLATE_INVALID_INDEX, index=array_match),
            )

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
        transform_opts: t.RecursiveContainerMapping,
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
        if FlextUtilitiesGuardsTypeCore.mapping(map_keys_val) and all(
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
    def _get_raw(
        data: p.AccessibleData | t.ConfigModelInput,
        key: str,
        *,
        default: t.RuntimeAtomic | None = None,
    ) -> t.RuntimeAtomic | t.RecursiveContainer:
        """Internal helper for raw get without DSL conversion."""
        fallback: t.RuntimeAtomic | t.RecursiveContainer = (
            default if default is not None else ""
        )
        match data:
            case dict() | Mapping():
                if key not in data:
                    return fallback
                return FlextUtilitiesMapper._normalize_accessible_value(data[key])
            case t.ConfigMap() | t.Dict():
                if key not in data.root:
                    return fallback
                return FlextUtilitiesMapper._normalize_accessible_value(
                    data.root[key],
                )
            case _ if hasattr(data, key):
                return FlextUtilitiesMapper._normalize_accessible_value(
                    getattr(data, key),
                )
            case _:
                return fallback

    @staticmethod
    def _get_numeric_field(
        item: t.ModelCarrier | t.RecursiveContainerMapping,
        field_name: str,
    ) -> t.Numeric | None:
        """Extract a numeric field value from a model or mapping-like object."""
        if isinstance(item, m.BaseModel):
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
            numeric_values = []
            for item in items_list:
                if isinstance(item, m.BaseModel):
                    field_value = FlextUtilitiesMapper._get_numeric_field(item, field)
                elif isinstance(item, Mapping):
                    field_value = FlextUtilitiesMapper._get_numeric_field(
                        cast("t.RecursiveContainerMapping", item),
                        field,
                    )
                else:
                    continue
                if field_value is not None:
                    numeric_values.append(field_value)
        agg_fn: Callable[[Sequence[t.Numeric]], t.Numeric] = (
            fn if fn is not None else sum
        )
        return agg_fn(numeric_values) if numeric_values else 0

    @staticmethod
    def _deep_eq_values(
        val_a: t.ValueOrModel,
        val_b: t.ValueOrModel,
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
                )
            case (list() as la, list() as lb):
                if len(la) != len(lb):
                    return False
                return all(
                    starmap(
                        FlextUtilitiesMapper._deep_eq_values,
                        zip(la, lb, strict=True),
                    ),
                )
            case _:
                return val_a == val_b

    @staticmethod
    def deep_eq(
        a: Mapping[str, t.ValueOrModel],
        b: Mapping[str, t.ValueOrModel],
    ) -> bool:
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

    @staticmethod
    def _extract_fail_or_default(
        msg: str,
        *,
        default: t.ValueOrModel | None,
        required: bool,
    ) -> r[t.ValueOrModel]:
        """Return fail (required) or ok(default) / fail (no default) for extract paths."""
        if required:
            return r[t.ValueOrModel].fail_op("extract required path", msg)
        if default is None:
            return r[t.ValueOrModel].fail_op(
                "extract path default",
                e.render_template(
                    c.ERR_TEMPLATE_MESSAGE_AND_DEFAULT_IS_NONE,
                    message=msg,
                ),
            )
        return FlextUtilitiesMapper._success_value_result(default)

    @staticmethod
    def _extract_resolve_path_part(
        current: t.ValueOrModel | None,
        part: str,
        *,
        path_context: str,
        default: t.ValueOrModel | None,
        required: bool,
    ) -> tuple[t.ValueOrModel | None, r[t.ValueOrModel] | None]:
        """Resolve one path segment; returns (next_current, None) or (None, early_result)."""
        found_none_prefix = "found_none:"
        key_part, array_match = FlextUtilitiesMapper._extract_parse_array_index(part)

        get_result = FlextUtilitiesMapper._extract_get_value(current, key_part)
        if get_result.failure:
            error_str = get_result.error or ""
            if found_none_prefix in error_str:
                next_val: t.ValueOrModel = None
            else:
                return None, FlextUtilitiesMapper._extract_fail_or_default(
                    e.render_template(
                        c.ERR_TEMPLATE_KEY_NOT_FOUND_AT_PATH,
                        key=key_part,
                        path=path_context,
                    ),
                    default=default,
                    required=required,
                )
        else:
            next_val = get_result.unwrap_or(None)

        if array_match and next_val is not None:
            narrowed_for_index = (
                next_val
                if isinstance(next_val, Sequence)
                and not isinstance(next_val, (str, bytes))
                else FlextRuntime.normalize_to_container(next_val)
            )
            index_result = FlextUtilitiesMapper._extract_handle_array_index(
                narrowed_for_index,
                array_match,
            )
            if index_result.failure:
                error_str = index_result.error or ""
                if found_none_prefix in error_str:
                    next_val = None
                else:
                    return None, FlextUtilitiesMapper._extract_fail_or_default(
                        e.render_template(
                            c.ERR_TEMPLATE_ARRAY_ERROR_AT_KEY,
                            key=key_part,
                            error=index_result.error,
                        ),
                        default=default,
                        required=required,
                    )
            else:
                next_val = index_result.unwrap_or(None)

        return next_val, None

    @staticmethod
    def extract(
        data: p.AccessibleData,
        path: str,
        *,
        default: t.ValueOrModel | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> r[t.ValueOrModel]:
        """Extract nested value via dot-notation path with array index support (e.g. "user.addresses[0].city")."""
        try:
            parts = path.split(separator)
            current: t.ValueOrModel | None = None
            if isinstance(data, m.BaseModel):
                current = data
            elif isinstance(data, Mapping):
                config_map = t.ConfigMap(
                    root={
                        str(k): FlextUtilitiesMapper._normalize_accessible_value(v)
                        for k, v in data.items()
                    },
                )
                current = FlextRuntime.normalize_to_container(config_map)
            else:
                model_dump_attr = getattr(data, "model_dump", None)
                if callable(model_dump_attr):
                    current = FlextRuntime.normalize_to_container(
                        t.ConfigMap.model_validate(model_dump_attr()),
                    )
                elif isinstance(data, p.ValidatorSpec):
                    current = str(data)
                elif data is None or isinstance(
                    data, (*t.SCALAR_TYPES, Path, list, tuple)
                ):
                    current = data

            for i, part in enumerate(parts):
                if current is None:
                    return FlextUtilitiesMapper._extract_fail_or_default(
                        e.render_template(
                            c.ERR_TEMPLATE_PATH_IS_NONE,
                            path=separator.join(parts[:i]),
                        ),
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
                    c.ERR_TEMPLATE_EXTRACTED_VALUE_IS_NONE,
                    default=default,
                    required=required,
                )
            if FlextUtilitiesGuards.container(current):
                return FlextUtilitiesMapper._success_value_result(current)
            return FlextUtilitiesMapper._success_value_result(str(current))
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as exc:
            params = m.OperationErrorParams(operation="extract", reason=str(exc))
            return r[t.ValueOrModel].fail_op(
                "extract path",
                e.render_error_template(
                    c.ERR_TEMPLATE_EXTRACT_FAILED,
                    operation="extract",
                    error=exc,
                    params=params,
                ),
            )

    # filter_dict removed (use collection.filter)

    @staticmethod
    def map_get(
        data: p.AccessibleData | t.ConfigModelInput,
        key: str,
        *,
        default: t.ValueOrModel | None = None,
    ) -> t.RuntimeAtomic | t.RecursiveContainer:
        """Get value by key from dict/object, returning default if missing."""
        runtime_default: t.RuntimeAtomic | None = (
            default
            if default is None or FlextUtilitiesGuards.container(default)
            else None
        )
        return FlextUtilitiesMapper._get_raw(data, key, default=runtime_default)

    @staticmethod
    def map_dict_keys(
        source: t.RecursiveContainerMapping,
        key_mapping: t.StrMapping,
        *,
        keep_unmapped: bool = True,
    ) -> r[t.MutableRecursiveContainerMapping]:
        """Rename dict keys using old_key->new_key mapping."""

        def _map_keys() -> t.MutableRecursiveContainerMapping:
            result: t.MutableRecursiveContainerMapping = {}
            for key, value in source.items():
                new_key = key_mapping.get(key)
                if new_key:
                    result[new_key] = value
                elif keep_unmapped:
                    result[key] = value
            return result

        mapped_result = r[t.MutableRecursiveContainerMapping].create_from_callable(
            _map_keys
        )
        return mapped_result.fold(
            on_failure=lambda exc: r[t.MutableRecursiveContainerMapping].fail_op(
                "map dict keys",
                e.render_error_template(
                    c.ERR_TEMPLATE_FAILED_TO_MAP_DICT_KEYS,
                    operation="map dict keys",
                    error=exc,
                    params=m.OperationErrorParams(
                        operation="map dict keys",
                        reason=str(exc),
                    ),
                ),
            ),
            on_success=lambda _: mapped_result,
        )

    # narrow_to_container suite removed (use FlextRuntime.normalize_to_container)

    @staticmethod
    def prop(
        key: str,
    ) -> Callable[[t.ConfigModelInput], t.RuntimeAtomic | t.RecursiveContainer]:
        """Return an accessor function that extracts the named property from an object."""

        def accessor(
            obj: t.ConfigModelInput,
        ) -> t.RuntimeAtomic | t.RecursiveContainer:
            """Access property from object."""
            result = FlextUtilitiesMapper.map_get(obj, key)
            return result if result is not None else ""

        return accessor

    @staticmethod
    def _take_by_key(
        data_or_items: Sequence[t.ValueOrModel]
        | Mapping[str, t.ValueOrModel]
        | t.ValueOrModel,
        key: str,
        *,
        as_type: type | None,
        default: t.ValueOrModel | None,
    ) -> t.ValueOrModel | None:
        """Extract a value by key from a Mapping or BaseModel."""
        fallback: t.ValueOrModel | None = default
        data: p.AccessibleData | t.ConfigModelInput
        if (
            isinstance(data_or_items, (t.ConfigMap, t.Dict, Mapping))
            and FlextUtilitiesGuardsTypeModel.configuration_mapping(data_or_items)
        ) or (isinstance(data_or_items, m.BaseModel)):
            data = data_or_items
        else:
            return fallback
        value = FlextUtilitiesMapper.map_get(data, key, default=default)
        if value is None:
            return fallback
        if as_type is not None and not isinstance(value, as_type):
            return fallback
        return value

    @staticmethod
    def _take_n_items(
        data_or_items: Sequence[t.ValueOrModel]
        | Mapping[str, t.ValueOrModel]
        | t.ValueOrModel,
        n: int,
        *,
        default: t.ValueOrModel | None,
        from_start: bool,
    ) -> (
        Sequence[t.ValueOrModel] | Mapping[str, t.ValueOrModel] | t.ValueOrModel | None
    ):
        """Take N items from a Mapping or Sequence."""
        match data_or_items:
            case Mapping():
                keys = list(data_or_items.keys())
                selected = keys[:n] if from_start else keys[-n:]
                return {k: data_or_items[k] for k in selected}
            case list() | tuple():
                items_list: Sequence[t.ValueOrModel] = data_or_items
                return items_list[:n] if from_start else items_list[-n:]
            case _:
                return default

    @staticmethod
    def take(
        data_or_items: Sequence[t.ValueOrModel]
        | Mapping[str, t.ValueOrModel]
        | t.ValueOrModel,
        key_or_n: str | int,
        *,
        as_type: type | None = None,
        default: t.ValueOrModel | None = None,
        from_start: bool = True,
    ) -> (
        Sequence[t.ValueOrModel] | Mapping[str, t.ValueOrModel] | t.ValueOrModel | None
    ):
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
    def _coerce_source_to_container_mapping(
        source: t.RecursiveContainerMapping | t.ConfigMap,
    ) -> t.RecursiveContainerMapping:
        """Coerce ConfigMap (whose root may include BaseModel values) to ContainerMapping."""
        if isinstance(source, t.ConfigMap):
            coerced: t.MutableRecursiveContainerMapping = {}
            for k, v in source.root.items():
                coerced[str(k)] = v if FlextUtilitiesGuards.container(v) else str(v)
            return coerced
        return source

    @staticmethod
    def transform(
        source: t.RecursiveContainerMapping | t.ConfigMap,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: t.StrMapping | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> r[t.MutableRecursiveContainerMapping | t.RecursiveContainerMapping]:
        """Apply normalize/strip_none/strip_empty/map_keys/filter_keys/exclude_keys to a dict."""
        coerced_source = FlextUtilitiesMapper._coerce_source_to_container_mapping(
            source,
        )
        transform_result = r[
            t.MutableRecursiveContainerMapping | t.RecursiveContainerMapping
        ].create_from_callable(
            lambda: FlextUtilitiesMapper._apply_transform_steps(
                dict(coerced_source),
                normalize=normalize,
                map_keys=map_keys,
                filter_keys=filter_keys,
                exclude_keys=exclude_keys,
                strip_none=strip_none,
                strip_empty=strip_empty,
            ),
        )
        return transform_result.fold(
            on_failure=lambda exc: r[
                t.MutableRecursiveContainerMapping | t.RecursiveContainerMapping
            ].fail_op("transform", exc),
            on_success=lambda _: transform_result,
        )


__all__: list[str] = ["FlextUtilitiesMapper"]
