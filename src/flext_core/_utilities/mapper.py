"""Utilities module - FlextUtilitiesMapper.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    MutableSequence,
    Sequence,
)
from itertools import starmap
from pathlib import Path

from flext_core import (
    FlextModelsContainers,
    FlextModelsExceptionParams,
    FlextModelsPydantic,
    FlextRuntime,
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


class FlextUtilitiesMapper:
    """Data structure mapping and transformation utilities.

    Provides generic methods for mapping between data structures, building
    objects from flags/mappings, and transforming dict/list structures.
    """

    @staticmethod
    def _apply_exclude_keys(
        result: Mapping[str, t.JsonValue],
        *,
        exclude_keys: set[str] | None,
    ) -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
        """Apply exclude keys step."""
        if exclude_keys:
            filtered_result: dict[str, t.JsonValue] = dict(result)
            for key in exclude_keys:
                _ = filtered_result.pop(key, None)
            return filtered_result
        return result

    @staticmethod
    def _apply_filter_keys(
        result: Mapping[str, t.JsonValue],
        *,
        filter_keys: set[str] | None,
    ) -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
        """Apply filter keys step."""
        if filter_keys:
            filtered_dict: dict[str, t.JsonValue] = {}
            for key in filter_keys:
                if key in result:
                    filtered_dict[key] = result[key]
            return filtered_dict
        return result

    @staticmethod
    def _apply_map_keys(
        result: Mapping[str, t.JsonValue],
        *,
        map_keys: t.StrMapping | None,
    ) -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
        """Apply map keys step."""
        if map_keys:
            mapped_result: p.Result[dict[str, t.JsonValue]] = (
                FlextUtilitiesMapper.map_dict_keys(
                    result,
                    map_keys,
                )
            )
            if mapped_result.success:
                return mapped_result.unwrap_or(dict(result))
        return result

    @staticmethod
    def _normalize_component(
        component: (t.JsonPayload | t.JsonMapping | set[t.JsonValue] | None),
    ) -> t.JsonValue:
        """Flat-Container normalization via canonical ``FlextRuntime.normalize_to_metadata``."""
        if component is None:
            return ""
        if isinstance(component, set):
            return [FlextRuntime.normalize_to_metadata(item) for item in component]
        return FlextRuntime.normalize_to_metadata(component)

    @staticmethod
    def _apply_normalize(
        result: Mapping[str, t.JsonValue],
        *,
        normalize: bool,
    ) -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
        """Apply normalize step."""
        if normalize:
            normalized = FlextUtilitiesMapper._normalize_component(
                {str(k): v for k, v in result.items()},
            )
            if FlextUtilitiesGuardsTypeCore.mapping(normalized):
                normalized_result: dict[str, t.JsonValue] = {}
                for key, value in normalized.items():
                    normalized_result[str(key)] = value
                return normalized_result
        return result

    @staticmethod
    def _apply_strip_empty(
        result: Mapping[str, t.JsonValue],
        *,
        strip_empty: bool,
    ) -> Mapping[str, t.JsonValue]:
        """Apply strip empty step."""
        if strip_empty:
            return FlextUtilitiesCollection.filter(
                result,
                lambda v: v not in ("", [], {}, None),
            )
        return result

    @staticmethod
    def _apply_strip_none(
        result: Mapping[str, t.JsonValue],
        *,
        strip_none: bool,
    ) -> Mapping[str, t.JsonValue]:
        """Apply strip none step."""
        if strip_none:
            return FlextUtilitiesCollection.filter(result, lambda v: v is not None)
        return result

    @staticmethod
    def _apply_transform_steps(
        result: Mapping[str, t.JsonValue],
        *,
        normalize: bool,
        map_keys: t.StrMapping | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
    ) -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
        """Apply transform steps to result dict."""
        step: dict[str, t.JsonValue] | Mapping[str, t.JsonValue] = (
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
    def _success_value_result(value: t.JsonPayload) -> p.Result[t.JsonPayload]:
        """Create a successful mapper result with concrete value typing."""
        return r[t.JsonPayload].ok(value)

    @staticmethod
    def _normalize_accessible_value(
        value: t.JsonPayload | p.Model | p.HasModelDump | p.ValidatorSpec | None,
    ) -> t.JsonPayload | t.JsonValue:
        """Normalize protocol-accessible values to canonical runtime/container shapes."""
        if value is None:
            return ""
        if isinstance(value, FlextModelsPydantic.BaseModel):
            return value
        model_dump_attr = getattr(value, "model_dump", None)
        if callable(model_dump_attr):
            return FlextRuntime.normalize_to_container(
                m.ConfigMap.model_validate(model_dump_attr()),
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
        item: (t.JsonPayload | Mapping[str, t.JsonValue] | Mapping[str, t.JsonPayload]),
        field_name: str,
    ) -> t.JsonValue:
        """Extract field value from dict or model for pyrefly type inference."""
        if FlextUtilitiesGuardsTypeCore.mapping(item):
            dict_item: dict[str, t.JsonValue] = {}
            for key, value in item.items():
                coerced_value: t.JsonValue = (
                    value if FlextUtilitiesGuards.container(value) else str(value)
                )
                dict_item[str(key)] = coerced_value
            found = dict_item.get(field_name)
            return found if found is not None else ""
        if hasattr(item, field_name):
            attr_value = getattr(item, field_name)
            if FlextUtilitiesGuards.container(attr_value):
                return attr_value
            return str(attr_value)
        return ""

    @staticmethod
    def _resolve_raw_value(
        raw: t.JsonPayload | None,
        key_part: str,
    ) -> p.Result[t.JsonPayload]:
        """Wrap a raw value into a Result: fail on None, narrow containers, stringify rest."""
        if raw is None:
            marker = "found_none:"
            return r[t.JsonPayload].fail_op(
                "resolve extracted value",
                marker + e.render_template(c.ERR_TEMPLATE_FOUND_NONE, key=key_part),
            )
        if FlextUtilitiesGuards.container(raw):
            return FlextUtilitiesMapper._success_value_result(raw)
        return FlextUtilitiesMapper._success_value_result(str(raw))

    @staticmethod
    def _extract_get_value(
        current: (
            t.JsonPayload
            | t.JsonMapping
            | FlextModelsContainers.ConfigMap
            | Sequence[t.JsonValue | t.JsonPayload]
            | None
        ),
        key_part: str,
    ) -> p.Result[t.JsonPayload]:
        """Get raw value from dict/object/model, returning found_none or not-found failures."""
        if isinstance(current, Mapping):
            mapping_obj: Mapping[str, t.JsonValue | t.JsonPayload] = current
            if key_part in mapping_obj:
                return FlextUtilitiesMapper._resolve_raw_value(
                    mapping_obj[key_part],
                    key_part,
                )
            return r[t.JsonPayload].fail_op(
                "extract mapping key",
                e.render_template(
                    c.ERR_TEMPLATE_KEY_NOT_FOUND,
                    key=key_part,
                )
                + " in Mapping",
            )
        # DEPRECATED: m.ConfigMap and m.Dict are no longer available
        # Refactor to use Pydantic models directly
        # if isinstance(current, (m.ConfigMap, m.Dict)):
        #     mapping_obj = current.root
        #     if key_part in mapping_obj:
        #         return FlextUtilitiesMapper._resolve_raw_value(
        #             mapping_obj[key_part],
        #             key_part,
        #         )
        #     return r[t.JsonPayload].fail_op(
        #         "extract config mapping key",
        #         e.render_template(
        #             c.ERR_TEMPLATE_KEY_NOT_FOUND,
        #             key=key_part,
        #         )
        #         + " in Mapping",
        #     )
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
                        return r[t.JsonPayload].fail_op(
                            "extract model key value",
                            marker
                            + e.render_template(
                                c.ERR_TEMPLATE_FOUND_NONE, key=key_part
                            ),
                        )
                    return FlextUtilitiesMapper._success_value_result(val)
        return r[t.JsonPayload].fail_op(
            "extract key",
            e.render_template(c.ERR_TEMPLATE_KEY_NOT_FOUND, key=key_part),
        )

    @staticmethod
    def _extract_handle_array_index(
        current: (
            t.JsonPayload
            | t.JsonMapping
            | Sequence[t.JsonValue | t.JsonPayload]
            | FlextModelsContainers.ObjectList
        ),
        array_match: str,
    ) -> p.Result[t.JsonPayload]:
        """Handle array indexing with negative index support."""
        sequence: Sequence[t.JsonValue | t.JsonPayload]
        if isinstance(current, FlextModelsContainers.ObjectList):
            sequence = current.root
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            sequence = current
        else:
            return r[t.JsonPayload].fail_op(
                "extract array index",
                c.ERR_MAPPER_NOT_A_SEQUENCE,
            )
        try:
            idx = int(array_match)
            if idx < 0:
                idx = len(sequence) + idx
            if 0 <= idx < len(sequence):
                item = sequence[idx]
                if item is None:
                    return r[t.JsonPayload].fail_op(
                        "extract array index value",
                        c.ERR_MAPPER_FOUND_NONE_INDEX,
                    )
                return FlextUtilitiesMapper._success_value_result(item)
            return r[t.JsonPayload].fail_op(
                "extract array index",
                e.render_template(
                    c.ERR_TEMPLATE_INDEX_OUT_OF_RANGE,
                    index=int(array_match),
                ),
            )
        except (ValueError, IndexError):
            return r[t.JsonPayload].fail_op(
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
    def _get_raw(
        data: p.AccessibleData | t.ConfigModelInput,
        key: str,
        *,
        default: t.JsonPayload | None = None,
    ) -> t.JsonPayload | t.JsonValue:
        """Internal helper for raw get without DSL conversion."""
        fallback: t.JsonPayload | t.JsonValue = default if default is not None else ""
        match data:
            case dict() | Mapping():
                if key not in data:
                    return fallback
                return FlextUtilitiesMapper._normalize_accessible_value(data[key])
            case m.ConfigMap() | m.Dict():
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
        item: m.BaseModel | Mapping[str, t.JsonValue],
        field_name: str,
    ) -> t.Numeric | None:
        """Extract a numeric field value from a model or mapping-like object."""
        if isinstance(item, FlextModelsPydantic.BaseModel):
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
                if isinstance(item, FlextModelsPydantic.BaseModel):
                    field_value = FlextUtilitiesMapper._get_numeric_field(item, field)
                elif isinstance(item, Mapping):
                    field_value = FlextUtilitiesMapper._get_numeric_field(
                        item,  # Mapping satisfies t.JsonMapping
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
        val_a: t.JsonPayload | t.JsonValue,
        val_b: t.JsonPayload | t.JsonValue,
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
        a: Mapping[str, t.JsonValue | t.JsonPayload],
        b: Mapping[str, t.JsonValue | t.JsonPayload],
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
        default: t.JsonPayload | None,
        required: bool,
    ) -> p.Result[t.JsonPayload]:
        """Return fail (required) or ok(default) / fail (no default) for extract paths."""
        if required:
            return r[t.JsonPayload].fail_op("extract required path", msg)
        if default is None:
            return r[t.JsonPayload].fail_op(
                "extract path default",
                e.render_template(
                    c.ERR_TEMPLATE_MESSAGE_AND_DEFAULT_IS_NONE,
                    message=msg,
                ),
            )
        return FlextUtilitiesMapper._success_value_result(default)

    @staticmethod
    def _extract_resolve_path_part(
        current: (
            t.JsonPayload
            | t.JsonMapping
            | FlextModelsContainers.ConfigMap
            | Sequence[t.JsonPayload]
            | None
        ),
        part: str,
        *,
        path_context: str,
        default: t.JsonPayload | None,
        required: bool,
    ) -> tuple[t.JsonPayload | None, p.Result[t.JsonPayload] | None]:
        """Resolve one path segment; returns (next_current, None) or (None, early_result)."""
        found_none_prefix = "found_none:"
        key_part, array_match = FlextUtilitiesMapper._extract_parse_array_index(part)

        get_result = FlextUtilitiesMapper._extract_get_value(current, key_part)
        next_val: t.JsonPayload | None
        if get_result.failure:
            error_str = get_result.error or ""
            if found_none_prefix in error_str:
                next_val = None
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
        default: t.JsonPayload | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> p.Result[t.JsonPayload]:
        """Extract nested value via dot-notation path with array index support (e.g. "user.addresses[0].city")."""
        try:
            parts = path.split(separator)
            current: (
                t.JsonPayload | t.JsonMapping | FlextModelsContainers.ConfigMap | None
            ) = None
            if isinstance(data, FlextModelsPydantic.BaseModel):
                current = data
            elif isinstance(data, Mapping):
                config_map = m.ConfigMap(
                    root={
                        str(k): FlextUtilitiesMapper._normalize_accessible_value(v)
                        for k, v in data.items()
                    },
                )
                current = config_map
            else:
                model_dump_attr = getattr(data, "model_dump", None)
                if callable(model_dump_attr):
                    result_map = m.ConfigMap.model_validate(model_dump_attr())
                    current = result_map
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
            params = FlextModelsExceptionParams.OperationErrorParams(
                operation="extract",
                reason=str(exc),
            )
            return r[t.JsonPayload].fail_op(
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
        default: t.JsonPayload | None = None,
    ) -> t.JsonPayload | t.JsonValue:
        """Get value by key from dict/object, returning default if missing."""
        runtime_default: t.JsonPayload | None = (
            default
            if default is None or FlextUtilitiesGuards.container(default)
            else None
        )
        return FlextUtilitiesMapper._get_raw(data, key, default=runtime_default)

    @staticmethod
    def map_dict_keys(
        source: Mapping[str, t.JsonValue],
        key_mapping: t.StrMapping,
        *,
        keep_unmapped: bool = True,
    ) -> p.Result[dict[str, t.JsonValue]]:
        """Rename dict keys using old_key->new_key mapping."""

        def _map_keys() -> dict[str, t.JsonValue]:
            result: dict[str, t.JsonValue] = {}
            for key, value in source.items():
                new_key = key_mapping.get(key)
                if new_key:
                    result[new_key] = value
                elif keep_unmapped:
                    result[key] = value
            return result

        mapped_result = r[dict[str, t.JsonValue]].create_from_callable(_map_keys)
        return mapped_result.fold(
            on_failure=lambda exc: r[dict[str, t.JsonValue]].fail_op(
                "map dict keys",
                e.render_error_template(
                    c.ERR_TEMPLATE_FAILED_TO_MAP_DICT_KEYS,
                    operation="map dict keys",
                    error=exc,
                    params=FlextModelsExceptionParams.OperationErrorParams(
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
    ) -> Callable[[t.ConfigModelInput], t.JsonPayload | t.JsonValue]:
        """Return an accessor function that extracts the named property from an object."""

        def accessor(
            obj: t.ConfigModelInput,
        ) -> t.JsonPayload | t.JsonValue:
            """Access property from object."""
            result = FlextUtilitiesMapper.map_get(obj, key)
            return result if result is not None else ""

        return accessor

    @staticmethod
    def _coerce_source_to_container_mapping(
        source: Mapping[str, t.JsonValue] | m.ConfigMap,
    ) -> Mapping[str, t.JsonValue]:
        """Coerce ConfigMap to a metadata-compatible mapping for transform steps."""
        if isinstance(source, m.ConfigMap):
            coerced: dict[str, t.JsonValue] = {}
            for k, v in source.root.items():
                coerced[str(k)] = FlextRuntime.normalize_to_metadata(v)
            return coerced
        return source

    @staticmethod
    def transform(
        source: Mapping[str, t.JsonValue] | m.ConfigMap,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: t.StrMapping | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> p.Result[dict[str, t.JsonValue] | Mapping[str, t.JsonValue]]:
        """Apply normalize/strip_none/strip_empty/map_keys/filter_keys/exclude_keys to a dict."""
        coerced_source = FlextUtilitiesMapper._coerce_source_to_container_mapping(
            source,
        )
        transform_result = r[
            dict[str, t.JsonValue] | Mapping[str, t.JsonValue]
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
            on_failure=lambda exc: p.Result[
                dict[str, t.JsonValue] | Mapping[str, t.JsonValue]
            ].fail_op("transform", exc),
            on_success=lambda _: transform_result,
        )


__all__: list[str] = ["FlextUtilitiesMapper"]
