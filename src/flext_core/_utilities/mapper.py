"""FlextUtilitiesMapper — data extraction, transformation, and aggregation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    """Data structure mapping, extraction, and transformation utilities."""

    @staticmethod
    def _apply_exclude_keys(
        result: Mapping[str, t.JsonValue],
        exclude_keys: set[str] | None,
    ) -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
        if not exclude_keys:
            return result
        return {k: v for k, v in result.items() if k not in exclude_keys}

    @staticmethod
    def _apply_filter_keys(
        result: Mapping[str, t.JsonValue],
        filter_keys: set[str] | None,
    ) -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
        if not filter_keys:
            return result
        return {k: result[k] for k in filter_keys if k in result}

    @staticmethod
    def _apply_map_keys(
        result: Mapping[str, t.JsonValue],
        map_keys: t.StrMapping | None,
    ) -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
        if not map_keys:
            return result
        return {map_keys.get(k, k): v for k, v in result.items()}

    @staticmethod
    def _apply_normalize(
        result: Mapping[str, t.JsonValue],
        *,
        normalize: bool,
    ) -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
        if not normalize:
            return result
        normalized = FlextRuntime.normalize_to_metadata(
            {str(k): v for k, v in result.items()},
        )
        if FlextUtilitiesGuardsTypeCore.mapping(normalized):
            return {str(k): v for k, v in normalized.items()}
        return result

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
        if isinstance(value, Mapping) and (
            FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)
        ):
            return value
        if isinstance(value, (list, tuple)) and all(
            FlextUtilitiesGuardsTypeCore.container(item) for item in value
        ):
            return value
        return str(value)

    @staticmethod
    def _resolve_raw_value(
        raw: t.JsonPayload | None,
        key_part: str,
    ) -> p.Result[t.JsonPayload]:
        """Wrap raw value in Result: fail on None, keep containers, stringify rest."""
        if raw is None:
            return r[t.JsonPayload].fail_op(
                "resolve extracted value",
                "found_none:"
                + e.render_template(c.ERR_TEMPLATE_FOUND_NONE, key=key_part),
            )
        return r[t.JsonPayload].ok(
            raw if FlextUtilitiesGuards.container(raw) else str(raw),
        )

    @staticmethod
    def _extract_get_value(
        current: t.JsonPayload
        | t.JsonMapping
        | FlextModelsContainers.ConfigMap
        | Sequence[t.JsonValue | t.JsonPayload]
        | None,
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
                e.render_template(c.ERR_TEMPLATE_KEY_NOT_FOUND, key=key_part)
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
                        return r[t.JsonPayload].fail_op(
                            "extract model key value",
                            "found_none:"
                            + e.render_template(
                                c.ERR_TEMPLATE_FOUND_NONE,
                                key=key_part,
                            ),
                        )
                    return r[t.JsonPayload].ok(val)
        return r[t.JsonPayload].fail_op(
            "extract key",
            e.render_template(c.ERR_TEMPLATE_KEY_NOT_FOUND, key=key_part),
        )

    @staticmethod
    def _extract_handle_array_index(
        current: t.JsonPayload
        | t.JsonMapping
        | Sequence[t.JsonValue | t.JsonPayload]
        | FlextModelsContainers.ObjectList,
        array_match: str,
    ) -> p.Result[t.JsonPayload]:
        """Handle array indexing with negative index support."""
        if isinstance(current, FlextModelsContainers.ObjectList):
            sequence: Sequence[t.JsonValue | t.JsonPayload] = current.root
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
                return r[t.JsonPayload].ok(item)
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
    def _get_raw(
        data: p.AccessibleData | t.ConfigModelInput,
        key: str,
    ) -> t.JsonPayload | t.JsonValue:
        """Internal helper for raw get without DSL conversion."""
        match data:
            case dict() | Mapping() if key in data:
                return FlextUtilitiesMapper._normalize_accessible_value(data[key])
            case m.ConfigMap() | m.Dict() if key in data.root:
                return FlextUtilitiesMapper._normalize_accessible_value(data.root[key])
            case _ if hasattr(data, key):
                return FlextUtilitiesMapper._normalize_accessible_value(
                    getattr(data, key),
                )
            case _:
                return ""

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
            numeric_values: list[t.Numeric] = [field(item) for item in items_list]
        else:
            numeric_values = []
            for item in items_list:
                raw: object | None
                if isinstance(item, FlextModelsPydantic.BaseModel):
                    raw = getattr(item, field, None)
                elif isinstance(item, Mapping):
                    raw = item.get(field)
                else:
                    continue
                if isinstance(raw, (int, float)):
                    numeric_values.append(raw)
        agg_fn = fn if fn is not None else sum
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
        return all(
            key in b and FlextUtilitiesMapper._deep_eq_values(val_a, b[key])
            for key, val_a in a.items()
        )

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
        return r[t.JsonPayload].ok(default)

    @staticmethod
    def _extract_resolve_path_part(
        current: t.JsonPayload
        | t.JsonMapping
        | FlextModelsContainers.ConfigMap
        | Sequence[t.JsonPayload]
        | None,
        part: str,
        *,
        path_context: str,
        default: t.JsonPayload | None,
        required: bool,
    ) -> tuple[t.JsonPayload | None, p.Result[t.JsonPayload] | None]:
        """Resolve one path segment; returns (next_current, None) or (None, early_result)."""
        if "[" in part and part.endswith("]"):
            bracket_pos = part.index("[")
            array_match = part[bracket_pos + 1 : -1]
            key_part = part[:bracket_pos]
        else:
            key_part, array_match = part, ""

        get_result = FlextUtilitiesMapper._extract_get_value(current, key_part)
        next_val: t.JsonPayload | None
        if get_result.failure:
            if "found_none:" in (get_result.error or ""):
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
                if "found_none:" in (index_result.error or ""):
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
        """Extract nested value via dot-notation path with array index support."""
        try:
            parts = path.split(separator)
            current: (
                t.JsonPayload | t.JsonMapping | FlextModelsContainers.ConfigMap | None
            )
            if isinstance(data, FlextModelsPydantic.BaseModel):
                current = data
            elif isinstance(data, Mapping):
                current = m.ConfigMap(
                    root={
                        str(k): FlextUtilitiesMapper._normalize_accessible_value(v)
                        for k, v in data.items()
                    },
                )
            else:
                model_dump_attr = getattr(data, "model_dump", None)
                if callable(model_dump_attr):
                    current = m.ConfigMap.model_validate(model_dump_attr())
                elif isinstance(data, p.ValidatorSpec):
                    current = str(data)
                elif data is None or isinstance(
                    data,
                    (*t.SCALAR_TYPES, Path, list, tuple),
                ):
                    current = data
                else:
                    current = None

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
            return r[t.JsonPayload].ok(
                current if FlextUtilitiesGuards.container(current) else str(current),
            )
        except (AttributeError, TypeError, ValueError, KeyError, IndexError) as exc:
            return r[t.JsonPayload].fail_op(
                "extract path",
                e.render_error_template(
                    c.ERR_TEMPLATE_EXTRACT_FAILED,
                    operation="extract",
                    error=exc,
                    params=FlextModelsExceptionParams.OperationErrorParams(
                        operation="extract",
                        reason=str(exc),
                    ),
                ),
            )

    @staticmethod
    def prop(
        key: str,
    ) -> Callable[[t.ConfigModelInput], t.JsonPayload | t.JsonValue]:
        """Return an accessor function that extracts the named property from an object."""

        def accessor(
            obj: t.ConfigModelInput,
        ) -> t.JsonPayload | t.JsonValue:
            result = FlextUtilitiesMapper._get_raw(obj, key)
            return result if result is not None else ""

        return accessor

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
        coerced: Mapping[str, t.JsonValue] = (
            {
                str(k): FlextRuntime.normalize_to_metadata(v)
                for k, v in source.root.items()
            }
            if isinstance(source, m.ConfigMap)
            else source
        )

        def _pipeline() -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
            step: dict[str, t.JsonValue] | Mapping[str, t.JsonValue] = dict(coerced)
            step = FlextUtilitiesMapper._apply_normalize(step, normalize=normalize)
            step = FlextUtilitiesMapper._apply_map_keys(step, map_keys)
            step = FlextUtilitiesMapper._apply_filter_keys(step, filter_keys)
            step = FlextUtilitiesMapper._apply_exclude_keys(step, exclude_keys)
            if strip_none:
                step = FlextUtilitiesCollection.filter(step, lambda v: v is not None)
            if strip_empty:
                step = FlextUtilitiesCollection.filter(
                    step,
                    lambda v: v not in ("", [], {}, None),
                )
            return step

        transform_result = r[
            dict[str, t.JsonValue] | Mapping[str, t.JsonValue]
        ].create_from_callable(_pipeline)
        return transform_result.fold(
            on_failure=lambda exc: p.Result[
                dict[str, t.JsonValue] | Mapping[str, t.JsonValue]
            ].fail_op("transform", exc),
            on_success=lambda _: transform_result,
        )


__all__: list[str] = ["FlextUtilitiesMapper"]
