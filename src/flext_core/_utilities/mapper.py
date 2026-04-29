"""FlextUtilitiesMapper — data extraction, transformation, and aggregation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import starmap

from flext_core import (
    FlextModelsPydantic,
    FlextRuntime,
    FlextUtilitiesCollection,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesMapperExtract,
    m,
    p,
    r,
    t,
)


class FlextUtilitiesMapper(FlextUtilitiesMapperExtract):
    """Data structure mapping, extraction, and transformation utilities."""

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
            {k: FlextRuntime.normalize_to_metadata(v) for k, v in source.root.items()}
            if isinstance(source, m.ConfigMap)
            else source
        )

        def _pipeline() -> dict[str, t.JsonValue] | Mapping[str, t.JsonValue]:
            step: dict[str, t.JsonValue] | Mapping[str, t.JsonValue] = dict(coerced)
            if normalize:
                normalized = FlextRuntime.normalize_to_metadata(
                    dict(step),
                )
                if FlextUtilitiesGuardsTypeCore.mapping(normalized):
                    step = dict(normalized)
            if map_keys:
                step = {map_keys.get(k, k): v for k, v in step.items()}
            if filter_keys:
                step = {k: step[k] for k in filter_keys if k in step}
            if exclude_keys:
                step = {k: v for k, v in step.items() if k not in exclude_keys}
            if strip_none:
                step = FlextUtilitiesCollection.filter(step, lambda v: v is not None)
            if strip_empty:
                step = FlextUtilitiesCollection.filter(
                    step,
                    lambda v: not FlextUtilitiesGuardsTypeCore.empty_value(v),
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
