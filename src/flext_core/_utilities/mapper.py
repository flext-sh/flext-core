"""FlextUtilitiesMapper — data extraction, transformation, and aggregation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import starmap

from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.mapper_extract import FlextUtilitiesMapperExtract
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilitiesMapper(FlextUtilitiesMapperExtract):
    """Data structure mapping, extraction, and transformation utilities."""

    @staticmethod
    def agg[T](
        items: t.SequenceOf[T] | tuple[T, ...],
        field: str | Callable[[T], t.Numeric],
        *,
        fn: Callable[[Sequence[t.Numeric]], t.Numeric] | None = None,
    ) -> t.Numeric:
        """Aggregate numeric field values from objects using fn (default: sum)."""
        items_list: t.SequenceOf[T] = list(items)
        if callable(field):
            numeric_values: list[t.Numeric] = [field(item) for item in items_list]
        else:
            numeric_values = []
            for item in items_list:
                raw: p.AttributeProbe | None
                if isinstance(item, FlextModelsPydantic.BaseModel):
                    raw = getattr(item, field, None)
                elif isinstance(item, Mapping):
                    raw = item.get(field)
                else:
                    continue
                if isinstance(raw, t.NUMERIC_TYPES):
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
        if val_a is None or val_b is None:
            return False
        if isinstance(val_a, Mapping) and isinstance(val_b, Mapping):
            return (
                hasattr(val_a, "items")
                and hasattr(val_b, "items")
                and FlextUtilitiesMapper.deep_eq(val_a, val_b)
            )
        if isinstance(val_a, list) and isinstance(val_b, list):
            return len(val_a) == len(val_b) and all(
                starmap(
                    FlextUtilitiesMapper._deep_eq_values,
                    zip(val_a, val_b, strict=True),
                ),
            )
        return val_a == val_b

    @staticmethod
    def deep_eq(
        a: t.MappingKV[str, t.JsonValue | t.JsonPayload],
        b: t.MappingKV[str, t.JsonValue | t.JsonPayload],
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
        source: t.JsonMapping | m.ConfigMap,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: t.StrMapping | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> p.Result[t.JsonMapping]:
        """Apply normalize/strip_none/strip_empty/map_keys/filter_keys/exclude_keys to a dict."""
        coerced: t.JsonMapping = (
            {k: FlextRuntime.normalize_to_metadata(v) for k, v in source.root.items()}
            if isinstance(source, m.ConfigMap)
            else source
        )

        def _pipeline() -> t.JsonDict:
            step: t.JsonDict = dict(coerced)
            if normalize:
                normalized = FlextRuntime.normalize_to_metadata(step)
                if FlextUtilitiesGuardsTypeCore.mapping(normalized):
                    step = dict(normalized)
            if map_keys:
                step = {map_keys.get(k, k): v for k, v in step.items()}
            if filter_keys:
                step = {k: step[k] for k in filter_keys if k in step}
            if exclude_keys:
                step = {k: v for k, v in step.items() if k not in exclude_keys}
            if strip_none:
                step = dict(
                    FlextUtilitiesCollection.filter(step, lambda v: v is not None)
                )
            if strip_empty:
                step = dict(
                    FlextUtilitiesCollection.filter(
                        step,
                        lambda v: not FlextUtilitiesGuardsTypeCore.empty_value(v),
                    )
                )
            return step

        transform_result: p.Result[t.JsonMapping] = r[
            t.JsonMapping
        ].create_from_callable(_pipeline)
        if transform_result.failure:
            failure_reason = (
                transform_result.exception
                if isinstance(transform_result.exception, Exception)
                else transform_result.error
            )
            return r[t.JsonMapping].fail_op(
                "transform",
                failure_reason,
            )
        return transform_result


__all__: list[str] = ["FlextUtilitiesMapper"]
