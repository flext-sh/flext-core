"""FlextUtilitiesMapper — data extraction, transformation, and aggregation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import starmap
from typing import TYPE_CHECKING, Annotated

from flext_core import c, m, r, t
from flext_core._models import FlextModelsPydantic
from flext_core._utilities import FlextUtilitiesCollection, FlextUtilitiesGuardsTypeCore
from flext_core._utilities.mapper_extract import FlextUtilitiesMapperExtract
from flext_core.runtime import FlextRuntime

if TYPE_CHECKING:
    from flext_core import p


def _transform_result(pipeline: Callable[[], t.JsonDict]) -> p.Result[t.JsonMapping]:
    """Run one transform pipeline, mapping failures to ``r``.

    Returns:
        The resulting ``p.Result[t.JsonMapping]``.

    """
    transform_result: p.Result[t.JsonMapping] = r[t.JsonMapping].create_from_callable(
        pipeline,
    )
    if transform_result.failure:
        failure_reason = (
            transform_result.exception
            if isinstance(transform_result.exception, Exception)
            else transform_result.error
        )
        return r[t.JsonMapping].fail_op("transform", failure_reason)
    return transform_result


def _normalize_step(step: t.JsonDict, *, normalize: bool) -> t.JsonDict:
    """Optionally normalize a step mapping to metadata form.

    Returns:
        The resulting ``t.JsonDict``.

    """
    if not normalize:
        return step
    normalized = FlextRuntime.normalize_to_metadata(step)
    if FlextUtilitiesGuardsTypeCore.mapping(normalized):
        return dict(normalized)
    return step


def _remap_step(
    step: t.JsonDict,
    map_keys: t.StrMapping | None,
    filter_keys: set[str] | None,
    exclude_keys: set[str] | None,
) -> t.JsonDict:
    """Apply key mapping, filtering, and exclusion to a step mapping.

    Returns:
        The resulting ``t.JsonDict``.

    """
    result = step
    if map_keys:
        result = {map_keys.get(k, k): v for k, v in result.items()}
    if filter_keys:
        result = {k: result[k] for k in filter_keys if k in result}
    if exclude_keys:
        result = {k: v for k, v in result.items() if k not in exclude_keys}
    return result


def _strip_step(
    step: t.JsonDict,
    *,
    strip_none: bool,
    strip_empty: bool,
) -> t.JsonDict:
    """Optionally strip None values and empty values from a step mapping.

    Returns:
        The resulting ``t.JsonDict``.

    """
    result = step
    if strip_none:
        result = dict(FlextUtilitiesCollection.filter(result, _is_not_none))
    if strip_empty:
        return dict(FlextUtilitiesCollection.filter(result, _is_not_empty))
    return result


def _is_not_none(value: t.JsonValue) -> bool:
    """Whether a value is not None.

    Returns:
        The resulting ``bool``.

    """
    return value is not None


def _is_not_empty(value: t.JsonValue) -> bool:
    """Whether a value is not an empty scalar or container.

    Returns:
        The resulting ``bool``.

    """
    return not FlextUtilitiesGuardsTypeCore.empty_value(value)


class FlextUtilitiesMapper(FlextUtilitiesMapperExtract):
    """Data structure mapping, extraction, and transformation utilities."""

    class TransformOptions(m.Value):
        """Validated option envelope for the ``transform`` value pipeline."""

        normalize: Annotated[
            bool, m.Field(description="Normalize values to metadata form")
        ] = False
        strip_none: Annotated[bool, m.Field(description="Strip ``None`` values")] = (
            False
        )
        strip_empty: Annotated[bool, m.Field(description="Strip empty values")] = False
        map_keys: Annotated[
            t.StrMapping | None,
            m.Field(description="Optional old-key to new-key mapping"),
        ] = None

    @staticmethod
    def agg[T](
        items: t.SequenceOf[T] | t.VariadicTuple[T],
        field: str | Callable[[T], t.Numeric],
        *,
        fn: Callable[[Sequence[t.Numeric]], t.Numeric] | None = None,
    ) -> t.Numeric:
        """Aggregate numeric field values from objects using fn (default: sum).

        Returns:
            The resulting ``t.Numeric``.

        """
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
                if isinstance(raw, c.NUMERIC_TYPES):
                    numeric_values.append(raw)
        agg_fn = fn if fn is not None else sum
        return agg_fn(numeric_values) if numeric_values else 0

    @staticmethod
    def _deep_eq_values(
        val_a: t.JsonPayload | t.JsonValue,
        val_b: t.JsonPayload | t.JsonValue,
    ) -> bool:
        """Recursive deep equality for any two nested items.

        Returns:
            The resulting ``bool``.

        """
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
        """Recursive deep equality for nested dicts/lists/primitives.

        Returns:
            The resulting ``bool``.

        """
        if a is b:
            return True
        if len(a) != len(b):
            return False
        return all(
            key in b and FlextUtilitiesMapper._deep_eq_values(val_a, b[key])
            for key, val_a in a.items()
        )

    @staticmethod
    def prop(key: str) -> Callable[[t.ConfigModelInput], t.JsonPayload | t.JsonValue]:
        """Return an accessor function that extracts the named property from an object.

        Returns:
            An accessor function that extracts the named property from an object.

        """

        def accessor(obj: t.ConfigModelInput) -> t.JsonPayload | t.JsonValue:
            result = FlextUtilitiesMapper._get_raw(obj, key)
            return result if result is not None else ""

        return accessor

    @staticmethod
    def transform(
        source: t.JsonMapping | m.ConfigMap,
        *,
        options: FlextUtilitiesMapper.TransformOptions | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> p.Result[t.JsonMapping]:
        """Apply the dict normalization pipeline to a mapping.

        Steps: normalize, strip_none, strip_empty, map_keys, filter_keys,
        exclude_keys. The first four steps are grouped in ``options``; the
        key-filter steps stay keyword-only for direct callers.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        resolved = (
            options
            if options is not None
            else (FlextUtilitiesMapper.TransformOptions())
        )
        coerced: t.JsonMapping = (
            {k: FlextRuntime.normalize_to_metadata(v) for k, v in source.root.items()}
            if isinstance(source, m.ConfigMap)
            else source
        )

        def _pipeline() -> t.JsonDict:
            normalized = _normalize_step(dict(coerced), normalize=resolved.normalize)
            step = _remap_step(normalized, resolved.map_keys, filter_keys, exclude_keys)
            return _strip_step(
                step,
                strip_none=resolved.strip_none,
                strip_empty=resolved.strip_empty,
            )

        return _transform_result(_pipeline)


__all__: list[str] = ["FlextUtilitiesMapper"]
