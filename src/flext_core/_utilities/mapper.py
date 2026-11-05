"""FlextUtilitiesMapper — data extraction, transformation, and aggregation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import starmap
from typing import TYPE_CHECKING, cast

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._utilities import (
    FlextUtilitiesCollection,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesMapperExtract,
    FlextUtilitiesPydantic,
)


class FlextUtilitiesMapper(FlextUtilitiesMapperExtract):
    """Data structure mapping, extraction, and transformation utilities."""

    @staticmethod
    def _transform_result(
        pipeline: Callable[[], flext_core.t.JsonDict],
    ) -> flext_core.p.Result[flext_core.t.JsonMapping]:
        """Run one transform pipeline, mapping failures to ``r``.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        transform_result: flext_core.p.Result[flext_core.t.JsonMapping] = (
            flext_core.FlextResult[flext_core.t.JsonMapping].create_from_callable(
                pipeline,
            )
        )
        if transform_result.failure:
            failure_reason = (
                transform_result.exception
                if isinstance(transform_result.exception, Exception)
                else transform_result.error
            )
            return flext_core.FlextResult[flext_core.t.JsonMapping].fail_op(
                "transform", failure_reason
            )
        return transform_result

    @staticmethod
    def _normalize_step(
        step: flext_core.t.JsonDict, *, normalize: bool
    ) -> flext_core.t.JsonDict:
        """Optionally normalize a step mapping to metadata form.

        Returns:
            The resulting ``t.JsonDict``.

        """
        if not normalize:
            return step
        normalized = FlextUtilitiesPydantic.normalize_to_json_value(step)
        if isinstance(normalized, Mapping):
            return dict(normalized)
        return step

    @staticmethod
    def _remap_step(
        step: flext_core.t.JsonDict,
        map_keys: Mapping[str, str] | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
    ) -> flext_core.t.JsonDict:
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

    @staticmethod
    def _strip_step(
        step: flext_core.t.JsonDict,
        *,
        strip_none: bool,
        strip_empty: bool,
    ) -> flext_core.t.JsonDict:
        """Optionally strip None values and empty values from a step mapping.

        Returns:
            The resulting ``t.JsonDict``.

        """
        result = step
        if strip_none:
            result = dict(
                FlextUtilitiesCollection.filter(
                    result, FlextUtilitiesMapper._is_not_none
                )
            )
        if strip_empty:
            return dict(
                FlextUtilitiesCollection.filter(
                    result, FlextUtilitiesMapper._is_not_empty
                )
            )
        return result

    @staticmethod
    def _is_not_none(value: flext_core.t.JsonValue) -> bool:
        """Whether a value is not None.

        Returns:
            The resulting ``bool``.

        """
        return value is not None

    @staticmethod
    def _is_not_empty(value: flext_core.t.JsonValue) -> bool:
        """Whether a value is not an empty scalar or container.

        Returns:
            The resulting ``bool``.

        """
        return not FlextUtilitiesGuardsTypeCore.empty_value(value)

    @staticmethod
    def agg[T](
        items: Sequence[T] | tuple[T, ...],
        field: str | Callable[[T], flext_core.t.Numeric],
        *,
        fn: Callable[[Sequence[flext_core.t.Numeric]], flext_core.t.Numeric]
        | None = None,
    ) -> flext_core.t.Numeric:
        """Aggregate numeric field values from objects using fn (default: sum).

        Returns:
            The resulting ``t.Numeric``.

        """
        items_list: Sequence[T] = list(items)
        if callable(field):
            numeric_values: list[flext_core.t.Numeric] = [
                field(item) for item in items_list
            ]
        else:
            numeric_values = []
            for item in items_list:
                raw: flext_core.p.AttributeProbe | None
                if isinstance(item, flext_core.m.BaseModel):
                    raw = getattr(item, field, None)
                elif isinstance(item, Mapping):
                    raw = cast(
                        "Mapping[str, flext_core.p.AttributeProbe]",
                        item,
                    ).get(field)
                else:
                    continue
                if isinstance(raw, flext_core.c.NUMERIC_TYPES):
                    numeric_values.append(raw)
        agg_fn = fn if fn is not None else sum
        return agg_fn(numeric_values) if numeric_values else 0

    @staticmethod
    def _deep_eq_values(
        val_a: flext_core.t.JsonPayload | flext_core.t.JsonValue,
        val_b: flext_core.t.JsonPayload | flext_core.t.JsonValue,
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
        a: Mapping[str, flext_core.t.JsonValue | flext_core.t.JsonPayload],
        b: Mapping[str, flext_core.t.JsonValue | flext_core.t.JsonPayload],
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
            for key, val_a in flext_core.a.items()
        )

    @staticmethod
    def prop(
        key: str,
    ) -> Callable[
        [flext_core.p.ConfigModelInput],
        flext_core.t.JsonPayload | flext_core.t.JsonValue,
    ]:
        """Return an accessor function that extracts the named property from an object.

        Returns:
            An accessor function that extracts the named property from an object.

        """

        def accessor(
            obj: flext_core.p.ConfigModelInput,
        ) -> flext_core.t.JsonPayload | flext_core.t.JsonValue:
            result = FlextUtilitiesMapper._get_raw(obj, key)
            return result if result is not None else ""

        return accessor

    @staticmethod
    def transform(
        source: flext_core.t.JsonMapping | flext_core.m.ConfigMap,
        *,
        options: flext_core.p.TransformOptions | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> flext_core.p.Result[flext_core.t.JsonMapping]:
        """Apply the dict normalization pipeline to a mapping.

        Steps: normalize, strip_none, strip_empty, map_keys, filter_keys,
        exclude_keys. The first four steps are grouped in ``options``; the
        key-filter steps stay keyword-only for direct callers.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        resolved = options if options is not None else (flext_core.m.TransformOptions())
        coerced: flext_core.t.JsonMapping = (
            {
                k: FlextUtilitiesPydantic.normalize_to_json_value(v)
                for k, v in source.root.items()
            }
            if isinstance(source, flext_core.m.ConfigMap)
            else source
        )

        def _pipeline() -> flext_core.t.JsonDict:
            normalized = FlextUtilitiesMapper._normalize_step(
                dict(coerced), normalize=resolved.normalize
            )
            step = FlextUtilitiesMapper._remap_step(
                normalized, resolved.map_keys, filter_keys, exclude_keys
            )
            return FlextUtilitiesMapper._strip_step(
                step,
                strip_none=resolved.strip_none,
                strip_empty=resolved.strip_empty,
            )

        return FlextUtilitiesMapper._transform_result(_pipeline)


__all__: list[str] = ["FlextUtilitiesMapper"]
