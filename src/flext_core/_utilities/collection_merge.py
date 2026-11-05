"""Mapping-merge strategies — replace, filter, append, deep.

Hosts the `merge_mappings` engine and per-strategy handlers so
`collection.py` can keep just iterator + normalization logic and stay
under the 200-LOC cap (logical LOC, AGENTS.md §3.1).

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._utilities import FlextUtilitiesGuardsTypeCore, FlextUtilitiesPydantic


class FlextUtilitiesCollectionMerge:
    """Mapping-merge strategies + dispatcher (`merge_mappings`)."""

    @staticmethod
    def _is_json_mapping(
        value: flext_core.p.AttributeProbe,
    ) -> TypeGuard[flext_core.t.JsonMapping]:
        return isinstance(value, Mapping)

    @staticmethod
    def _is_json_list(
        value: flext_core.p.AttributeProbe,
    ) -> TypeGuard[flext_core.t.MutableJsonList]:
        return isinstance(value, list)

    @staticmethod
    def _merge_deep_single_key(
        result: flext_core.t.MutableJsonMapping,
        key: str,
        value: flext_core.t.JsonValue,
    ) -> flext_core.p.Result[bool]:
        """Merge single key in deep merge strategy.

        Returns:
            The resulting ``p.Result[bool]``.

        """
        current_val = result.get(key)
        if FlextUtilitiesCollectionMerge._is_json_mapping(
            current_val,
        ) and FlextUtilitiesCollectionMerge._is_json_mapping(value):
            result[key] = FlextUtilitiesPydantic.normalize_to_json_value({
                **current_val,
                **value,
            })
            return flext_core.FlextResult[bool].ok(value=True)
        result[key] = value
        return flext_core.FlextResult[bool].ok(value=True)

    @staticmethod
    def _merge_replace(
        other: flext_core.t.JsonMapping,
        base: flext_core.t.JsonMapping,
    ) -> flext_core.p.Result[flext_core.t.JsonMapping]:
        """Replace strategy: base values overwrite other.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: flext_core.t.MutableJsonMapping = dict(other)
        result.update(base)
        return flext_core.FlextResult[flext_core.t.JsonMapping].ok(result)

    @staticmethod
    def _merge_filter_none(
        other: flext_core.t.JsonMapping,
        base: flext_core.t.JsonMapping,
    ) -> flext_core.p.Result[flext_core.t.JsonMapping]:
        """Filter-none strategy: skip None values from base.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: flext_core.t.MutableJsonMapping = dict(other)
        result.update({k: v for k, v in base.items() if v is not None})
        return flext_core.FlextResult[flext_core.t.JsonMapping].ok(result)

    @staticmethod
    def _merge_filter_empty(
        other: flext_core.t.JsonMapping,
        base: flext_core.t.JsonMapping,
    ) -> flext_core.p.Result[flext_core.t.JsonMapping]:
        """Filter-empty strategy: skip empty values from base.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: flext_core.t.MutableJsonMapping = dict(other)
        result.update({
            k: v
            for k, v in base.items()
            if not FlextUtilitiesGuardsTypeCore.empty_value(v)
        })
        return flext_core.FlextResult[flext_core.t.JsonMapping].ok(result)

    @staticmethod
    def _merge_append(
        other: flext_core.t.JsonMapping,
        base: flext_core.t.JsonMapping,
    ) -> flext_core.p.Result[flext_core.t.JsonMapping]:
        """Append strategy: concatenate lists instead of replacing.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: flext_core.t.MutableJsonMapping = dict(other)
        for key, value in base.items():
            current_val = result.get(key)
            if FlextUtilitiesCollectionMerge._is_json_list(
                current_val,
            ) and FlextUtilitiesCollectionMerge._is_json_list(value):
                result[key] = FlextUtilitiesPydantic.normalize_to_json_value([
                    *current_val,
                    *value,
                ])
                continue
            result[key] = value
        return flext_core.FlextResult[flext_core.t.JsonMapping].ok(result)

    @staticmethod
    def _merge_deep(
        other: flext_core.t.JsonMapping,
        base: flext_core.t.JsonMapping,
    ) -> flext_core.p.Result[flext_core.t.JsonMapping]:
        """Deep strategy: recursively merge nested dicts.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: flext_core.t.MutableJsonMapping = dict(other)
        for key, value in base.items():
            merge_result = FlextUtilitiesCollectionMerge._merge_deep_single_key(
                result,
                key,
                value,
            )
            if merge_result.failure:
                return flext_core.FlextResult[flext_core.t.JsonMapping].from_failure(
                    merge_result
                )
        return flext_core.FlextResult[flext_core.t.JsonMapping].ok(result)

    _MERGE_STRATEGIES: ClassVar[
        Mapping[
            str,
            Callable[
                [flext_core.t.JsonMapping, flext_core.t.JsonMapping],
                flext_core.p.Result[flext_core.t.JsonMapping],
            ],
        ]
    ] = {
        flext_core.c.MergeStrategy.REPLACE: _merge_replace,
        flext_core.c.MergeStrategy.OVERRIDE: _merge_replace,
        flext_core.c.MergeStrategy.FILTER_NONE: _merge_filter_none,
        flext_core.c.MergeStrategy.FILTER_EMPTY: _merge_filter_empty,
        flext_core.c.MergeStrategy.FILTER_BOTH: _merge_filter_empty,
        flext_core.c.MergeStrategy.APPEND: _merge_append,
        flext_core.c.MergeStrategy.DEEP: _merge_deep,
    }

    @staticmethod
    def merge_mappings(
        other: flext_core.t.JsonMapping | None,
        base: flext_core.t.JsonMapping,
        *,
        strategy: str = flext_core.c.MergeStrategy.DEEP,
    ) -> flext_core.p.Result[flext_core.t.JsonMapping]:
        """Merge two dictionaries with configurable strategy.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        Raises:
            TypeError: If merge_mappings requires an iterable mapping for 'other', got
                None.

        """
        if other is None:
            msg = "merge_mappings requires an iterable mapping for 'other', got None"
            raise TypeError(msg)
        handler = FlextUtilitiesCollectionMerge._MERGE_STRATEGIES.get(strategy)
        if handler is None:
            return flext_core.FlextResult[flext_core.t.JsonMapping].fail(
                f"Unknown merge strategy: {strategy}"
            )
        return handler(other, base)


__all__: list[str] = ["FlextUtilitiesCollectionMerge"]
