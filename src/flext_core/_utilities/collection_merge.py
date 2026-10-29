"""Mapping-merge strategies — replace, filter, append, deep.

Hosts the `merge_mappings` engine and per-strategy handlers so
`collection.py` can keep just iterator + normalization logic and stay
under the 200-LOC cap (logical LOC, AGENTS.md §3.1).

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import ClassVar, TypeGuard

from flext_core import FlextResult, c, p, t
from flext_core._utilities import FlextUtilitiesGuardsTypeCore, FlextUtilitiesPydantic


class FlextUtilitiesCollectionMerge:
    """Mapping-merge strategies + dispatcher (`merge_mappings`)."""

    @staticmethod
    def _is_json_mapping(value: p.AttributeProbe) -> TypeGuard[t.JsonMapping]:
        return isinstance(value, Mapping)

    @staticmethod
    def _is_json_list(value: p.AttributeProbe) -> TypeGuard[t.MutableJsonList]:
        return isinstance(value, list)

    @staticmethod
    def _merge_deep_single_key(
        result: t.MutableJsonMapping,
        key: str,
        value: t.JsonValue,
    ) -> p.Result[bool]:
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
            return FlextResult[bool].ok(value=True)
        result[key] = value
        return FlextResult[bool].ok(value=True)

    @staticmethod
    def _merge_replace(
        other: t.JsonMapping,
        base: t.JsonMapping,
    ) -> p.Result[t.JsonMapping]:
        """Replace strategy: base values overwrite other.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: t.MutableJsonMapping = dict(other)
        result.update(base)
        return FlextResult[t.JsonMapping].ok(result)

    @staticmethod
    def _merge_filter_none(
        other: t.JsonMapping,
        base: t.JsonMapping,
    ) -> p.Result[t.JsonMapping]:
        """Filter-none strategy: skip None values from base.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: t.MutableJsonMapping = dict(other)
        result.update({k: v for k, v in base.items() if v is not None})
        return FlextResult[t.JsonMapping].ok(result)

    @staticmethod
    def _merge_filter_empty(
        other: t.JsonMapping,
        base: t.JsonMapping,
    ) -> p.Result[t.JsonMapping]:
        """Filter-empty strategy: skip empty values from base.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: t.MutableJsonMapping = dict(other)
        result.update({
            k: v
            for k, v in base.items()
            if not FlextUtilitiesGuardsTypeCore.empty_value(v)
        })
        return FlextResult[t.JsonMapping].ok(result)

    @staticmethod
    def _merge_append(
        other: t.JsonMapping,
        base: t.JsonMapping,
    ) -> p.Result[t.JsonMapping]:
        """Append strategy: concatenate lists instead of replacing.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: t.MutableJsonMapping = dict(other)
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
        return FlextResult[t.JsonMapping].ok(result)

    @staticmethod
    def _merge_deep(
        other: t.JsonMapping,
        base: t.JsonMapping,
    ) -> p.Result[t.JsonMapping]:
        """Deep strategy: recursively merge nested dicts.

        Returns:
            The resulting ``p.Result[t.JsonMapping]``.

        """
        result: t.MutableJsonMapping = dict(other)
        for key, value in base.items():
            merge_result = FlextUtilitiesCollectionMerge._merge_deep_single_key(
                result,
                key,
                value,
            )
            if merge_result.failure:
                return FlextResult[t.JsonMapping].from_failure(merge_result)
        return FlextResult[t.JsonMapping].ok(result)

    _MERGE_STRATEGIES: ClassVar[
        Mapping[str, Callable[[t.JsonMapping, t.JsonMapping], p.Result[t.JsonMapping]]]
    ] = {
        c.MergeStrategy.REPLACE: _merge_replace,
        c.MergeStrategy.OVERRIDE: _merge_replace,
        c.MergeStrategy.FILTER_NONE: _merge_filter_none,
        c.MergeStrategy.FILTER_EMPTY: _merge_filter_empty,
        c.MergeStrategy.FILTER_BOTH: _merge_filter_empty,
        c.MergeStrategy.APPEND: _merge_append,
        c.MergeStrategy.DEEP: _merge_deep,
    }

    @staticmethod
    def merge_mappings(
        other: t.JsonMapping | None,
        base: t.JsonMapping,
        *,
        strategy: str = c.MergeStrategy.DEEP,
    ) -> p.Result[t.JsonMapping]:
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
            return FlextResult[t.JsonMapping].fail(
                f"Unknown merge strategy: {strategy}"
            )
        return handler(other, base)


__all__: list[str] = ["FlextUtilitiesCollectionMerge"]
