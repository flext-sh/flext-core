"""Mapping-merge strategies — replace, filter, append, deep.

Hosts the `merge_mappings` engine and per-strategy handlers so
`collection.py` can keep just iterator + normalization logic and stay
under the 200-LOC cap (logical LOC, AGENTS.md §3.1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import ClassVar

from flext_core import (
    FlextRuntime,
    FlextUtilitiesGuardsTypeCore,
    p,
    r,
    t,
)
from flext_core._constants.cqrs import FlextConstantsCqrs as _c_cqrs


class FlextUtilitiesCollectionMerge:
    """Mapping-merge strategies + dispatcher (`merge_mappings`)."""

    @staticmethod
    def _merge_deep_single_key(
        result: dict[str, t.JsonValue],
        key: str,
        value: t.JsonValue,
    ) -> p.Result[bool]:
        """Merge single key in deep merge strategy."""
        current_val = result.get(key)
        if (
            current_val is not None
            and isinstance(current_val, Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = FlextRuntime.normalize_to_metadata({**current_val, **value})
            return r[bool].ok(True)
        result[key] = value
        return r[bool].ok(True)

    @staticmethod
    def _merge_replace(
        other: Mapping[str, t.JsonValue],
        base: Mapping[str, t.JsonValue],
    ) -> p.Result[Mapping[str, t.JsonValue]]:
        """Replace strategy: base values overwrite other."""
        result: dict[str, t.JsonValue] = dict(other)
        result.update(base)
        return r[Mapping[str, t.JsonValue]].ok(result)

    @staticmethod
    def _merge_filter_none(
        other: Mapping[str, t.JsonValue],
        base: Mapping[str, t.JsonValue],
    ) -> p.Result[Mapping[str, t.JsonValue]]:
        """Filter-none strategy: skip None values from base."""
        result: dict[str, t.JsonValue] = dict(other)
        result.update({k: v for k, v in base.items() if v is not None})
        return r[Mapping[str, t.JsonValue]].ok(result)

    @staticmethod
    def _merge_filter_empty(
        other: Mapping[str, t.JsonValue],
        base: Mapping[str, t.JsonValue],
    ) -> p.Result[Mapping[str, t.JsonValue]]:
        """Filter-empty strategy: skip empty values from base."""
        result: dict[str, t.JsonValue] = dict(other)
        result.update({
            k: v
            for k, v in base.items()
            if not FlextUtilitiesGuardsTypeCore.empty_value(v)
        })
        return r[Mapping[str, t.JsonValue]].ok(result)

    @staticmethod
    def _merge_append(
        other: Mapping[str, t.JsonValue],
        base: Mapping[str, t.JsonValue],
    ) -> p.Result[Mapping[str, t.JsonValue]]:
        """Append strategy: concatenate lists instead of replacing."""
        result: dict[str, t.JsonValue] = dict(other)
        for key, value in base.items():
            current_val = result.get(key)
            if (
                current_val is not None
                and isinstance(current_val, list)
                and isinstance(value, list)
            ):
                result[key] = FlextRuntime.normalize_to_metadata(
                    [*current_val, *value],
                )
            else:
                result[key] = value
        return r[Mapping[str, t.JsonValue]].ok(result)

    @staticmethod
    def _merge_deep(
        other: Mapping[str, t.JsonValue],
        base: Mapping[str, t.JsonValue],
    ) -> p.Result[Mapping[str, t.JsonValue]]:
        """Deep strategy: recursively merge nested dicts."""
        result: dict[str, t.JsonValue] = dict(other)
        for key, value in base.items():
            merge_result = FlextUtilitiesCollectionMerge._merge_deep_single_key(
                result,
                key,
                value,
            )
            if merge_result.failure:
                return r[Mapping[str, t.JsonValue]].fail(
                    merge_result.error or "Unknown error",
                )
        return r[Mapping[str, t.JsonValue]].ok(result)

    _MergeHandler = Callable[
        [Mapping[str, t.JsonValue], Mapping[str, t.JsonValue]],
        "p.Result[Mapping[str, t.JsonValue]]",
    ]

    _MERGE_STRATEGIES: ClassVar[Mapping[str, _MergeHandler]] = {
        _c_cqrs.MergeStrategy.REPLACE: _merge_replace,
        _c_cqrs.MergeStrategy.OVERRIDE: _merge_replace,
        _c_cqrs.MergeStrategy.FILTER_NONE: _merge_filter_none,
        _c_cqrs.MergeStrategy.FILTER_EMPTY: _merge_filter_empty,
        _c_cqrs.MergeStrategy.FILTER_BOTH: _merge_filter_empty,
        _c_cqrs.MergeStrategy.APPEND: _merge_append,
        _c_cqrs.MergeStrategy.DEEP: _merge_deep,
    }

    @staticmethod
    def merge_mappings(
        other: Mapping[str, t.JsonValue] | None,
        base: Mapping[str, t.JsonValue],
        *,
        strategy: str = _c_cqrs.MergeStrategy.DEEP,
    ) -> p.Result[Mapping[str, t.JsonValue]]:
        """Merge two dictionaries with configurable strategy."""
        if other is None:
            msg = "merge_mappings requires an iterable mapping for 'other', got None"
            raise TypeError(msg)
        handler = FlextUtilitiesCollectionMerge._MERGE_STRATEGIES.get(strategy)
        if handler is None:
            return r[Mapping[str, t.JsonValue]].fail(
                f"Unknown merge strategy: {strategy}",
            )
        return handler(other, base)


__all__: list[str] = ["FlextUtilitiesCollectionMerge"]
