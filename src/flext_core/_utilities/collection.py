"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from typing import overload

from flext_core import (
    FlextModelsContainers as mc,
    FlextRuntime,
    c,
    p,
    r,
    t,
)


class FlextUtilitiesCollection:
    """Utilities for collection operations with full generic type support."""

    @staticmethod
    def normalize_domain_event_data(
        value: mc.ConfigMap | t.JsonPayload | None,
    ) -> Mapping[str, t.JsonValue]:
        """Normalize domain event payloads into plain flat mappings.

        Moved from FlextUtilitiesDomain so the DomainEvent model can consume it
        without creating a utility→model→utility cycle.

        Used as a pydantic ``BeforeValidator`` — raw input may be any
        ``JsonPayload``. Non-mapping inputs (scalars, sequences) are
        re-validated through ``mc.ConfigMap.model_validate`` so pydantic
        surfaces a canonical ``ValidationError`` instead of a raw
        ``AttributeError`` from downstream iteration.
        """
        if value is None:
            return {}
        raw_source = value.root if isinstance(value, mc.ConfigMap) else value
        if not isinstance(raw_source, Mapping):
            validated = mc.ConfigMap.model_validate(raw_source)
            return FlextUtilitiesCollection.normalize_domain_event_data(validated)
        normalized: MutableMapping[str, t.JsonValue] = {}
        for key, item in raw_source.items():
            if item is None:
                continue
            normalized[str(key)] = FlextRuntime.normalize_to_metadata(item)
        return normalized

    @staticmethod
    def count[TItem](
        items: Sequence[TItem],
        predicate: Callable[[TItem], bool] | None = None,
    ) -> int:
        """Count items, optionally matching predicate."""
        if predicate is None:
            return len(items)
        return sum(1 for item in items if predicate(item))

    @overload
    @staticmethod
    def filter[TItem](
        items: Sequence[TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: None = None,
    ) -> Sequence[TItem]: ...

    @overload
    @staticmethod
    def filter[TItem, TMapped](
        items: Sequence[TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: Callable[[TItem], TMapped],
    ) -> Sequence[TMapped]: ...

    @overload
    @staticmethod
    def filter[TItem](
        items: tuple[TItem, ...],
        predicate: Callable[[TItem], bool],
        *,
        mapper: None = None,
    ) -> tuple[TItem, ...]: ...

    @overload
    @staticmethod
    def filter[TItem, TMapped](
        items: tuple[TItem, ...],
        predicate: Callable[[TItem], bool],
        *,
        mapper: Callable[[TItem], TMapped],
    ) -> tuple[TMapped, ...]: ...

    @overload
    @staticmethod
    def filter[TItem](
        items: Mapping[str, TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: None = None,
    ) -> Mapping[str, TItem]: ...

    @overload
    @staticmethod
    def filter[TItem, TMapped](
        items: Mapping[str, TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: Callable[[TItem], TMapped],
    ) -> Mapping[str, TMapped]: ...

    @staticmethod
    def filter[TItem, TMapped](
        items: Sequence[TItem] | tuple[TItem, ...] | Mapping[str, TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: Callable[[TItem], TMapped] | None = None,
    ) -> (
        Sequence[TItem]
        | Sequence[TMapped]
        | tuple[TItem, ...]
        | tuple[TMapped, ...]
        | Mapping[str, TItem]
        | Mapping[str, TMapped]
    ):
        """Unified filter function with generic type support.

        Filters elements based on predicate while preserving container type.
        Optionally maps filtered items with mapper function.
        Supports lists, tuples, and dicts.
        """
        if isinstance(items, list):
            if mapper is not None:
                return [mapper(item) for item in items if predicate(item)]
            return [item for item in items if predicate(item)]
        if isinstance(items, tuple):
            if mapper is not None:
                mapped_items = [mapper(item) for item in items if predicate(item)]
                return (*mapped_items,)
            filtered_items = [item for item in items if predicate(item)]
            return (*filtered_items,)
        if isinstance(items, Mapping):
            filtered: MutableMapping[str, TItem] = {
                k: v for k, v in items.items() if predicate(v)
            }
            if mapper is not None:
                return {k: mapper(v) for k, v in filtered.items()}
            return filtered
        return [item for item in items if predicate(item)]

    @staticmethod
    def find[TItem](
        items: Sequence[TItem] | tuple[TItem, ...] | Mapping[str, TItem],
        predicate: Callable[[TItem], bool],
    ) -> p.Result[TItem]:
        """Find first item matching predicate with generic type support.

        Returns first item where predicate returns True, or None.
        """
        if isinstance(items, (list, tuple)):
            for item in items:
                if predicate(item):
                    return r[TItem].ok(item)
            return r[TItem].fail(c.ERR_COLLECTION_NO_MATCHING_ITEM_FOUND)
        if isinstance(items, Mapping):
            for v in items.values():
                if predicate(v):
                    return r[TItem].ok(v)
        return r[TItem].fail(c.ERR_COLLECTION_NO_MATCHING_ITEM_FOUND)

    @overload
    @staticmethod
    def map[TItem, TMapped](
        items: Sequence[TItem],
        mapper: Callable[[TItem], TMapped],
    ) -> Sequence[TMapped]: ...

    @overload
    @staticmethod
    def map[TItem, TMapped](
        items: tuple[TItem, ...],
        mapper: Callable[[TItem], TMapped],
    ) -> tuple[TMapped, ...]: ...

    @overload
    @staticmethod
    def map[TItem, TMapped](
        items: Mapping[str, TItem],
        mapper: Callable[[TItem], TMapped],
    ) -> Mapping[str, TMapped]: ...

    @overload
    @staticmethod
    def map[TItem, TMapped](
        items: set[TItem],
        mapper: Callable[[TItem], TMapped],
    ) -> set[TMapped]: ...

    @overload
    @staticmethod
    def map[TItem, TMapped](
        items: frozenset[TItem],
        mapper: Callable[[TItem], TMapped],
    ) -> frozenset[TMapped]: ...

    @staticmethod
    def map[TItem, TMapped](
        items: Sequence[TItem]
        | tuple[TItem, ...]
        | Mapping[str, TItem]
        | set[TItem]
        | frozenset[TItem],
        mapper: Callable[[TItem], TMapped],
    ) -> (
        Sequence[TMapped]
        | tuple[TMapped, ...]
        | Mapping[str, TMapped]
        | set[TMapped]
        | frozenset[TMapped]
    ):
        """Unified map function with generic type support.

        Transforms elements using mapper function while preserving container type.
        Supports lists, tuples, dicts, sets, and frozensets.
        """
        if isinstance(items, list):
            return [mapper(item) for item in items]
        if isinstance(items, tuple):
            return tuple(mapper(item) for item in items)
        if isinstance(items, Mapping):
            return {k: mapper(v) for k, v in items.items()}
        if isinstance(items, set):
            return {mapper(item) for item in items}
        return frozenset(mapper(item) for item in items)

    @staticmethod
    def process[TItem, TMapped](
        items: Sequence[TItem],
        processor: Callable[[TItem], TMapped],
        *,
        predicate: Callable[[TItem], bool] | None = None,
        on_error: str = "fail",
    ) -> p.Result[Sequence[TMapped]]:
        """Process items with optional filtering; on_error="skip" to skip failures."""
        results: MutableSequence[TMapped] = []
        for item in items:
            item_typed: TItem = item
            if predicate is not None and (not predicate(item_typed)):
                continue
            process_result = r[TMapped].create_from_callable(
                lambda current_item=item_typed: processor(current_item),
            )
            if process_result.failure:
                if on_error == "skip":
                    continue
                return r[Sequence[TMapped]].fail(
                    c.ERR_COLLECTION_PROCESSING_FAILED_FOR_ITEM.format(item=item),
                )
            processed_item: TMapped = process_result.unwrap()
            results.append(processed_item)
        return r[Sequence[TMapped]].ok(results)


__all__: list[str] = ["FlextUtilitiesCollection"]
