"""Public collection facade composing iter + merge layers via MRO.

The heavy lifting lives in `collection_iter.py` (filter/map overloads) and
`collection_merge.py` (merge strategies). This file keeps just the small
helpers (count, find, process, normalize_domain_event_data) so it remains
under the 200-LOC cap (logical LOC, AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence

from flext_core import (
    FlextModelsContainers as mc,
    FlextRuntime,
    c,
    p,
    r,
    t,
)
from flext_core._utilities.collection_iter import FlextUtilitiesCollectionIter


class FlextUtilitiesCollection(FlextUtilitiesCollectionIter):
    """Facade composing iter + merge utilities; small helpers live here."""

    @staticmethod
    def normalize_domain_event_data(
        value: mc.ConfigMap | t.JsonPayload | None,
    ) -> Mapping[str, t.JsonValue]:
        """Normalize domain event payloads into plain flat mappings."""
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

    @staticmethod
    def find[TItem](
        items: Sequence[TItem] | tuple[TItem, ...] | Mapping[str, TItem],
        predicate: Callable[[TItem], bool],
    ) -> p.Result[TItem]:
        """Find first item matching predicate; returns r[T]."""
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

    @staticmethod
    def process[TItem, TMapped](
        items: Sequence[TItem],
        processor: Callable[[TItem], TMapped],
        *,
        predicate: Callable[[TItem], bool] | None = None,
        on_error: str = "fail",
    ) -> p.Result[Sequence[TMapped]]:
        """Process items with optional filter; ``on_error="skip"`` skips failures."""
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
