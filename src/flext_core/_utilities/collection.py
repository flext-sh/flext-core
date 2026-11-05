"""Public collection facade composing iter + merge layers via MRO.

The heavy lifting lives in `collection_iter.py` (filter/map overloads) and
`collection_merge.py` (merge strategies). This file keeps just the small
helpers (count, find, process) so it remains
under the 200-LOC cap (logical LOC, AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSequence, Sequence
from functools import partial
from typing import TYPE_CHECKING

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._utilities.collection_iter import FlextUtilitiesCollectionIter
from flext_core._utilities.collection_merge import FlextUtilitiesCollectionMerge


class FlextUtilitiesCollection(
    FlextUtilitiesCollectionIter,
    FlextUtilitiesCollectionMerge,
):
    """Facade composing iter + merge utilities; small helpers live here."""

    @staticmethod
    def count[TItem](
        items: Sequence[TItem],
        predicate: Callable[[TItem], bool] | None = None,
    ) -> int:
        """Count items, optionally matching predicate.

        Returns:
            The resulting ``int``.

        """
        if predicate is None:
            return len(items)
        return sum(1 for item in items if predicate(item))

    @staticmethod
    def find[TItem](
        items: Sequence[TItem] | tuple[TItem] | Mapping[str, TItem],
        predicate: Callable[[TItem], bool],
    ) -> flext_core.p.Result[TItem]:
        """Find first item matching predicate; returns r[T].

        Returns:
            The resulting ``p.Result[TItem]``.

        """
        if isinstance(items, Mapping):
            for v in items.values():
                if predicate(v):
                    return flext_core.FlextResult[TItem].ok(v)
            return flext_core.FlextResult[TItem].fail(
                flext_core.c.ERR_COLLECTION_NO_MATCHING_ITEM_FOUND
            )
        if isinstance(items, flext_core.c.SEQUENCE_PAIR_TYPES):
            for item in items:
                if predicate(item):
                    return flext_core.FlextResult[TItem].ok(item)
            return flext_core.FlextResult[TItem].fail(
                flext_core.c.ERR_COLLECTION_NO_MATCHING_ITEM_FOUND
            )
        return flext_core.FlextResult[TItem].fail(
            flext_core.c.ERR_COLLECTION_NO_MATCHING_ITEM_FOUND
        )

    @staticmethod
    def process[TItem, TMapped](
        items: Sequence[TItem],
        processor: Callable[[TItem], TMapped],
        *,
        predicate: Callable[[TItem], bool] | None = None,
    ) -> flext_core.p.Result[Sequence[TMapped]]:
        """Map items (optionally filtered); the first item failure ends the run.

        The returned failure carries the processor's exception and error code
        so callers keep the originating cause.

        Returns:
            The resulting ``p.Result[Sequence[TMapped]]``.

        """
        results: MutableSequence[TMapped] = []
        for item in items:
            if predicate is not None and not predicate(item):
                continue
            process_result = flext_core.FlextResult[TMapped].create_from_callable(
                partial(processor, item)
            )
            if process_result.failure:
                return flext_core.FlextResult[Sequence[TMapped]].fail(
                    flext_core.c.ERR_COLLECTION_PROCESSING_FAILED_FOR_ITEM.format(
                        item=item,
                        error=process_result.error,
                    ),
                    error_code=process_result.error_code,
                    exception=process_result.exception,
                )
            results.append(process_result.unwrap())
        return flext_core.FlextResult[Sequence[TMapped]].ok(results)


__all__: list[str] = ["FlextUtilitiesCollection"]
