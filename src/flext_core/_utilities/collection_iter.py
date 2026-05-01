"""Generic-iterator utilities — `filter` and `map` (overloaded by container).

Hosts the heavy `@overload` matrices for `filter` and `map` so the public
collection facade stays under the 200-LOC cap (logical LOC, AGENTS.md §3.1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import overload

from flext_core import t
from flext_core._utilities.collection_merge import FlextUtilitiesCollectionMerge


class FlextUtilitiesCollectionIter(FlextUtilitiesCollectionMerge):
    """Generic, container-preserving `filter` and `map` helpers."""

    @overload
    @staticmethod
    def filter[TItem](
        items: t.SequenceOf[TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: None = None,
    ) -> t.SequenceOf[TItem]: ...

    @overload
    @staticmethod
    def filter[TItem, TMapped](
        items: t.SequenceOf[TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: Callable[[TItem], TMapped],
    ) -> t.SequenceOf[TMapped]: ...

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
        items: t.MappingKV[str, TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: None = None,
    ) -> t.MappingKV[str, TItem]: ...

    @overload
    @staticmethod
    def filter[TItem, TMapped](
        items: t.MappingKV[str, TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: Callable[[TItem], TMapped],
    ) -> t.MappingKV[str, TMapped]: ...

    @staticmethod
    def filter[TItem, TMapped](
        items: t.SequenceOf[TItem] | tuple[TItem, ...] | t.MappingKV[str, TItem],
        predicate: Callable[[TItem], bool],
        *,
        mapper: Callable[[TItem], TMapped] | None = None,
    ) -> (
        t.SequenceOf[TItem]
        | t.SequenceOf[TMapped]
        | tuple[TItem, ...]
        | tuple[TMapped, ...]
        | t.MappingKV[str, TItem]
        | t.MappingKV[str, TMapped]
    ):
        """Unified filter function — preserves container type, optional mapper."""
        filtered_output: (
            t.SequenceOf[TItem]
            | t.SequenceOf[TMapped]
            | tuple[TItem, ...]
            | tuple[TMapped, ...]
            | t.MappingKV[str, TItem]
            | t.MappingKV[str, TMapped]
        )
        if isinstance(items, Mapping):
            filtered_mapping = {
                key: value for key, value in items.items() if predicate(value)
            }
            filtered_output = (
                {key: mapper(value) for key, value in filtered_mapping.items()}
                if mapper is not None
                else filtered_mapping
            )
        else:
            sequence_items: t.SequenceOf[TItem] = items
            filtered_sequence: list[TItem] = [
                sequence_items[index]
                for index in range(len(sequence_items))
                if predicate(sequence_items[index])
            ]
            match sequence_items:
                case tuple():
                    filtered_output = (
                        tuple(mapper(item) for item in filtered_sequence)
                        if mapper is not None
                        else tuple(filtered_sequence)
                    )
                case _:
                    filtered_output = (
                        [mapper(item) for item in filtered_sequence]
                        if mapper is not None
                        else filtered_sequence
                    )
        return filtered_output

    @overload
    @staticmethod
    def map[TItem, TMapped](
        items: t.SequenceOf[TItem],
        mapper: Callable[[TItem], TMapped],
    ) -> t.SequenceOf[TMapped]: ...

    @overload
    @staticmethod
    def map[TItem, TMapped](
        items: tuple[TItem, ...],
        mapper: Callable[[TItem], TMapped],
    ) -> tuple[TMapped, ...]: ...

    @overload
    @staticmethod
    def map[TItem, TMapped](
        items: t.MappingKV[str, TItem],
        mapper: Callable[[TItem], TMapped],
    ) -> t.MappingKV[str, TMapped]: ...

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
        items: t.SequenceOf[TItem]
        | tuple[TItem, ...]
        | t.MappingKV[str, TItem]
        | set[TItem]
        | frozenset[TItem],
        mapper: Callable[[TItem], TMapped],
    ) -> (
        t.SequenceOf[TMapped]
        | tuple[TMapped, ...]
        | t.MappingKV[str, TMapped]
        | set[TMapped]
        | frozenset[TMapped]
    ):
        """Unified map function — preserves container type."""
        if isinstance(items, list):
            return [mapper(item) for item in items]
        if isinstance(items, tuple):
            return tuple(mapper(item) for item in items)
        if isinstance(items, Mapping):
            return {k: mapper(v) for k, v in items.items()}
        if isinstance(items, set):
            return {mapper(item) for item in items}
        return frozenset(mapper(item) for item in items)


__all__: list[str] = ["FlextUtilitiesCollectionIter"]
