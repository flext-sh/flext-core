"""Typed immutable defaults shared by Pydantic model fields."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Never, override


class ImmutableEmptyMapping[K, V](Mapping[K, V]):
    """Fully typed immutable empty mapping used as a field factory.

    Kept beside the ``immutable_empty_mapping`` factory: every consumer
    imported on 0.12.0-dev still binds the class directly.
    """

    @override
    def __getitem__(self, key: K) -> V:
        """Reject every key because the mapping is empty."""
        raise KeyError(key)

    @override
    def __iter__(self) -> Iterator[K]:
        """Iterate over no keys."""
        return iter(())

    @override
    def __len__(self) -> int:
        """Return the invariant empty size."""
        return 0


def immutable_empty_mapping() -> Mapping[Any, Never]:
    """Return a fresh immutable empty mapping."""
    empty: dict[Any, Never] = {}
    return MappingProxyType(empty)


# Internal owner: direct module imports are intentional; no facade ABI is published.
__all__: tuple[str, ...] = ()
