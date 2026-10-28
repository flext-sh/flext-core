"""Bootstrap-safe native value descriptor preserving complete callable types.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)


class FlextTypingNativeExport[T]:
    """Expose a native value without binding or narrowing callable overloads."""

    __slots__ = ("_value",)

    def __init__(self, value: T) -> None:
        """Retain the original value and its complete inferred type."""
        self._value = value

    def __get__[OwnerT](
        self,
        instance: OwnerT | None,
        owner: type[OwnerT] | None = None,
    ) -> T:
        """Return the identical native value for class and instance access.

        Returns:
            The unchanged original value.

        """
        return self._value


__all__: list[str] = ["FlextTypingNativeExport"]
