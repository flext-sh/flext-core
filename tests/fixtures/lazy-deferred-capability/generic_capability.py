"""PEP 695 capability a lazy package defers until its first member access.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class GenericCapability[T: int]:
    """Generic members a deferred stand-in resolves on first access."""

    VALUE: ClassVar[int] = 7

    @staticmethod
    def first(items: tuple[T, ...]) -> T:
        """Return the first item.

        Returns:
            The first of ``items``.

        """
        return items[0]


__all__: list[str] = ["GenericCapability"]
