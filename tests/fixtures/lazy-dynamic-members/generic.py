"""PEP 695 generic part, whose class body sits inside its type-parameter scope.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class Generic[T]:
    """Members of a generic part."""

    SIZE: ClassVar[int] = 3

    @staticmethod
    def first(items: tuple[T, ...]) -> T:
        """Return the first item.

        Returns:
            The first of ``items``.

        """
        return items[0]


__all__: list[str] = ["Generic"]
