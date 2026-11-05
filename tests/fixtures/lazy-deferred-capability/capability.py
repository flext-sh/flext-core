"""Capability a lazy package defers until its first member access.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class Capability:
    """Members a deferred stand-in resolves from this module on first access."""

    VALUE: ClassVar[int] = 7

    class Nested:
        """Nested declaration the stand-in resolves like any other member."""

    @staticmethod
    def twice(value: int) -> int:
        """Double ``value``.

        Returns:
            Twice ``value``.

        """
        return 2 * value

    @classmethod
    def who(cls) -> str:
        """Name the class the member is read through.

        Returns:
            The receiving class name.

        """
        return cls.__name__


__all__: list[str] = ["Capability"]
