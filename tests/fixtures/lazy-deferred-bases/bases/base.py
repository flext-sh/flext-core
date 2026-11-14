"""Base part a stand-in shares through the initializer that exports it.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class Base:
    """Members the deriving stand-in reads through its base stand-in."""

    VALUE: ClassVar[int] = 1

    @staticmethod
    def twice(value: int) -> int:
        """Double ``value``.

        Returns:
            Twice ``value``.

        """
        return 2 * value
