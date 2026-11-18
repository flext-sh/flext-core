"""Part whose members a stand-in resolves from this module's code.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class Part:
    """Members bound by plain stores, by unpacking and by decorators."""

    VALUE: ClassVar[int] = 7
    ADDED: ClassVar[str] = "added"
    LEFT, RIGHT = 1, 2

    @classmethod
    def who(cls) -> str:
        """Name the class the member is read through.

        Returns:
            The receiving class name.

        """
        return cls.__name__

    @staticmethod
    def twice(value: int) -> int:
        """Double ``value``.

        Returns:
            Twice ``value``.

        """
        return 2 * value


__all__: list[str] = ["Part"]
