"""Result combinator callables for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests import p, r


class TestsFlextUtilitiesResultCombinators:
    """Pure callables exercising result combinator contracts."""

    @staticmethod
    def double(value: int) -> int:
        """Double one integer value.

        Returns:
            The doubled value.

        """
        return value * 2

    @staticmethod
    def is_even(value: int) -> bool:
        """Return whether an integer is even."""
        return value % 2 == 0

    @staticmethod
    def is_odd(value: int) -> bool:
        """Return whether an integer is odd."""
        return value % 2 != 0

    @staticmethod
    def increment(value: int) -> p.Result[int]:
        """Increment one integer into a successful result.

        Returns:
            The successful incremented result.

        """
        return r[int].ok(value + 1)

    @staticmethod
    def shout(error: str) -> str:
        """Uppercase one error message.

        Returns:
            The uppercased error message.

        """
        return error.upper()

    @staticmethod
    def forty_two(_error: str) -> int:
        """Recover any failure with a fixed successful value.

        Returns:
            The fixed recovery value.

        """
        return 42

    @staticmethod
    def zero(_error: str) -> int:
        """Recover any failure with zero.

        Returns:
            The fixed zero value.

        """
        return 0


__all__: list[str] = ["TestsFlextUtilitiesResultCombinators"]
