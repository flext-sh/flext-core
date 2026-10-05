"""Shared exception-carrying r fixtures.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from tests.models import m


class TestsFlextResultExceptionCarrying:
    class BrokenSized:
        """Sized t.JsonValue that raises on __len__."""

        @staticmethod
        def __len__() -> int:
            """Raise TypeError on length call.

            Raises:
                TypeError: If no length.

            """
            msg = "no length"
            raise TypeError(msg)

    class UserModel(m.Value):
        """User model for testing."""

        name: Annotated[str, m.Field(description="User name")]
        age: Annotated[int, m.Field(description="User age")]
