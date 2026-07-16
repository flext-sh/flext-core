"""Shared exception-carrying r fixtures."""

from __future__ import annotations

from typing import Annotated

from tests.models import m


class TestsFlextResultExceptionCarrying:
    class BrokenSized:
        """Sized t.JsonValue that raises on __len__."""

        def __len__(self) -> int:
            """Raise TypeError on length call."""
            msg = "no length"
            raise TypeError(msg)

    class UserModel(m.Value):
        """User model for testing."""

        name: Annotated[str, m.Field(description="User name")]
        age: Annotated[int, m.Field(description="User age")]
