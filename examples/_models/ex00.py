"""Example 00 models."""

from __future__ import annotations

from typing import Annotated

from examples import ExamplesFlextCoreModelsErrors as _err
from flext_core import c, m, p, r, t, u


class ExamplesFlextCoreModelsEx00:
    """Example 00 model namespace."""

    class UserProfile(m.Entity):
        """User profile transport model."""

        name: Annotated[str, u.Field(min_length=1, description="User display name")]
        email: Annotated[str, u.Field(min_length=1, description="User email address")]
        status: Annotated[c.Status, u.Field(description="User account status")] = (
            c.Status.ACTIVE
        )

        def activate(self) -> p.Result[None]:
            """Activate user once."""
            if self.status == c.Status.ACTIVE:
                return r[None].fail("Already active")
            return r[None].ok(None)

    class UserInput(m.Value):
        """Raw user input model."""

        name: Annotated[str, u.Field(min_length=1, description="User display name")]
        email: Annotated[str, u.Field(min_length=1, description="User email address")]

        @u.field_validator("name", "email", mode="before")
        @classmethod
        def validate_non_empty_text(cls, value: t.JsonPayload) -> str:
            """Validate text input."""
            if not isinstance(value, str):
                raise TypeError(_err.Examples.ErrorMessages.EXPECTED_TEXT_INPUT)
            normalized = value.strip()
            if not normalized:
                raise ValueError(_err.Examples.ErrorMessages.TEXT_INPUT_CANNOT_BE_EMPTY)
            return normalized
