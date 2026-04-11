"""Example 00 models."""

from __future__ import annotations

from pydantic import Field, field_validator

from examples import c
from examples._models.errors import ExamplesFlextCoreModelsErrors as _err
from flext_core import m, r, t


class ExamplesFlextCoreModelsEx00:
    """Example 00 model namespace."""

    class Ex00UserProfile(m.Entity):
        """User profile transport model."""

        name: str = Field(min_length=1)
        email: str = Field(min_length=1)
        status: c.Status = c.Status.ACTIVE

        def activate(self) -> r[None]:
            """Activate user once."""
            if self.status == c.Status.ACTIVE:
                return r[None].fail("Already active")
            return r[None].ok(None)

    class Ex00UserInput(m.Value):
        """Raw user input model."""

        name: str = Field(min_length=1)
        email: str = Field(min_length=1)

        @field_validator("name", "email", mode="before")
        @classmethod
        def validate_non_empty_text(cls, value: t.RuntimeData) -> str:
            """Validate text input."""
            if not isinstance(value, str):
                raise TypeError(_err.Examples.ErrorMessages.EXPECTED_TEXT_INPUT)
            normalized = value.strip()
            if not normalized:
                raise ValueError(_err.Examples.ErrorMessages.TEXT_INPUT_CANNOT_BE_EMPTY)
            return normalized
