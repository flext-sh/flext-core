"""Example 00 models."""

from __future__ import annotations

from pydantic import Field, field_validator

from flext_core import c, m, r, t


class Ex00UserProfile(m.Entity):
    """User profile transport model."""

    name: str = Field(min_length=1)
    email: str = Field(min_length=1)
    status: c.Domain.Status = c.Domain.Status.ACTIVE

    def activate(self) -> r[None]:
        """Activate user once."""
        if self.status == c.Domain.Status.ACTIVE:
            return r[None].fail("Already active")
        return r.ok(None)


class Ex00UserInput(m.Value):
    """Raw user input model."""

    name: str = Field(min_length=1)
    email: str = Field(min_length=1)

    @field_validator("name", "email", mode="before")
    @classmethod
    def validate_non_empty_text(cls, value: t.ContainerValue) -> str:
        """Validate text input."""
        if not isinstance(value, str):
            msg = "Expected text input"
            raise TypeError(msg)
        normalized = value.strip()
        if not normalized:
            msg = "Text input cannot be empty"
            raise ValueError(msg)
        return normalized
