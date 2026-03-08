"""Shared example helper models."""

from __future__ import annotations

from pydantic import BaseModel


class SharedPerson(BaseModel):
    """Shared random person model."""

    name: str
    age: int


class SharedHandle(BaseModel):
    """Shared handle model."""

    value: int
    cleaned: bool = False
