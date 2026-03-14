"""Shared example helper models."""

from __future__ import annotations

from flext_core import m


class SharedPerson(m.Value):
    name: str
    age: int


class SharedHandle(m.Value):
    value: int
    cleaned: bool = False
