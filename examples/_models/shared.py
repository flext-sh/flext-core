"""Shared example helper models."""

from __future__ import annotations

from typing import Annotated

from examples import t
from flext_core import m, u


class ExamplesFlextSharedPerson(m.Value):
    """Shared Person value model used across public examples."""

    name: Annotated[str, u.Field(description="Given name of the person.")]
    age: Annotated[int, u.Field(description="Age in whole years.")]


class ExamplesFlextSharedHandle(m.Value):
    """Shared resource-handle value model used across public examples."""

    value: Annotated[int, u.Field(description="Opaque integer handle identifier.")]
    cleaned: Annotated[
        bool,
        u.Field(default=False, description="Whether the handle has been released."),
    ] = False


__all__: t.MutableSequenceOf[str] = [
    "ExamplesFlextSharedHandle",
    "ExamplesFlextSharedPerson",
]
