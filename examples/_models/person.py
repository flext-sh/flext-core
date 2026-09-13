"""Shared example Person model."""

from __future__ import annotations

from typing import Annotated

from flext_core import m, u


class ExamplesFlextSharedPerson(m.Value):
    """Shared Person value model used across public examples."""

    name: Annotated[str, u.Field(description="Given name of the person.")]
    age: Annotated[int, u.Field(description="Age in whole years.")]


__all__: list[str] = ["ExamplesFlextSharedPerson"]
