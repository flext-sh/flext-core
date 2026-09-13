"""Shared example Person model."""

from __future__ import annotations

from typing import Annotated

from flext_core import m


class ExamplesFlextSharedPerson(m.Value):
    """Shared Person value model used across public examples."""

    name: Annotated[str, m.Field(description="Given name of the person.")]
    age: Annotated[int, m.Field(description="Age in whole years.")]


__all__: list[str] = ["ExamplesFlextSharedPerson"]
