"""Shared example resource-handle model."""

from __future__ import annotations

from typing import Annotated

from flext_core import m, u


class ExamplesFlextSharedHandle(m.Value):
    """Shared resource-handle value model used across public examples."""

    value: Annotated[int, u.Field(description="Opaque integer handle identifier.")]
    cleaned: Annotated[
        bool,
        u.Field(default=False, description="Whether the handle has been released."),
    ] = False


__all__: list[str] = ["ExamplesFlextSharedHandle"]
