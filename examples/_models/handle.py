"""Shared example resource-handle model.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import m


class ExamplesFlextSharedHandle(m.Value):
    """Shared resource-handle value model used across public examples."""

    value: Annotated[int, m.Field(description="Opaque integer handle identifier.")]
    cleaned: Annotated[
        bool,
        m.Field(default=False, description="Whether the handle has been released."),
    ] = False


__all__: list[str] = ["ExamplesFlextSharedHandle"]
