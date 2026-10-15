"""Composed private generic result base over the _result pyramid.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._result import (
    FlextResultBehavior,
    FlextResultComposition,
    FlextResultConstruction,
    FlextResultTransforms,
    FlextResultUnwrap,
)


class FlextResultComposed[T](
    FlextResultUnwrap[T],
    FlextResultComposition[T],
    FlextResultTransforms[T],
    FlextResultConstruction[T],
    FlextResultBehavior[T],
):
    """Type-safe result with monadic railway-oriented operations."""


__all__: list[str] = ["FlextResultComposed"]
