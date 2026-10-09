"""Composed private generic result base over the _result pyramid.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._result.behavior import FlextResultBehavior
from flext_core._result.composition import FlextResultComposition
from flext_core._result.construction import FlextResultConstruction
from flext_core._result.transforms import FlextResultTransforms
from flext_core._result.unwrap import FlextResultUnwrap


class _FlextResult[T](
    FlextResultUnwrap[T],
    FlextResultComposition[T],
    FlextResultTransforms[T],
    FlextResultConstruction[T],
    FlextResultBehavior[T],
):
    """Type-safe result with monadic railway-oriented operations."""


__all__: list[str] = ["_FlextResult"]
