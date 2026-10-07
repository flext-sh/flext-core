"""Composed private generic result base over the _result pyramid.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._result.base import JsonDict
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

    def __init__(  # ruff: ignore[too-many-arguments] -- the keyword contract mirrors the public constructor; every argument is a distinct documented field.
        self,
        error_code: str | None = None,
        error_data: JsonDict | None = None,
        *,
        value: T | None = None,
        error: str | None = None,
        success: bool = True,
        exception: BaseException | None = None,
    ) -> None:
        """Initialize a result with value, error, or exception state."""
        super().__init__(
            error_code=error_code,
            error_data=error_data,
            value=value,
            error=error,
            success=success,
            exception=exception,
        )


__all__: list[str] = ["_FlextResult"]
