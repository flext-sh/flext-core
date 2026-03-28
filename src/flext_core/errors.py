"""Structured error handling for Result types.

Provides FlextErrorDomain enum and FlextError model for categorized error handling
with proper error routing and metadata support across the FLEXT ecosystem.

The canonical definitions live in the namespace facades:
- ``c.ErrorDomain`` (FlextConstantsErrors.ErrorDomain)
- ``m.Error`` (FlextModelsErrors.Error)

This module re-exports them for package-level access via ``__init__.py``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core._constants.errors import (
        FlextConstantsErrors as _FlextConstantsErrors,
    )
    from flext_core._models.errors import FlextModelsErrors as _FlextModelsErrors

    FlextErrorDomain: type[_FlextConstantsErrors.ErrorDomain]
    FlextError: type[_FlextModelsErrors.Error]


def __getattr__(name: str) -> type:
    """Lazy re-export canonical error types on first access."""
    if name == "FlextErrorDomain":
        from flext_core._constants.errors import FlextConstantsErrors  # noqa: PLC0415

        return FlextConstantsErrors.ErrorDomain
    if name == "FlextError":
        from flext_core._models.errors import FlextModelsErrors  # noqa: PLC0415

        return FlextModelsErrors.Error
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["FlextError", "FlextErrorDomain"]
