"""Exception hierarchy — thin MRO facade over private _exceptions/ modules.

Provides structured exceptions with error codes and correlation tracking
for consistent error handling across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextExceptionsBase,
    FlextExceptionsFactories,
    FlextExceptionsHelpers,
    FlextExceptionsMetrics,
    FlextExceptionsTemplate,
    FlextExceptionsTypes,
)


class FlextExceptions(
    FlextExceptionsFactories,
    FlextExceptionsTemplate,
    FlextExceptionsHelpers,
    FlextExceptionsMetrics,
    FlextExceptionsTypes,
    FlextExceptionsBase,
):
    """Exception types with correlation metadata — MRO facade.

    Provides structured exceptions with error codes and correlation tracking
    for consistent error handling and logging.
    """


e = FlextExceptions

__all__: list[str] = ["FlextExceptions", "e"]
