"""Exception hierarchy — thin MRO facade over private _exceptions/ modules.

Provides structured exceptions with error codes and correlation tracking
for consistent error handling across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core._constants.enforcement import FlextMroViolation, FlextSmellViolation
from flext_core._exceptions.base import FlextExceptionsBase
from flext_core._exceptions.exception_types import FlextExceptionsTypes
from flext_core._exceptions.factories import FlextExceptionsFactories
from flext_core._exceptions.helpers import FlextExceptionsHelpers
from flext_core._exceptions.metrics import FlextExceptionsMetrics
from flext_core._exceptions.template import FlextExceptionsTemplate


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

    MroViolation: ClassVar[type[FlextMroViolation]] = FlextMroViolation
    SmellViolation: ClassVar[type[FlextSmellViolation]] = FlextSmellViolation


e = FlextExceptions

__all__: list[str] = ["FlextExceptions", "e"]
