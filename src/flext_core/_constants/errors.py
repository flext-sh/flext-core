"""FlextConstantsErrors - error domain constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._errors_parts import (
    FlextConstantsErrorsDomainParser,
    FlextConstantsErrorsMessages,
    FlextConstantsErrorsRuntimeExceptions,
    FlextConstantsErrorsRuntimeSettings,
    FlextConstantsErrorsValidationExceptions,
)


class FlextConstantsErrors(
    FlextConstantsErrorsMessages,
    FlextConstantsErrorsRuntimeExceptions,
    FlextConstantsErrorsValidationExceptions,
    FlextConstantsErrorsDomainParser,
    FlextConstantsErrorsRuntimeSettings,
):
    """Error domain constants for structured error routing."""


__all__ = ["FlextConstantsErrors"]
