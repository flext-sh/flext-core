"""FlextConstantsErrors - error domain constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._errors_parts.flextconstantserrors_part_01 import FlextConstantsErrorsMessages
from ._errors_parts.flextconstantserrors_part_02 import (
    FlextConstantsErrorsRuntimeExceptions,
)
from ._errors_parts.flextconstantserrors_part_03 import (
    FlextConstantsErrorsValidationExceptions,
)
from ._errors_parts.flextconstantserrors_part_04 import FlextConstantsErrorsDomainParser
from ._errors_parts.flextconstantserrors_part_05 import (
    FlextConstantsErrorsRuntimeSettings,
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
