"""Pydantic v2 structural contracts and handlers exported via FlextProtocols.

Including: ValidationInfo, ModelWrapValidatorHandler, GetCoreSchemaHandler, etc.

Architecture: Abstraction boundary - protocols layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import (
    EncoderProtocol,
    ModelWrapValidatorHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
)


class FlextProtocolsPydantic:
    """Structural contracts exported from pydantic.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use p.* instead.
    """

    # Protocols
    EncoderProtocol = EncoderProtocol
    ModelWrapValidatorHandler = ModelWrapValidatorHandler
    ValidationInfo = ValidationInfo
    ValidatorFunctionWrapHandler = ValidatorFunctionWrapHandler
