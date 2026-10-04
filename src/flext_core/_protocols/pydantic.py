"""Pydantic v2 structural contracts and handlers exported via FlextProtocols.

Including: ValidationInfo, ModelWrapValidatorHandler, GetCoreSchemaHandler, etc.

Architecture: Abstraction boundary - protocols layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pydantic


class FlextProtocolsPydantic:
    """Structural contracts exported from pydantic.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use p.* instead. Each name is a type for annotations.
    """

    type EncoderProtocol = pydantic.EncoderProtocol
    type ModelWrapValidatorHandler[T] = pydantic.ModelWrapValidatorHandler[T]
    type ValidationInfo = pydantic.ValidationInfo
    type ValidatorFunctionWrapHandler = pydantic.ValidatorFunctionWrapHandler
