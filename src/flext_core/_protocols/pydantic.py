"""Pydantic v2 structural contracts and handlers exported via FlextProtocols.

Including: ValidationInfo, ModelWrapValidatorHandler, GetCoreSchemaHandler, etc.

Architecture: Abstraction boundary - protocols layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol

from pydantic import (
    EncoderProtocol,
    ModelWrapValidatorHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
)

from .base import FlextProtocolsBase


class FlextProtocolsPydantic:
    """Structural contracts exported from pydantic.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use p.* instead.
    """

    class TypeAdapter[ValidatedT](Protocol):
        """Structural validation capability implemented by Pydantic adapters."""

        def validate_python(
            self, value: FlextProtocolsBase.AttributeProbe, /
        ) -> ValidatedT:
            """Validate a Python value into the adapter's declared output type."""
            ...

        def validate_json(self, data: str | bytes | bytearray, /) -> ValidatedT:
            """Validate serialized JSON into the adapter's declared output type."""
            ...

    # Protocols
    EncoderProtocol = EncoderProtocol
    ModelWrapValidatorHandler = ModelWrapValidatorHandler
    ValidationInfo = ValidationInfo
    ValidatorFunctionWrapHandler = ValidatorFunctionWrapHandler
