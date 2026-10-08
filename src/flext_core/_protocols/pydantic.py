"""Pydantic v2 structural contracts and handlers exported via FlextProtocols.

Including: ValidationInfo, ModelWrapValidatorHandler, GetCoreSchemaHandler, etc.

Architecture: Abstraction boundary - protocols layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, overload

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

    class FunctionDecorator(Protocol):
        """Decorator preserving each function's parameter and return types."""

        def __call__[**ParametersT, ReturnT](
            self,
            func: Callable[ParametersT, ReturnT],
            /,
        ) -> Callable[ParametersT, ReturnT]: ...

    class ValidateCall(Protocol):
        """Pydantic's bare call validator and keyword-only decorator factory."""

        @overload
        def __call__[**ParametersT, ReturnT](
            self,
            func: Callable[ParametersT, ReturnT],
            /,
        ) -> Callable[ParametersT, ReturnT]: ...

        @overload
        def __call__(
            self,
            *,
            config: pydantic.ConfigDict | None = None,
            validate_return: bool = False,
        ) -> FlextProtocolsPydantic.FunctionDecorator: ...
