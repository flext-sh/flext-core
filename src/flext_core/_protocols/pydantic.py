"""Pydantic v2 structural contracts and handlers exported via FlextProtocols.

Including: ValidationInfo, ModelWrapValidatorHandler, GetCoreSchemaHandler, etc.

Architecture: Abstraction boundary - protocols layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, Protocol, overload, runtime_checkable

import pydantic
from pydantic_core import PydanticUndefinedType

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsPydantic:
    """Structural contracts exported from pydantic.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use p.* instead. Each name is a type for annotations.
    """

    type EncoderProtocol = pydantic.EncoderProtocol
    type ModelWrapValidatorHandler[T] = pydantic.ModelWrapValidatorHandler[T]
    type ValidationInfo = pydantic.ValidationInfo
    type ValidatorFunctionWrapHandler = pydantic.ValidatorFunctionWrapHandler

    @runtime_checkable
    class ModelField(Protocol):
        """Declared field annotation consumed by schema and protocol generators."""

        @property
        def annotation(self) -> t.TypeFormSpecifier | None: ...

        @property
        def frozen(self) -> bool | None: ...

    @runtime_checkable
    class ModelType(FlextProtocolsBase.ModuleOwned, Protocol):
        """Native class metadata; instance operations remain on Model."""

        __name__: str
        __qualname__: str

        @property
        def model_fields(self) -> Mapping[str, FlextProtocolsPydantic.ModelField]:
            """Read the class-level field descriptor without requiring storage."""
            ...

        @property
        def model_config(self) -> pydantic.ConfigDict:
            """Read the model owner's native configuration."""
            ...

    @runtime_checkable
    class FunctionDecorator(Protocol):
        """Decorator preserving each function's parameter and return types."""

        def __call__[CallT: Callable[..., FlextProtocolsBase.AttributeProbe]](
            self,
            func: CallT,
            /,
        ) -> CallT: ...

    @runtime_checkable
    class ValidateCall(Protocol):
        """Pydantic's bare call validator and keyword-only decorator factory."""

        @overload
        def __call__(
            self,
            *,
            config: pydantic.ConfigDict | None = None,
            validate_return: bool = False,
        ) -> FlextProtocolsPydantic.FunctionDecorator: ...

        @overload
        def __call__[CallT: Callable[..., FlextProtocolsBase.AttributeProbe]](
            self,
            func: CallT,
            /,
        ) -> CallT: ...

    @runtime_checkable
    class UnsetPrivateAttribute(Protocol):
        """Native private descriptor created without a default or factory."""

        def get_default(
            self, *, call_default_factory: bool = False
        ) -> PydanticUndefinedType: ...

    @runtime_checkable
    class PrivateAttrFactory(Protocol):
        """Receiver-free native private attribute metadata factory."""

        @overload
        def __call__[PrivateT](
            self,
            default: PrivateT,
            *,
            default_factory: None = ...,
            init: Literal[False] = ...,
        ) -> PrivateT: ...

        @overload
        def __call__[FactoryT](
            self,
            *,
            default_factory: Callable[[], FactoryT]
            | Callable[[Mapping[str, FlextProtocolsBase.AttributeProbe]], FactoryT],
            init: Literal[False] = ...,
        ) -> FactoryT: ...

        @overload
        def __call__(
            self,
            *,
            default_factory: None = ...,
            init: Literal[False] = ...,
        ) -> FlextProtocolsPydantic.UnsetPrivateAttribute: ...


__all__: tuple[str, ...] = ("FlextProtocolsPydantic",)
