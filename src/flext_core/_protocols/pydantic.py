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
    from flext_core._typings import FlextTypingBase


class FlextProtocolsPydantic:
    """Structural contracts exported from pydantic.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use p.* instead. Each name is a type for annotations.
    """

    type EncoderProtocol = pydantic.EncoderProtocol
    type ModelWrapValidatorHandler[T] = pydantic.ModelWrapValidatorHandler[T]
    type ValidationInfo = pydantic.ValidationInfo
    type ValidatorFunctionWrapHandler = pydantic.ValidatorFunctionWrapHandler

    class ModelField(Protocol):
        """Declared field annotation consumed by schema and protocol generators."""

        @property
        def annotation(self) -> FlextTypingBase.TypeHintSpecifier | None: ...

        @property
        def frozen(self) -> bool | None: ...

    class ModelConfiguration(Protocol):
        """Native configuration flags consumed by structural model projections."""

        def get(
            self, key: Literal["frozen"], default: bool = False, /
        ) -> bool | None: ...

    class ModelType(FlextProtocolsBase.ModuleOwned, Protocol):
        """Read-only model declaration metadata; instance operations stay on Model."""

        __name__: str
        __qualname__: str

        @property
        def model_fields(
            self,
        ) -> Mapping[str, FlextProtocolsPydantic.ModelField]: ...

        @property
        def model_config(self) -> FlextProtocolsPydantic.ModelConfiguration: ...

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
    class FieldFactory(Protocol):
        """Receiver-free native field metadata factory for model declarations.

        Declares the polymorphic call contract ``pydantic.Field`` exposes so a
        checker resolves each ``m.Field(...)`` call site without a first-class
        reference to the overloaded native function.
        """

        @overload
        def __call__[DefaultT](
            self,
            default: DefaultT,
            *,
            default_factory: Callable[[], DefaultT]
            | Callable[[Mapping[str, FlextProtocolsBase.AttributeProbe]], DefaultT]
            | PydanticUndefinedType
            | None = ...,
            alias: str | PydanticUndefinedType | None = ...,
            **kwargs: FlextProtocolsBase.AttributeProbe,
        ) -> DefaultT: ...

        @overload
        def __call__(
            self,
            *,
            default_factory: Callable[..., FlextProtocolsBase.AttributeProbe]
            | PydanticUndefinedType
            | None = ...,
            alias: str | PydanticUndefinedType | None = ...,
            **kwargs: FlextProtocolsBase.AttributeProbe,
        ) -> FlextTypingBase.UnconstrainedValue: ...

    @runtime_checkable
    class PrivateAttrFactory(Protocol):
        """Receiver-free native private attribute metadata factory."""

        @overload
        def __call__[PrivateT](
            self,
            default: PrivateT,
            *,
            default_factory: Callable[..., PrivateT] | None = ...,
            init: Literal[False] = ...,
        ) -> PrivateT: ...

        @overload
        def __call__(
            self,
            *,
            default_factory: Callable[..., FlextProtocolsBase.AttributeProbe]
            | None = ...,
            init: Literal[False] = ...,
        ) -> FlextTypingBase.UnconstrainedValue: ...


__all__: tuple[str, ...] = ("FlextProtocolsPydantic",)
