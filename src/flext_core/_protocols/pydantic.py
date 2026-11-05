"""Pydantic v2 structural contracts and handlers exported via FlextProtocols.

Including: ValidationInfo, ModelWrapValidatorHandler, GetCoreSchemaHandler, etc.

Architecture: Abstraction boundary - protocols layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, Protocol, Self, overload, runtime_checkable

import pydantic
from pydantic_core import PydanticUndefinedType

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsPydantic:
    """Structural contracts exported from pydantic.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use p.* instead. Each name is a type for annotations.
    """

    type ConfigDict = pydantic.ConfigDict
    type EncoderProtocol = pydantic.EncoderProtocol
    type ModelWrapValidatorHandler[T] = pydantic.ModelWrapValidatorHandler[T]
    type ValidationInfo = pydantic.ValidationInfo
    type ValidatorFunctionWrapHandler = pydantic.ValidatorFunctionWrapHandler

    @runtime_checkable
    class Base(Protocol):
        """Base protocol for FLEXT structural types."""

    @runtime_checkable
    class AttributeProbe(Protocol):
        """Structural marker for values inspected only via ``hasattr``/``getattr``.

        Empty Protocol body — accepts any runtime value structurally. Used by
        ``FlextDecoratorsRuntime`` and other infrastructure helpers as a typed
        parameter annotation when the actual contract is "any value that
        may carry inspectable attributes" without loose top-type annotations.
        """

    @runtime_checkable
    class ModuleOwned(Protocol):
        """Structural contract for runtime symbols that expose owner module."""

        __module__: str

    @runtime_checkable
    class ModelCopy(Protocol):
        """Copy a validated model without constraining its serialized payload."""

        def model_copy(
            self,
            *,
            update: Mapping[str, object] | None = None,
            deep: bool = False,
        ) -> Self:
            """Copy through the native model's update and depth contract."""
            ...

    @runtime_checkable
    class BaseModel[DumpT: t.JsonPayload = t.JsonDict](Base, ModelCopy, Protocol):
        """Structural typing protocol for Pydantic v2 models.

        Ensures types have Pydantic signatures without importing BaseModel directly
        in typings.py, preventing circular dependencies.
        """

        # NOTE (multi-agent, mro-wkii.17 / agent: codex): declare only the
        # instance capability consumed by interfaces; Pydantic class APIs stay
        # on ``ModelType``.

        def model_dump(
            self,
            *,
            mode: str = "python",
            include: t.IncEx | None = None,
            exclude: t.IncEx | None = None,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
        ) -> DumpT:
            """Dump the validated model at an external serialization boundary."""
            ...

        def model_dump_json(
            self,
            *,
            indent: int | None = None,
            exclude: set[str] | None = None,
            by_alias: bool | None = None,
            exclude_none: bool = False,
            round_trip: bool = False,
        ) -> str:
            """Serialize through the native JSON options structural consumers use."""
            ...

    @runtime_checkable
    class ModelField(Protocol):
        """Declared field annotation consumed by schema and protocol generators."""

        @property
        def annotation(self) -> t.TypeFormSpecifier | None: ...

        @property
        def frozen(self) -> bool | None: ...

    @runtime_checkable
    class ModelType(ModuleOwned, Protocol):
        """Native class metadata; instance operations remain on Model."""

        __name__: str
        __qualname__: str

        @property
        def model_fields(self) -> Mapping[str, FlextProtocolsPydantic.ModelField]:
            """Read the class-level field descriptor without requiring storage."""
            ...

        @property
        def model_config(self) -> t.ConfigDict:
            """Read the model owner's native configuration."""
            ...

    @runtime_checkable
    class FunctionDecorator(Protocol):
        """Decorator preserving each function's parameter and return types."""

        def __call__[CallT: Callable[..., FlextProtocolsPydantic.AttributeProbe]](
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
            config: t.ConfigDict | None = None,
            validate_return: bool = False,
        ) -> FlextProtocolsPydantic.FunctionDecorator: ...

        @overload
        def __call__[CallT: Callable[..., FlextProtocolsPydantic.AttributeProbe]](
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
            | Callable[[Mapping[str, FlextProtocolsPydantic.AttributeProbe]], FactoryT],
            init: Literal[False] = ...,
        ) -> FactoryT: ...

        @overload
        def __call__(
            self,
            *,
            default_factory: None = ...,
            init: Literal[False] = ...,
        ) -> FlextProtocolsPydantic.UnsetPrivateAttribute: ...

    @runtime_checkable
    class BaseSettings(BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class RootModel[RootValueT](BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        root: RootValueT


__all__: tuple[str, ...] = ("FlextProtocolsPydantic",)
