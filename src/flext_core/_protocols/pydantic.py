"""Pydantic v2 structural contracts and handlers exported via FlextProtocols.

Including: ValidationInfo, ModelWrapValidatorHandler, GetCoreSchemaHandler, etc.

Architecture: Abstraction boundary - protocols layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set as AbstractSet
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import (
    EncoderProtocol,
    ModelWrapValidatorHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
)

if TYPE_CHECKING:
    from flext_core import t

    from .base import FlextProtocolsBase


type _IncEx = (
    AbstractSet[int]
    | AbstractSet[str]
    | Mapping[int, bool | _IncEx]
    | Mapping[str, bool | _IncEx]
)


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

        def dump_json(
            self,
            obj: ValidatedT,
            /,
            *,
            indent: int | None = None,
            ensure_ascii: bool = False,
            include: _IncEx | None = None,
            exclude: _IncEx | None = None,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
            warnings: bool | Literal["none", "warn", "error"] = True,
            fallback: Callable[[t.JsonPayload], t.JsonPayload] | None = None,
            serialize_as_any: bool = False,
            polymorphic_serialization: bool | None = None,
            context: t.JsonMapping | None = None,
        ) -> bytes:
            """Serialize a validated object to JSON bytes."""
            ...

    # Protocols
    EncoderProtocol = EncoderProtocol
    ModelWrapValidatorHandler = ModelWrapValidatorHandler
    ValidationInfo = ValidationInfo
    ValidatorFunctionWrapHandler = ValidatorFunctionWrapHandler
