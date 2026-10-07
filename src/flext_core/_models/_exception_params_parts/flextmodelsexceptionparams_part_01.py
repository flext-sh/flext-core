"""FlextModelsExceptionParams - validated params for typed exception hierarchy.

Canonical home for exception parameter models. Used by:
- FlextExceptions (exceptions.py) for __init__ validation
- FlextModelsSettings ErrorConfig models (settings.py) via inheritance

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import c, t
from flext_core._models.base import FlextModelsBase
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._typings.pydantic import FlextTypesPydantic


class FlextModelsExceptionParams:
    """Validated parameter models for the FLEXT exception hierarchy.

    Field annotations spell their type expressions fully through the imported
    typing facade (``FlextTypesPydantic.*`` / ``t.*``): nested class bodies
    have no lexical access to the namespace-class scope, so class-scope field
    aliases would be invisible to static analysis. Each per-field annotation
    stacks an outer ``Annotated[..., Field(...)]`` — Pydantic v2 merges the
    ``FieldInfo`` layers automatically (default+strict from the type
    expression, description/title/examples from the outer Field).
    """

    class ParamsModel(FlextModelsBase.ArbitraryTypesModel):
        """Shared strict params model for exception helpers."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                extra="forbid",
                strict=True,
                validate_assignment=True,
                arbitrary_types_allowed=True,
                use_enum_values=True,
            )
        )

    class ExceptionFactoryOptions(ParamsModel):
        """Shared factory options for exception failures."""

        error: Annotated[
            Exception | str | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Optional underlying error cause for this failure.",
            ),
        ] = None
        error_code: Annotated[
            c.ErrorCode | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Optional override for the canonical failure error code.",
            ),
        ] = None

    class ResourceIdentityParams(ParamsModel):
        """Shared resource identity fields for resource-oriented errors."""

        resource_type: Annotated[
            FlextTypesPydantic.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Domain resource type associated with the failure.",
            ),
        ] = None
        resource_id: Annotated[
            FlextTypesPydantic.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Identifier of the resource associated with the failure.",
            ),
        ] = None

    class ExpectedActualTypeParams(ParamsModel):
        """Shared expected/actual runtime type fields."""

        expected_type: Annotated[
            FlextTypesPydantic.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Expected runtime type name for the failing value.",
            ),
        ] = None
        actual_type: Annotated[
            FlextTypesPydantic.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Actual runtime type name received at runtime.",
            ),
        ] = None

    class ValidationErrorParams(ParamsModel):
        """Validated params for ValidationError."""

        field: Annotated[
            FlextTypesPydantic.StrictStr | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Name of the input field that failed validation.",
                title="Field",
                examples=["email"],
            ),
        ] = None
        value: Annotated[
            t.RuntimeData | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Rejected input value that triggered the validation error.",
            ),
        ] = None

    class ConfigurationErrorParams(ParamsModel):
        """Validated params for ConfigurationError."""

        config_key: Annotated[
            FlextTypesPydantic.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Settings key associated with the error.",
            ),
        ] = None
        config_source: Annotated[
            FlextTypesPydantic.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Settings source where the invalid value originated.",
            ),
        ] = None

    class ConnectionErrorParams(ParamsModel):
        """Validated params for ConnectionError."""

        host: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description=(
                    "Hostname or address used for the failed connection attempt."
                ),
            ),
        ] = None
        port: Annotated[
            int | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Network port used for the failed connection attempt.",
            ),
        ] = None
        timeout: Annotated[
            t.Numeric | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Connection timeout threshold in seconds.",
            ),
        ] = None

        @property
        def connection_target(self) -> str:
            """Human-readable host:port string for log messages."""
            host = self.host or c.IDENTIFIER_UNKNOWN
            if self.port is None:
                return host
            return f"{host}:{self.port}"


__all__: list[str] = ["FlextModelsExceptionParams"]
