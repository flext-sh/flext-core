"""FlextModelsExceptionParams - validated params for typed exception hierarchy.

Canonical home for exception parameter models. Used by:
- FlextExceptions (exceptions.py) for __init__ validation
- FlextModelsSettings ErrorConfig models (settings.py) via inheritance

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, ClassVar

from flext_core import c, p, t
from flext_core._models import FlextModelsBase, FlextModelsPydantic


class FlextModelsExceptionParamsBase:
    """Validated parameter models for the FLEXT exception hierarchy."""

    class ParamsModel(FlextModelsBase.ArbitraryTypesModel):
        """Shared strict params model for exception helpers."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                extra="forbid",
                strict=True,
                validate_assignment=True,
                use_enum_values=True,
            )
        )

    class ExceptionFactoryOptions(ParamsModel):
        """Shared factory options for exception failures."""

        error: Annotated[
            t.Port[Exception] | str | None,
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

    class ExceptionInitOptions(ParamsModel):
        """Shared initialization options for BaseError construction."""

        error_code: Annotated[
            str,
            FlextModelsPydantic.Field(
                default=c.ErrorCode.UNKNOWN_ERROR,
                description="Canonical error code carried by the exception.",
            ),
        ] = c.ErrorCode.UNKNOWN_ERROR
        context: Annotated[
            Mapping[str, t.JsonPayload | None] | t.Port[p.HasModelDump] | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Structured context mapping or dumpable payload.",
            ),
        ] = None
        metadata: Annotated[
            t.Port[p.HasModelDump] | t.JsonValue | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Metadata payload or dumpable model for the exception.",
            ),
        ] = None
        correlation_id: Annotated[
            t.StrictStr | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Explicit correlation identifier for the exception.",
            ),
        ] = None
        auto_correlation: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=False,
                description="Generate a correlation identifier when none is given.",
            ),
        ] = False
        auto_log: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True,
                description="Log the exception automatically on construction.",
            ),
        ] = True
        merged_kwargs: Annotated[
            Mapping[str, t.JsonPayload | None] | t.Port[p.HasModelDump] | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Pre-merged keyword payload forwarded by factories.",
            ),
        ] = None

    class ResourceIdentityParams(ParamsModel):
        """Shared resource identity fields for resource-oriented errors."""

        resource_type: Annotated[
            t.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Domain resource type associated with the failure.",
            ),
        ] = None
        resource_id: Annotated[
            t.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Identifier of the resource associated with the failure.",
            ),
        ] = None

    class ExpectedActualTypeParams(ParamsModel):
        """Shared expected/actual runtime type fields."""

        expected_type: Annotated[
            t.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Expected runtime type name for the failing value.",
            ),
        ] = None
        actual_type: Annotated[
            t.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Actual runtime type name received at runtime.",
            ),
        ] = None

    class ValidationErrorParams(ParamsModel):
        """Validated params for ValidationError."""

        field: Annotated[
            t.StrictStr | None,
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
            t.StrictStr | None,
            FlextModelsPydantic.Field(
                description="Settings key associated with the error.",
            ),
        ] = None
        config_source: Annotated[
            t.StrictStr | None,
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


__all__: list[str] = ["FlextModelsExceptionParamsBase"]
