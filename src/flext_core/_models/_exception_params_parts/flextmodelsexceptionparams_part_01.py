"""FlextModelsExceptionParams - validated params for typed exception hierarchy.

Canonical home for exception parameter models. Used by:
- FlextExceptions (exceptions.py) for __init__ validation
- FlextModelsSettings ErrorConfig models (settings.py) via inheritance

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import (
    FlextConstants as c,
    FlextTypes as t,
)
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.pydantic import FlextTypesPydantic as tp


class FlextModelsExceptionParams:
    """Validated parameter models for the FLEXT exception hierarchy.

    Field-builder type aliases (``OptStrictStr`` / ``OptStrictInt`` /
    ``OptNumeric``) live here as ``ClassVar`` to keep all model surface inside
    the namespace class (per AGENTS.md §3.1: no loose module-level objects).
    Each per-field annotation stacks an outer ``Annotated[..., Field(...)]``
    over these aliases — Pydantic v2 merges the two ``FieldInfo`` layers
    automatically (default+strict from the alias, description/title/examples
    from the outer Field).
    """

    type OptStrictStr = tp.StrictStr | None

    type OptStrictInt = tp.StrictInt | None

    type OptNumeric = t.Numeric | None

    class ParamsModel(m.ArbitraryTypesModel):
        """Shared strict params model for exception helpers."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            extra="forbid",
            strict=True,
            validate_assignment=True,
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class ExceptionFactoryOptions(ParamsModel):
        """Shared factory options for exception failures."""

        error: Annotated[
            Exception | str | None,
            mp.Field(
                default=None,
                description="Optional underlying error cause for this failure.",
            ),
        ] = None
        error_code: Annotated[
            c.ErrorCode | None,
            mp.Field(
                default=None,
                description="Optional override for the canonical failure error code.",
            ),
        ] = None

    class ResourceIdentityParams(ParamsModel):
        """Shared resource identity fields for resource-oriented errors."""

        resource_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            mp.Field(
                description="Domain resource type associated with the failure.",
            ),
        ] = None
        resource_id: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            mp.Field(
                description="Identifier of the resource associated with the failure.",
            ),
        ] = None

    class ExpectedActualTypeParams(ParamsModel):
        """Shared expected/actual runtime type fields."""

        expected_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            mp.Field(
                description="Expected runtime type name for the failing value.",
            ),
        ] = None
        actual_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            mp.Field(
                description="Actual runtime type name received at runtime.",
            ),
        ] = None

    class ValidationErrorParams(ParamsModel):
        """Validated params for ValidationError."""

        field: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            mp.Field(
                default=None,
                description="Name of the input field that failed validation.",
                title="Field",
                examples=["email"],
            ),
        ] = None
        value: Annotated[
            t.RuntimeData | None,
            mp.Field(
                default=None,
                description="Rejected input value that triggered the validation error.",
            ),
        ] = None

    class ConfigurationErrorParams(ParamsModel):
        """Validated params for ConfigurationError."""

        config_key: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            mp.Field(
                description="Settings key associated with the error.",
            ),
        ] = None
        config_source: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            mp.Field(
                description="Settings source where the invalid value originated.",
            ),
        ] = None

    class ConnectionErrorParams(ParamsModel):
        """Validated params for ConnectionError."""

        host: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Hostname or address used for the failed connection attempt.",
            ),
        ] = None
        port: Annotated[
            int | None,
            mp.Field(
                default=None,
                description="Network port used for the failed connection attempt.",
            ),
        ] = None
        timeout: Annotated[
            t.Numeric | None,
            mp.Field(
                default=None,
                description="Connection timeout threshold in seconds.",
            ),
        ] = None

        @property
        def connection_target(self) -> str:
            """Human-readable host:port string for log messages."""
            host = self.host or "unknown"
            if self.port is None:
                return host
            return f"{host}:{self.port}"


__all__: list[str] = ["FlextModelsExceptionParams"]
