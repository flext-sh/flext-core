# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_exception_params/base
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._constants import FlextConstantsValidation
    from flext_core._protocols import FlextProtocolsResult
    from flext_core._typings import (
        FlextTypesPydantic,
        FlextTypesServices,
        FlextTypingBase,
    )


class FlextProtocolsExceptionParamsBase:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ParamsModel(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class ExceptionFactoryOptions(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def error(self) -> Exception | str | None:
            """Optional underlying error cause for this failure."""
            ...

        @property
        def error_code(self) -> FlextConstantsValidation.ErrorCode | None:
            """Optional override for the canonical failure error code."""
            ...

    @runtime_checkable
    class ExceptionInitOptions(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def error_code(self) -> str:
            """Canonical error code carried by the exception."""
            ...

        @property
        def context(
            self,
        ) -> (
            Mapping[str, FlextTypesServices.JsonPayload | None]
            | FlextProtocolsResult.HasModelDump
            | None
        ):
            """Structured context mapping or dumpable payload."""
            ...

        @property
        def metadata(
            self,
        ) -> FlextProtocolsResult.HasModelDump | FlextTypesPydantic.JsonValue | None:
            """Metadata payload or dumpable model for the exception."""
            ...

        @property
        def correlation_id(self) -> FlextTypesPydantic.StrictStr | None:
            """Explicit correlation identifier for the exception."""
            ...

        @property
        def auto_correlation(self) -> bool:
            """Generate a correlation identifier when none is given."""
            ...

        @property
        def auto_log(self) -> bool:
            """Log the exception automatically on construction."""
            ...

        @property
        def merged_kwargs(
            self,
        ) -> (
            Mapping[str, FlextTypesServices.JsonPayload | None]
            | FlextProtocolsResult.HasModelDump
            | None
        ):
            """Pre-merged keyword payload forwarded by factories."""
            ...

    @runtime_checkable
    class ResourceIdentityParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def resource_type(self) -> FlextTypesPydantic.StrictStr | None:
            """Domain resource type associated with the failure."""
            ...

        @property
        def resource_id(self) -> FlextTypesPydantic.StrictStr | None:
            """Identifier of the resource associated with the failure."""
            ...

    @runtime_checkable
    class ExpectedActualTypeParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def expected_type(self) -> FlextTypesPydantic.StrictStr | None:
            """Expected runtime type name for the failing value."""
            ...

        @property
        def actual_type(self) -> FlextTypesPydantic.StrictStr | None:
            """Actual runtime type name received at runtime."""
            ...

    @runtime_checkable
    class ValidationErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def field(self) -> FlextTypesPydantic.StrictStr | None:
            """Name of the input field that failed validation."""
            ...

        @property
        def value(self) -> FlextTypingBase.RuntimeData | None:
            """Rejected input value that triggered the validation error."""
            ...

    @runtime_checkable
    class ConfigurationErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def config_key(self) -> FlextTypesPydantic.StrictStr | None:
            """Settings key associated with the error."""
            ...

        @property
        def config_source(self) -> FlextTypesPydantic.StrictStr | None:
            """Settings source where the invalid value originated."""
            ...

    @runtime_checkable
    class ConnectionErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def host(self) -> str | None:
            """Hostname or address used for the failed connection attempt."""
            ...

        @property
        def port(self) -> int | None:
            """Network port used for the failed connection attempt."""
            ...

        @property
        def timeout(self) -> FlextTypingBase.Numeric | None:
            """Connection timeout threshold in seconds."""
            ...

        @property
        def connection_target(self) -> str:
            """Human-readable host:port string for log messages."""


__all__: list[str] = [
    "FlextProtocolsExceptionParamsBase",
]
