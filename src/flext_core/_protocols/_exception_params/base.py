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
    from flext_core import c, t
    from flext_core._protocols import FlextProtocolsResult


class FlextProtocolsExceptionParamsBase:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ParamsModel(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class ExceptionFactoryOptions(ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def error(self) -> Exception | str | None:
            """Optional underlying error cause for this failure."""
            ...

        error_code: c.ErrorCode | None

    @runtime_checkable
    class ExceptionInitOptions(ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        error_code: str

        @property
        def context(
            self,
        ) -> (
            Mapping[str, t.JsonPayload | None]
            | FlextProtocolsResult.HasModelDump
            | None
        ):
            """Structured context mapping or dumpable payload."""
            ...

        @property
        def metadata(self) -> FlextProtocolsResult.HasModelDump | t.JsonValue | None:
            """Metadata payload or dumpable model for the exception."""
            ...

        correlation_id: t.StrictStr | None
        auto_correlation: bool
        auto_log: bool

        @property
        def merged_kwargs(
            self,
        ) -> (
            Mapping[str, t.JsonPayload | None]
            | FlextProtocolsResult.HasModelDump
            | None
        ):
            """Pre-merged keyword payload forwarded by factories."""
            ...

    @runtime_checkable
    class ResourceIdentityParams(ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        resource_type: t.StrictStr | None
        resource_id: t.StrictStr | None

    @runtime_checkable
    class ExpectedActualTypeParams(ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        expected_type: t.StrictStr | None
        actual_type: t.StrictStr | None

    @runtime_checkable
    class ValidationErrorParams(ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        field: t.StrictStr | None
        value: t.RuntimeData | None

    @runtime_checkable
    class ConfigurationErrorParams(ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        config_key: t.StrictStr | None
        config_source: t.StrictStr | None

    @runtime_checkable
    class ConnectionErrorParams(ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        host: str | None
        port: int | None
        timeout: t.Numeric | None

        @property
        def connection_target(self) -> str:
            """Human-readable host:port string for log messages."""
            ...


__all__: list[str] = [
    "FlextProtocolsExceptionParamsBase",
]
