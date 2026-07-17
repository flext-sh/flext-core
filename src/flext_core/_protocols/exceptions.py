"""Structural contracts for validated exception parameter models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .base import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core import FlextConstants as c, FlextTypes as t


class FlextProtocolsExceptions:
    """Runtime-checkable protocols for exception parameter models."""

    @runtime_checkable
    class ExceptionFactoryOptions(FlextProtocolsBase.BaseModel, Protocol):
        """Shared factory options for exception failures."""

        @property
        def error(self) -> Exception | str | None: ...

        @property
        def error_code(self) -> c.ErrorCode | None: ...

    @runtime_checkable
    class ResourceIdentityParams(FlextProtocolsBase.BaseModel, Protocol):
        """Shared resource identity fields for resource-oriented errors."""

        @property
        def resource_type(self) -> str | None: ...

        @property
        def resource_id(self) -> str | None: ...

    @runtime_checkable
    class ExpectedActualTypeParams(FlextProtocolsBase.BaseModel, Protocol):
        """Shared expected/actual runtime type fields."""

        @property
        def expected_type(self) -> str | None: ...

        @property
        def actual_type(self) -> str | None: ...

    @runtime_checkable
    class ValidationErrorParams(FlextProtocolsBase.BaseModel, Protocol):
        """Validated params for ValidationError."""

        @property
        def field(self) -> str | None: ...

        @property
        def value(self) -> t.RuntimeData | None: ...

    @runtime_checkable
    class ConfigurationErrorParams(FlextProtocolsBase.BaseModel, Protocol):
        """Validated params for ConfigurationError."""

        @property
        def config_key(self) -> str | None: ...

        @property
        def config_source(self) -> str | None: ...

    @runtime_checkable
    class ConnectionErrorParams(FlextProtocolsBase.BaseModel, Protocol):
        """Validated params for ConnectionError."""

        @property
        def host(self) -> str | None: ...

        @property
        def port(self) -> int | None: ...

        @property
        def timeout(self) -> t.Numeric | None: ...

    @runtime_checkable
    class ServiceLookupParams(ExpectedActualTypeParams, Protocol):
        """Validated params for service lookup and narrowing failures."""

        @property
        def service_name(self) -> str | None: ...


__all__: list[str] = ["FlextProtocolsExceptions"]
