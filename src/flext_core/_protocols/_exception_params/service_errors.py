# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_exception_params/service_errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import flext_core
from flext_core._protocols._exception_params import FlextProtocolsExceptionParamsBase

if TYPE_CHECKING:
    pass


class FlextProtocolsExceptionParamsServiceErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class TimeoutErrorParams(FlextProtocolsExceptionParamsBase.ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        timeout_seconds: flext_core.t.Numeric | None
        operation: str | None

    @runtime_checkable
    class AuthenticationErrorParams(
        FlextProtocolsExceptionParamsBase.ParamsModel, Protocol
    ):
        """Structural data contract projected from the owning model."""

        auth_method: str | None
        user_id: str | None

    @runtime_checkable
    class AuthorizationErrorParams(
        FlextProtocolsExceptionParamsBase.ParamsModel, Protocol
    ):
        """Structural data contract projected from the owning model."""

        user_id: str | None
        resource: str | None
        permission: str | None

    @runtime_checkable
    class NotFoundErrorParams(
        FlextProtocolsExceptionParamsBase.ResourceIdentityParams, Protocol
    ):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class ConflictErrorParams(
        FlextProtocolsExceptionParamsBase.ResourceIdentityParams, Protocol
    ):
        """Structural data contract projected from the owning model."""

        conflict_reason: str | None

    @runtime_checkable
    class RateLimitErrorParams(FlextProtocolsExceptionParamsBase.ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        limit: int | None
        window_seconds: int | None
        retry_after: flext_core.t.Numeric | None

    @runtime_checkable
    class CircuitBreakerErrorParams(
        FlextProtocolsExceptionParamsBase.ParamsModel, Protocol
    ):
        """Structural data contract projected from the owning model."""

        service_name: str | None
        failure_count: int | None
        reset_timeout: flext_core.t.Numeric | None

    @runtime_checkable
    class TypeErrorParams(
        FlextProtocolsExceptionParamsBase.ExpectedActualTypeParams, Protocol
    ):
        """Structural data contract projected from the owning model."""


__all__: list[str] = [
    "FlextProtocolsExceptionParamsServiceErrors",
]
