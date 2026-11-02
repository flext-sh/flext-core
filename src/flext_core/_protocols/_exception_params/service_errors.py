# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_exception_params/service_errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import t


class FlextProtocolsExceptionParamsServiceErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class TimeoutErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        timeout_seconds: t.Numeric | None
        operation: str | None

    @runtime_checkable
    class AuthenticationErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        auth_method: str | None
        user_id: str | None

    @runtime_checkable
    class AuthorizationErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        user_id: str | None
        resource: str | None
        permission: str | None

    @runtime_checkable
    class NotFoundErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        resource_type: t.StrictStr | None
        resource_id: t.StrictStr | None

    @runtime_checkable
    class ConflictErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        resource_type: t.StrictStr | None
        resource_id: t.StrictStr | None
        conflict_reason: str | None

    @runtime_checkable
    class RateLimitErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        limit: int | None
        window_seconds: int | None
        retry_after: t.Numeric | None

    @runtime_checkable
    class CircuitBreakerErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        service_name: str | None
        failure_count: int | None
        reset_timeout: t.Numeric | None

    @runtime_checkable
    class TypeErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        expected_type: t.StrictStr | None
        actual_type: t.StrictStr | None


__all__: list[str] = [
    "FlextProtocolsExceptionParamsServiceErrors",
]
