# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_exception_params/service_errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._typings import FlextTypesPydantic, FlextTypingBase


class FlextProtocolsExceptionParamsServiceErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class TimeoutErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def timeout_seconds(self) -> FlextTypingBase.Numeric | None:
            """Timeout duration in seconds that triggered this exception."""
            ...

        @property
        def operation(self) -> str | None:
            """Operation name that exceeded the configured timeout."""
            ...

    @runtime_checkable
    class AuthenticationErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def auth_method(self) -> str | None:
            """Authentication method used when the failure occurred."""
            ...

        @property
        def user_id(self) -> str | None:
            """User identifier associated with the authentication attempt."""
            ...

    @runtime_checkable
    class AuthorizationErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def user_id(self) -> str | None:
            """User identifier denied access to a protected resource."""
            ...

        @property
        def resource(self) -> str | None:
            """Protected resource that triggered the authorization failure."""
            ...

        @property
        def permission(self) -> str | None:
            """Missing permission required to complete the requested action."""
            ...

    @runtime_checkable
    class NotFoundErrorParams(FlextProtocolsBase.Model, Protocol):
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
    class ConflictErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def resource_type(self) -> FlextTypesPydantic.StrictStr | None:
            """Domain resource type associated with the failure."""
            ...

        @property
        def resource_id(self) -> FlextTypesPydantic.StrictStr | None:
            """Identifier of the resource associated with the failure."""
            ...

        @property
        def conflict_reason(self) -> str | None:
            """Human-readable explanation for why the conflict occurred."""
            ...

    @runtime_checkable
    class RateLimitErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def limit(self) -> int | None:
            """Maximum request count allowed within the configured window."""
            ...

        @property
        def window_seconds(self) -> int | None:
            """Duration, in seconds, of the rate-limit window."""
            ...

        @property
        def retry_after(self) -> FlextTypingBase.Numeric | None:
            """Time in seconds clients should wait before retrying."""
            ...

    @runtime_checkable
    class CircuitBreakerErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def service_name(self) -> str | None:
            """External service monitored by the circuit breaker."""
            ...

        @property
        def failure_count(self) -> int | None:
            """Consecutive failure count at the moment the breaker opened."""
            ...

        @property
        def reset_timeout(self) -> FlextTypingBase.Numeric | None:
            """Seconds before allowing a circuit breaker reset attempt."""
            ...

    @runtime_checkable
    class TypeErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def expected_type(self) -> FlextTypesPydantic.StrictStr | None:
            """Expected runtime type name for the failing value."""
            ...

        @property
        def actual_type(self) -> FlextTypesPydantic.StrictStr | None:
            """Actual runtime type name received at runtime."""
            ...


__all__: list[str] = [
    "FlextProtocolsExceptionParamsServiceErrors",
]
