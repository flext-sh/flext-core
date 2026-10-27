"""FlextModelsExceptionParams - validated params for typed exception hierarchy.

Canonical home for exception parameter models. Used by:
- FlextExceptions (exceptions.py) for __init__ validation
- FlextModelsSettings ErrorConfig models (settings.py) via inheritance

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import t
from flext_core._models import FlextModelsPydantic
from flext_core._models._exception_params import base


class FlextModelsExceptionParamsServiceErrors(base.FlextModelsExceptionParamsBase):
    class TimeoutErrorParams(base.FlextModelsExceptionParamsBase.ParamsModel):
        """Validated params for TimeoutError."""

        timeout_seconds: Annotated[
            t.Numeric | None,
            FlextModelsPydantic.Field(
                default=None,
                description=(
                    "Timeout duration in seconds that triggered this exception."
                ),
                title="Timeout Seconds",
                examples=[30, 30.0],
            ),
        ] = None
        operation: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Operation name that exceeded the configured timeout.",
                title="Operation",
                examples=["dispatch"],
            ),
        ] = None

    class AuthenticationErrorParams(base.FlextModelsExceptionParamsBase.ParamsModel):
        """Validated params for AuthenticationError."""

        auth_method: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                description="Authentication method used when the failure occurred.",
            ),
        ] = None
        user_id: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                description=(
                    "User identifier associated with the authentication attempt."
                ),
            ),
        ] = None

    class AuthorizationErrorParams(base.FlextModelsExceptionParamsBase.ParamsModel):
        """Validated params for AuthorizationError."""

        user_id: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description="User identifier denied access to a protected resource.",
                title="User ID",
                examples=["user-123"],
            ),
        ] = None
        resource: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description=(
                    "Protected resource that triggered the authorization failure."
                ),
                title="Resource",
                examples=["invoice:12345"],
            ),
        ] = None
        permission: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description=(
                    "Missing permission required to complete the requested action."
                ),
                title="Permission",
                examples=["write"],
            ),
        ] = None

    class NotFoundErrorParams(
        base.FlextModelsExceptionParamsBase.ResourceIdentityParams
    ):
        """Validated params for NotFoundError."""

    class ConflictErrorParams(
        base.FlextModelsExceptionParamsBase.ResourceIdentityParams
    ):
        """Validated params for ConflictError."""

        conflict_reason: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Human-readable explanation for why the conflict occurred.",
                title="Conflict Reason",
                examples=["version_mismatch"],
            ),
        ] = None

    class RateLimitErrorParams(base.FlextModelsExceptionParamsBase.ParamsModel):
        """Validated params for RateLimitError."""

        limit: Annotated[
            int | None,
            FlextModelsPydantic.Field(
                default=None,
                description=(
                    "Maximum request count allowed within the configured window."
                ),
                title="Limit",
                examples=[100],
            ),
        ] = None
        window_seconds: Annotated[
            int | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Duration, in seconds, of the rate-limit window.",
                title="Window Seconds",
                examples=[60],
            ),
        ] = None
        retry_after: Annotated[
            t.Numeric | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Time in seconds clients should wait before retrying.",
                title="Retry After",
                examples=[1, 1.5],
            ),
        ] = None

    class CircuitBreakerErrorParams(base.FlextModelsExceptionParamsBase.ParamsModel):
        """Validated params for CircuitBreakerError."""

        service_name: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description="External service monitored by the circuit breaker.",
            ),
        ] = None
        failure_count: Annotated[
            int | None,
            FlextModelsPydantic.Field(
                default=None,
                description=(
                    "Consecutive failure count at the moment the breaker opened."
                ),
            ),
        ] = None
        reset_timeout: Annotated[
            t.Numeric | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Seconds before allowing a circuit breaker reset attempt.",
            ),
        ] = None

    class TypeErrorParams(base.FlextModelsExceptionParamsBase.ExpectedActualTypeParams):
        """Validated params for TypeError."""


__all__: list[str] = ["FlextModelsExceptionParamsServiceErrors"]
