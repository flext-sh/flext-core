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
    FlextModelsBase as m,
    FlextModelsPydantic,
    FlextUtilitiesPydantic,
    t,
)


class FlextModelsExceptionParams:
    """Validated parameter models for the FLEXT exception hierarchy."""

    class ParamsModel(m.ArbitraryTypesModel):
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

    class ValidationErrorParams(ParamsModel):
        """Validated params for ValidationError."""

        field: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Name of the input field that failed validation.",
                title="Field",
                examples=["email"],
            ),
        ]
        value: Annotated[
            t.Scalar | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                description="Rejected input value that triggered the validation error.",
            ),
        ] = None

    class ConfigurationErrorParams(ParamsModel):
        """Validated params for ConfigurationError."""

        config_key: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Settings key associated with the error.",
                title="Settings Key",
                examples=["database_url"],
            ),
        ]
        config_source: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Settings source where the invalid value originated.",
                title="Settings Source",
                examples=[".env"],
            ),
        ]

    class ConnectionErrorParams(ParamsModel):
        """Validated params for ConnectionError."""

        host: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Hostname or address used for the failed connection attempt.",
                title="Host",
                examples=["db.internal"],
            ),
        ]
        port: Annotated[
            int | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Network port used for the failed connection attempt.",
                title="Port",
                examples=[5432],
            ),
        ]
        timeout: Annotated[
            t.Numeric | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                description="Connection timeout threshold in seconds.",
                title="Timeout",
                examples=[5, 5.5],
            ),
        ]

    class TimeoutErrorParams(ParamsModel):
        """Validated params for TimeoutError."""

        timeout_seconds: Annotated[
            t.Numeric | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                description="Timeout duration in seconds that triggered this exception.",
                title="Timeout Seconds",
                examples=[30, 30.0],
            ),
        ]
        operation: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Operation name that exceeded the configured timeout.",
                title="Operation",
                examples=["dispatch"],
            ),
        ]

    class AuthenticationErrorParams(ParamsModel):
        """Validated params for AuthenticationError."""

        auth_method: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Authentication method used when the failure occurred.",
                title="Auth Method",
                examples=["token"],
            ),
        ]
        user_id: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="User identifier associated with the authentication attempt.",
                title="User ID",
                examples=["user-123"],
            ),
        ]

    class AuthorizationErrorParams(ParamsModel):
        """Validated params for AuthorizationError."""

        user_id: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="User identifier denied access to a protected resource.",
                title="User ID",
                examples=["user-123"],
            ),
        ]
        resource: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Protected resource that triggered the authorization failure.",
                title="Resource",
                examples=["invoice:12345"],
            ),
        ]
        permission: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Missing permission required to complete the requested action.",
                title="Permission",
                examples=["write"],
            ),
        ]

    class NotFoundErrorParams(ParamsModel):
        """Validated params for NotFoundError."""

        resource_type: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Domain resource type that could not be located.",
                title="Resource Type",
                examples=["user"],
            ),
        ]
        resource_id: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Unique identifier of the missing resource.",
                title="Resource ID",
                examples=["42"],
            ),
        ]

    class ConflictErrorParams(ParamsModel):
        """Validated params for ConflictError."""

        resource_type: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Domain resource type involved in the conflict.",
                title="Resource Type",
                examples=["order"],
            ),
        ]
        resource_id: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Identifier of the resource that caused the conflict.",
                title="Resource ID",
                examples=["ord-1001"],
            ),
        ]
        conflict_reason: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Human-readable explanation for why the conflict occurred.",
                title="Conflict Reason",
                examples=["version_mismatch"],
            ),
        ]

    class RateLimitErrorParams(ParamsModel):
        """Validated params for RateLimitError."""

        limit: Annotated[
            int | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Maximum request count allowed within the configured window.",
                title="Limit",
                examples=[100],
            ),
        ]
        window_seconds: Annotated[
            int | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Duration, in seconds, of the rate-limit window.",
                title="Window Seconds",
                examples=[60],
            ),
        ]
        retry_after: Annotated[
            t.Numeric | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                description="Time in seconds clients should wait before retrying.",
                title="Retry After",
                examples=[1, 1.5],
            ),
        ]

    class CircuitBreakerErrorParams(ParamsModel):
        """Validated params for CircuitBreakerError."""

        service_name: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="External service monitored by the circuit breaker.",
                title="Service Name",
                examples=["payments-api"],
            ),
        ]
        failure_count: Annotated[
            int | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Consecutive failure count at the moment the breaker opened.",
                title="Failure Count",
                examples=[5],
            ),
        ]
        reset_timeout: Annotated[
            t.Numeric | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                description="Seconds before allowing a circuit breaker reset attempt.",
                title="Reset Timeout",
                examples=[30, 30.0],
            ),
        ]

    class TypeErrorParams(ParamsModel):
        """Validated params for TypeError."""

        expected_type: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Expected type name for the failing value.",
                title="Expected Type",
                examples=["str"],
            ),
        ]
        actual_type: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Actual type name received at runtime.",
                title="Actual Type",
                examples=["int"],
            ),
        ]

    class OperationErrorParams(ParamsModel):
        """Validated params for OperationError."""

        operation: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Operation name associated with the failure.",
                title="Operation",
                examples=["publish_events"],
            ),
        ]
        reason: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Short reason explaining the operation failure.",
                title="Reason",
                examples=["transient_backend_error"],
            ),
        ]

    class ServiceLookupParams(ParamsModel):
        """Validated params for service lookup and narrowing failures."""

        service_name: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Service identifier requested from the container/registry.",
                title="Service Name",
                examples=["command_bus"],
            ),
        ]
        expected_type: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Expected runtime type name for the resolved service.",
                title="Expected Type",
                examples=["FlextDispatcher"],
            ),
        ]
        actual_type: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Actual runtime type name returned by resolution.",
                title="Actual Type",
                examples=["str"],
            ),
        ]

    class RegistryPluginParams(ParamsModel):
        """Validated params for registry plugin registration lifecycle."""

        category: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Registry category for plugin registration.",
                title="Category",
                examples=["validators"],
            ),
        ]
        name: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Plugin name within a category.",
                title="Plugin Name",
                examples=["strict_typing"],
            ),
        ]
        scope: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Registration scope used by registry operations.",
                title="Scope",
                examples=["instance", "class"],
            ),
        ]

    class AttributeAccessErrorParams(ParamsModel):
        """Validated params for AttributeAccessError."""

        attribute_name: Annotated[
            str | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                strict=True,
                description="Attribute name that could not be accessed safely.",
                title="Attribute Name",
                examples=["token"],
            ),
        ]
        attribute_context: Annotated[
            t.MetadataValue | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                description="Context payload describing the state during access failure.",
                title="Attribute Context",
                examples=[{"owner": "session"}],
            ),
        ]


__all__: list[str] = ["FlextModelsExceptionParams"]
