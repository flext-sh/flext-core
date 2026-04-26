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
    FlextModelsPydantic as mp,
    FlextUtilitiesPydantic as up,
    t,
)


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

    OptStrictStr: ClassVar = Annotated[
        str | None,
        up.Field(default=None, strict=True),
    ]
    OptStrictInt: ClassVar = Annotated[
        int | None,
        up.Field(default=None, strict=True),
    ]
    OptNumeric: ClassVar = Annotated[
        t.Numeric | None,
        up.Field(default=None),
    ]

    class ParamsModel(m.ArbitraryTypesModel):
        """Shared strict params model for exception helpers."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            extra="forbid",
            strict=True,
            validate_assignment=True,
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class ValidationErrorParams(ParamsModel):
        """Validated params for ValidationError."""

        field: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Name of the input field that failed validation.",
                title="Field",
                examples=["email"],
            ),
        ]
        value: Annotated[
            t.JsonPayload | t.Scalar | None,
            up.Field(
                default=None,
                description="Rejected input value that triggered the validation error.",
            ),
        ] = None

    class ConfigurationErrorParams(ParamsModel):
        """Validated params for ConfigurationError."""

        config_key: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Settings key associated with the error.",
                title="Settings Key",
                examples=["database_url"],
            ),
        ]
        config_source: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Settings source where the invalid value originated.",
                title="Settings Source",
                examples=[".env"],
            ),
        ]

    class ConnectionErrorParams(ParamsModel):
        """Validated params for ConnectionError."""

        host: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Hostname or address used for the failed connection attempt.",
                title="Host",
                examples=["db.internal"],
            ),
        ]
        port: Annotated[
            FlextModelsExceptionParams.OptStrictInt,
            up.Field(
                description="Network port used for the failed connection attempt.",
                title="Port",
                examples=[5432],
            ),
        ]
        timeout: Annotated[
            FlextModelsExceptionParams.OptNumeric,
            up.Field(
                description="Connection timeout threshold in seconds.",
                title="Timeout",
                examples=[5, 5.5],
            ),
        ]

    class TimeoutErrorParams(ParamsModel):
        """Validated params for TimeoutError."""

        timeout_seconds: Annotated[
            FlextModelsExceptionParams.OptNumeric,
            up.Field(
                description="Timeout duration in seconds that triggered this exception.",
                title="Timeout Seconds",
                examples=[30, 30.0],
            ),
        ]
        operation: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Operation name that exceeded the configured timeout.",
                title="Operation",
                examples=["dispatch"],
            ),
        ]

    class AuthenticationErrorParams(ParamsModel):
        """Validated params for AuthenticationError."""

        auth_method: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Authentication method used when the failure occurred.",
                title="Auth Method",
                examples=["token"],
            ),
        ]
        user_id: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="User identifier associated with the authentication attempt.",
                title="User ID",
                examples=["user-123"],
            ),
        ]

    class AuthorizationErrorParams(ParamsModel):
        """Validated params for AuthorizationError."""

        user_id: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="User identifier denied access to a protected resource.",
                title="User ID",
                examples=["user-123"],
            ),
        ]
        resource: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Protected resource that triggered the authorization failure.",
                title="Resource",
                examples=["invoice:12345"],
            ),
        ]
        permission: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Missing permission required to complete the requested action.",
                title="Permission",
                examples=["write"],
            ),
        ]

    class NotFoundErrorParams(ParamsModel):
        """Validated params for NotFoundError."""

        resource_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Domain resource type that could not be located.",
                title="Resource Type",
                examples=["user"],
            ),
        ]
        resource_id: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Unique identifier of the missing resource.",
                title="Resource ID",
                examples=["42"],
            ),
        ]

    class ConflictErrorParams(ParamsModel):
        """Validated params for ConflictError."""

        resource_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Domain resource type involved in the conflict.",
                title="Resource Type",
                examples=["order"],
            ),
        ]
        resource_id: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Identifier of the resource that caused the conflict.",
                title="Resource ID",
                examples=["ord-1001"],
            ),
        ]
        conflict_reason: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Human-readable explanation for why the conflict occurred.",
                title="Conflict Reason",
                examples=["version_mismatch"],
            ),
        ]

    class RateLimitErrorParams(ParamsModel):
        """Validated params for RateLimitError."""

        limit: Annotated[
            FlextModelsExceptionParams.OptStrictInt,
            up.Field(
                description="Maximum request count allowed within the configured window.",
                title="Limit",
                examples=[100],
            ),
        ]
        window_seconds: Annotated[
            FlextModelsExceptionParams.OptStrictInt,
            up.Field(
                description="Duration, in seconds, of the rate-limit window.",
                title="Window Seconds",
                examples=[60],
            ),
        ]
        retry_after: Annotated[
            FlextModelsExceptionParams.OptNumeric,
            up.Field(
                description="Time in seconds clients should wait before retrying.",
                title="Retry After",
                examples=[1, 1.5],
            ),
        ]

    class CircuitBreakerErrorParams(ParamsModel):
        """Validated params for CircuitBreakerError."""

        service_name: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="External service monitored by the circuit breaker.",
                title="Service Name",
                examples=["payments-api"],
            ),
        ]
        failure_count: Annotated[
            FlextModelsExceptionParams.OptStrictInt,
            up.Field(
                description="Consecutive failure count at the moment the breaker opened.",
                title="Failure Count",
                examples=[5],
            ),
        ]
        reset_timeout: Annotated[
            FlextModelsExceptionParams.OptNumeric,
            up.Field(
                description="Seconds before allowing a circuit breaker reset attempt.",
                title="Reset Timeout",
                examples=[30, 30.0],
            ),
        ]

    class TypeErrorParams(ParamsModel):
        """Validated params for TypeError."""

        expected_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Expected type name for the failing value.",
                title="Expected Type",
                examples=["str"],
            ),
        ]
        actual_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Actual type name received at runtime.",
                title="Actual Type",
                examples=["int"],
            ),
        ]

    class OperationErrorParams(ParamsModel):
        """Validated params for OperationError."""

        operation: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Operation name associated with the failure.",
                title="Operation",
                examples=["publish_events"],
            ),
        ]
        reason: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Short reason explaining the operation failure.",
                title="Reason",
                examples=["transient_backend_error"],
            ),
        ]

    class ServiceLookupParams(ParamsModel):
        """Validated params for service lookup and narrowing failures."""

        service_name: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Service identifier requested from the container/registry.",
                title="Service Name",
                examples=["command_bus"],
            ),
        ]
        expected_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Expected runtime type name for the resolved service.",
                title="Expected Type",
                examples=["FlextDispatcher"],
            ),
        ]
        actual_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Actual runtime type name returned by resolution.",
                title="Actual Type",
                examples=["str"],
            ),
        ]

    class RegistryPluginParams(ParamsModel):
        """Validated params for registry plugin registration lifecycle."""

        category: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Registry category for plugin registration.",
                title="Category",
                examples=["validators"],
            ),
        ]
        name: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Plugin name within a category.",
                title="Plugin Name",
                examples=["strict_typing"],
            ),
        ]
        scope: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Registration scope used by registry operations.",
                title="Scope",
                examples=["instance", "class"],
            ),
        ]

    class AttributeAccessErrorParams(ParamsModel):
        """Validated params for AttributeAccessError."""

        attribute_name: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Attribute name that could not be accessed safely.",
                title="Attribute Name",
                examples=["token"],
            ),
        ]
        attribute_context: Annotated[
            t.JsonValue | None,
            up.Field(
                default=None,
                description="Context payload describing the state during access failure.",
                title="Attribute Context",
                examples=[{"owner": "session"}],
            ),
        ] = None


__all__: t.MutableSequenceOf[str] = ["FlextModelsExceptionParams"]
