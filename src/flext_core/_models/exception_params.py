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

    class ResourceIdentityParams(ParamsModel):
        """Shared resource identity fields for resource-oriented errors."""

        resource_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Domain resource type associated with the failure.",
            ),
        ] = None
        resource_id: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Identifier of the resource associated with the failure.",
            ),
        ] = None

    class ExpectedActualTypeParams(ParamsModel):
        """Shared expected/actual runtime type fields."""

        expected_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Expected runtime type name for the failing value.",
            ),
        ] = None
        actual_type: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Actual runtime type name received at runtime.",
            ),
        ] = None

    class ValidationErrorParams(ParamsModel):
        """Validated params for ValidationError."""

        field: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Name of the input field that failed validation.",
                title="Field",
                examples=["email"],
            ),
        ] = None
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
            ),
        ] = None
        config_source: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Settings source where the invalid value originated.",
            ),
        ] = None

    class ConnectionErrorParams(ParamsModel):
        """Validated params for ConnectionError."""

        host: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Hostname or address used for the failed connection attempt.",
            ),
        ] = None
        port: Annotated[
            FlextModelsExceptionParams.OptStrictInt,
            up.Field(
                description="Network port used for the failed connection attempt.",
            ),
        ] = None
        timeout: Annotated[
            FlextModelsExceptionParams.OptNumeric,
            up.Field(
                description="Connection timeout threshold in seconds.",
            ),
        ] = None

    class TimeoutErrorParams(ParamsModel):
        """Validated params for TimeoutError."""

        timeout_seconds: Annotated[
            FlextModelsExceptionParams.OptNumeric,
            up.Field(
                default=None,
                description="Timeout duration in seconds that triggered this exception.",
                title="Timeout Seconds",
                examples=[30, 30.0],
            ),
        ] = None
        operation: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Operation name that exceeded the configured timeout.",
                title="Operation",
                examples=["dispatch"],
            ),
        ] = None

    class AuthenticationErrorParams(ParamsModel):
        """Validated params for AuthenticationError."""

        auth_method: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Authentication method used when the failure occurred.",
            ),
        ] = None
        user_id: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="User identifier associated with the authentication attempt.",
            ),
        ] = None

    class AuthorizationErrorParams(ParamsModel):
        """Validated params for AuthorizationError."""

        user_id: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="User identifier denied access to a protected resource.",
                title="User ID",
                examples=["user-123"],
            ),
        ] = None
        resource: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Protected resource that triggered the authorization failure.",
                title="Resource",
                examples=["invoice:12345"],
            ),
        ] = None
        permission: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Missing permission required to complete the requested action.",
                title="Permission",
                examples=["write"],
            ),
        ] = None

    class NotFoundErrorParams(ResourceIdentityParams):
        """Validated params for NotFoundError."""

    class ConflictErrorParams(ResourceIdentityParams):
        """Validated params for ConflictError."""

        conflict_reason: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Human-readable explanation for why the conflict occurred.",
                title="Conflict Reason",
                examples=["version_mismatch"],
            ),
        ] = None

    class RateLimitErrorParams(ParamsModel):
        """Validated params for RateLimitError."""

        limit: Annotated[
            FlextModelsExceptionParams.OptStrictInt,
            up.Field(
                default=None,
                description="Maximum request count allowed within the configured window.",
                title="Limit",
                examples=[100],
            ),
        ] = None
        window_seconds: Annotated[
            FlextModelsExceptionParams.OptStrictInt,
            up.Field(
                default=None,
                description="Duration, in seconds, of the rate-limit window.",
                title="Window Seconds",
                examples=[60],
            ),
        ] = None
        retry_after: Annotated[
            FlextModelsExceptionParams.OptNumeric,
            up.Field(
                default=None,
                description="Time in seconds clients should wait before retrying.",
                title="Retry After",
                examples=[1, 1.5],
            ),
        ] = None

    class CircuitBreakerErrorParams(ParamsModel):
        """Validated params for CircuitBreakerError."""

        service_name: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="External service monitored by the circuit breaker.",
            ),
        ] = None
        failure_count: Annotated[
            FlextModelsExceptionParams.OptStrictInt,
            up.Field(
                description="Consecutive failure count at the moment the breaker opened.",
            ),
        ] = None
        reset_timeout: Annotated[
            FlextModelsExceptionParams.OptNumeric,
            up.Field(
                description="Seconds before allowing a circuit breaker reset attempt.",
            ),
        ] = None

    class TypeErrorParams(ExpectedActualTypeParams):
        """Validated params for TypeError."""

    class OperationErrorParams(ParamsModel):
        """Validated params for OperationError."""

        operation: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Operation name associated with the failure.",
            ),
        ] = None
        reason: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                description="Short reason explaining the operation failure.",
            ),
        ] = None

    class ServiceLookupParams(ExpectedActualTypeParams):
        """Validated params for service lookup and narrowing failures."""

        service_name: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Service identifier requested from the container/registry.",
                title="Service Name",
                examples=["command_bus"],
            ),
        ] = None

    class RegistryPluginParams(ParamsModel):
        """Validated params for registry plugin registration lifecycle."""

        category: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Registry category for plugin registration.",
                title="Category",
                examples=["validators"],
            ),
        ] = None
        name: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Plugin name within a category.",
                title="Plugin Name",
                examples=["strict_typing"],
            ),
        ] = None
        scope: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Registration scope used by registry operations.",
                title="Scope",
                examples=["instance", "class"],
            ),
        ] = None

    class AttributeAccessErrorParams(ParamsModel):
        """Validated params for AttributeAccessError."""

        attribute_name: Annotated[
            FlextModelsExceptionParams.OptStrictStr,
            up.Field(
                default=None,
                description="Attribute name that could not be accessed safely.",
                title="Attribute Name",
                examples=["token"],
            ),
        ] = None
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
