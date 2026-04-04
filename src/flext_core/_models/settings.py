"""Configuration patterns extracted from FlextModels.

This module contains the FlextModelsConfig class with all configuration-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Config instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, ClassVar, Final, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings

from flext_core import (
    FlextModelsBase,
    FlextModelsCollections,
    FlextModelsExceptionParams,
    FlextRuntime,
    c,
    p,
    t,
)


class FlextModelsConfig:
    """Configuration pattern container class.

    This class acts as a namespace container for configuration patterns.
    All nested classes are accessed via FlextModels.Config.* in the main models.py.
    """

    class AutoConfig(FlextModelsBase.ArbitraryTypesModel):
        """Automatic settings wrapper for BaseSettings classes."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        config_class: Annotated[
            type[BaseSettings],
            Field(description="Settings class to instantiate"),
        ]
        env_prefix: Annotated[
            str,
            Field(
                default=c.ENV_PREFIX,
                description="Environment variable prefix for settings resolution",
            ),
        ] = c.ENV_PREFIX
        env_file: Annotated[
            str | None,
            Field(
                default=None,
                description="Path to .env file for environment variable loading",
            ),
        ] = None

        def create_config(self) -> BaseSettings:
            return self.config_class()

    class ProcessingRequest(FlextModelsBase.ArbitraryTypesModel):
        """Enhanced processing request with advanced validation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=False,
            use_enum_values=True,
            arbitrary_types_allowed=True,
        )
        operation_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Unique operation identifier",
            ),
        ] = Field(default_factory=FlextRuntime.generate_id)
        data: Annotated[
            t.ConfigMap,
            Field(
                description="Primary request payload passed to the processing operation.",
                title="Processing Data",
                examples=[{"record_id": "123", "status": "pending"}],
            ),
        ] = Field(default_factory=t.ConfigMap)
        context: Annotated[
            t.ConfigMap,
            Field(
                description="Execution context metadata used for traceability and request scoping.",
                title="Processing Context",
                examples=[{"correlation_id": "corr-123"}],
            ),
        ] = Field(default_factory=t.ConfigMap)
        timeout_seconds: Annotated[
            t.PositiveFloat,
            Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                le=c.MAX_TIMEOUT_SECONDS,
                description="Operation timeout from c (Constants default)",
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        retry_attempts: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.MAX_RETRY_ATTEMPTS,
                le=c.MAX_RETRY_ATTEMPTS,
                description="Maximum retry attempts from c (Constants default)",
            ),
        ] = c.MAX_RETRY_ATTEMPTS
        enable_validation: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to run input validation before processing the request.",
            ),
        ] = True

        @field_validator("context", mode="before")
        @classmethod
        def validate_context(
            cls,
            v: BaseModel | t.ScalarMapping | t.Scalar | None,
        ) -> t.StrMapping:
            """Ensure context has required fields (using FlextRuntime).

            Returns t.StrMapping because ensure_trace_context generates
            string trace IDs. This is compatible with the field type
            ConfigurationDict since str is a subtype.
            """
            if v is None:
                return dict[str, str]()
            return FlextRuntime.ensure_trace_context(
                v,
                include_correlation_id=True,
                include_timestamp=True,
            )

    class RetryConfiguration(
        FlextModelsBase.ArbitraryTypesModel,
        FlextModelsBase.RetryConfigurationMixin,
    ):
        """Retry configuration with advanced validation."""

        max_retries: Annotated[
            t.PositiveInt,
            Field(
                default=c.MAX_RETRY_ATTEMPTS,
                le=c.MAX_RETRY_ATTEMPTS,
                alias="max_attempts",
                validation_alias=AliasChoices("max_attempts", "max_retries"),
                serialization_alias="max_attempts",
                description="Maximum retry attempts from c (Constants default)",
            ),
        ] = c.MAX_RETRY_ATTEMPTS
        exponential_backoff: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to use exponential backoff between retry attempts.",
            ),
        ] = True
        backoff_multiplier: Annotated[
            t.BackoffMultiplier,
            Field(
                default=c.DEFAULT_BACKOFF_MULTIPLIER,
                description="Backoff multiplier for exponential backoff",
            ),
        ] = c.DEFAULT_BACKOFF_MULTIPLIER
        retry_on_exceptions: Annotated[
            Sequence[type[BaseException]],
            Field(
                description="Exception types to retry on",
            ),
        ] = Field(default_factory=list[type[BaseException]])
        retry_on_status_codes: Annotated[
            Sequence[int],
            Field(
                max_length=c.HTTP_STATUS_MIN,
                description="HTTP status codes to retry on",
            ),
        ] = Field(default_factory=list[int])

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency."""
            if self.max_delay_seconds < self.initial_delay_seconds:
                msg = "max_delay_seconds must be >= initial_delay_seconds"
                raise ValueError(msg)
            return self

    class ValidationConfiguration(FlextModelsBase.ArbitraryTypesModel):
        """Validation configuration."""

        max_validation_errors: Annotated[
            t.PositiveInt,
            Field(
                default=c.DEFAULT_PAGE_SIZE,
                le=c.HTTP_STATUS_MIN,
                description="Maximum validation errors",
            ),
        ] = c.DEFAULT_PAGE_SIZE
        validate_on_assignment: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to re-validate field values on assignment.",
            ),
        ] = True
        validate_on_read: Annotated[
            bool,
            Field(
                default=False,
                description="Whether to validate field values when read.",
            ),
        ] = False
        custom_validators: Annotated[
            Sequence[p.ValidatorSpec],
            Field(
                max_length=c.MAX_CONTEXT_KEYS,
                description="Custom validator callables",
            ),
        ] = Field(default_factory=list[p.ValidatorSpec])

        @field_validator("custom_validators", mode="after")
        @classmethod
        def validate_additional_validators(
            cls,
            v: Sequence[p.ValidatorSpec],
        ) -> Sequence[p.ValidatorSpec]:
            """Validate custom validators are callable."""
            for validator in v:
                if not callable(validator):
                    base_msg = "Validator must be callable"
                    error_msg = f"{base_msg}: got {validator.__class__.__name__}"
                    raise TypeError(error_msg)
            return v

    class BatchProcessingConfig(FlextModelsCollections.Config):
        """Enhanced batch processing configuration."""

        batch_size: Annotated[
            t.PositiveInt,
            Field(
                default=c.MAX_ITEMS,
                le=c.DEFAULT_SIZE,
                description="Batch size from c (Constants default)",
            ),
        ] = c.MAX_ITEMS
        max_workers: Annotated[
            t.PositiveInt,
            Field(
                default=c.DEFAULT_MAX_WORKERS,
                le=c.MAX_CONTEXT_KEYS,
                description="Maximum workers (Config has priority over Constants)",
            ),
        ] = c.DEFAULT_MAX_WORKERS
        timeout_per_item: Annotated[
            t.PositiveFloat,
            Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                description="Timeout per item (Config has priority over Constants)",
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        continue_on_error: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to continue processing remaining items after an error.",
            ),
        ] = True
        data_items: Annotated[
            Sequence[t.ValueOrModel],
            Field(
                max_length=c.MAX_ITEMS,
                description="Ordered list of items to process in this batch; bounded by MAX_ITEMS performance constant.",
                title="Data Items",
                examples=[["item-a", "item-b"]],
            ),
        ] = Field(default_factory=list)

        @classmethod
        def validate_batch(
            cls,
            models: Sequence[t.ValueOrModel],
        ) -> Sequence[FlextModelsConfig.BatchProcessingConfig]:
            return FlextRuntime.validate_model_sequence(models, cls)

        @model_validator(mode="after")
        def validate_cross_fields(self) -> Self:
            adjusted_workers = min(self.max_workers, self.batch_size)
            if adjusted_workers != self.max_workers:
                self.max_workers = adjusted_workers
            return self

    class HandlerExecutionConfig(FlextModelsCollections.Config):
        """Enhanced handler execution configuration."""

        handler_name: Annotated[
            str,
            Field(
                pattern=c.PATTERN_IDENTIFIER,
                description="Handler identifier used to route execution in the dispatcher.",
                title="Handler Name",
                examples=["process_order", "sync_inventory"],
            ),
        ]
        input_data: Annotated[
            t.ConfigMap,
            Field(
                description="Input payload supplied to the handler during execution.",
                title="Input Data",
                examples=[{"order_id": "ord-1001"}],
            ),
        ] = Field(default_factory=t.ConfigMap)
        execution_context: Annotated[
            t.ConfigMap,
            Field(
                description="Context values provided to the handler for tracing and runtime behavior.",
                title="Execution Context",
                examples=[{"correlation_id": "corr-abc"}],
            ),
        ] = Field(default_factory=t.ConfigMap)
        timeout_seconds: Annotated[
            t.PositiveFloat,
            Field(
                default=c.DEFAULT_TIMEOUT_SECONDS,
                le=c.MAX_TIMEOUT_SECONDS,
                description="Timeout in seconds (default from constants)",
            ),
        ] = c.DEFAULT_TIMEOUT_SECONDS
        retry_on_failure: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to retry the handler execution on failure.",
            ),
        ] = True
        max_retries: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.MAX_RETRY_ATTEMPTS,
                description="Max retries (default from constants)",
            ),
        ] = c.MAX_RETRY_ATTEMPTS

    class MiddlewareConfig:
        """Configuration for middleware execution.

        Provides configuration options for middleware ordering and priority
        within request/response processing pipeline.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True,
            json_schema_extra={
                "title": "MiddlewareConfig",
                "description": "Configuration for middleware execution in request processing",
            },
        )
        enabled: Annotated[
            bool,
            Field(default=True, description="Whether middleware is enabled"),
        ] = True
        order: Annotated[
            int,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Execution order in middleware chain",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        name: Annotated[
            str | None,
            Field(default=None, description="Optional middleware name"),
        ] = None
        config: Annotated[
            t.ConfigMap,
            Field(
                description="Middleware-specific configuration",
            ),
        ] = Field(default_factory=t.ConfigMap)

    class DispatcherMiddlewareConfig(MiddlewareConfig):
        """Internal configuration for dispatcher middleware."""

        middleware_id: str
        middleware_type: str

    class RateLimiterState:
        """State tracking for rate limiter functionality.

        Tracks request counts, windows, and blocking state for rate limiting
        operations within the FLEXT request processing pipeline.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "RateLimiterState",
                "description": "State tracking for rate limiter functionality",
            },
        )
        processor_name: Annotated[
            str,
            Field(default="", description="Name of the rate limiter processor"),
        ] = ""
        count: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Current request count in window",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        window_start: Annotated[
            float,
            Field(
                default=c.INITIAL_TIME,
                ge=c.INITIAL_TIME,
                description="Timestamp when current window started",
            ),
        ] = c.INITIAL_TIME
        limit: Annotated[
            t.PositiveInt,
            Field(
                default=c.HTTP_STATUS_MIN,
                description="Maximum requests allowed per window",
            ),
        ] = c.HTTP_STATUS_MIN
        window_seconds: Annotated[
            t.PositiveInt,
            Field(
                default=c.DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
                description="Duration of rate limit window in seconds",
            ),
        ] = c.DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        block_until: Annotated[
            float,
            Field(
                default=c.INITIAL_TIME,
                ge=c.INITIAL_TIME,
                description="Timestamp until which requests are blocked",
            ),
        ] = c.INITIAL_TIME

    class ExternalCommandConfig(FlextModelsCollections.Config):
        """Configuration for external command execution (Pydantic v2).

        Reduces parameter count for u.CommandExecution
        run_external_command using canonical config-container patterns.
        Reuses timeout pattern from ProcessingRequest and HandlerExecutionConfig.
        """

        capture_output: Annotated[
            bool,
            Field(default=True, description="Whether to capture stdout/stderr"),
        ] = True
        check: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to raise exception on non-zero exit code",
            ),
        ] = True
        env: Annotated[
            t.ConfigMap | None,
            Field(
                default=None,
                description="Environment variables for the command",
            ),
        ] = None
        cwd: Annotated[
            str | None,
            Field(
                default=None,
                description="Working directory for command execution",
            ),
        ] = None
        timeout_seconds: Annotated[
            t.PositiveFloat | None,
            Field(
                default=None,
                le=c.MAX_TIMEOUT_SECONDS,
                description="Command timeout in seconds (max 5 min)",
            ),
        ] = None
        command_input: Annotated[
            str | bytes | None,
            Field(
                default=None,
                description="Input to send to command stdin",
            ),
        ] = None
        text: Annotated[
            bool | None,
            Field(
                default=None,
                description="Whether to decode stdout/stderr as text",
            ),
        ] = None

    class StructlogConfig(FlextModelsCollections.Config):
        """Configuration for structlog setup (Pydantic v2).

        Reduces parameter count for FlextRuntime.configure_structlog.
        Allows validation and composition of logging configuration.
        """

        log_level: Annotated[
            t.NonNegativeInt,
            Field(
                le=c.MAX_CONTEXT_KEYS,
                description="Numeric log level (DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50) - default from constants",
            ),
        ] = Field(default_factory=FlextRuntime.get_log_level_from_config)
        console_renderer: Annotated[
            bool,
            Field(
                default=True,
                description="Use console renderer (True) or JSON renderer (False)",
            ),
        ] = True
        additional_processors: Annotated[
            Sequence[t.StructlogProcessor],
            Field(
                description="Optional extra processors after standard FLEXT processors",
            ),
        ] = Field(default_factory=list[t.StructlogProcessor])
        wrapper_class_factory: Annotated[
            Callable[[], type] | None,
            Field(
                default=None,
                description="Custom wrapper factory for structlog",
            ),
        ] = None
        logger_factory: Annotated[
            Callable[..., p.OutputLogger] | None,
            Field(
                default=None,
                description="Custom logger factory for structlog",
            ),
        ] = None
        cache_logger_on_first_use: Annotated[
            bool,
            Field(
                default=True,
                description="Cache logger on first use (performance optimization)",
            ),
        ] = True
        async_logging: Annotated[
            bool,
            Field(
                default=True,
                description="Enable asynchronous buffered logging backend",
            ),
        ] = True

    class LoggerConfig(FlextModelsCollections.Config):
        """Configuration for FlextLogger initialization (Pydantic v2).

        Reduces parameter count for FlextLogger.__init__ from 6 to 2 params.
        Groups optional logger context and configuration.
        """

        level: Annotated[
            str,
            Field(
                default=c.DEFAULT_LEVEL,
                description="Log level (default from constants, can be overridden)",
            ),
        ] = c.DEFAULT_LEVEL
        service_name: Annotated[
            str | None,
            Field(
                default=None,
                description="Service name for distributed tracing context",
            ),
        ] = None
        service_version: Annotated[
            str | None,
            Field(
                default=None,
                description="Service version for distributed tracing context",
            ),
        ] = None
        correlation_id: Annotated[
            str | None,
            Field(
                default=None,
                description="Correlation ID for distributed tracing",
            ),
        ] = None
        force_new: Annotated[
            bool,
            Field(
                default=False,
                description="Force creation of new logger instance (for testing)",
            ),
        ] = False

    class DispatchConfig(FlextModelsCollections.Config):
        """Configuration for FlextDispatcher.dispatch (Pydantic v2).

        Reduces parameter count for dispatch from 5 to 3 params (message, data, config).
        Groups optional dispatch context and overrides.
        """

        metadata: Annotated[
            FlextModelsBase.Metadata | None,
            Field(
                default=None,
                description="Optional execution context metadata (Pydantic model)",
            ),
        ] = None
        correlation_id: Annotated[
            str | None,
            Field(
                default=None,
                description="Optional correlation ID for distributed tracing",
            ),
        ] = None
        timeout_override: Annotated[
            t.NonNegativeInt | None,
            Field(
                default=None,
                description="Optional timeout override in seconds",
            ),
        ] = None

    class ExecuteDispatchAttemptOptions(FlextModelsCollections.Config):
        """Options for _execute_dispatch_attempt (Pydantic v2).

        Reduces parameter count from 6 to 2 params (message, options).
        Groups execution context parameters.
        """

        message_type: Annotated[
            t.NonEmptyStr,
            Field(
                description="Message type name for routing and circuit breaker",
            ),
        ]
        metadata: Annotated[
            t.ValueOrModel | None,
            Field(
                default=None,
                description="Optional execution context metadata",
            ),
        ] = None
        correlation_id: Annotated[
            str | None,
            Field(
                default=None,
                description="Optional correlation ID for distributed tracing",
            ),
        ] = None
        timeout_override: Annotated[
            t.NonNegativeInt | None,
            Field(
                default=None,
                description="Optional timeout override in seconds",
            ),
        ] = None
        operation_id: Annotated[
            t.NonEmptyStr,
            Field(description="Operation ID for timeout tracking"),
        ]

    class RuntimeScopeOptions(FlextModelsCollections.Config):
        """Options for runtime_scope (Pydantic v2).

        Reduces parameter count from 7 to 2 params (self, options).
        Groups runtime scope configuration parameters.
        """

        config_overrides: Annotated[
            Mapping[str, t.ValueOrModel] | None,
            Field(default=None, description="Optional configuration overrides"),
        ] = None
        context: Annotated[
            p.Context | None,
            Field(default=None, description="Optional context protocol instance"),
        ] = None
        subproject: Annotated[
            str | None,
            Field(default=None, description="Optional subproject name"),
        ] = None
        services: Annotated[
            Mapping[str, t.RegisterableService] | None,
            Field(default=None, description="Optional container services mapping"),
        ] = None
        factories: Annotated[
            Mapping[str, t.FactoryCallable] | None,
            Field(default=None, description="Optional container factories mapping"),
        ] = None
        container_services: Annotated[
            Mapping[str, t.RegisterableService] | None,
            Field(
                default=None,
                description="Optional container services (alias for services)",
            ),
        ] = None
        container_factories: Annotated[
            Mapping[str, t.FactoryCallable] | None,
            Field(
                default=None,
                description="Optional container factories (alias for factories)",
            ),
        ] = None

    class NestedExecutionOptions(FlextModelsCollections.Config):
        """Options for nested_execution (Pydantic v2).

        Reduces parameter count from 6 to 2 params (self, options).
        Groups nested execution configuration parameters.
        """

        config_overrides: Annotated[
            Mapping[str, t.ValueOrModel] | None,
            Field(default=None, description="Optional configuration overrides"),
        ] = None
        service_name: Annotated[
            str | None,
            Field(default=None, description="Optional service name"),
        ] = None
        version: Annotated[
            str | None,
            Field(default=None, description="Optional version string"),
        ] = None
        correlation_id: Annotated[
            str | None,
            Field(default=None, description="Optional correlation ID for tracing"),
        ] = None
        container_services: Annotated[
            Mapping[str, t.RegisterableService] | None,
            Field(default=None, description="Optional container services mapping"),
        ] = None
        container_factories: Annotated[
            Mapping[str, t.FactoryCallable] | None,
            Field(default=None, description="Optional container factories mapping"),
        ] = None

    class ExceptionConfig(FlextModelsCollections.Config):
        """Configuration for e.__init__ (Pydantic v2).

        Reduces parameter count for exception initialization from 7 to 2 params
        (message, config). Groups optional exception context and behavior.
        """

        error_code: Annotated[
            str | None,
            Field(default=None, description="Error code for categorization"),
        ] = None
        correlation_id: Annotated[
            str | None,
            Field(default=None, description="Correlation ID for distributed tracing"),
        ] = None
        metadata: Annotated[
            FlextModelsBase.Metadata | None,
            Field(default=None, description="Additional metadata (Pydantic model)"),
        ] = None
        auto_log: Annotated[
            bool,
            Field(default=False, description="Whether to automatically log exception"),
        ] = False
        auto_correlation: Annotated[
            bool,
            Field(
                default=False,
                description="Whether to auto-generate correlation ID",
            ),
        ] = False
        extra_kwargs: Annotated[
            t.Dict,
            Field(
                description="Additional keyword arguments for metadata",
            ),
        ] = Field(default_factory=t.Dict)

    class ResultConfig(FlextModelsCollections.Config):
        """Configuration for p.Result failure case (Pydantic v2).

        Groups optional error context for result failures.
        """

        error: Annotated[
            str | None,
            Field(default=None, description="Error message for failure case"),
        ] = None
        error_code: Annotated[
            str | None,
            Field(default=None, description="Error code for categorization"),
        ] = None
        error_data: Annotated[
            FlextModelsBase.Metadata | None,
            Field(default=None, description="Additional error data (Pydantic model)"),
        ] = None

    class ValidationErrorConfig(
        FlextModelsExceptionParams.ValidationErrorParams, ExceptionConfig
    ):
        """Configuration for ValidationError (Pydantic v2)."""

    class ConfigurationErrorConfig(
        FlextModelsExceptionParams.ConfigurationErrorParams, ExceptionConfig
    ):
        """Configuration for ConfigurationError (Pydantic v2)."""

    class ConnectionErrorConfig(
        FlextModelsExceptionParams.ConnectionErrorParams, ExceptionConfig
    ):
        """Configuration for ConnectionError (Pydantic v2)."""

    class TimeoutErrorConfig(
        FlextModelsExceptionParams.TimeoutErrorParams, ExceptionConfig
    ):
        """Configuration for TimeoutError (Pydantic v2)."""

    class AuthenticationErrorConfig(
        FlextModelsExceptionParams.AuthenticationErrorParams, ExceptionConfig
    ):
        """Configuration for AuthenticationError (Pydantic v2)."""

    class AuthorizationErrorConfig(
        FlextModelsExceptionParams.AuthorizationErrorParams, ExceptionConfig
    ):
        """Configuration for AuthorizationError (Pydantic v2)."""

    class NotFoundErrorConfig(
        FlextModelsExceptionParams.NotFoundErrorParams, ExceptionConfig
    ):
        """Configuration for NotFoundError (Pydantic v2)."""

    class ConflictErrorConfig(
        FlextModelsExceptionParams.ConflictErrorParams, ExceptionConfig
    ):
        """Configuration for ConflictError (Pydantic v2)."""

    class RateLimitErrorConfig(
        FlextModelsExceptionParams.RateLimitErrorParams, ExceptionConfig
    ):
        """Configuration for RateLimitError (Pydantic v2)."""

    class InternalErrorConfig(ExceptionConfig):
        """Configuration for InternalError (Pydantic v2)."""

        component: Annotated[
            str | None,
            Field(default=None, description="Component where internal error occurred"),
        ] = None
        operation: Annotated[
            str | None,
            Field(default=None, description="Operation that caused internal error"),
        ] = None

    class TypeErrorConfig(FlextModelsExceptionParams.TypeErrorParams, ExceptionConfig):
        """Configuration for TypeError (Pydantic v2)."""

    class TypeErrorOptions(FlextModelsCollections.Config):
        """Options for TypeError initialization (Pydantic v2).

        Groups TypeError constructor parameters for cleaner initialization.
        """

        expected_type: Annotated[
            type | None,
            Field(default=None, description="Expected type class"),
        ] = None
        actual_type: Annotated[
            type | None,
            Field(default=None, description="Actual type class"),
        ] = None
        context: Annotated[
            Mapping[str, t.ValueOrModel] | None,
            Field(default=None, description="Additional context for error"),
        ] = None
        metadata: Annotated[
            FlextModelsBase.Metadata | Mapping[str, t.ValueOrModel] | None,
            Field(default=None, description="Metadata for error"),
        ] = None

    class ValueErrorConfig(ExceptionConfig):
        """Configuration for ValueError (Pydantic v2)."""

        expected_value: Annotated[
            str | None,
            Field(default=None, description="Expected value description"),
        ] = None
        actual_value: Annotated[
            t.ValueOrModel | None,
            Field(default=None, description="Actual value that caused error"),
        ] = None

    class CircuitBreakerErrorConfig(
        FlextModelsExceptionParams.CircuitBreakerErrorParams, ExceptionConfig
    ):
        """Configuration for CircuitBreakerError (Pydantic v2)."""

    class OperationErrorConfig(
        FlextModelsExceptionParams.OperationErrorParams, ExceptionConfig
    ):
        """Configuration for OperationError (Pydantic v2)."""

    class AttributeAccessErrorConfig(ExceptionConfig):
        """Configuration for AttributeAccessError (Pydantic v2)."""

        attribute_name: Annotated[
            str | None,
            Field(default=None, description="Attribute name that access failed for"),
        ] = None
        object_type: Annotated[
            str | None,
            Field(
                default=None,
                description="Type of canonical value that attribute access failed on",
            ),
        ] = None

    class OperationExtraConfig(FlextModelsCollections.Config):
        """Configuration for operation logging extra data (Pydantic v2).

        Reduces parameter count for _build_operation_extra from 8 to 2 params.
        Groups operation context and performance tracking.
        """

        func_name: Annotated[
            t.NonEmptyStr,
            Field(description="Function name for logging"),
        ]
        func_module: Annotated[
            t.NonEmptyStr,
            Field(description="Function module for logging"),
        ]
        correlation_id: Annotated[
            str | None,
            Field(default=None, description="Correlation ID for distributed tracing"),
        ] = None
        success: Annotated[
            bool | None,
            Field(default=None, description="Operation success status"),
        ] = None
        error: Annotated[
            str | None,
            Field(default=None, description="Error message if operation failed"),
        ] = None
        error_type: Annotated[
            str | None,
            Field(default=None, description="Error type name"),
        ] = None
        start_time: Annotated[
            float,
            Field(
                default=c.INITIAL_TIME,
                ge=c.INITIAL_TIME,
                description="Operation start time for performance tracking",
            ),
        ] = c.INITIAL_TIME
        track_perf: Annotated[
            bool,
            Field(default=False, description="Whether to track performance metrics"),
        ] = False

    class LogOperationFailureConfig(FlextModelsCollections.Config):
        """Configuration for logging operation failures (Pydantic v2).

        Reduces parameter count for _log_operation_failure from 8 to 3 params.
        Groups logger, operation context, and exception details.
        """

        op_name: Annotated[t.NonEmptyStr, Field(description="Operation name")]
        func_name: Annotated[t.NonEmptyStr, Field(description="Function name")]
        func_module: Annotated[t.NonEmptyStr, Field(description="Function module")]
        correlation_id: Annotated[
            str | None,
            Field(default=None, description="Correlation ID for distributed tracing"),
        ] = None
        exc: Annotated[Exception, Field(description="Exception that caused failure")]
        start_time: Annotated[
            float,
            Field(
                default=c.INITIAL_TIME,
                ge=c.INITIAL_TIME,
                description="Operation start time",
            ),
        ] = c.INITIAL_TIME
        track_perf: Annotated[
            bool,
            Field(default=False, description="Whether to track performance metrics"),
        ] = False

    class RetryLoopConfig(FlextModelsBase.ArbitraryTypesModel):
        """Configuration for retry loop execution (Pydantic v2).

        Reduces parameter count for _execute_retry_loop from 8 to 3 params.
        Groups function, arguments, logger, and retry configuration.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)
        func: Annotated[
            Callable[..., t.ValueOrModel],
            Field(description="Function to execute"),
        ]
        args: Annotated[
            tuple[t.ValueOrModel, ...],
            Field(
                description="Positional arguments for function",
            ),
        ] = Field(default_factory=tuple)
        call_kwargs: Annotated[
            t.ConfigMap,
            Field(
                description="Keyword arguments for function",
            ),
        ] = Field(default_factory=t.ConfigMap)
        retry_config: FlextModelsConfig.RetryConfiguration | None = Field(
            default=None,
            description="Retry configuration (takes priority over individual params)",
        )
        attempts: Annotated[
            t.PositiveInt,
            Field(
                default=c.MAX_RETRY_ATTEMPTS,
                description="Number of retry attempts (used if retry_config is None)",
            ),
        ] = c.MAX_RETRY_ATTEMPTS
        delay: Annotated[
            t.PositiveFloat,
            Field(
                default=float(c.DEFAULT_RETRY_DELAY_SECONDS),
                description="Initial delay between retries (used if retry_config is None)",
            ),
        ] = float(c.DEFAULT_RETRY_DELAY_SECONDS)
        strategy: Annotated[
            str,
            Field(
                default=c.DEFAULT_BACKOFF_STRATEGY,
                description="Retry strategy: 'exponential' or 'linear' (used if retry_config is None)",
            ),
        ] = c.DEFAULT_BACKOFF_STRATEGY

    class DispatcherConfig(FlextModelsBase.ArbitraryTypesModel):
        """Configuration for message dispatcher.

        Replaces legacy dispatcher config mapping from typings.py.
        Provides type-safe configuration for message dispatcher behavior.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            use_enum_values=True,
            extra="forbid",
        )
        dispatcher_timeout_seconds: Annotated[
            t.PositiveFloat,
            Field(
                default=30.0,
                description="Timeout in seconds for dispatcher operations",
            ),
        ] = 30.0
        executor_workers: Annotated[
            t.WorkerCount,
            Field(
                default=4,
                le=256,
                description="Number of executor worker threads",
            ),
        ] = 4
        circuit_breaker_threshold: Annotated[
            t.PositiveInt,
            Field(
                default=5,
                description="Circuit breaker failure threshold",
            ),
        ] = 5
        rate_limit_max_requests: Annotated[
            t.PositiveInt,
            Field(
                default=1000,
                description="Maximum requests for rate limiting",
            ),
        ] = 1000
        rate_limit_window_seconds: Annotated[
            t.PositiveFloat,
            Field(
                default=60.0,
                description="Rate limit window in seconds",
            ),
        ] = 60.0
        max_retry_attempts: Annotated[
            t.RetryCount,
            Field(
                default=3,
                description="Maximum retry attempts",
            ),
        ] = 3
        retry_delay: Annotated[
            t.NonNegativeFloat,
            Field(
                default=1.0,
                description="Delay between retries in seconds",
            ),
        ] = 1.0
        enable_timeout_executor: Annotated[
            bool,
            Field(default=True, description="Enable timeout executor"),
        ] = True
        dispatcher_enable_logging: Annotated[
            bool,
            Field(default=True, description="Enable dispatcher logging"),
        ] = True
        dispatcher_auto_context: Annotated[
            bool,
            Field(
                default=True,
                description="Automatically add context to messages",
            ),
        ] = True
        dispatcher_enable_metrics: Annotated[
            bool,
            Field(
                default=True,
                description="Enable dispatcher metrics collection",
            ),
        ] = True

    DOMAIN_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        validate_return=True,
        validate_default=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=False,
        extra="forbid",
    )
    "Domain model configuration defaults.\n\n    Moved from FlextConstants.DOMAIN_MODEL_CONFIG because\n    constants.py cannot import ConfigDict from pydantic.\n\n    Use m.DOMAIN_MODEL_CONFIG instead of c.DOMAIN_MODEL_CONFIG.\n    "


__all__ = ["FlextModelsConfig"]
