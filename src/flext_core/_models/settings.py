"""Configuration patterns extracted from FlextModels.

This module contains the FlextModelsConfig class with all configuration-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Config instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, ClassVar, Final, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from flext_core import c, p, r, t
from flext_core._models import (
    FlextModelFoundation,
    FlextModelsCollections,
)
from flext_core.runtime import FlextRuntime


class FlextModelsConfig:
    """Configuration pattern container class.

    This class acts as a namespace container for configuration patterns.
    All nested classes are accessed via FlextModels.Config.* in the main models.py.
    """

    @staticmethod
    def _get_log_level_from_config() -> int:
        """Get default log level from configuration constants.

        Returns:
            int: Numeric logging level (e.g., logging.INFO = 20)

        """
        return FlextRuntime.get_log_level_from_config()

    class ProcessingRequest(FlextModelFoundation.ArbitraryTypesModel):
        """Enhanced processing request with advanced validation."""

        model_config = ConfigDict(
            validate_assignment=False,
            use_enum_values=True,
            arbitrary_types_allowed=True,
        )
        operation_id: Annotated[
            str,
            Field(
                default_factory=FlextRuntime.generate_id,
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Unique operation identifier",
            ),
        ]
        data: Annotated[
            t.ConfigMap,
            Field(
                default_factory=t.ConfigMap,
                description="Primary request payload passed to the processing operation.",
                title="Processing Data",
                examples=[{"record_id": "123", "status": "pending"}],
            ),
        ]
        context: Annotated[
            t.ConfigMap,
            Field(
                default_factory=t.ConfigMap,
                description="Execution context metadata used for traceability and request scoping.",
                title="Processing Context",
                examples=[{"correlation_id": "corr-123"}],
            ),
        ]
        timeout_seconds: Annotated[
            float,
            Field(
                default=c.Defaults.TIMEOUT,
                gt=c.ZERO,
                le=c.Performance.MAX_TIMEOUT_SECONDS,
                description="Operation timeout from c (Constants default)",
            ),
        ] = c.Defaults.TIMEOUT
        retry_attempts: Annotated[
            int,
            Field(
                default=c.Reliability.MAX_RETRY_ATTEMPTS,
                ge=c.ZERO,
                le=c.Reliability.MAX_RETRY_ATTEMPTS,
                description="Maximum retry attempts from c (Constants default)",
            ),
        ] = c.Reliability.MAX_RETRY_ATTEMPTS
        enable_validation: bool = True

        @field_validator("context", mode="before")
        @classmethod
        def validate_context(
            cls, v: BaseModel | Mapping[str, t.Scalar] | t.Scalar | None
        ) -> Mapping[str, str]:
            """Ensure context has required fields (using FlextRuntime).

            Returns Mapping[str, str] because ensure_trace_context generates
            string trace IDs. This is compatible with the field type
            ConfigurationDict since str is a subtype.
            """
            if v is None:
                return {}
            return FlextRuntime.ensure_trace_context(
                v, include_correlation_id=True, include_timestamp=True
            )

    class RetryConfiguration(
        FlextModelFoundation.ArbitraryTypesModel,
        FlextModelFoundation.RetryConfigurationMixin,
    ):
        """Retry configuration with advanced validation."""

        max_retries: Annotated[
            int,
            Field(
                default=c.Reliability.MAX_RETRY_ATTEMPTS,
                ge=c.Reliability.RETRY_COUNT_MIN,
                le=c.Reliability.MAX_RETRY_ATTEMPTS,
                alias="max_attempts",
                validation_alias=AliasChoices("max_attempts", "max_retries"),
                serialization_alias="max_attempts",
                description="Maximum retry attempts from c (Constants default)",
            ),
        ] = c.Reliability.MAX_RETRY_ATTEMPTS
        exponential_backoff: bool = True
        backoff_multiplier: Annotated[
            float,
            Field(
                default=c.DEFAULT_BACKOFF_MULTIPLIER,
                ge=float(c.Reliability.RETRY_COUNT_MIN),
                description="Backoff multiplier for exponential backoff",
            ),
        ] = c.DEFAULT_BACKOFF_MULTIPLIER
        retry_on_exceptions: Annotated[
            list[type[BaseException]],
            Field(
                default_factory=list,
                description="Exception types to retry on",
            ),
        ]
        retry_on_status_codes: Annotated[
            list[int],
            Field(
                default_factory=list,
                max_length=c.Validation.MAX_RETRY_STATUS_CODES,
                description="HTTP status codes to retry on",
            ),
        ]

        @field_validator("retry_on_status_codes", mode="after")
        @classmethod
        def validate_backoff_strategy(cls, v: list[int] | list[t.Scalar]) -> list[int]:
            """Validate status codes are valid HTTP codes."""
            codes_for_validation: list[int] = []
            for item in v:
                if isinstance(item, bool):
                    msg = "retry_on_status_codes item must be int or str, got bool"
                    raise TypeError(msg)
                if isinstance(item, int):
                    codes_for_validation.append(item)
                else:
                    try:
                        parsed_code = int(str(item))
                    except (TypeError, ValueError) as parse_exc:
                        msg = f"retry_on_status_codes item must be int or str: {parse_exc}"
                        raise TypeError(msg) from parse_exc
                    codes_for_validation.append(parsed_code)
            result = FlextRuntime.validate_http_status_codes(codes_for_validation)
            if result.is_failure:
                base_msg = "HTTP status code validation failed"
                error_msg = (
                    f"{base_msg}: {result.error}"
                    if result.error
                    else f"{base_msg} (invalid status code)"
                )
                raise ValueError(error_msg)
            validated_codes: list[int] = result.value
            return validated_codes

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency."""
            if self.max_delay_seconds < self.initial_delay_seconds:
                msg = "max_delay_seconds must be >= initial_delay_seconds"
                raise ValueError(msg)
            return self

    class ValidationConfiguration(FlextModelFoundation.ArbitraryTypesModel):
        """Validation configuration."""

        max_validation_errors: Annotated[
            int,
            Field(
                default=c.Cqrs.DEFAULT_MAX_VALIDATION_ERRORS,
                ge=c.Reliability.RETRY_COUNT_MIN,
                le=c.Validation.MAX_RETRY_STATUS_CODES,
                description="Maximum validation errors",
            ),
        ] = c.Cqrs.DEFAULT_MAX_VALIDATION_ERRORS
        validate_on_assignment: bool = True
        validate_on_read: bool = False
        custom_validators: Annotated[
            list[p.ValidatorSpec],
            Field(
                default_factory=list,
                max_length=c.Validation.MAX_CUSTOM_VALIDATORS,
                description="Custom validator callables",
            ),
        ]

        @field_validator("custom_validators", mode="after")
        @classmethod
        def validate_additional_validators(
            cls, v: list[p.ValidatorSpec]
        ) -> list[p.ValidatorSpec]:
            """Validate custom validators are callable."""
            for validator in v:
                if not callable(validator):
                    base_msg = "Validator must be callable"
                    error_msg = f"{base_msg}: got {validator.__class__.__name__}"
                    raise TypeError(error_msg)
            return v

    class BatchProcessingConfig(FlextModelsCollections.Config):
        """Enhanced batch processing configuration."""

        _batch_list_adapter: ClassVar[
            TypeAdapter[list[FlextModelsConfig.BatchProcessingConfig]] | None
        ] = None

        batch_size: Annotated[
            int,
            Field(
                default=c.Performance.MAX_BATCH_SIZE,
                le=c.Performance.BatchProcessing.MAX_VALIDATION_SIZE,
                description="Batch size from c (Constants default)",
            ),
        ] = c.Performance.MAX_BATCH_SIZE
        max_workers: Annotated[
            int,
            Field(
                default=c.Processing.DEFAULT_MAX_WORKERS,
                le=c.Settings.MAX_WORKERS_THRESHOLD,
                description="Maximum workers (Config has priority over Constants)",
            ),
        ] = c.Processing.DEFAULT_MAX_WORKERS
        timeout_per_item: Annotated[
            float,
            Field(
                default=c.Defaults.TIMEOUT,
                description="Timeout per item (Config has priority over Constants)",
            ),
        ] = c.Defaults.TIMEOUT
        continue_on_error: bool = True
        data_items: Annotated[
            list[t.ValueOrModel],
            Field(
                default_factory=list,
                max_length=c.Performance.BatchProcessing.MAX_ITEMS,
                description="Ordered list of items to process in this batch; bounded by MAX_ITEMS performance constant.",
                title="Data Items",
                examples=[["item-a", "item-b"]],
            ),
        ]

        @classmethod
        def _batch_adapter(
            cls,
        ) -> TypeAdapter[list[FlextModelsConfig.BatchProcessingConfig]]:
            if cls._batch_list_adapter is None:
                cls._batch_list_adapter = TypeAdapter(
                    list[FlextModelsConfig.BatchProcessingConfig]
                )
            return cls._batch_list_adapter

        @classmethod
        def validate_batch(
            cls, models: list[t.ValueOrModel]
        ) -> list[FlextModelsConfig.BatchProcessingConfig]:
            batch_result = r[
                list[FlextModelsConfig.BatchProcessingConfig]
            ].create_from_callable(lambda: cls._batch_adapter().validate_python(models))
            if batch_result.is_success:
                return batch_result.value
            exc = getattr(batch_result, "_exception", None)
            if isinstance(exc, ValidationError):
                item_errors = [
                    f"{'.'.join(str(part) for part in err.get('loc', ()))}: {err.get('msg', 'validation error')}"
                    for err in exc.errors()
                ]
                msg = f"Batch validation failed: {'; '.join(item_errors)}"
                raise TypeError(msg)
            raise ValueError(batch_result.error or "Batch validation failed")

        @model_validator(mode="after")
        def validate_cross_fields(self) -> Self:
            adjusted_workers = min(self.max_workers, self.batch_size)
            self.max_workers = adjusted_workers
            return self

    class HandlerExecutionConfig(FlextModelsCollections.Config):
        """Enhanced handler execution configuration."""

        handler_name: Annotated[
            str,
            Field(
                pattern=c.Platform.PATTERN_IDENTIFIER,
                description="Handler identifier used to route execution in the dispatcher.",
                title="Handler Name",
                examples=["process_order", "sync_inventory"],
            ),
        ]
        input_data: Annotated[
            t.ConfigMap,
            Field(
                default_factory=t.ConfigMap,
                description="Input payload supplied to the handler during execution.",
                title="Input Data",
                examples=[{"order_id": "ord-1001"}],
            ),
        ]
        execution_context: Annotated[
            t.ConfigMap,
            Field(
                default_factory=t.ConfigMap,
                description="Context values provided to the handler for tracing and runtime behavior.",
                title="Execution Context",
                examples=[{"correlation_id": "corr-abc"}],
            ),
        ]
        timeout_seconds: Annotated[
            float,
            Field(
                default=c.Defaults.TIMEOUT,
                le=c.Performance.MAX_TIMEOUT_SECONDS,
                description="Timeout in seconds (default from constants)",
            ),
        ] = c.Defaults.TIMEOUT
        retry_on_failure: bool = True
        max_retries: Annotated[
            int,
            Field(
                default=c.Reliability.MAX_RETRY_ATTEMPTS,
                description="Max retries (default from constants)",
            ),
        ] = c.Reliability.MAX_RETRY_ATTEMPTS

    class MiddlewareConfig:
        """Configuration for middleware execution.

        Provides configuration options for middleware ordering and priority
        within request/response processing pipeline.
        """

        model_config = ConfigDict(
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
                default=c.Defaults.DEFAULT_MIDDLEWARE_ORDER,
                description="Execution order in middleware chain",
            ),
        ] = c.Defaults.DEFAULT_MIDDLEWARE_ORDER
        name: Annotated[
            str | None,
            Field(default=None, description="Optional middleware name"),
        ] = None
        config: Annotated[
            t.ConfigMap,
            Field(
                default_factory=t.ConfigMap,
                description="Middleware-specific configuration",
            ),
        ]

    class DispatcherMiddlewareConfig(MiddlewareConfig):
        """Internal configuration for dispatcher middleware."""

        middleware_id: str
        middleware_type: str

    class RateLimiterState:
        """State tracking for rate limiter functionality.

        Tracks request counts, windows, and blocking state for rate limiting
        operations within the FLEXT request processing pipeline.
        """

        model_config = ConfigDict(
            json_schema_extra={
                "title": "RateLimiterState",
                "description": "State tracking for rate limiter functionality",
            }
        )
        processor_name: Annotated[
            str,
            Field(default="", description="Name of the rate limiter processor"),
        ] = ""
        count: Annotated[
            int,
            Field(
                default=c.ZERO,
                ge=c.ZERO,
                description="Current request count in window",
            ),
        ] = c.ZERO
        window_start: Annotated[
            float,
            Field(
                default=c.INITIAL_TIME,
                ge=c.INITIAL_TIME,
                description="Timestamp when current window started",
            ),
        ] = c.INITIAL_TIME
        limit: Annotated[
            int,
            Field(
                default=c.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
                ge=c.Reliability.RETRY_COUNT_MIN,
                description="Maximum requests allowed per window",
            ),
        ] = c.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS
        window_seconds: Annotated[
            int,
            Field(
                default=c.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
                ge=c.Reliability.RETRY_COUNT_MIN,
                description="Duration of rate limit window in seconds",
            ),
        ] = c.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS
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
        run_external_command using config object pattern.
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
            float | None,
            Field(
                default=None,
                gt=c.ZERO,
                le=c.Performance.MAX_TIMEOUT_SECONDS,
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
            int,
            Field(
                default_factory=FlextRuntime.get_log_level_from_config,
                ge=c.ZERO,
                le=c.Validation.MAX_CUSTOM_VALIDATORS,
                description="Numeric log level (DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50) - default from constants",
            ),
        ]
        console_renderer: Annotated[
            bool,
            Field(
                default=True,
                description="Use console renderer (True) or JSON renderer (False)",
            ),
        ] = True
        additional_processors: Annotated[
            list[Callable[..., t.Container]],
            Field(
                default_factory=list,
                description="Optional extra processors after standard FLEXT processors",
            ),
        ]
        wrapper_class_factory: Annotated[
            Callable[[], type] | None,
            Field(
                default=None,
                description="Custom wrapper factory for structlog",
            ),
        ] = None
        logger_factory: Annotated[
            Callable[..., p.Logger] | None,
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

    class LoggerConfig(FlextModelsCollections.Config):
        """Configuration for FlextLogger initialization (Pydantic v2).

        Reduces parameter count for FlextLogger.__init__ from 6 to 2 params.
        Groups optional logger context and configuration.
        """

        level: Annotated[
            str,
            Field(
                default=c.Logging.DEFAULT_LEVEL,
                description="Log level (default from constants, can be overridden)",
            ),
        ] = c.Logging.DEFAULT_LEVEL
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
            FlextModelFoundation.Metadata | None,
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
            int | None,
            Field(
                default=None,
                ge=c.ZERO,
                description="Optional timeout override in seconds",
            ),
        ] = None

    class ExecuteDispatchAttemptOptions(FlextModelsCollections.Config):
        """Options for _execute_dispatch_attempt (Pydantic v2).

        Reduces parameter count from 6 to 2 params (message, options).
        Groups execution context parameters.
        """

        message_type: Annotated[
            str,
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
            int | None,
            Field(
                default=None,
                ge=c.ZERO,
                description="Optional timeout override in seconds",
            ),
        ] = None
        operation_id: Annotated[
            str, Field(description="Operation ID for timeout tracking")
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
            FlextModelFoundation.Metadata | None,
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
                default_factory=t.Dict,
                description="Additional keyword arguments for metadata",
            ),
        ]

    class ResultConfig(FlextModelsCollections.Config):
        """Configuration for r failure case (Pydantic v2).

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
            FlextModelFoundation.Metadata | None,
            Field(default=None, description="Additional error data (Pydantic model)"),
        ] = None

    class ValidationErrorConfig(ExceptionConfig):
        """Configuration for ValidationError (Pydantic v2)."""

        field: Annotated[
            str | None,
            Field(default=None, description="Field name that failed validation"),
        ] = None
        value: Annotated[
            t.ValueOrModel | None,
            Field(default=None, description="Value that failed validation"),
        ] = None

    class ConfigurationErrorConfig(ExceptionConfig):
        """Configuration for ConfigurationError (Pydantic v2)."""

        config_key: Annotated[
            str | None,
            Field(default=None, description="Configuration key that caused error"),
        ] = None
        config_source: Annotated[
            str | None,
            Field(
                default=None, description="Source of configuration (file, env, etc.)"
            ),
        ] = None

    class ConnectionErrorConfig(ExceptionConfig):
        """Configuration for ConnectionError (Pydantic v2)."""

        host: Annotated[
            str | None,
            Field(default=None, description="Host that connection failed to"),
        ] = None
        port: Annotated[
            int | None,
            Field(default=None, description="Port that connection failed to"),
        ] = None
        timeout: Annotated[
            float | None,
            Field(default=None, description="Timeout value that was exceeded"),
        ] = None

    class TimeoutErrorConfig(ExceptionConfig):
        """Configuration for TimeoutError (Pydantic v2)."""

        timeout_seconds: Annotated[
            float | None,
            Field(default=None, description="Timeout in seconds that was exceeded"),
        ] = None
        operation: Annotated[
            str | None,
            Field(default=None, description="Operation that timed out"),
        ] = None

    class AuthenticationErrorConfig(ExceptionConfig):
        """Configuration for AuthenticationError (Pydantic v2)."""

        auth_method: Annotated[
            str | None,
            Field(default=None, description="Authentication method that failed"),
        ] = None
        user_id: Annotated[
            str | None,
            Field(default=None, description="User ID that authentication failed for"),
        ] = None

    class AuthorizationErrorConfig(ExceptionConfig):
        """Configuration for AuthorizationError (Pydantic v2)."""

        user_id: Annotated[
            str | None,
            Field(default=None, description="User ID that authorization failed for"),
        ] = None
        resource: Annotated[
            str | None,
            Field(default=None, description="Resource that access was denied to"),
        ] = None
        permission: Annotated[
            str | None,
            Field(default=None, description="Permission that was denied"),
        ] = None

    class NotFoundErrorConfig(ExceptionConfig):
        """Configuration for NotFoundError (Pydantic v2)."""

        resource_type: Annotated[
            str | None,
            Field(default=None, description="Type of resource that was not found"),
        ] = None
        resource_id: Annotated[
            str | None,
            Field(default=None, description="ID of resource that was not found"),
        ] = None

    class ConflictErrorConfig(ExceptionConfig):
        """Configuration for ConflictError (Pydantic v2)."""

        resource_type: Annotated[
            str | None,
            Field(default=None, description="Type of resource that conflicted"),
        ] = None
        resource_id: Annotated[
            str | None,
            Field(default=None, description="ID of resource that conflicted"),
        ] = None
        conflict_reason: Annotated[
            str | None,
            Field(default=None, description="Reason for the conflict"),
        ] = None

    class RateLimitErrorConfig(ExceptionConfig):
        """Configuration for RateLimitError (Pydantic v2)."""

        limit: Annotated[
            int | None,
            Field(default=None, description="Rate limit that was exceeded"),
        ] = None
        window_seconds: Annotated[
            int | None,
            Field(default=None, description="Time window for rate limit"),
        ] = None
        retry_after: Annotated[
            float | None,
            Field(default=None, description="Seconds to wait before retrying"),
        ] = None

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

    class TypeErrorConfig(ExceptionConfig):
        """Configuration for TypeError (Pydantic v2)."""

        expected_type: Annotated[
            str | None,
            Field(default=None, description="Expected type name"),
        ] = None
        actual_type: Annotated[
            str | None,
            Field(default=None, description="Actual type name"),
        ] = None

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
            FlextModelFoundation.Metadata | Mapping[str, t.ValueOrModel] | None,
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

    class CircuitBreakerErrorConfig(ExceptionConfig):
        """Configuration for CircuitBreakerError (Pydantic v2)."""

        service_name: Annotated[
            str | None,
            Field(
                default=None, description="Service name where circuit breaker opened"
            ),
        ] = None
        failure_count: Annotated[
            int | None,
            Field(
                default=None,
                description="Number of failures that triggered circuit breaker",
            ),
        ] = None
        reset_timeout: Annotated[
            float | None,
            Field(default=None, description="Timeout before circuit breaker resets"),
        ] = None

    class OperationErrorConfig(ExceptionConfig):
        """Configuration for OperationError (Pydantic v2)."""

        operation: Annotated[
            str | None,
            Field(default=None, description="Operation that failed"),
        ] = None
        reason: Annotated[
            str | None,
            Field(default=None, description="Reason for operation failure"),
        ] = None

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
                description="Type of object that attribute access failed on",
            ),
        ] = None

    class OperationExtraConfig(FlextModelsCollections.Config):
        """Configuration for operation logging extra data (Pydantic v2).

        Reduces parameter count for _build_operation_extra from 8 to 2 params.
        Groups operation context and performance tracking.
        """

        func_name: Annotated[str, Field(description="Function name for logging")]
        func_module: Annotated[str, Field(description="Function module for logging")]
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

        op_name: Annotated[str, Field(description="Operation name")]
        func_name: Annotated[str, Field(description="Function name")]
        func_module: Annotated[str, Field(description="Function module")]
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

    class RetryLoopConfig(FlextModelFoundation.ArbitraryTypesModel):
        """Configuration for retry loop execution (Pydantic v2).

        Reduces parameter count for _execute_retry_loop from 8 to 3 params.
        Groups function, arguments, logger, and retry configuration.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)
        func: Annotated[
            Callable[..., t.ValueOrModel],
            Field(description="Function to execute"),
        ]
        args: Annotated[
            tuple[t.ValueOrModel, ...],
            Field(
                default_factory=tuple,
                description="Positional arguments for function",
            ),
        ]
        call_kwargs: Annotated[
            t.ConfigMap,
            Field(
                default_factory=t.ConfigMap,
                description="Keyword arguments for function",
            ),
        ]
        retry_config: Annotated[
            FlextModelsConfig.RetryConfiguration | None,
            Field(
                default=None,
                description="Retry configuration (takes priority over individual params)",
            ),
        ] = None
        attempts: Annotated[
            int,
            Field(
                default=c.Reliability.MAX_RETRY_ATTEMPTS,
                ge=c.Reliability.RETRY_COUNT_MIN,
                description="Number of retry attempts (used if retry_config is None)",
            ),
        ] = c.Reliability.MAX_RETRY_ATTEMPTS
        delay: Annotated[
            float,
            Field(
                default=float(c.Reliability.DEFAULT_RETRY_DELAY_SECONDS),
                gt=c.INITIAL_TIME,
                description="Initial delay between retries (used if retry_config is None)",
            ),
        ] = float(c.Reliability.DEFAULT_RETRY_DELAY_SECONDS)
        strategy: Annotated[
            str,
            Field(
                default=c.Reliability.DEFAULT_BACKOFF_STRATEGY,
                description="Retry strategy: 'exponential' or 'linear' (used if retry_config is None)",
            ),
        ] = c.Reliability.DEFAULT_BACKOFF_STRATEGY

    class DispatcherConfig(FlextModelFoundation.ArbitraryTypesModel):
        """Configuration for message dispatcher.

        Replaces legacy dispatcher config mapping from typings.py.
        Provides type-safe configuration for message dispatcher behavior.
        """

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, extra="forbid"
        )
        dispatcher_timeout_seconds: Annotated[
            float,
            Field(
                default=30.0,
                gt=0,
                description="Timeout in seconds for dispatcher operations",
            ),
        ] = 30.0
        executor_workers: Annotated[
            int,
            Field(
                default=4,
                ge=1,
                le=256,
                description="Number of executor worker threads",
            ),
        ] = 4
        circuit_breaker_threshold: Annotated[
            int,
            Field(
                default=5,
                ge=1,
                description="Circuit breaker failure threshold",
            ),
        ] = 5
        rate_limit_max_requests: Annotated[
            int,
            Field(
                default=1000,
                ge=1,
                description="Maximum requests for rate limiting",
            ),
        ] = 1000
        rate_limit_window_seconds: Annotated[
            float,
            Field(
                default=60.0,
                gt=0,
                description="Rate limit window in seconds",
            ),
        ] = 60.0
        max_retry_attempts: Annotated[
            int,
            Field(
                default=3,
                ge=0,
                le=10,
                description="Maximum retry attempts",
            ),
        ] = 3
        retry_delay: Annotated[
            float,
            Field(
                default=1.0,
                ge=0,
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
    "Domain model configuration defaults.\n\n    Moved from FlextConstants.Domain.DOMAIN_MODEL_CONFIG because\n    constants.py cannot import ConfigDict from pydantic.\n\n    Use m.DOMAIN_MODEL_CONFIG instead of c.Domain.DOMAIN_MODEL_CONFIG.\n    "


__all__ = ["FlextModelsConfig"]
