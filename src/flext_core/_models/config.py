"""Configuration patterns extracted from FlextModels.

This module contains the FlextModelsConfig class with all configuration-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Config instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flext_core._models.base import FlextModelsBase
from flext_core._models.collections import FlextModelsCollections
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.validation import FlextUtilitiesValidation
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.typings import t

# NOTE: models.py cannot import utilities - use direct imports from _utilities/* instead


def _get_log_level_from_config() -> int:
    """Get log level from default constant (avoids circular import with config.py)."""
    # Use default log level from constants to avoid circular import
    # config.py -> runtime.py -> models.py -> _models/config.py -> config.py
    default_log_level = c.Logging.DEFAULT_LEVEL.upper()
    return getattr(logging, default_log_level, logging.INFO)


class FlextModelsConfig:
    """Configuration pattern container class.

    This class acts as a namespace container for configuration patterns.
    All nested classes are accessed via FlextModels.Config.* in the main models.py.
    """

    class ProcessingRequest(FlextModelsBase.ArbitraryTypesModel):
        """Enhanced processing request with advanced validation."""

        model_config = ConfigDict(
            validate_assignment=False,  # Allow invalid values to be set for testing
            use_enum_values=True,
            arbitrary_types_allowed=True,
        )

        # Note: default_factory requires a callable that returns a value
        # Using lambda is necessary here as Pydantic calls the factory function
        operation_id: str = Field(
            default_factory=FlextUtilitiesGenerators.generate,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Unique operation identifier",
        )
        data: t.Types.ConfigurationDict = Field(default_factory=dict)
        context: t.Types.ConfigurationDict = Field(default_factory=dict)
        timeout_seconds: float = Field(
            default=c.Defaults.TIMEOUT,
            gt=c.ZERO,
            le=c.Performance.MAX_TIMEOUT_SECONDS,
            description=("Operation timeout from c (Constants default)"),
        )
        retry_attempts: int = Field(
            default=c.Reliability.MAX_RETRY_ATTEMPTS,
            ge=c.ZERO,
            le=c.Reliability.MAX_RETRY_ATTEMPTS,
            description=("Maximum retry attempts from c (Constants default)"),
        )
        enable_validation: bool = True

        @field_validator("context", mode="before")
        @classmethod
        def validate_context(cls, v: t.GeneralValueType) -> t.Types.StringDict:
            """Ensure context has required fields (using FlextUtilitiesGenerators).

            Returns t.Types.StringDict because ensure_trace_context generates
            string trace IDs. This is compatible with the field type
            ConfigurationDict since str is a subtype.
            """
            return FlextUtilitiesGenerators.ensure_trace_context(
                v,
                include_correlation_id=True,
                include_timestamp=True,
            )

        def validate_processing_constraints(self) -> r[bool]:
            """Validate constraints that should be checked during processing."""
            max_timeout_seconds = c.Utilities.MAX_TIMEOUT_SECONDS
            if self.timeout_seconds > max_timeout_seconds:
                return r[bool].fail(
                    f"Timeout cannot exceed {max_timeout_seconds} seconds",
                )

            return r[bool].ok(True)

    class RetryConfiguration(FlextModelsBase.ArbitraryTypesModel):
        """Retry configuration with advanced validation."""

        max_attempts: int = Field(
            default=c.Reliability.MAX_RETRY_ATTEMPTS,
            ge=c.Reliability.RETRY_COUNT_MIN,
            le=c.Reliability.MAX_RETRY_ATTEMPTS,
            description=("Maximum retry attempts from c (Constants default)"),
        )
        initial_delay_seconds: float = Field(
            default=c.Performance.DEFAULT_INITIAL_DELAY_SECONDS,
            gt=c.ZERO,
            description="Initial delay between retries",
        )
        max_delay_seconds: float = Field(
            default=c.DEFAULT_MAX_DELAY_SECONDS,
            gt=c.ZERO,
            description="Maximum delay between retries",
        )
        exponential_backoff: bool = True
        backoff_multiplier: float = Field(
            default=c.DEFAULT_BACKOFF_MULTIPLIER,
            ge=float(c.Reliability.RETRY_COUNT_MIN),
            description="Backoff multiplier for exponential backoff",
        )
        retry_on_exceptions: list[type[BaseException]] = Field(
            default_factory=list,
            description="Exception types to retry on",
        )
        retry_on_status_codes: list[int] = Field(
            default_factory=list,
            max_length=c.Validation.MAX_RETRY_STATUS_CODES,
            description="HTTP status codes to retry on",
        )

        @field_validator("retry_on_status_codes", mode="after")
        @classmethod
        def validate_backoff_strategy(cls, v: list[int] | list[object]) -> list[int]:
            """Validate status codes are valid HTTP codes."""
            # Use default HTTP status code range (100-599) - domain-specific validation
            # removed from flext-core per domain violation rules
            # Convert to list[object] for validation function (accepts object)
            codes_for_validation: list[object] = list(v)
            result = FlextUtilitiesValidation.validate_http_status_codes(
                codes_for_validation,
            )
            if result.is_failure:
                base_msg = "HTTP status code validation failed"
                error_msg = (
                    f"{base_msg}: {result.error}"
                    if result.error
                    else f"{base_msg} (invalid status code)"
                )
                raise ValueError(error_msg)
            # Return validated list[int] - FlextResult never returns None on success
            # Use .value directly - FlextResult never returns None on success
            validated_codes: list[int] = result.value
            return validated_codes

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency."""
            if self.max_delay_seconds < self.initial_delay_seconds:
                msg = "max_delay_seconds must be >= initial_delay_seconds"
                raise ValueError(msg)
            return self

    class ValidationConfiguration(FlextModelsBase.ArbitraryTypesModel):
        """Validation configuration."""

        enable_strict_mode: bool = Field(default=True)
        max_validation_errors: int = Field(
            default=c.Cqrs.DEFAULT_MAX_VALIDATION_ERRORS,
            ge=c.Reliability.RETRY_COUNT_MIN,
            le=c.Validation.MAX_RETRY_STATUS_CODES,
            description="Maximum validation errors",
        )
        validate_on_assignment: bool = True
        validate_on_read: bool = False
        custom_validators: Annotated[
            list[object],
            Field(
                default_factory=list,
                max_length=c.Validation.MAX_CUSTOM_VALIDATORS,
                description="Custom validator callables",
            ),
        ]

        @field_validator("custom_validators", mode="after")
        @classmethod
        def validate_additional_validators(cls, v: list[object]) -> list[object]:
            """Validate custom validators are callable."""
            for validator in v:
                # Direct callable check - object can be any callable,
                # not just GeneralValueType
                if not callable(validator):
                    base_msg = "Validator must be callable"
                    error_msg = f"{base_msg}: got {type(validator).__name__}"
                    raise TypeError(error_msg)
            return v

    class BatchProcessingConfig(FlextModelsCollections.Config):
        """Enhanced batch processing configuration."""

        batch_size: int = Field(
            default=c.Performance.MAX_BATCH_SIZE,
            description=("Batch size from c (Constants default)"),
        )
        max_workers: int = Field(
            default=c.Processing.DEFAULT_MAX_WORKERS,
            le=c.Settings.MAX_WORKERS_THRESHOLD,
            description="Maximum workers (Config has priority over Constants)",
        )
        timeout_per_item: float = Field(
            default=c.Defaults.TIMEOUT,
            description="Timeout per item (Config has priority over Constants)",
        )
        continue_on_error: bool = True
        data_items: Annotated[
            list[object],
            Field(
                default_factory=list,
                max_length=c.Performance.BatchProcessing.MAX_ITEMS,
            ),
        ]

        @model_validator(mode="after")
        def validate_batch(self) -> Self:
            """Validate batch configuration consistency."""
            max_batch_size = c.Performance.BatchProcessing.MAX_VALIDATION_SIZE
            if self.batch_size > max_batch_size:
                msg = f"Batch size cannot exceed {max_batch_size}"
                raise ValueError(msg)

            # Adjust max_workers to not exceed batch_size without triggering validation
            adjusted_workers = min(self.max_workers, self.batch_size)
            # Use direct assignment to __dict__ to bypass Pydantic validation
            self.__dict__["max_workers"] = adjusted_workers

            return self

    class HandlerExecutionConfig(FlextModelsCollections.Config):
        """Enhanced handler execution configuration."""

        handler_name: str = Field(pattern=c.Platform.PATTERN_IDENTIFIER)
        input_data: t.Types.ConfigurationDict = Field(default_factory=dict)
        execution_context: t.Types.ConfigurationDict = Field(
            default_factory=dict,
        )
        timeout_seconds: float = Field(
            default=c.Defaults.TIMEOUT,
            le=c.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout in seconds (default from constants)",
        )
        retry_on_failure: bool = True
        max_retries: int = Field(
            default=c.Reliability.MAX_RETRY_ATTEMPTS,
            description="Max retries (default from constants)",
        )

    class MiddlewareConfig(BaseModel):
        """Configuration for middleware execution.

        Provides configuration options for middleware ordering and priority
        within request/response processing pipeline.
        """

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_schema_extra={
                "title": "MiddlewareConfig",
                "description": (
                    "Configuration for middleware execution in request processing"
                ),
            },
        )

        enabled: bool = Field(default=True, description="Whether middleware is enabled")
        order: int = Field(
            default=c.Defaults.DEFAULT_MIDDLEWARE_ORDER,
            description="Execution order in middleware chain",
        )
        name: str | None = Field(default=None, description="Optional middleware name")
        config: t.Types.ConfigurationDict = Field(
            default_factory=dict,
            description="Middleware-specific configuration",
        )

    class RateLimiterState(BaseModel):
        """State tracking for rate limiter functionality.

        Tracks request counts, windows, and blocking state for rate limiting
        operations within the FLEXT request processing pipeline.
        """

        model_config = ConfigDict(
            json_schema_extra={
                "title": "RateLimiterState",
                "description": "State tracking for rate limiter functionality",
            },
        )

        processor_name: str = Field(
            default="",
            description="Name of the rate limiter processor",
        )
        count: int = Field(
            default=c.ZERO,
            ge=c.ZERO,
            description="Current request count in window",
        )
        window_start: float = Field(
            default=c.INITIAL_TIME,
            ge=c.INITIAL_TIME,
            description="Timestamp when current window started",
        )
        limit: int = Field(
            default=c.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
            ge=c.Reliability.RETRY_COUNT_MIN,
            description="Maximum requests allowed per window",
        )
        window_seconds: int = Field(
            default=c.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
            ge=c.Reliability.RETRY_COUNT_MIN,
            description="Duration of rate limit window in seconds",
        )
        block_until: float = Field(
            default=c.INITIAL_TIME,
            ge=c.INITIAL_TIME,
            description="Timestamp until which requests are blocked",
        )

    class ExternalCommandConfig(FlextModelsCollections.Config):
        """Configuration for external command execution (Pydantic v2).

        Reduces parameter count for u.CommandExecution
        run_external_command using config object pattern.
        Reuses timeout pattern from ProcessingRequest and HandlerExecutionConfig.
        """

        capture_output: bool = Field(
            default=True,
            description="Whether to capture stdout/stderr",
        )
        check: bool = Field(
            default=True,
            description="Whether to raise exception on non-zero exit code",
        )
        env: t.Types.StringDict | None = Field(
            default=None,
            description="Environment variables for the command",
        )
        cwd: str | None = Field(
            default=None,
            description="Working directory for command execution",
        )
        timeout_seconds: float | None = Field(
            default=None,
            gt=c.ZERO,
            le=c.Performance.MAX_TIMEOUT_SECONDS,
            description="Command timeout in seconds (max 5 min)",
        )
        command_input: str | bytes | None = Field(
            default=None,
            description="Input to send to command stdin",
        )
        text: bool | None = Field(
            default=None,
            description="Whether to decode stdout/stderr as text",
        )

    class StructlogConfig(FlextModelsCollections.Config):
        """Configuration for structlog setup (Pydantic v2).

        Reduces parameter count for FlextRuntime.configure_structlog.
        Allows validation and composition of logging configuration.
        """

        log_level: int = Field(
            default_factory=_get_log_level_from_config,
            ge=c.ZERO,
            le=c.Validation.MAX_CUSTOM_VALIDATORS,
            description=(
                "Numeric log level (DEBUG=10, INFO=20, WARNING=30, "
                "ERROR=40, CRITICAL=50) - default from constants"
            ),
        )
        console_renderer: bool = Field(
            default=True,
            description="Use console renderer (True) or JSON renderer (False)",
        )
        additional_processors: list[Callable[..., object]] = Field(
            default_factory=list,
            description="Optional extra processors after standard FLEXT processors",
        )
        wrapper_class_factory: Callable[[], type] | None = Field(
            default=None,
            description="Custom wrapper factory for structlog",
        )
        logger_factory: p.VariadicCallable[t.GeneralValueType] | None = Field(
            default=None,
            description="Custom logger factory for structlog",
        )
        cache_logger_on_first_use: bool = Field(
            default=True,
            description="Cache logger on first use (performance optimization)",
        )

    class LoggerConfig(FlextModelsCollections.Config):
        """Configuration for FlextLogger initialization (Pydantic v2).

        Reduces parameter count for FlextLogger.__init__ from 6 to 2 params.
        Groups optional logger context and configuration.
        """

        level: str = Field(
            default=c.Logging.DEFAULT_LEVEL,
            description="Log level (default from constants, can be overridden)",
        )
        service_name: str | None = Field(
            default=None,
            description="Service name for distributed tracing context",
        )
        service_version: str | None = Field(
            default=None,
            description="Service version for distributed tracing context",
        )
        correlation_id: str | None = Field(
            default=None,
            description="Correlation ID for distributed tracing",
        )
        force_new: bool = Field(
            default=False,
            description="Force creation of new logger instance (for testing)",
        )

    class DispatchConfig(FlextModelsCollections.Config):
        """Configuration for FlextDispatcher.dispatch (Pydantic v2).

        Reduces parameter count for dispatch from 5 to 3 params (message, data, config).
        Groups optional dispatch context and overrides.
        """

        metadata: FlextModelsBase.Metadata | None = Field(
            default=None,
            description="Optional execution context metadata (Pydantic model)",
        )
        correlation_id: str | None = Field(
            default=None,
            description="Optional correlation ID for distributed tracing",
        )
        timeout_override: int | None = Field(
            default=None,
            ge=c.ZERO,
            description="Optional timeout override in seconds",
        )

    class ExecuteDispatchAttemptOptions(FlextModelsCollections.Config):
        """Options for _execute_dispatch_attempt (Pydantic v2).

        Reduces parameter count from 6 to 2 params (message, options).
        Groups execution context parameters.
        """

        message_type: str = Field(
            description="Message type name for routing and circuit breaker",
        )
        metadata: t.GeneralValueType | None = Field(
            default=None,
            description="Optional execution context metadata",
        )
        correlation_id: str | None = Field(
            default=None,
            description="Optional correlation ID for distributed tracing",
        )
        timeout_override: int | None = Field(
            default=None,
            ge=c.ZERO,
            description="Optional timeout override in seconds",
        )
        operation_id: str = Field(
            description="Operation ID for timeout tracking",
        )

    class RuntimeScopeOptions(FlextModelsCollections.Config):
        """Options for runtime_scope (Pydantic v2).

        Reduces parameter count from 7 to 2 params (self, options).
        Groups runtime scope configuration parameters.
        """

        config_overrides: Mapping[str, t.FlexibleValue] | None = Field(
            default=None,
            description="Optional configuration overrides",
        )
        context: p.Ctx | None = Field(
            default=None,
            description="Optional context protocol instance",
        )
        subproject: str | None = Field(
            default=None,
            description="Optional subproject name",
        )
        services: Mapping[str, t.FlexibleValue] | None = Field(
            default=None,
            description="Optional container services mapping",
        )
        factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = Field(
            default=None,
            description="Optional container factories mapping",
        )
        container_services: Mapping[str, t.FlexibleValue] | None = Field(
            default=None,
            description="Optional container services (alias for services)",
        )
        container_factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = Field(
            default=None,
            description="Optional container factories (alias for factories)",
        )

    class NestedExecutionOptions(FlextModelsCollections.Config):
        """Options for nested_execution (Pydantic v2).

        Reduces parameter count from 6 to 2 params (self, options).
        Groups nested execution configuration parameters.
        """

        config_overrides: Mapping[str, t.FlexibleValue] | None = Field(
            default=None,
            description="Optional configuration overrides",
        )
        service_name: str | None = Field(
            default=None,
            description="Optional service name",
        )
        version: str | None = Field(
            default=None,
            description="Optional version string",
        )
        correlation_id: str | None = Field(
            default=None,
            description="Optional correlation ID for tracing",
        )
        container_services: Mapping[str, t.FlexibleValue] | None = Field(
            default=None,
            description="Optional container services mapping",
        )
        container_factories: Mapping[str, Callable[[], t.FlexibleValue]] | None = Field(
            default=None,
            description="Optional container factories mapping",
        )

    class ExceptionConfig(FlextModelsCollections.Config):
        """Configuration for e.__init__ (Pydantic v2).

        Reduces parameter count for exception initialization from 7 to 2 params
        (message, config). Groups optional exception context and behavior.
        """

        error_code: str | None = Field(
            default=None,
            description="Error code for categorization",
        )
        correlation_id: str | None = Field(
            default=None,
            description="Correlation ID for distributed tracing",
        )
        metadata: FlextModelsBase.Metadata | None = Field(
            default=None,
            description="Additional metadata (Pydantic model)",
        )
        auto_log: bool = Field(
            default=False,
            description="Whether to automatically log exception",
        )
        auto_correlation: bool = Field(
            default=False,
            description="Whether to auto-generate correlation ID",
        )
        extra_kwargs: t.Types.ConfigurationDict = Field(
            default_factory=dict,
            description="Additional keyword arguments for metadata",
        )

    class ResultConfig(FlextModelsCollections.Config):
        """Configuration for r failure case (Pydantic v2).

        Groups optional error context for result failures.
        """

        error: str | None = Field(
            default=None,
            description="Error message for failure case",
        )
        error_code: str | None = Field(
            default=None,
            description="Error code for categorization",
        )
        error_data: FlextModelsBase.Metadata | None = Field(
            default=None,
            description="Additional error data (Pydantic model)",
        )

    # Exception-specific configs that extend ExceptionConfig
    class ValidationErrorConfig(ExceptionConfig):
        """Configuration for ValidationError (Pydantic v2)."""

        field: str | None = Field(
            default=None,
            description="Field name that failed validation",
        )
        value: t.GeneralValueType | None = Field(
            default=None,
            description="Value that failed validation",
        )

    class ConfigurationErrorConfig(ExceptionConfig):
        """Configuration for ConfigurationError (Pydantic v2)."""

        config_key: str | None = Field(
            default=None,
            description="Configuration key that caused error",
        )
        config_source: str | None = Field(
            default=None,
            description="Source of configuration (file, env, etc.)",
        )

    class ConnectionErrorConfig(ExceptionConfig):
        """Configuration for ConnectionError (Pydantic v2)."""

        host: str | None = Field(
            default=None,
            description="Host that connection failed to",
        )
        port: int | None = Field(
            default=None,
            description="Port that connection failed to",
        )
        timeout: float | None = Field(
            default=None,
            description="Timeout value that was exceeded",
        )

    class TimeoutErrorConfig(ExceptionConfig):
        """Configuration for TimeoutError (Pydantic v2)."""

        timeout_seconds: float | None = Field(
            default=None,
            description="Timeout in seconds that was exceeded",
        )
        operation: str | None = Field(
            default=None,
            description="Operation that timed out",
        )

    class AuthenticationErrorConfig(ExceptionConfig):
        """Configuration for AuthenticationError (Pydantic v2)."""

        auth_method: str | None = Field(
            default=None,
            description="Authentication method that failed",
        )
        user_id: str | None = Field(
            default=None,
            description="User ID that authentication failed for",
        )

    class AuthorizationErrorConfig(ExceptionConfig):
        """Configuration for AuthorizationError (Pydantic v2)."""

        user_id: str | None = Field(
            default=None,
            description="User ID that authorization failed for",
        )
        resource: str | None = Field(
            default=None,
            description="Resource that access was denied to",
        )
        permission: str | None = Field(
            default=None,
            description="Permission that was denied",
        )

    class NotFoundErrorConfig(ExceptionConfig):
        """Configuration for NotFoundError (Pydantic v2)."""

        resource_type: str | None = Field(
            default=None,
            description="Type of resource that was not found",
        )
        resource_id: str | None = Field(
            default=None,
            description="ID of resource that was not found",
        )

    class ConflictErrorConfig(ExceptionConfig):
        """Configuration for ConflictError (Pydantic v2)."""

        resource_type: str | None = Field(
            default=None,
            description="Type of resource that conflicted",
        )
        resource_id: str | None = Field(
            default=None,
            description="ID of resource that conflicted",
        )
        conflict_reason: str | None = Field(
            default=None,
            description="Reason for the conflict",
        )

    class RateLimitErrorConfig(ExceptionConfig):
        """Configuration for RateLimitError (Pydantic v2)."""

        limit: int | None = Field(
            default=None,
            description="Rate limit that was exceeded",
        )
        window_seconds: int | None = Field(
            default=None,
            description="Time window for rate limit",
        )
        retry_after: float | None = Field(
            default=None,
            description="Seconds to wait before retrying",
        )

    class InternalErrorConfig(ExceptionConfig):
        """Configuration for InternalError (Pydantic v2)."""

        component: str | None = Field(
            default=None,
            description="Component where internal error occurred",
        )
        operation: str | None = Field(
            default=None,
            description="Operation that caused internal error",
        )

    class TypeErrorConfig(ExceptionConfig):
        """Configuration for TypeError (Pydantic v2)."""

        expected_type: str | None = Field(
            default=None,
            description="Expected type name",
        )
        actual_type: str | None = Field(
            default=None,
            description="Actual type name",
        )

    class TypeErrorOptions(FlextModelsCollections.Config):
        """Options for TypeError initialization (Pydantic v2).

        Groups TypeError constructor parameters for cleaner initialization.
        """

        expected_type: type | None = Field(
            default=None,
            description="Expected type class",
        )
        actual_type: type | None = Field(
            default=None,
            description="Actual type class",
        )
        context: Mapping[str, t.MetadataAttributeValue] | None = Field(
            default=None,
            description="Additional context for error",
        )
        metadata: (
            FlextModelsBase.Metadata | Mapping[str, t.MetadataAttributeValue] | None
        ) = Field(
            default=None,
            description="Metadata for error",
        )

    class ValueErrorConfig(ExceptionConfig):
        """Configuration for ValueError (Pydantic v2)."""

        expected_value: str | None = Field(
            default=None,
            description="Expected value description",
        )
        actual_value: t.GeneralValueType | None = Field(
            default=None,
            description="Actual value that caused error",
        )

    class CircuitBreakerErrorConfig(ExceptionConfig):
        """Configuration for CircuitBreakerError (Pydantic v2)."""

        service_name: str | None = Field(
            default=None,
            description="Service name where circuit breaker opened",
        )
        failure_count: int | None = Field(
            default=None,
            description="Number of failures that triggered circuit breaker",
        )
        reset_timeout: float | None = Field(
            default=None,
            description="Timeout before circuit breaker resets",
        )

    class OperationErrorConfig(ExceptionConfig):
        """Configuration for OperationError (Pydantic v2)."""

        operation: str | None = Field(
            default=None,
            description="Operation that failed",
        )
        reason: str | None = Field(
            default=None,
            description="Reason for operation failure",
        )

    class AttributeAccessErrorConfig(ExceptionConfig):
        """Configuration for AttributeAccessError (Pydantic v2)."""

        attribute_name: str | None = Field(
            default=None,
            description="Attribute name that access failed for",
        )
        object_type: str | None = Field(
            default=None,
            description="Type of object that attribute access failed on",
        )

    class OperationExtraConfig(FlextModelsCollections.Config):
        """Configuration for operation logging extra data (Pydantic v2).

        Reduces parameter count for _build_operation_extra from 8 to 2 params.
        Groups operation context and performance tracking.
        """

        func_name: str = Field(description="Function name for logging")
        func_module: str = Field(description="Function module for logging")
        correlation_id: str | None = Field(
            default=None,
            description="Correlation ID for distributed tracing",
        )
        success: bool | None = Field(
            default=None,
            description="Operation success status",
        )
        error: str | None = Field(
            default=None,
            description="Error message if operation failed",
        )
        error_type: str | None = Field(
            default=None,
            description="Error type name",
        )
        start_time: float = Field(
            default=c.INITIAL_TIME,
            ge=c.INITIAL_TIME,
            description="Operation start time for performance tracking",
        )
        track_perf: bool = Field(
            default=False,
            description="Whether to track performance metrics",
        )

    class LogOperationFailureConfig(FlextModelsCollections.Config):
        """Configuration for logging operation failures (Pydantic v2).

        Reduces parameter count for _log_operation_failure from 8 to 3 params.
        Groups logger, operation context, and exception details.
        """

        op_name: str = Field(description="Operation name")
        func_name: str = Field(description="Function name")
        func_module: str = Field(description="Function module")
        correlation_id: str | None = Field(
            default=None,
            description="Correlation ID for distributed tracing",
        )
        exc: Exception = Field(description="Exception that caused failure")
        start_time: float = Field(
            default=c.INITIAL_TIME,
            ge=c.INITIAL_TIME,
            description="Operation start time",
        )
        track_perf: bool = Field(
            default=False,
            description="Whether to track performance metrics",
        )

    class RetryLoopConfig(FlextModelsBase.ArbitraryTypesModel):
        """Configuration for retry loop execution (Pydantic v2).

        Reduces parameter count for _execute_retry_loop from 8 to 3 params.
        Groups function, arguments, logger, and retry configuration.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        func: p.VariadicCallable[t.GeneralValueType] = Field(
            description="Function to execute",
        )
        args: tuple[t.GeneralValueType, ...] = Field(
            default_factory=tuple,
            description="Positional arguments for function",
        )
        kwargs: t.Types.ConfigurationMapping = Field(
            default_factory=dict,
            description="Keyword arguments for function",
        )
        retry_config: FlextModelsConfig.RetryConfiguration | None = Field(
            default=None,
            description="Retry configuration (takes priority over individual params)",
        )
        attempts: int = Field(
            default=c.Reliability.MAX_RETRY_ATTEMPTS,
            ge=c.Reliability.RETRY_COUNT_MIN,
            description="Number of retry attempts (used if retry_config is None)",
        )
        delay: float = Field(
            default=float(c.Reliability.DEFAULT_RETRY_DELAY_SECONDS),
            gt=c.INITIAL_TIME,
            description="Initial delay between retries (used if retry_config is None)",
        )
        strategy: str = Field(
            default=c.Reliability.DEFAULT_BACKOFF_STRATEGY,
            description=(
                "Retry strategy: 'exponential' or 'linear' "
                "(used if retry_config is None)"
            ),
        )

    # Domain model configuration - moved from constants.py
    # constants.py cannot import ConfigDict, so this belongs here
    DOMAIN_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        validate_return=True,
        validate_default=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=False,
        extra="forbid",
    )
    """Domain model configuration defaults.

    Moved from FlextConstants.Domain.DOMAIN_MODEL_CONFIG because
    constants.py cannot import ConfigDict from pydantic.

    Use m.Config.DOMAIN_MODEL_CONFIG instead of c.Domain.DOMAIN_MODEL_CONFIG.
    """


__all__ = ["FlextModelsConfig"]
