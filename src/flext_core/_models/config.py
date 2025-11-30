"""Configuration patterns extracted from FlextModels.

This module contains the FlextModelsConfig class with all configuration-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Config instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flext_core._models.base import FlextModelsBase
from flext_core._models.collections import FlextModelsCollections
from flext_core._models.metadata import Metadata
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.result import FlextResult
from flext_core.utilities import FlextUtilities


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

        operation_id: str = Field(
            default_factory=FlextUtilities.Generators.generate_id,
            min_length=1,
            description="Unique operation identifier",
        )
        data: dict[str, FlextTypes.GeneralValueType] = Field(default_factory=dict)
        context: dict[str, FlextTypes.GeneralValueType] = Field(default_factory=dict)
        timeout_seconds: float = Field(
            default=FlextConstants.Defaults.TIMEOUT,
            gt=0,
            le=FlextConstants.Performance.MAX_TIMEOUT_SECONDS,
            description=("Operation timeout from FlextConstants (Constants default)"),
        )
        retry_attempts: int = Field(
            default=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
            ge=0,
            le=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
            description=(
                "Maximum retry attempts from FlextConstants (Constants default)"
            ),
        )
        enable_validation: bool = True

        @field_validator("context", mode="before")
        @classmethod
        def validate_context(cls, v: object) -> dict[str, str]:
            """Ensure context has required fields (using FlextUtilities.Generators).

            Returns dict[str, str] because ensure_trace_context generates string trace IDs.
            This is compatible with the field type dict[str, FlextTypes.GeneralValueType] since str is a subtype.
            """
            return FlextUtilities.Generators.ensure_trace_context(
                v,
                include_correlation_id=True,
                include_timestamp=True,
            )

        def validate_processing_constraints(self) -> FlextResult[bool]:
            """Validate constraints that should be checked during processing."""
            max_timeout_seconds = FlextConstants.Utilities.MAX_TIMEOUT_SECONDS
            if self.timeout_seconds > max_timeout_seconds:
                return FlextResult[bool].fail(
                    f"Timeout cannot exceed {max_timeout_seconds} seconds",
                )

            return FlextResult[bool].ok(True)

    class RetryConfiguration(FlextModelsBase.ArbitraryTypesModel):
        """Retry configuration with advanced validation."""

        max_attempts: int = Field(
            default=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
            ge=FlextConstants.Reliability.RETRY_COUNT_MIN,
            le=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
            description=(
                "Maximum retry attempts from FlextConstants (Constants default)"
            ),
        )
        initial_delay_seconds: float = Field(
            default=FlextConstants.Performance.DEFAULT_INITIAL_DELAY_SECONDS,
            gt=0,
            description="Initial delay between retries",
        )
        max_delay_seconds: float = Field(
            default=FlextConstants.Performance.DEFAULT_MAX_DELAY_SECONDS,
            gt=0,
            description="Maximum delay between retries",
        )
        exponential_backoff: bool = True
        backoff_multiplier: float = Field(
            default=FlextConstants.Performance.DEFAULT_BACKOFF_MULTIPLIER,
            ge=1.0,
            description="Backoff multiplier for exponential backoff",
        )
        retry_on_exceptions: list[type[BaseException]] = Field(
            default_factory=list,
            description="Exception types to retry on",
        )
        retry_on_status_codes: list[object] = Field(
            default_factory=list,
            max_length=100,
            description="HTTP status codes to retry on",
        )

        @field_validator("retry_on_status_codes", mode="after")
        @classmethod
        def validate_backoff_strategy(cls, v: list[object]) -> list[object]:
            """Validate status codes are valid HTTP codes."""
            result = FlextUtilities.Validation.validate_http_status_codes(
                v,
                min_code=FlextConstants.FlextWeb.HTTP_STATUS_MIN,
                max_code=FlextConstants.FlextWeb.HTTP_STATUS_MAX,
            )
            if result.is_failure:
                base_msg = "HTTP status code validation failed"
                error_msg = (
                    f"{base_msg}: {result.error}"
                    if result.error
                    else f"{base_msg} (invalid status code)"
                )
                raise ValueError(error_msg)
            # Return as list[object] to match Pydantic field type
            return [int(code) for code in result.unwrap()]

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency."""
            if self.max_delay_seconds < self.initial_delay_seconds:
                msg = "max_delay_seconds must be >= initial_delay_seconds"
                raise ValueError(msg)
            return self

    class ValidationConfiguration(FlextModelsBase.ArbitraryTypesModel):
        """Validation configuration."""

        enable_strict_mode: bool = Field(default_factory=lambda: True)
        max_validation_errors: int = Field(
            default_factory=lambda: FlextConstants.Cqrs.DEFAULT_MAX_VALIDATION_ERRORS,
            ge=1,
            le=100,
            description="Maximum validation errors",
        )
        validate_on_assignment: bool = True
        validate_on_read: bool = False
        custom_validators: Annotated[
            list[object],
            Field(
                default_factory=list,
                max_length=50,
                description="Custom validator callables",
            ),
        ]

        @field_validator("custom_validators", mode="after")
        @classmethod
        def validate_additional_validators(cls, v: list[object]) -> list[object]:
            """Validate custom validators are callable."""
            for validator in v:
                # Direct callable check - object can be any callable, not just GeneralValueType
                if not callable(validator):
                    base_msg = "Validator must be callable"
                    error_msg = f"{base_msg}: got {type(validator).__name__}"
                    raise FlextExceptions.TypeError(
                        error_msg,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
            return v

    class BatchProcessingConfig(FlextModelsCollections.Config):
        """Enhanced batch processing configuration."""

        batch_size: int = Field(
            default=FlextConstants.Processing.MAX_BATCH_SIZE,
            description=("Batch size from FlextConstants (Constants default)"),
        )
        max_workers: int = Field(
            default=FlextConstants.Processing.DEFAULT_MAX_WORKERS,
            le=FlextConstants.Settings.MAX_WORKERS_THRESHOLD,
            description="Maximum workers (Config has priority over Constants)",
        )
        timeout_per_item: float = Field(
            default=FlextConstants.Defaults.TIMEOUT,
            description="Timeout per item (Config has priority over Constants)",
        )
        continue_on_error: bool = True
        data_items: Annotated[
            list[object],
            Field(
                default_factory=list,
                max_length=FlextConstants.Performance.BatchProcessing.MAX_ITEMS,
            ),
        ]

        @model_validator(mode="after")
        def validate_batch(self) -> Self:
            """Validate batch configuration consistency."""
            max_batch_size = (
                FlextConstants.Performance.BatchProcessing.MAX_VALIDATION_SIZE
            )
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

        handler_name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
        input_data: dict[str, FlextTypes.GeneralValueType] = Field(default_factory=dict)
        execution_context: dict[str, FlextTypes.GeneralValueType] = Field(
            default_factory=dict,
        )
        timeout_seconds: float = Field(
            default=FlextConstants.Defaults.TIMEOUT,
            le=FlextConstants.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout from FlextConfig",
        )
        retry_on_failure: bool = True
        max_retries: int = Field(
            default=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
            description="Max retries from FlextConfig",
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
        order: int = Field(default=0, description="Execution order in middleware chain")
        name: str | None = Field(default=None, description="Optional middleware name")
        config: dict[str, FlextTypes.GeneralValueType] = Field(
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
            default=0,
            ge=0,
            description="Current request count in window",
        )
        window_start: float = Field(
            default=0.0,
            ge=0.0,
            description="Timestamp when current window started",
        )
        limit: int = Field(
            default=100,
            ge=1,
            description="Maximum requests allowed per window",
        )
        window_seconds: int = Field(
            default=60,
            ge=1,
            description="Duration of rate limit window in seconds",
        )
        block_until: float = Field(
            default=0.0,
            ge=0.0,
            description="Timestamp until which requests are blocked",
        )

    class ExternalCommandConfig(FlextModelsCollections.Config):
        """Configuration for external command execution (Pydantic v2).

        Reduces parameter count for FlextUtilities.CommandExecution
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
        env: dict[str, str] | None = Field(
            default=None,
            description="Environment variables for the command",
        )
        cwd: str | None = Field(
            default=None,
            description="Working directory for command execution",
        )
        timeout_seconds: float | None = Field(
            default=None,
            gt=0,
            le=FlextConstants.Performance.MAX_TIMEOUT_SECONDS,
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
            default_factory=lambda: getattr(
                logging,
                FlextConfig().log_level.upper(),
                logging.INFO,
            ),
            ge=0,
            le=50,
            description=(
                "Numeric log level from FlextConfig (DEBUG=10, INFO=20, WARNING=30, "
                "ERROR=40, CRITICAL=50)"
            ),
        )
        console_renderer: bool = Field(
            default=True,
            description="Use console renderer (True) or JSON renderer (False)",
        )
        additional_processors: list[object] = Field(
            default_factory=list,
            description="Optional extra processors after standard FLEXT processors",
        )
        wrapper_class_factory: object | None = Field(
            default=None,
            description="Custom wrapper factory for structlog",
        )
        logger_factory: object | None = Field(
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
            default=FlextConstants.Logging.DEFAULT_LEVEL,
            description="Log level from FlextConfig (can be overridden)",
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

        metadata: Metadata | None = Field(
            default=None,
            description="Optional execution context metadata (Pydantic model)",
        )
        correlation_id: str | None = Field(
            default=None,
            description="Optional correlation ID for distributed tracing",
        )
        timeout_override: int | None = Field(
            default=None,
            ge=0,
            description="Optional timeout override in seconds",
        )

    class ExceptionConfig(FlextModelsCollections.Config):
        """Configuration for FlextExceptions.__init__ (Pydantic v2).

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
        metadata: Metadata | None = Field(
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
        extra_kwargs: dict[str, FlextTypes.GeneralValueType] = Field(
            default_factory=dict,
            description="Additional keyword arguments for metadata",
        )

    class ResultConfig(FlextModelsCollections.Config):
        """Configuration for FlextResult failure case (Pydantic v2).

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
        error_data: Metadata | None = Field(
            default=None,
            description="Additional error data (Pydantic model)",
        )


__all__ = ["FlextModelsConfig"]
