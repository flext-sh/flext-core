"""Configuration patterns extracted from FlextModels.

This module contains the FlextModelsConfig class with all configuration-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Config instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flext_core._models.entity import FlextModelsEntity
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.result import FlextResult


class FlextModelsConfig:
    """Configuration pattern container class.

    This class acts as a namespace container for configuration patterns.
    All nested classes are accessed via FlextModels.Config.* in the main models.py.
    """

    class ProcessingRequest(FlextModelsEntity.ArbitraryTypesModel):
        """Enhanced processing request with advanced validation."""

        model_config = ConfigDict(
            validate_assignment=False,  # Allow invalid values to be set for testing
            use_enum_values=True,
            arbitrary_types_allowed=True,
        )

        operation_id: str = Field(
            default_factory=lambda: str(uuid.uuid4()),
            min_length=1,
            description="Unique operation identifier",
        )
        data: dict[str, object] = Field(default_factory=dict)
        context: dict[str, object] = Field(default_factory=dict)
        timeout_seconds: float = Field(
            default_factory=lambda: FlextConfig().timeout_seconds,
            gt=0,
            le=FlextConstants.Performance.MAX_TIMEOUT_SECONDS,
            description="Operation timeout from FlextConfig",
        )
        retry_attempts: int = Field(
            default_factory=lambda: FlextConfig().max_retry_attempts,
            ge=0,
            le=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
            description="Maximum retry attempts from FlextConfig",
        )
        enable_validation: bool = True

        @field_validator("context", mode="before")
        @classmethod
        def validate_context(cls, v: object) -> dict[str, object]:
            """Validate context has required fields (Pydantic v2 mode='before')."""
            if not isinstance(v, dict):
                v = {}
            context: dict[str, object] = dict(v)
            if "correlation_id" not in context:
                context["correlation_id"] = str(uuid.uuid4())
            if "timestamp" not in context:
                context["timestamp"] = datetime.now(UTC).isoformat()
            return context

        def validate_processing_constraints(self) -> FlextResult[None]:
            """Validate constraints that should be checked during processing."""
            max_timeout_seconds = FlextConstants.Utilities.MAX_TIMEOUT_SECONDS
            if self.timeout_seconds > max_timeout_seconds:
                return FlextResult[None].fail(
                    f"Timeout cannot exceed {max_timeout_seconds} seconds"
                )

            return FlextResult[None].ok(None)

    class RetryConfiguration(FlextModelsEntity.ArbitraryTypesModel):
        """Retry configuration with advanced validation."""

        max_attempts: int = Field(
            default_factory=lambda: FlextConfig().max_retry_attempts,
            ge=FlextConstants.Reliability.RETRY_COUNT_MIN,
            le=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
            description="Maximum retry attempts from FlextConfig",
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
        retry_on_exceptions: Annotated[
            list[type[BaseException]],
            Field(
                default_factory=list,
                description="Exception types to retry on",
            ),
        ]
        retry_on_status_codes: Annotated[
            list[object],
            Field(
                default_factory=list,
                max_length=100,
                description="HTTP status codes to retry on",
            ),
        ]

        @field_validator("retry_on_status_codes", mode="after")
        @classmethod
        def validate_backoff_strategy(cls, v: list[object]) -> list[object]:
            """Validate status codes are valid HTTP codes (Pydantic v2 mode='after')."""
            validated_codes: list[object] = []
            for code in v:
                try:
                    if isinstance(code, (int, str)):
                        code_int = int(str(code))
                        if (
                            not FlextConstants.FlextWeb.HTTP_STATUS_MIN
                            <= code_int
                            <= FlextConstants.FlextWeb.HTTP_STATUS_MAX
                        ):
                            msg = f"Invalid HTTP status code: {code}"
                            raise FlextExceptions.ValidationError(
                                message=msg,
                                error_code=FlextConstants.Errors.VALIDATION_ERROR,
                            )
                        validated_codes.append(code_int)
                    else:
                        msg = f"Invalid HTTP status code type: {type(code)}"
                        raise FlextExceptions.TypeError(
                            message=msg,
                            error_code=FlextConstants.Errors.TYPE_ERROR,
                        )
                except (ValueError, TypeError) as e:
                    msg = f"Invalid HTTP status code: {code}"
                    raise FlextExceptions.ValidationError(
                        message=msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    ) from e
            return validated_codes

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency."""
            if self.max_delay_seconds < self.initial_delay_seconds:
                msg = "max_delay_seconds must be >= initial_delay_seconds"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return self

    class ValidationConfiguration(FlextModelsEntity.ArbitraryTypesModel):
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
            """Validate custom validators are callable (Pydantic v2 mode='after')."""
            for validator in v:
                if not callable(validator):
                    msg = "All validators must be callable"
                    raise FlextExceptions.TypeError(
                        message=msg,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
            return v

    class BatchProcessingConfig(FlextModelsEntity.ArbitraryTypesModel):
        """Enhanced batch processing configuration."""

        batch_size: int = Field(
            default_factory=lambda: FlextConfig().max_batch_size,
            description="Batch size from FlextConfig",
        )
        max_workers: int = Field(
            default_factory=lambda: FlextConfig().max_workers,
            le=FlextConstants.Settings.MAX_WORKERS_THRESHOLD,
            description="Maximum workers from FlextConfig",
        )
        timeout_per_item: float = Field(
            default_factory=lambda: FlextConfig().timeout_seconds,
            description="Timeout per item from FlextConfig",
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
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Adjust max_workers to not exceed batch_size without triggering validation
            adjusted_workers = min(self.max_workers, self.batch_size)
            # Use direct assignment to __dict__ to bypass Pydantic validation
            self.__dict__["max_workers"] = adjusted_workers

            return self

    class HandlerExecutionConfig(FlextModelsEntity.ArbitraryTypesModel):
        """Enhanced handler execution configuration."""

        handler_name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
        input_data: dict[str, object] = Field(default_factory=dict)
        execution_context: dict[str, object] = Field(default_factory=dict)
        timeout_seconds: float = Field(
            default_factory=lambda: FlextConfig().timeout_seconds,
            le=FlextConstants.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout from FlextConfig",
        )
        retry_on_failure: bool = True
        max_retries: int = Field(
            default_factory=lambda: FlextConfig().max_retry_attempts,
            description="Max retries from FlextConfig",
        )
        fallback_handlers: list[str] = Field(default_factory=list)

    class MiddlewareConfig(BaseModel):
        """Configuration for middleware execution.

        Provides configuration options for middleware ordering and priority
        within request/response processing pipeline.
        """

        model_config = ConfigDict(
            json_schema_extra={
                "title": "MiddlewareConfig",
                "description": "Configuration for middleware execution in request processing",
            },
        )

        enabled: bool = Field(default=True, description="Whether middleware is enabled")
        order: int = Field(default=0, description="Execution order in middleware chain")
        priority: int = Field(
            default=0,
            ge=0,
            le=100,
            description="Priority level for execution ordering (0-100)",
        )
        name: str | None = Field(default=None, description="Optional middleware name")
        config: dict[str, object] = Field(
            default_factory=dict, description="Middleware-specific configuration"
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
            default="", description="Name of the rate limiter processor"
        )
        count: int = Field(
            default=0, ge=0, description="Current request count in window"
        )
        window_start: float = Field(
            default=0.0, ge=0.0, description="Timestamp when current window started"
        )
        limit: int = Field(
            default=100, ge=1, description="Maximum requests allowed per window"
        )
        window_seconds: int = Field(
            default=60, ge=1, description="Duration of rate limit window in seconds"
        )
        block_until: float = Field(
            default=0.0,
            ge=0.0,
            description="Timestamp until which requests are blocked",
        )


__all__ = ["FlextModelsConfig"]
