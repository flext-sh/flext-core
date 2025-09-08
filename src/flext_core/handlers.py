"""Handler system for request processing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final, cast, override

from flext_core.constants import FlextConstants
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


def _convert_config_for_pydantic(config: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
    """Convert config dict to proper types for Pydantic models.

    This function safely converts object values to the expected types
    that Pydantic models require, handling type conversion errors gracefully.
    """
    converted: FlextTypes.Core.Dict = {}

    for key, value in config.items():
        if value is None:
            converted[key] = None
        elif isinstance(value, (str, int, float, bool)):
            converted[key] = value
        elif isinstance(value, str):
            # Try to convert string to appropriate type
            if key in {
                "timeout",
                "max_retries",
                "max_validation_errors",
                "max_cache_size",
            }:
                try:
                    converted[key] = int(value)
                except (ValueError, TypeError):
                    converted[key] = value  # Keep as string if conversion fails
            elif key in {
                "enable_metrics",
                "enable_validation",
                "validation_strict",
                "enable_authorization",
                "enable_events",
                "enable_caching",
            }:
                if value.lower() in {"true", "1", "yes", "on"}:
                    converted[key] = True
                elif value.lower() in {"false", "0", "no", "off"}:
                    converted[key] = False
                else:
                    converted[key] = value  # Keep as string if conversion fails
            else:
                converted[key] = value
        else:
            # For any other type, convert to string
            converted[key] = str(value)

    return converted


def _create_pydantic_config(config_class: type, **kwargs: object) -> object:
    """Create Pydantic config instance with proper type handling."""
    return config_class(**kwargs)


class FlextHandlers:
    """Handler system for request processing with CQRS and metrics."""

    # =========================================================================
    # CONSTANTS - Handler system constants extending FlextConstants
    # =========================================================================

    class Constants(FlextConstants):
        """Handler execution constants and limits."""

        # Inherit all base constants from FlextConstants
        # Add handler-specific constant categories

        class Handler:
            """Handler execution timeouts, retries, and performance limits."""

            # Execution parameters from centralized constants
            DEFAULT_TIMEOUT: Final[int] = FlextConstants.Network.DEFAULT_TIMEOUT
            MAX_RETRIES: Final[int] = FlextConstants.Defaults.MAX_RETRIES
            RETRY_DELAY: Final[float] = FlextConstants.Handlers.RETRY_DELAY
            DEFAULT_BATCH_SIZE: Final[int] = (
                FlextConstants.Performance.DEFAULT_BATCH_SIZE
            )

            # Chain limits from centralized constants
            MAX_CHAIN_HANDLERS: Final[int] = FlextConstants.Handlers.MAX_CHAIN_HANDLERS
            MAX_PIPELINE_STAGES: Final[int] = (
                FlextConstants.Handlers.MAX_PIPELINE_STAGES
            )

            # Performance thresholds from centralized constants
            SLOW_HANDLER_THRESHOLD: Final[float] = (
                FlextConstants.Handlers.SLOW_HANDLER_THRESHOLD
            )
            MEMORY_THRESHOLD_MB: Final[int] = (
                FlextConstants.Handlers.MEMORY_THRESHOLD_MB
            )
            METRICS_COLLECTION_INTERVAL: Final[int] = (
                FlextConstants.Handlers.METRICS_COLLECTION_INTERVAL
            )

            class States:
                """Handler execution states."""

                IDLE: Final[str] = FlextConstants.Handlers.HANDLER_STATE_IDLE
                PROCESSING: Final[str] = (
                    FlextConstants.Handlers.HANDLER_STATE_PROCESSING
                )
                COMPLETED: Final[str] = FlextConstants.Handlers.HANDLER_STATE_COMPLETED
                FAILED: Final[str] = FlextConstants.Handlers.HANDLER_STATE_FAILED
                TIMEOUT: Final[str] = FlextConstants.Handlers.HANDLER_STATE_TIMEOUT
                PAUSED: Final[str] = FlextConstants.Handlers.HANDLER_STATE_PAUSED

            class Types:
                """Handler type classifications."""

                BASIC: Final[str] = FlextConstants.Handlers.HANDLER_TYPE_BASIC
                VALIDATING: Final[str] = FlextConstants.Handlers.HANDLER_TYPE_VALIDATING
                AUTHORIZING: Final[str] = (
                    FlextConstants.Handlers.HANDLER_TYPE_AUTHORIZING
                )
                METRICS: Final[str] = FlextConstants.Handlers.HANDLER_TYPE_METRICS
                COMMAND: Final[str] = FlextConstants.Handlers.HANDLER_TYPE_COMMAND
                QUERY: Final[str] = FlextConstants.Handlers.HANDLER_TYPE_QUERY
                EVENT: Final[str] = FlextConstants.Handlers.HANDLER_TYPE_EVENT
                PIPELINE: Final[str] = FlextConstants.Handlers.HANDLER_TYPE_PIPELINE
                CHAIN: Final[str] = FlextConstants.Handlers.HANDLER_TYPE_CHAIN

    # =========================================================================
    # TYPES - Handler type definitions extending FlextTypes
    # =========================================================================

    class Types(FlextTypes):
        """Type definitions for handler system."""

        class HandlerTypes:
            """Type aliases for handler components."""

            # Core handler types
            type Name = str  # Handler identifier
            type State = str  # Handler execution state
            type Metadata = FlextTypes.Core.Dict  # Handler configuration metadata

            # Metrics and performance types
            type Metrics = FlextTypes.Core.Dict  # Flexible metrics storage
            type Counter = int  # Simple counter metric
            type Timing = float  # Timing measurements
            type ErrorMap = FlextTypes.Core.CounterDict  # Error type counters
            type SizeList = list[int]  # Size measurements
            type PerformanceMap = dict[
                str,
                dict[str, int | float],
            ]  # Nested performance data

            # Handler function types
            type HandlerFunction = Callable[[object], FlextResult[object]]
            type ValidatorFunction = Callable[[object], bool | FlextResult[None]]
            type AuthorizerFunction = FlextTypes.Validation.Validator
            type ProcessorFunction = Callable[[object], FlextResult[object]]

        class Message:
            """Type aliases for message components."""

            type Data = FlextTypes.Core.Dict  # Message payload
            type Type = str  # Message type identifier
            type Headers = FlextTypes.Core.Headers  # Message headers
            type Context = FlextTypes.Core.Dict  # Processing context

    # =========================================================================
    # PROTOCOLS - Handler protocol definitions for type safety
    # =========================================================================

    class Protocols:
        """Protocol interfaces for handlers."""

        # Alias core protocols for handler use
        Validator = FlextProtocols.Foundation.Validator
        MessageHandler = FlextProtocols.Application.MessageHandler
        ValidatingHandler = FlextProtocols.Application.ValidatingHandler
        AuthorizingHandler = FlextProtocols.Application.AuthorizingHandler

        # Handler-specific protocol extensions
        class MetricsHandler(FlextProtocols.Application.MessageHandler):
            """Handler protocol with metrics collection."""

            @abstractmethod
            def get_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get handler metrics."""
                ...

            @abstractmethod
            def reset_metrics(self) -> None:
                """Reset all collected metrics."""
                ...

        class ChainableHandler(FlextProtocols.Application.MessageHandler):
            """Handler protocol for chain participation."""

            @property
            @abstractmethod
            def handler_name(self) -> FlextHandlers.Types.HandlerTypes.Name:
                """Get handler identifier."""
                ...

            @abstractmethod
            def can_handle(self, message_type: type) -> bool:
                """Check if can handle message type."""
                ...

    # =========================================================================
    # INFRASTRUCTURE - Thread-safe operations and utilities
    # =========================================================================

    # Class-level thread safety infrastructure
    _handlers_lock: Final[threading.RLock] = threading.RLock()

    @staticmethod
    @contextmanager
    def thread_safe_operation() -> Iterator[None]:
        """Context manager for thread-safe operations."""
        with FlextHandlers._handlers_lock:
            yield

    # =========================================================================
    # IMPLEMENTATION - Core handler implementations
    # =========================================================================

    class Implementation:
        """Concrete handler implementations."""

        class AbstractHandler[TInput, TOutput](ABC):
            """Abstract base class defining handler contract."""

            @property
            @abstractmethod
            def handler_name(self) -> FlextHandlers.Types.HandlerTypes.Name:
                """Get handler identifier."""
                ...

            @abstractmethod
            def handle(self, request: TInput) -> FlextResult[TOutput]:
                """Process request."""
                ...

            @abstractmethod
            def can_handle(self, message_type: type) -> bool:
                """Check if can handle message type."""
                ...

        class BasicHandler:
            """Basic handler with metrics collection and state tracking."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
                config: FlextTypes.Core.Dict | None = None,
            ) -> None:
                """Initialize handler with optional Pydantic configuration."""
                # Support both dict and Pydantic config for backward compatibility
                if config is None:
                    config = {}

                # If dict provided, validate with Pydantic model
                if isinstance(config, dict):
                    # Convert to proper types for Pydantic
                    typed_cfg = _convert_config_for_pydantic(config)
                    if (
                        "handler_name" in typed_cfg
                        and typed_cfg["handler_name"] is not None
                    ):
                        pydantic_config = _create_pydantic_config(
                            FlextModels.SystemConfigs.BasicHandlerConfig, **typed_cfg
                        )
                    else:
                        pydantic_config = _create_pydantic_config(
                            FlextModels.SystemConfigs.BasicHandlerConfig,
                            handler_name=name,
                            **typed_cfg,
                        )
                    # Convert back to dict for internal use
                    self._config = cast(
                        "FlextModels.SystemConfigs.BasicHandlerConfig", pydantic_config
                    ).model_dump()

                self._handler_name: Final[FlextHandlers.Types.HandlerTypes.Name] = (
                    self._config.get("handler_name") or name or self.__class__.__name__
                )
                self._state: FlextHandlers.Types.HandlerTypes.State = (
                    FlextHandlers.Constants.Handler.States.IDLE
                )
                self._metrics: FlextHandlers.Types.HandlerTypes.Metrics = {
                    "requests_processed": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "average_processing_time": 0.0,
                    "total_processing_time": 0.0,
                    "error_count": 0,
                }

            @property
            def handler_name(self) -> FlextHandlers.Types.HandlerTypes.Name:
                """Get handler identifier."""
                return self._handler_name

            @property
            def state(self) -> FlextHandlers.Types.HandlerTypes.State:
                """Get current handler execution state."""
                return self._state

            def configure(
                self,
                config: FlextTypes.Core.Dict
                | FlextModels.SystemConfigs.BasicHandlerConfig,
            ) -> FlextResult[None]:
                """Configure handler with dict or Pydantic model."""
                try:
                    # Support both dict and Pydantic config
                    if isinstance(config, dict):
                        # Convert to proper types for Pydantic
                        typed_cfg = _convert_config_for_pydantic(config)
                        pydantic_config = _create_pydantic_config(
                            FlextModels.SystemConfigs.BasicHandlerConfig, **typed_cfg
                        )
                        validated_config = cast(
                            "FlextModels.SystemConfigs.BasicHandlerConfig",
                            pydantic_config,
                        ).model_dump()
                    else:
                        # Already a Pydantic model
                        validated_config = config.model_dump()

                    # Update configuration with validated values
                    # Pydantic has already performed all validation
                    self._config.update(validated_config)
                    return FlextResult[None].ok(None)

                except Exception as e:
                    return FlextResult[None].fail(f"Handler configuration failed: {e}")

            def handle(self, request: object) -> FlextResult[object]:
                """Handle request with metrics and state management."""
                start_time = time.time()

                # Update state to processing with thread safety
                with FlextHandlers.thread_safe_operation():
                    self._state = FlextHandlers.Constants.Handler.States.PROCESSING
                    self._metrics["requests_processed"] = (
                        cast("int", self._metrics["requests_processed"]) + 1
                    )

                try:
                    # Process request using template method pattern
                    result = self._process_request(request)

                    # Update metrics based on result
                    with FlextHandlers.thread_safe_operation():
                        if result.success:
                            self._metrics["successful_requests"] = (
                                cast("int", self._metrics["successful_requests"]) + 1
                            )
                            self._state = (
                                FlextHandlers.Constants.Handler.States.COMPLETED
                            )
                        else:
                            self._metrics["failed_requests"] = (
                                cast("int", self._metrics["failed_requests"]) + 1
                            )
                            self._metrics["error_count"] = (
                                cast("int", self._metrics["error_count"]) + 1
                            )
                            self._state = FlextHandlers.Constants.Handler.States.FAILED

                    return result

                except Exception as e:
                    # Handle unexpected exceptions
                    with FlextHandlers.thread_safe_operation():
                        self._metrics["failed_requests"] = (
                            cast("int", self._metrics["failed_requests"]) + 1
                        )
                        self._metrics["error_count"] = (
                            cast("int", self._metrics["error_count"]) + 1
                        )
                        self._state = FlextHandlers.Constants.Handler.States.FAILED

                    return FlextResult[object].fail(
                        f"Handler execution failed: {e}",
                        error_code=FlextConstants.Errors.OPERATION_ERROR,
                    )

                finally:
                    # Update timing metrics
                    processing_time = time.time() - start_time
                    with FlextHandlers.thread_safe_operation():
                        # Type-safe arithmetic operations
                        current_total = self._metrics["total_processing_time"]
                        if isinstance(current_total, (int, float)):
                            self._metrics["total_processing_time"] = (
                                current_total + processing_time
                            )

                        # Calculate average processing time
                        requests_count = cast(
                            "float",
                            self._metrics["requests_processed"],
                        )
                        if requests_count > 0:
                            total_time = cast(
                                "float",
                                self._metrics["total_processing_time"],
                            )
                            self._metrics["average_processing_time"] = (
                                total_time / requests_count
                            )

                    # Reset state to idle if still processing
                    if self._state == FlextHandlers.Constants.Handler.States.PROCESSING:
                        self._state = FlextHandlers.Constants.Handler.States.IDLE

            def _process_request(self, request: object) -> FlextResult[object]:
                """Process request."""
                return FlextResult[object].ok(request)

            def can_handle(self, message_type: type) -> bool:
                """Check if handler can process message type."""
                _ = message_type  # Base handler accepts all message types
                return True

            def get_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get handler metrics."""
                with FlextHandlers.thread_safe_operation():
                    return dict(self._metrics)

            def reset_metrics(self) -> None:
                """Reset handler metrics."""
                with FlextHandlers.thread_safe_operation():
                    self._metrics = {
                        "requests_processed": 0,
                        "successful_requests": 0,
                        "failed_requests": 0,
                        "average_processing_time": 0.0,
                        "total_processing_time": 0.0,
                        "error_count": 0,
                    }

            # =============================================================================
            # CONFIGURATION MANAGEMENT - FlextTypes.Config Integration
            # =============================================================================

            def get_handler_config(self) -> FlextResult[FlextTypes.Config.ConfigDict]:
                """Get current handler configuration."""
                try:
                    current_config: FlextTypes.Config.ConfigDict = {
                        "handler_name": self._handler_name,
                        "handler_state": self._state,
                        "state": self._state,  # Include both forms for compatibility
                        "log_level": self._config.get(
                            "log_level",
                            FlextConstants.Config.LogLevel.INFO.value,
                        ),
                        "environment": self._config.get(
                            "environment",
                            FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                        ),
                        "validation_level": self._config.get(
                            "validation_level",
                            FlextConstants.Config.ValidationLevel.NORMAL.value,
                        ),
                        "timeout": self._config.get("timeout", 30000),
                        "max_retries": self._config.get("max_retries", 3),
                        "requests_processed": cast(
                            "int",
                            self._metrics["requests_processed"],
                        ),
                        "success_rate": (
                            (
                                cast("int", self._metrics["successful_requests"])
                                / max(
                                    cast("int", self._metrics["requests_processed"]),
                                    1,
                                )
                            )
                            * 100
                        ),
                        "average_response_time_ms": cast(
                            "float",
                            self._metrics["average_processing_time"],
                        ),
                        "supported_handler_types": [
                            "BASIC",
                            "VALIDATING",
                            "AUTHORIZING",
                            "METRICS",
                            "EVENT",
                        ],
                        "available_configurations": [
                            "log_level",
                            "environment",
                            "validation_level",
                            "timeout",
                            "max_retries",
                        ],
                        "metrics": {
                            "requests_processed": cast(
                                "int",
                                self._metrics["requests_processed"],
                            ),
                            "successful_requests": cast(
                                "int",
                                self._metrics["successful_requests"],
                            ),
                            "average_processing_time": cast(
                                "float",
                                self._metrics["average_processing_time"],
                            ),
                        },
                    }

                    return FlextResult[FlextTypes.Config.ConfigDict].ok(current_config)

                except Exception as e:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Failed to get handler config: {e}",
                    )

            @classmethod
            def create_environment_handler_config(
                cls,
                environment: FlextTypes.Config.Environment,
            ) -> FlextResult[FlextTypes.Config.ConfigDict]:
                """Create environment-specific handler configuration."""
                try:
                    # Validate environment
                    valid_environments = [
                        e.value for e in FlextConstants.Config.ConfigEnvironment
                    ]
                    if environment not in valid_environments:
                        return FlextResult[FlextTypes.Config.ConfigDict].fail(
                            f"Invalid environment '{environment}'. Valid options: {valid_environments}",
                        )

                    # Create environment-specific configuration
                    if environment == "production":
                        config: FlextTypes.Config.ConfigDict = {
                            "environment": environment,
                            "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                            "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                            "timeout": 30000,  # Reasonable timeout for production
                            "max_retries": 1,  # Fewer retries for production
                            "enable_detailed_logging": False,
                            "enable_performance_metrics": True,
                            "enable_error_reporting": True,
                        }
                    elif environment == "development":
                        config = {
                            "environment": environment,
                            "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                            "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                            "timeout": 60000,  # Longer timeout for development
                            "max_retries": 5,  # More retries for development
                            "enable_detailed_logging": True,
                            "enable_performance_metrics": True,
                            "enable_error_reporting": True,
                            "enable_debugging": True,  # Enable debugging for development environment
                        }
                    elif environment == "test":
                        config = {
                            "environment": environment,
                            "log_level": FlextConstants.Config.LogLevel.ERROR.value,
                            "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                            "timeout": 5000,  # Short timeout for tests
                            "max_retries": 0,  # No retries in tests
                            "enable_detailed_logging": False,
                            "enable_performance_metrics": False,  # No metrics in tests
                            "enable_performance_tracking": False,  # No performance tracking in tests
                            "enable_error_reporting": False,
                        }
                    else:  # staging, local, etc.
                        config = {
                            "environment": environment,
                            "log_level": FlextConstants.Config.LogLevel.INFO.value,
                            "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                            "timeout": 30000,
                            "max_retries": 3,
                            "enable_detailed_logging": True,
                            "enable_performance_metrics": True,
                            "enable_error_reporting": True,
                        }

                    return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

                except Exception as e:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Environment handler config failed: {e}",
                    )

            @classmethod
            def optimize_handler_performance(
                cls,
                config: FlextTypes.Config.ConfigDict,
            ) -> FlextResult[FlextTypes.Config.ConfigDict]:
                """Optimize handler performance."""
                try:
                    # Optimize based on performance level
                    performance_level = config.get("performance_level", "standard")

                    # Default performance configuration
                    optimized_config: FlextTypes.Config.ConfigDict = {
                        "performance_level": performance_level,  # Include performance level in result
                        "concurrent_handlers": config.get("concurrent_handlers", 10),
                        "max_concurrent_requests": config.get(
                            "max_concurrent_requests",
                            100,
                        ),  # Add max concurrent requests
                        "request_queue_size": config.get(
                            "request_queue_size",
                            1000,
                        ),  # Add request queue size
                        "processing_timeout": config.get(
                            "processing_timeout",
                            30000,
                        ),  # Add processing timeout
                        "request_batch_size": config.get("request_batch_size", 100),
                        "response_caching_enabled": config.get(
                            "response_caching_enabled",
                            False,
                        ),
                        "metrics_collection_interval": config.get(
                            "metrics_collection_interval",
                            5000,
                        ),
                        "enable_request_pooling": config.get(
                            "enable_request_pooling",
                            False,
                        ),
                        "thread_pool_size": config.get("thread_pool_size", 4),
                        "memory_limit_mb": config.get("memory_limit_mb", 512),
                        "enable_compression": config.get("enable_compression", False),
                    }

                    if performance_level == "high":
                        optimized_config.update(
                            {
                                "concurrent_handlers": 50,
                                "request_batch_size": 500,
                                "response_caching_enabled": True,
                                "metrics_collection_interval": 1000,
                                "enable_request_pooling": True,
                                "thread_pool_size": 16,
                                "memory_limit_mb": 2048,
                                "enable_compression": True,
                            },
                        )
                    elif performance_level == "low":
                        optimized_config.update(
                            {
                                "concurrent_handlers": 2,
                                "request_batch_size": 10,
                                "response_caching_enabled": False,
                                "metrics_collection_interval": 10000,
                                "enable_request_pooling": False,
                                "thread_pool_size": 1,
                                "memory_limit_mb": 128,
                                "enable_compression": False,
                            },
                        )

                    return FlextResult[FlextTypes.Config.ConfigDict].ok(
                        optimized_config,
                    )

                except Exception as e:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Handler performance optimization failed: {e}",
                    )

        class ValidatingHandler(BasicHandler):
            """Handler that validates requests before processing."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
                validators: list[FlextProtocols.Foundation.Validator[object]]
                | None = None,
                config: FlextTypes.Core.Dict | None = None,
            ) -> None:
                """Initialize with optional validators and Pydantic configuration."""
                # Validate config with ValidatingHandlerConfig
                if config is None:
                    config = {}

                if isinstance(config, dict):
                    # Convert to proper types for Pydantic
                    typed_cfg = _convert_config_for_pydantic(config)
                    pydantic_config = _create_pydantic_config(
                        FlextModels.SystemConfigs.ValidatingHandlerConfig,
                        handler_name=name,
                        **typed_cfg,
                    )
                    validated_config = cast(
                        "FlextModels.SystemConfigs.ValidatingHandlerConfig",
                        pydantic_config,
                    ).model_dump()

                super().__init__(name, validated_config)
                self._validators: Final[
                    list[FlextProtocols.Foundation.Validator[object]]
                ] = validators or []

            @override
            def handle(self, request: object) -> FlextResult[object]:
                """Handle with validation before processing."""
                # Validate request first
                validation_result = self.validate(request)
                if validation_result.is_failure:
                    return FlextResult[object].fail(
                        f"Validation failed: {validation_result.error}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Delegate to parent handler if validation succeeds
                return super().handle(request)

            def validate(self, request: object) -> FlextResult[None]:
                """Validate request."""
                for validator in self._validators:
                    try:
                        validation_result = validator.validate(request)
                        # Handle both bool and FlextResult return types
                        if isinstance(validation_result, bool):
                            if not validation_result:
                                return FlextResult[None].fail(
                                    "Validation failed",
                                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                                )
                        elif hasattr(validation_result, "is_failure") and getattr(
                            validation_result,
                            "is_failure",
                            False,
                        ):
                            return cast("FlextResult[None]", validation_result)
                    except Exception as e:
                        return FlextResult[None].fail(
                            f"Validation error: {e}",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )

                return FlextResult[None].ok(None)

            def add_validator(
                self,
                validator: FlextProtocols.Foundation.Validator[object],
            ) -> None:
                """Add validator to chain."""
                self._validators.append(validator)

        class AuthorizingHandler(BasicHandler):
            """Handler with authorization checking."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
                authorization_check: FlextHandlers.Types.HandlerTypes.AuthorizerFunction
                | None = None,
                config: FlextTypes.Core.Dict | None = None,
            ) -> None:
                """Initialize with authorization and Pydantic configuration."""
                # Validate config with AuthorizingHandlerConfig
                if config is None:
                    config = {}

                if isinstance(config, dict):
                    # Convert to proper types for Pydantic
                    typed_cfg = _convert_config_for_pydantic(config)
                    pydantic_config = _create_pydantic_config(
                        FlextModels.SystemConfigs.AuthorizingHandlerConfig,
                        handler_name=name,
                        **typed_cfg,
                    )
                    validated_config = cast(
                        "FlextModels.SystemConfigs.AuthorizingHandlerConfig",
                        pydantic_config,
                    ).model_dump()

                super().__init__(name, validated_config)
                self._authorization_check: FlextHandlers.Types.HandlerTypes.AuthorizerFunction = (
                    authorization_check or self._default_authorization
                )

            @override
            def handle(self, request: object) -> FlextResult[object]:
                """Handle with authorization check."""
                # Check authorization first
                if not self._authorization_check(request):
                    return FlextResult[object].fail(
                        "Authorization failed for request",
                        error_code=FlextConstants.Errors.AUTHORIZATION_DENIED,
                    )

                # Delegate to parent handler if authorized
                return super().handle(request)

            def _default_authorization(self, _request: object) -> bool:
                """Default authorization allows all."""
                return True

        class MetricsHandler(BasicHandler):
            """Handler with enhanced metrics collection."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
                config: FlextTypes.Core.Dict | None = None,
            ) -> None:
                """Initialize with enhanced metrics and Pydantic configuration."""
                # Validate config with MetricsHandlerConfig
                if config is None:
                    config = {}

                if isinstance(config, dict):
                    # Convert to proper types for Pydantic
                    typed_cfg = _convert_config_for_pydantic(config)
                    pydantic_config = _create_pydantic_config(
                        FlextModels.SystemConfigs.MetricsHandlerConfig,
                        handler_name=name,
                        **typed_cfg,
                    )
                    validated_config = cast(
                        "FlextModels.SystemConfigs.MetricsHandlerConfig",
                        pydantic_config,
                    ).model_dump()

                super().__init__(name, validated_config)
                self._error_types: FlextHandlers.Types.HandlerTypes.ErrorMap = {}
                self._request_sizes: FlextHandlers.Types.HandlerTypes.SizeList = []
                self._response_sizes: FlextHandlers.Types.HandlerTypes.SizeList = []
                self._peak_memory_usage: FlextHandlers.Types.HandlerTypes.Counter = 0

            @override
            def handle(self, request: object) -> FlextResult[object]:
                """Handle with enhanced metrics."""
                # Track request size
                request_size = len(str(request)) if request else 0
                self._request_sizes.append(request_size)

                # Delegate to parent handler
                result = super().handle(request)

                # Track result-specific metrics
                if result.is_failure and result.error:
                    error_type = type(result.error).__name__
                    self._error_types[error_type] = (
                        self._error_types.get(error_type, 0) + 1
                    )
                elif result.success and result.value:
                    response_size = len(str(result.value))
                    self._response_sizes.append(response_size)

                return result

            @override
            def get_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get enhanced metrics."""
                base_metrics = super().get_metrics()

                enhanced_metrics: FlextHandlers.Types.HandlerTypes.Metrics = {
                    "error_types": self._error_types,
                    "request_sizes": self._request_sizes,
                    "response_sizes": self._response_sizes,
                    "peak_memory_usage": self._peak_memory_usage,
                }

                # Calculate averages
                if self._request_sizes:
                    enhanced_metrics["average_request_size"] = sum(
                        self._request_sizes,
                    ) / len(self._request_sizes)

                if self._response_sizes:
                    enhanced_metrics["average_response_size"] = sum(
                        self._response_sizes,
                    ) / len(self._response_sizes)

                return {**base_metrics, **enhanced_metrics}

        class EventHandler:
            """Handler for domain events."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
                config: FlextTypes.Core.Dict | None = None,
            ) -> None:
                """Initialize event handler with Pydantic configuration."""
                # Validate config with EventHandlerConfig
                if config is None:
                    config = {}

                if isinstance(config, dict):
                    # Convert to proper types for Pydantic
                    typed_cfg = _convert_config_for_pydantic(config)
                    pydantic_config = _create_pydantic_config(
                        FlextModels.SystemConfigs.EventHandlerConfig,
                        handler_name=name,
                        **typed_cfg,
                    )
                    self._config = cast(
                        "FlextModels.SystemConfigs.EventHandlerConfig", pydantic_config
                    ).model_dump()

                self._name: FlextHandlers.Types.HandlerTypes.Name = (
                    self._config.get("handler_name") or name or "EventHandler"
                )
                self._events_processed: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._event_types: FlextHandlers.Types.HandlerTypes.ErrorMap = {}

            def handle_event(self, event: object) -> FlextResult[None]:
                """Handle domain event."""
                with FlextHandlers.thread_safe_operation():
                    self._events_processed += 1
                    event_type = type(event).__name__
                    self._event_types[event_type] = (
                        self._event_types.get(event_type, 0) + 1
                    )

                return self._process_event(event)

            def _process_event(self, _event: object) -> FlextResult[None]:
                """Process event."""
                return FlextResult[None].ok(None)

            def get_event_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get event metrics."""
                with FlextHandlers.thread_safe_operation():
                    return {
                        "events_processed": self._events_processed,
                        "event_types": dict(self._event_types),
                    }

    # =========================================================================
    # CQRS - Command/Query responsibility segregation handlers
    # =========================================================================

    class CQRS:
        """CQRS pattern implementations."""

        class CommandHandler[TCommand, TResult](ABC):
            """Abstract command handler."""

            @abstractmethod
            def handle_command(self, command: TCommand) -> FlextResult[TResult]:
                """Handle command."""
                ...

            @abstractmethod
            def can_handle(self, command_type: type) -> bool:
                """Check if handler can execute command type."""

        class QueryHandler[TQuery, TResult](ABC):
            """Abstract query handler."""

            @abstractmethod
            def handle_query(self, query: TQuery) -> FlextResult[TResult]:
                """Handle query."""

        class CommandBus:
            """Command bus for routing commands."""

            def __init__(self) -> None:
                """Initialize command bus."""
                self._handlers: dict[type, object] = {}
                self._commands_processed: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._successful_commands: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._failed_commands: FlextHandlers.Types.HandlerTypes.Counter = 0

            def register(
                self,
                command_type: type,
                handler: object,
            ) -> FlextResult[None]:
                """Register command handler."""
                with FlextHandlers.thread_safe_operation():
                    self._handlers[command_type] = handler
                    return FlextResult[None].ok(None)

            def send(self, command: object) -> FlextResult[object]:
                """Send command to handler."""
                command_type = type(command)

                with FlextHandlers.thread_safe_operation():
                    handler = self._handlers.get(command_type)

                    if not handler:
                        return FlextResult[object].fail(
                            f"No handler registered for command: {command_type.__name__}",
                            error_code=FlextConstants.Errors.RESOURCE_NOT_FOUND,
                        )

                    self._commands_processed += 1

                # Execute handler outside lock to avoid deadlock
                try:
                    # Handler is known to have handle_command method from registration
                    handler_with_method = cast(
                        "FlextHandlers.CQRS.CommandHandler[object, object]",
                        handler,
                    )
                    result = handler_with_method.handle_command(command)

                    with FlextHandlers.thread_safe_operation():
                        if result.success:
                            self._successful_commands += 1
                        else:
                            self._failed_commands += 1

                    return result
                except Exception as e:
                    with FlextHandlers.thread_safe_operation():
                        self._failed_commands += 1

                    return FlextResult[object].fail(
                        f"Command execution failed: {e}",
                        error_code=FlextConstants.Errors.OPERATION_ERROR,
                    )

            def get_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get command bus metrics."""
                with FlextHandlers.thread_safe_operation():
                    return {
                        "commands_processed": self._commands_processed,
                        "successful_commands": self._successful_commands,
                        "failed_commands": self._failed_commands,
                        "registered_handlers": len(self._handlers),
                    }

        class QueryBus:
            """Query bus for routing queries."""

            def __init__(self) -> None:
                """Initialize query bus."""
                self._handlers: dict[
                    type,
                    FlextProtocols.Application.MessageHandler,
                ] = {}
                self._queries_processed: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._successful_queries: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._failed_queries: FlextHandlers.Types.HandlerTypes.Counter = 0

            def register(
                self,
                query_type: type,
                handler: FlextProtocols.Application.MessageHandler,
            ) -> FlextResult[None]:
                """Register query handler."""
                with FlextHandlers.thread_safe_operation():
                    self._handlers[query_type] = handler
                    return FlextResult[None].ok(None)

            def send(self, query: object) -> FlextResult[object]:
                """Send query to handler."""
                query_type = type(query)

                with FlextHandlers.thread_safe_operation():
                    handler = self._handlers.get(query_type)

                    if not handler:
                        return FlextResult[object].fail(
                            f"No handler registered for query: {query_type.__name__}",
                            error_code=FlextConstants.Errors.RESOURCE_NOT_FOUND,
                        )

                    self._queries_processed += 1

                # Execute handler outside lock
                try:
                    result = cast("FlextResult[object]", handler.handle(query))

                    with FlextHandlers.thread_safe_operation():
                        if result.success:
                            self._successful_queries += 1
                        else:
                            self._failed_queries += 1

                    return result
                except Exception as e:
                    with FlextHandlers.thread_safe_operation():
                        self._failed_queries += 1

                    return FlextResult[object].fail(
                        f"Query execution failed: {e}",
                        error_code=FlextConstants.Errors.OPERATION_ERROR,
                    )

            def get_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get query bus metrics."""
                with FlextHandlers.thread_safe_operation():
                    return {
                        "queries_processed": self._queries_processed,
                        "successful_queries": self._successful_queries,
                        "failed_queries": self._failed_queries,
                        "registered_handlers": len(self._handlers),
                    }

    # =========================================================================
    # PATTERNS - Enterprise design pattern implementations
    # =========================================================================

    class Patterns:
        """Enterprise design pattern implementations."""

        class HandlerChain:
            """Chain of Responsibility pattern."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
            ) -> None:
                """Initialize handler chain."""
                self._name: FlextHandlers.Types.HandlerTypes.Name = (
                    name or "HandlerChain"
                )
                self._handlers: list[FlextHandlers.Protocols.ChainableHandler] = []
                self._chain_executions: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._successful_chains: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._handler_performance: FlextHandlers.Types.HandlerTypes.PerformanceMap = {}

            def add_handler(
                self,
                handler: FlextHandlers.Protocols.ChainableHandler,
            ) -> FlextResult[None]:
                """Add handler to chain."""
                if (
                    len(self._handlers)
                    >= FlextHandlers.Constants.Handler.MAX_CHAIN_HANDLERS
                ):
                    return FlextResult[None].fail(
                        f"Chain exceeds maximum handlers limit: {FlextHandlers.Constants.Handler.MAX_CHAIN_HANDLERS}",
                        error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                    )

                with FlextHandlers.thread_safe_operation():
                    self._handlers.append(handler)
                    return FlextResult[None].ok(None)

            def handle(self, request: object) -> FlextResult[object]:
                """Execute chain of handlers."""
                with FlextHandlers.thread_safe_operation():
                    self._chain_executions += 1

                for handler in self._handlers:
                    if handler.can_handle(type(request)):
                        start_time = time.time()

                        try:
                            result = cast(
                                "FlextResult[object]",
                                handler.handle(request),
                            )
                            processing_time = time.time() - start_time

                            # Update performance metrics
                            handler_name = handler.handler_name
                            with FlextHandlers.thread_safe_operation():
                                if handler_name not in self._handler_performance:
                                    self._handler_performance[handler_name] = {
                                        "executions": 0,
                                        "total_time": 0.0,
                                        "average_time": 0.0,
                                    }

                                perf = self._handler_performance[handler_name]
                                perf["executions"] = cast("int", perf["executions"]) + 1
                                perf["total_time"] = (
                                    cast("float", perf["total_time"]) + processing_time
                                )
                                perf["average_time"] = (
                                    perf["total_time"] / perf["executions"]
                                )

                                if result.success:
                                    self._successful_chains += 1

                            return result
                        except Exception as e:
                            return FlextResult[object].fail(
                                f"Handler {handler.handler_name} failed: {e}",
                                error_code=FlextConstants.Errors.OPERATION_ERROR,
                            )

                return FlextResult[object].fail(
                    "No handler in chain could process the request",
                    error_code=FlextConstants.Errors.RESOURCE_NOT_FOUND,
                )

            def get_chain_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get chain metrics."""
                with FlextHandlers.thread_safe_operation():
                    return {
                        "chain_executions": self._chain_executions,
                        "successful_chains": self._successful_chains,
                        "handler_count": len(self._handlers),
                        "handler_performance": dict(self._handler_performance),
                    }

        class Pipeline:
            """Pipeline pattern for data transformation."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
            ) -> None:
                """Initialize pipeline."""
                self._name: Final[FlextHandlers.Types.HandlerTypes.Name] = (
                    name or "Pipeline"
                )
                self._stages: list[
                    FlextHandlers.Types.HandlerTypes.ProcessorFunction
                ] = []
                self._pipeline_executions: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._successful_pipelines: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._stage_performance: FlextHandlers.Types.HandlerTypes.PerformanceMap = {}

            def add_stage(
                self,
                stage: FlextHandlers.Types.HandlerTypes.ProcessorFunction,
            ) -> FlextResult[None]:
                """Add stage to pipeline."""
                if (
                    len(self._stages)
                    >= FlextHandlers.Constants.Handler.MAX_PIPELINE_STAGES
                ):
                    return FlextResult[None].fail(
                        f"Pipeline exceeds maximum stages limit: {FlextHandlers.Constants.Handler.MAX_PIPELINE_STAGES}",
                        error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                    )

                with FlextHandlers.thread_safe_operation():
                    self._stages.append(stage)
                    return FlextResult[None].ok(None)

            def process(self, data: object) -> FlextResult[object]:
                """Process data through pipeline."""
                with FlextHandlers.thread_safe_operation():
                    self._pipeline_executions += 1

                current_data = data

                for i, stage in enumerate(self._stages):
                    stage_name = f"stage_{i}"
                    start_time = time.time()

                    try:
                        result = stage(current_data)
                        processing_time = time.time() - start_time

                        # Update stage performance metrics
                        with FlextHandlers.thread_safe_operation():
                            if stage_name not in self._stage_performance:
                                self._stage_performance[stage_name] = {
                                    "executions": 0,
                                    "total_time": 0.0,
                                    "average_time": 0.0,
                                }

                            perf = self._stage_performance[stage_name]
                            perf["executions"] = cast("int", perf["executions"]) + 1
                            perf["total_time"] = (
                                cast("float", perf["total_time"]) + processing_time
                            )
                            perf["average_time"] = (
                                perf["total_time"] / perf["executions"]
                            )

                        if result.is_failure:
                            return FlextResult[object].fail(
                                f"Pipeline failed at stage {i}: {result.error}",
                                error_code=FlextConstants.Errors.OPERATION_ERROR,
                            )

                        current_data = result.value

                    except Exception as e:
                        return FlextResult[object].fail(
                            f"Pipeline stage {i} raised exception: {e}",
                            error_code=FlextConstants.Errors.OPERATION_ERROR,
                        )

                with FlextHandlers.thread_safe_operation():
                    self._successful_pipelines += 1

                return FlextResult[object].ok(current_data)

            def get_pipeline_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get pipeline metrics."""
                with FlextHandlers.thread_safe_operation():
                    return {
                        "pipeline_executions": self._pipeline_executions,
                        "successful_pipelines": self._successful_pipelines,
                        "stage_count": len(self._stages),
                        "stage_performance": dict(self._stage_performance),
                    }

    # =========================================================================
    # MANAGEMENT - Handler registry and lifecycle management
    # =========================================================================

    class Management:
        """Handler lifecycle and registry management."""

        class HandlerRegistry:
            """Registry for handler management."""

            def __init__(self) -> None:
                """Initialize handler registry."""
                self._handlers: dict[FlextHandlers.Types.HandlerTypes.Name, object] = {}
                self._registrations: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._lookups: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._successful_lookups: FlextHandlers.Types.HandlerTypes.Counter = 0

            def register(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name,
                handler: object,
            ) -> FlextResult[object]:
                """Register handler."""
                if not name or not name.strip():
                    return FlextResult[object].fail(
                        "Handler name cannot be empty",
                        error_code=FlextConstants.Errors.INVALID_ARGUMENT,
                    )

                with FlextHandlers.thread_safe_operation():
                    if name in self._handlers:
                        return FlextResult[object].fail(
                            f"Handler with name '{name}' already registered",
                            error_code=FlextConstants.Errors.DUPLICATE_RESOURCE,
                        )

                    self._handlers[name] = handler
                    self._registrations += 1
                    return FlextResult[object].ok(handler)

            def get_handler(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name,
            ) -> FlextResult[object]:
                """Get handler by name."""
                with FlextHandlers.thread_safe_operation():
                    self._lookups += 1
                    handler = self._handlers.get(name)

                    if handler:
                        self._successful_lookups += 1
                        return FlextResult[object].ok(handler)

                return FlextResult[object].fail(
                    f"Handler '{name}' not found in registry",
                    error_code=FlextConstants.Errors.RESOURCE_NOT_FOUND,
                )

            def get_all_handlers(
                self,
            ) -> dict[FlextHandlers.Types.HandlerTypes.Name, object]:
                """Get all registered handlers."""
                with FlextHandlers.thread_safe_operation():
                    return dict(self._handlers)

            def unregister(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name,
            ) -> FlextResult[None]:
                """Unregister handler."""
                with FlextHandlers.thread_safe_operation():
                    if name in self._handlers:
                        del self._handlers[name]
                        return FlextResult[None].ok(None)

                return FlextResult[None].fail(
                    f"Handler '{name}' not found for unregistration",
                    error_code=FlextConstants.Errors.RESOURCE_NOT_FOUND,
                )

            def get_registry_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get registry metrics."""
                with FlextHandlers.thread_safe_operation():
                    return {
                        "total_handlers": len(self._handlers),
                        "total_registrations": self._registrations,
                        "total_lookups": self._lookups,
                        "successful_lookups": self._successful_lookups,
                        "lookup_success_rate": (
                            (self._successful_lookups / self._lookups * 100)
                            if self._lookups > 0
                            else 0.0
                        ),
                    }


# =============================================================================
# MODULE EXPORTS - Only FlextHandlers (ABI compatibility in legacy.py)
# =============================================================================

__all__ = [
    "FlextHandlers",  # Single consolidated export following FLEXT patterns
]
