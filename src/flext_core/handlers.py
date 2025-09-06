"""Handler system for request processing.

Provides FlextHandlers implementing Chain of Responsibility, CQRS,
command buses, and validation handlers with metrics collection.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final, cast, override

from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextHandlers:
    """Comprehensive handler system providing production-ready request processing capabilities.

    This class serves as the central container for all handler-related functionality
    in the FLEXT ecosystem, implementing sophisticated patterns for request processing,
    validation, authorization, CQRS, and metrics collection. The system is designed
    for high-performance, thread-safe operation in enterprise environments.

    """

    # =========================================================================
    # CONSTANTS - Handler system constants extending FlextConstants
    # =========================================================================

    class Constants(FlextConstants):
        """Handler execution constants and limits extending FlextConstants.

        This class provides handler-specific constants while inheriting all base
        constants from FlextConstants. It organizes constants into logical categories
        for handler execution parameters, state definitions, and type classifications.
        """

        # Inherit all base constants from FlextConstants
        # Add handler-specific constant categories

        class Handler:
            """Handler execution timeouts, retries, and performance limits from FlextConstants.

            This class provides execution parameters and performance thresholds for
            handler operations, sourced from the centralized FlextConstants system
            to ensure consistency across the FLEXT ecosystem.
            """

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
                """Handler execution states (IDLE, PROCESSING, COMPLETED, etc.)."""

                IDLE: Final[str] = FlextConstants.Handlers.HANDLER_STATE_IDLE
                PROCESSING: Final[str] = (
                    FlextConstants.Handlers.HANDLER_STATE_PROCESSING
                )
                COMPLETED: Final[str] = FlextConstants.Handlers.HANDLER_STATE_COMPLETED
                FAILED: Final[str] = FlextConstants.Handlers.HANDLER_STATE_FAILED
                TIMEOUT: Final[str] = FlextConstants.Handlers.HANDLER_STATE_TIMEOUT
                PAUSED: Final[str] = FlextConstants.Handlers.HANDLER_STATE_PAUSED

            class Types:
                """Handler type classifications (BASIC, VALIDATING, COMMAND, etc.)."""

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
        """Type definitions for handler names, states, metrics, and functions."""

        class HandlerTypes:
            """Type aliases for handler identifiers, states, metrics, and functions."""

            # Core handler types
            type Name = str  # Handler identifier
            type State = str  # Handler execution state
            type Metadata = dict[str, object]  # Handler configuration metadata

            # Metrics and performance types
            type Metrics = dict[str, object]  # Flexible metrics storage
            type Counter = int  # Simple counter metric
            type Timing = float  # Timing measurements
            type ErrorMap = dict[str, int]  # Error type counters
            type SizeList = list[int]  # Size measurements
            type PerformanceMap = dict[
                str,
                dict[str, int | float],
            ]  # Nested performance data

            # Handler function types
            type HandlerFunction = Callable[[object], FlextResult[object]]
            type ValidatorFunction = Callable[[object], bool | FlextResult[None]]
            type AuthorizerFunction = Callable[[object], bool]
            type ProcessorFunction = Callable[[object], FlextResult[object]]

        class Message:
            """Type aliases for message data, headers, and context."""

            type Data = dict[str, object]  # Message payload
            type Type = str  # Message type identifier
            type Headers = dict[str, str]  # Message headers
            type Context = dict[str, object]  # Processing context

    # =========================================================================
    # PROTOCOLS - Handler protocol definitions for type safety
    # =========================================================================

    class Protocols:
        """Protocol interfaces for handlers, validators, and metrics collection."""

        # Alias core protocols for handler use
        Validator = FlextProtocols.Foundation.Validator
        MessageHandler = FlextProtocols.Application.MessageHandler
        ValidatingHandler = FlextProtocols.Application.ValidatingHandler
        AuthorizingHandler = FlextProtocols.Application.AuthorizingHandler

        # Handler-specific protocol extensions
        class MetricsHandler(FlextProtocols.Application.MessageHandler):
            """Handler protocol that provides metrics collection and reset methods."""

            @abstractmethod
            def get_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get efficient handler metrics."""
                ...

            @abstractmethod
            def reset_metrics(self) -> None:
                """Reset all collected metrics."""
                ...

        class ChainableHandler(FlextProtocols.Application.MessageHandler):
            """Handler protocol for chain participation with name and type checking."""

            @property
            @abstractmethod
            def handler_name(self) -> FlextHandlers.Types.HandlerTypes.Name:
                """Get unique handler identifier."""
                ...

            @abstractmethod
            def can_handle(self, message_type: type) -> bool:
                """Check if handler can process given message type."""
                ...

    # =========================================================================
    # INFRASTRUCTURE - Thread-safe operations and utilities
    # =========================================================================

    # Class-level thread safety infrastructure
    _handlers_lock: Final[threading.RLock] = threading.RLock()

    @staticmethod
    @contextmanager
    def thread_safe_operation() -> Iterator[None]:
        """Context manager for thread-safe handler state modifications."""
        with FlextHandlers._handlers_lock:
            yield

    # =========================================================================
    # IMPLEMENTATION - Core handler implementations
    # =========================================================================

    class Implementation:
        """Concrete handler implementations with production-ready capabilities."""

        class AbstractHandler[TInput, TOutput](ABC):
            """Abstract base class defining handler contract.

            Requires implementation of handler_name property, handle method,
            and can_handle method for type checking.
            """

            @property
            @abstractmethod
            def handler_name(self) -> FlextHandlers.Types.HandlerTypes.Name:
                """Get unique handler identifier."""
                ...

            @abstractmethod
            def handle(self, request: TInput) -> FlextResult[TOutput]:
                """Process request with railway-oriented programming."""
                ...

            @abstractmethod
            def can_handle(self, message_type: type) -> bool:
                """Check if handler can process given message type."""
                ...

        class BasicHandler:
            """Basic handler with metrics collection and state tracking.

            Tracks request counts, success/failure rates, processing times,
            and execution state. Uses template method pattern - override
            _process_request() for custom logic.
            """

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
            ) -> None:
                """Initialize handler with name and metrics infrastructure."""
                self._handler_name: Final[FlextHandlers.Types.HandlerTypes.Name] = (
                    name or self.__class__.__name__
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

                # Initialize handler configuration with FlextTypes.Config
                self._config: FlextTypes.Config.ConfigDict = {
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "timeout": 30000,  # milliseconds
                    "max_retries": 3,
                }

            @property
            def handler_name(self) -> FlextHandlers.Types.HandlerTypes.Name:
                """Get handler unique identifier."""
                return self._handler_name

            @property
            def state(self) -> FlextHandlers.Types.HandlerTypes.State:
                """Get current handler execution state.

                Returns:
                    FlextHandlers.Types.HandlerTypes.State: Current state

                """
                return self._state

            def configure(
                self,
                config: FlextTypes.Config.ConfigDict,
            ) -> FlextResult[None]:
                """Configure handler with FlextTypes.Config integration.

                Args:
                    config: Configuration dictionary with FlextTypes.Config validation

                Returns:
                    FlextResult[None]: Success or failure with validation errors

                """
                try:
                    # Validate log level
                    if "log_level" in config:
                        log_level = config["log_level"]
                        valid_levels = [
                            level.value for level in FlextConstants.Config.LogLevel
                        ]
                        if log_level not in valid_levels:
                            return FlextResult[None].fail(
                                f"Invalid log_level: {log_level}",
                            )

                    # Validate environment
                    if "environment" in config:
                        env = config["environment"]
                        valid_envs = [
                            e.value for e in FlextConstants.Config.ConfigEnvironment
                        ]
                        if env not in valid_envs:
                            return FlextResult[None].fail(f"Invalid environment: {env}")

                    # Validate validation level
                    if "validation_level" in config:
                        val_level = config["validation_level"]
                        valid_levels = [
                            v.value for v in FlextConstants.Config.ValidationLevel
                        ]
                        if val_level not in valid_levels:
                            return FlextResult[None].fail(
                                f"Invalid validation_level: {val_level}",
                            )

                    # Update configuration
                    self._config.update(config)
                    return FlextResult[None].ok(None)

                except Exception as e:
                    return FlextResult[None].fail(f"Handler configuration failed: {e}")

            def handle(self, request: object) -> FlextResult[object]:
                """Handle request with efficient metrics and state management.

                Provides production-ready request processing with:
                    - Thread-safe metrics collection
                    - State management throughout request lifecycle
                    - Performance monitoring with timing
                    - Error handling with FlextResult patterns

                Args:
                    request: Request data to process

                Returns:
                    FlextResult[object]: Success with result or failure with error

                """
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
                """Process request - Template method pattern for extensibility.

                Override this method in subclasses to implement specific processing logic.
                Default implementation returns the request unchanged as success.

                Args:
                    request: Request data to process

                Returns:
                    FlextResult[object]: Processing result

                """
                return FlextResult[object].ok(request)

            def can_handle(self, message_type: type) -> bool:
                """Check if handler can process given message type.

                Args:
                    message_type: Type of message to evaluate

                Returns:
                    bool: True if handler can process this message type

                """
                _ = message_type  # Base handler accepts all message types
                return True

            def get_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get efficient handler metrics with thread safety.

                Returns:
                    FlextHandlers.Types.HandlerTypes.Metrics: Current metrics snapshot

                """
                with FlextHandlers.thread_safe_operation():
                    return dict(self._metrics)

            def reset_metrics(self) -> None:
                """Reset all handler metrics to initial state."""
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
                """Get current handler configuration with system information.

                Returns:
                    FlextResult containing current handler configuration with runtime state

                """
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
                """Create environment-specific handler configuration.

                Args:
                    environment: Target environment for handler configuration

                Returns:
                    FlextResult containing environment-optimized handler configuration

                """
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
                """Optimize handler performance based on configuration.

                Args:
                    config: Performance optimization configuration dictionary

                Returns:
                    FlextResult containing optimized handler configuration

                """
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
            """Handler that validates requests before processing using validator functions."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
                validators: list[FlextProtocols.Foundation.Validator[object]]
                | None = None,
            ) -> None:
                """Initialize with optional validators.

                Args:
                    name: Optional handler name
                    validators: List of validator protocols to apply

                """
                super().__init__(name)
                self._validators: Final[
                    list[FlextProtocols.Foundation.Validator[object]]
                ] = validators or []

            @override
            def handle(self, request: object) -> FlextResult[object]:
                """Handle with validation before processing.

                Validates request using all configured validators before
                delegating to parent handler for processing.

                Args:
                    request: Request data to validate and process

                Returns:
                    FlextResult[object]: Validation and processing result

                """
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
                """Validate request using configured validators.

                Args:
                    request: Request data to validate

                Returns:
                    FlextResult[None]: Validation result

                """
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
                """Add validator to the validation chain.

                Args:
                    validator: Validator protocol to add

                """
                self._validators.append(validator)

        class AuthorizingHandler(BasicHandler):
            """Handler that checks authorization before processing requests."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
                authorization_check: FlextHandlers.Types.HandlerTypes.AuthorizerFunction
                | None = None,
            ) -> None:
                """Initialize with optional authorization function.

                Args:
                    name: Optional handler name
                    authorization_check: Authorization function, defaults to allow-all

                """
                super().__init__(name)
                self._authorization_check: FlextHandlers.Types.HandlerTypes.AuthorizerFunction = (
                    authorization_check or self._default_authorization
                )

            @override
            def handle(self, request: object) -> FlextResult[object]:
                """Handle with authorization check before processing.

                Args:
                    request: Request data to authorize and process

                Returns:
                    FlextResult[object]: Authorization and processing result

                """
                # Check authorization first
                if not self._authorization_check(request):
                    return FlextResult[object].fail(
                        "Authorization failed for request",
                        error_code=FlextConstants.Errors.AUTHORIZATION_DENIED,
                    )

                # Delegate to parent handler if authorized
                return super().handle(request)

            def _default_authorization(self, _request: object) -> bool:
                """Default authorization - allows all requests.

                Args:
                    _request: Request data (unused in default implementation)

                Returns:
                    bool: True (allow all)

                """
                return True

        class MetricsHandler(BasicHandler):
            """Handler with enhanced metrics including request/response sizes and error types."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
            ) -> None:
                """Initialize with enhanced metrics infrastructure.

                Args:
                    name: Optional handler name

                """
                super().__init__(name)
                self._error_types: FlextHandlers.Types.HandlerTypes.ErrorMap = {}
                self._request_sizes: FlextHandlers.Types.HandlerTypes.SizeList = []
                self._response_sizes: FlextHandlers.Types.HandlerTypes.SizeList = []
                self._peak_memory_usage: FlextHandlers.Types.HandlerTypes.Counter = 0

            @override
            def handle(self, request: object) -> FlextResult[object]:
                """Handle with enhanced metrics collection.

                Args:
                    request: Request data to process with metrics

                Returns:
                    FlextResult[object]: Processing result with metrics

                """
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
                """Get enhanced metrics including base metrics.

                Returns:
                    FlextHandlers.Types.HandlerTypes.Metrics: Complete metrics data

                """
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
            """Specialized handler for domain events following CQRS patterns."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
            ) -> None:
                """Initialize event handler.

                Args:
                    name: Optional handler name, defaults to EventHandler

                """
                self._name: FlextHandlers.Types.HandlerTypes.Name = (
                    name or "EventHandler"
                )
                self._events_processed: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._event_types: FlextHandlers.Types.HandlerTypes.ErrorMap = {}

            def handle_event(self, event: object) -> FlextResult[None]:
                """Handle domain event with type tracking.

                Args:
                    event: Domain event to process

                Returns:
                    FlextResult[None]: Event processing result

                """
                with FlextHandlers.thread_safe_operation():
                    self._events_processed += 1
                    event_type = type(event).__name__
                    self._event_types[event_type] = (
                        self._event_types.get(event_type, 0) + 1
                    )

                return self._process_event(event)

            def _process_event(self, _event: object) -> FlextResult[None]:
                """Process event - Template method for extensibility.

                Args:
                    _event: Event to process

                Returns:
                    FlextResult[None]: Processing result

                """
                return FlextResult[None].ok(None)

            def get_event_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get event-specific metrics.

                Returns:
                    FlextHandlers.Types.HandlerTypes.Metrics: Event metrics

                """
                with FlextHandlers.thread_safe_operation():
                    return {
                        "events_processed": self._events_processed,
                        "event_types": dict(self._event_types),
                    }

    # =========================================================================
    # CQRS - Command/Query responsibility segregation handlers
    # =========================================================================

    class CQRS:
        """CQRS pattern implementations for command and query separation.

        Provides production-ready CQRS infrastructure with:
            - Command bus for write operations
            - Query bus for read operations
            - Handler registration and routing
            - Performance metrics and monitoring
        """

        class CommandHandler[TCommand, TResult](ABC):
            """Abstract command handler for CQRS write operations."""

            @abstractmethod
            def handle_command(self, command: TCommand) -> FlextResult[TResult]:
                """Handle command execution.

                Args:
                    command: Command to execute

                Returns:
                    FlextResult[TResult]: Command execution result

                """
                ...

            @abstractmethod
            def can_handle(self, command_type: type) -> bool:
                """Check if handler can execute command type.

                Args:
                    command_type: Type of command to check

                Returns:
                    bool: True if handler can execute this command type

                """

        class QueryHandler[TQuery, TResult](ABC):
            """Abstract query handler for CQRS read operations."""

            @abstractmethod
            def handle_query(self, query: TQuery) -> FlextResult[TResult]:
                """Handle query execution.

                Args:
                    query: Query to execute

                Returns:
                    FlextResult[TResult]: Query execution result

                """

        class CommandBus:
            """Command bus for routing commands to registered handlers."""

            def __init__(self) -> None:
                """Initialize command bus with metrics infrastructure."""
                self._handlers: dict[type, object] = {}
                self._commands_processed: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._successful_commands: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._failed_commands: FlextHandlers.Types.HandlerTypes.Counter = 0

            def register(
                self,
                command_type: type,
                handler: object,
            ) -> FlextResult[None]:
                """Register command handler for specific command type.

                Args:
                    command_type: Type of command to handle
                    handler: Handler instance for the command

                Returns:
                    FlextResult[None]: Registration result

                """
                with FlextHandlers.thread_safe_operation():
                    self._handlers[command_type] = handler
                    return FlextResult[None].ok(None)

            def send(self, command: object) -> FlextResult[object]:
                """Send command to registered handler.

                Args:
                    command: Command instance to execute

                Returns:
                    FlextResult[object]: Command execution result

                """
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
                """Get command bus metrics.

                Returns:
                    FlextHandlers.Types.HandlerTypes.Metrics: Bus metrics

                """
                with FlextHandlers.thread_safe_operation():
                    return {
                        "commands_processed": self._commands_processed,
                        "successful_commands": self._successful_commands,
                        "failed_commands": self._failed_commands,
                        "registered_handlers": len(self._handlers),
                    }

        class QueryBus:
            """Query bus for routing queries to registered handlers."""

            def __init__(self) -> None:
                """Initialize query bus with metrics infrastructure."""
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
                """Register query handler for specific query type.

                Args:
                    query_type: Type of query to handle
                    handler: Handler instance for the query

                Returns:
                    FlextResult[None]: Registration result

                """
                with FlextHandlers.thread_safe_operation():
                    self._handlers[query_type] = handler
                    return FlextResult[None].ok(None)

            def send(self, query: object) -> FlextResult[object]:
                """Send query to registered handler.

                Args:
                    query: Query instance to execute

                Returns:
                    FlextResult[object]: Query execution result

                """
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
                """Get query bus metrics.

                Returns:
                    FlextHandlers.Types.HandlerTypes.Metrics: Bus metrics

                """
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
        """Enterprise design pattern implementations for handler composition.

        Provides production-ready implementations of common enterprise patterns:
            - Chain of Responsibility for request processing pipelines
            - Pipeline for sequential data transformation
            - Registry for handler lifecycle management
        """

        class HandlerChain:
            """Chain of Responsibility pattern with performance monitoring."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
            ) -> None:
                """Initialize handler chain.

                Args:
                    name: Optional chain name, defaults to HandlerChain

                """
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
                """Add handler to the processing chain.

                Args:
                    handler: Handler to add to chain

                Returns:
                    FlextResult[None]: Addition result

                """
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
                """Execute chain of handlers until one handles the request.

                Args:
                    request: Request data to process through chain

                Returns:
                    FlextResult[object]: Processing result from first applicable handler

                """
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
                """Get chain performance metrics.

                Returns:
                    FlextHandlers.Types.HandlerTypes.Metrics: Chain metrics

                """
                with FlextHandlers.thread_safe_operation():
                    return {
                        "chain_executions": self._chain_executions,
                        "successful_chains": self._successful_chains,
                        "handler_count": len(self._handlers),
                        "handler_performance": dict(self._handler_performance),
                    }

        class Pipeline:
            """Pipeline pattern for sequential data transformation."""

            def __init__(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name | None = None,
            ) -> None:
                """Initialize processing pipeline.

                Args:
                    name: Optional pipeline name, defaults to Pipeline

                """
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
                """Add processing stage to pipeline.

                Args:
                    stage: Processing function to add to pipeline

                Returns:
                    FlextResult[None]: Addition result

                """
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
                """Process data through all pipeline stages sequentially.

                Args:
                    data: Data to process through pipeline

                Returns:
                    FlextResult[object]: Final processing result

                """
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
                """Get pipeline performance metrics.

                Returns:
                    FlextHandlers.Types.HandlerTypes.Metrics: Pipeline metrics

                """
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
        """Handler lifecycle and registry management infrastructure.

        Provides production-ready handler management with:
            - Handler registration and discovery
            - Lifecycle management
            - Performance monitoring
            - Resource cleanup
        """

        class HandlerRegistry:
            """Registry for handler lifecycle management and discovery."""

            def __init__(self) -> None:
                """Initialize handler registry with management infrastructure."""
                self._handlers: dict[FlextHandlers.Types.HandlerTypes.Name, object] = {}
                self._registrations: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._lookups: FlextHandlers.Types.HandlerTypes.Counter = 0
                self._successful_lookups: FlextHandlers.Types.HandlerTypes.Counter = 0

            def register(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name,
                handler: object,
            ) -> FlextResult[object]:
                """Register handler in the registry.

                Args:
                    name: Unique handler identifier
                    handler: Handler instance to register

                Returns:
                    FlextResult[object]: Registration result with registered handler

                """
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
                """Get handler by name from registry.

                Args:
                    name: Handler identifier to lookup

                Returns:
                    FlextResult[object]: Retrieved handler or failure

                """
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
                """Get all registered handlers.

                Returns:
                    dict[str, object]: Copy of all registered handlers

                """
                with FlextHandlers.thread_safe_operation():
                    return dict(self._handlers)

            def unregister(
                self,
                name: FlextHandlers.Types.HandlerTypes.Name,
            ) -> FlextResult[None]:
                """Unregister handler from registry.

                Args:
                    name: Handler identifier to unregister

                Returns:
                    FlextResult[None]: Unregistration result

                """
                with FlextHandlers.thread_safe_operation():
                    if name in self._handlers:
                        del self._handlers[name]
                        return FlextResult[None].ok(None)

                return FlextResult[None].fail(
                    f"Handler '{name}' not found for unregistration",
                    error_code=FlextConstants.Errors.RESOURCE_NOT_FOUND,
                )

            def get_registry_metrics(self) -> FlextHandlers.Types.HandlerTypes.Metrics:
                """Get registry performance and usage metrics.

                Returns:
                    FlextHandlers.Types.HandlerTypes.Metrics: Registry metrics

                """
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
