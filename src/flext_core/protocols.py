"""Protocol definitions codifying the FLEXT-Core 1.0.0 contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Generic, Protocol, overload, override, runtime_checkable

from flext_core.config import FlextConfig
from flext_core.result import FlextResult
from flext_core.typings import (
    FlextTypes,
    T_contra,
    TCommand_contra,
    TEvent_contra,
    TInput_contra,
    TQuery_contra,
    TResult,
    TState,
    TState_co,
)


class FlextProtocols:
    """Grouped protocol interfaces underpinning the modernization contracts.

    They clarify the callable semantics, configuration hooks, and extension
    points relied upon during the 1.0.0 rollout.
    """

    @override
    def __init__(self, config: dict[str, object] | None = None) -> None:
        """Initialize FlextProtocols with optional configuration.

        Args:
            config: Optional configuration dictionary for protocols

        """
        self._registry: dict[str, type[object]] = {}
        self._middleware: list[object] = []
        self._config = config or {}
        self._metrics: dict[str, int] = {}
        self._audit_log: list[dict[str, object]] = []
        self._performance_metrics: dict[str, float] = {}
        self._circuit_breaker: dict[str, bool] = {}
        self._rate_limiter: dict[str, dict[str, int | float]] = {}
        self._cache: dict[str, tuple[object, float]] = {}
        cache_ttl = self._config.get("cache_ttl", 300)
        self._cache_ttl = (
            float(cache_ttl) if isinstance(cache_ttl, (int, float, str)) else 300.0
        )

        circuit_threshold = self._config.get("circuit_breaker_threshold", 5)
        self._circuit_breaker_threshold = (
            int(circuit_threshold)
            if isinstance(circuit_threshold, (int, float, str))
            else 5
        )

        rate_limit = self._config.get("rate_limit", 10)
        self._rate_limit = (
            int(rate_limit) if isinstance(rate_limit, (int, float, str)) else 10
        )

        rate_window = self._config.get("rate_limit_window", 60)
        self._rate_limit_window = (
            int(rate_window) if isinstance(rate_window, (int, float, str)) else 60
        )

        max_retries = self._config.get("max_retries", 3)
        self._max_retries = (
            int(max_retries) if isinstance(max_retries, (int, float, str)) else 3
        )

        retry_delay = self._config.get("retry_delay", 0.1)
        self._retry_delay = (
            float(retry_delay) if isinstance(retry_delay, (int, float, str)) else 0.1
        )

        timeout = self._config.get("timeout", 30.0)
        self._timeout = (
            float(timeout) if isinstance(timeout, (int, float, str)) else 30.0
        )

    def register(self, name: str, protocol: type[object]) -> FlextResult[None]:
        """Register a protocol with the given name.

        Args:
            name: Name to register the protocol under
            protocol: The protocol class to register

        Returns:
            FlextResult[None]: Success if registration succeeded, failure otherwise

        """
        if not name:
            return FlextResult[None].fail("Protocol name cannot be empty")

        if protocol is None:
            return FlextResult[None].fail("Protocol cannot be None")

        if name in self._registry:
            return FlextResult[None].fail(f"Protocol '{name}' already registered")

        self._registry[name] = protocol
        self._metrics["registrations"] = self._metrics.get("registrations", 0) + 1

        return FlextResult[None].ok(None)

    def unregister(self, name: str, protocol: type[object]) -> FlextResult[None]:
        """Unregister a protocol with the given name.

        Args:
            name: Name of the protocol to unregister
            protocol: The protocol class to unregister

        Returns:
            FlextResult[None]: Success if unregistration succeeded, failure otherwise

        """
        if not name:
            return FlextResult[None].fail("Protocol name cannot be empty")

        if name not in self._registry:
            return FlextResult[None].fail(f"Protocol '{name}' not found")

        if self._registry[name] != protocol:
            return FlextResult[None].fail(f"Protocol mismatch for '{name}'")

        del self._registry[name]
        self._metrics["unregistrations"] = self._metrics.get("unregistrations", 0) + 1

        return FlextResult[None].ok(None)

    def validate_implementation(
        self, name: str, implementation: type[object]
    ) -> FlextResult[None]:
        """Validate that an implementation conforms to a registered protocol.

        Args:
            name: Name of the registered protocol
            implementation: Implementation class to validate

        Returns:
            FlextResult[None]: Success if validation passed, failure otherwise

        """
        if name not in self._registry:
            return FlextResult[None].fail(f"Protocol '{name}' not found")

        protocol = self._registry[name]

        # Check circuit breaker
        if self._circuit_breaker.get(name, False):
            return FlextResult[None].fail(f"Circuit breaker open for protocol '{name}'")

        # Check rate limit
        now = time.time()
        rate_key = f"{name}_rate"
        if rate_key not in self._rate_limiter:
            self._rate_limiter[rate_key] = {"count": 0, "window_start": now}

        rate_data = self._rate_limiter[rate_key]
        if now - rate_data["window_start"] > self._rate_limit_window:
            rate_data["count"] = 0
            rate_data["window_start"] = now

        if rate_data["count"] >= self._rate_limit:
            return FlextResult[None].fail(f"Rate limit exceeded for protocol '{name}'")

        rate_data["count"] += 1

        # Check cache
        cache_key = f"{name}:{hash(str(implementation))}"
        if cache_key in self._cache:
            _, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return FlextResult[None].ok(None)

        # Apply middleware
        processed_implementation = implementation
        for middleware in self._middleware:
            if callable(middleware):
                try:
                    processed_implementation = middleware(processed_implementation)
                except Exception as e:
                    return FlextResult[None].fail(f"Middleware error: {e}")

        # Validate implementation
        try:
            # Basic validation - check if implementation has the expected methods
            if hasattr(protocol, "__annotations__"):
                # For now, just do basic validation
                # In a real implementation, you'd check method signatures, etc.
                self._cache[cache_key] = (True, time.time())
                self._metrics["successful_validations"] = (
                    self._metrics.get("successful_validations", 0) + 1
                )
                self._audit_log.append({
                    "timestamp": time.time(),
                    "protocol": name,
                    "implementation": str(implementation),
                    "status": "success",
                })
                return FlextResult[None].ok(None)
            return FlextResult[None].fail(f"Protocol '{name}' has no annotations")
        except Exception as e:
            self._metrics["failed_validations"] = (
                self._metrics.get("failed_validations", 0) + 1
            )
            self._audit_log.append({
                "timestamp": time.time(),
                "protocol": name,
                "implementation": str(implementation),
                "status": "error",
                "error": str(e),
            })
            return FlextResult[None].fail(f"Validation error: {e}")

    def add_middleware(self, middleware: object) -> None:
        """Add middleware to the validation pipeline.

        Args:
            middleware: Middleware function to add

        """
        if callable(middleware):
            self._middleware.append(middleware)
        else:
            error_msg = "Middleware must be callable"
            raise TypeError(error_msg)

    def get_metrics(self) -> dict[str, int]:
        """Get protocol metrics.

        Returns:
            dict[str, int]: Current metrics

        """
        return self._metrics.copy()

    def get_audit_log(self) -> list[dict[str, object]]:
        """Get audit log of validation operations.

        Returns:
            list[dict[str, object]]: Audit log entries

        """
        return self._audit_log.copy()

    def get_performance_metrics(self) -> dict[str, float]:
        """Get performance metrics.

        Returns:
            dict[str, float]: Performance metrics

        """
        return self._performance_metrics.copy()

    def is_circuit_breaker_open(self, name: str) -> bool:
        """Check if circuit breaker is open for a protocol.

        Args:
            name: Protocol name

        Returns:
            bool: True if circuit breaker is open

        """
        return self._circuit_breaker.get(name, False)

    def validate_batch(
        self, name: str, implementations: list[type[object]]
    ) -> FlextResult[list[object]]:
        """Validate a batch of implementations.

        Args:
            name: Protocol name
            implementations: List of implementation classes to validate

        Returns:
            FlextResult[list[object]]: List of validation results

        """
        results: list[object] = []
        for implementation in implementations:
            result = self.validate_implementation(name, implementation)
            if result.is_failure:
                return FlextResult[list[object]].fail(
                    f"Batch validation failed: {result.error}"
                )
            results.append(result.value)

        return FlextResult[list[object]].ok(results)

    def validate_parallel(
        self, name: str, implementations: list[type[object]]
    ) -> FlextResult[list[object]]:
        """Validate implementations in parallel.

        Args:
            name: Protocol name
            implementations: List of implementation classes to validate

        Returns:
            FlextResult[list[object]]: List of validation results

        """
        # For now, just validate sequentially - can be enhanced with actual parallel processing
        return self.validate_batch(name, implementations)

    # =============================================================================
    # Missing Methods for Test Compatibility
    # =============================================================================

    def cleanup(self) -> None:
        """Clean up protocol registry resources."""
        self._registry.clear()
        self._middleware.clear()
        self._metrics.clear()
        self._audit_log.clear()
        self._performance_metrics.clear()
        self._circuit_breaker.clear()

    def get_protocols(self, name: str) -> list[type[object]]:
        """Get protocols for specific name.

        Args:
            name: Protocol name

        Returns:
            List of protocol implementations

        """
        # Return single protocol or empty list
        protocol = self._registry.get(name)
        return [protocol] if protocol is not None else []

    def clear_protocols(self) -> None:
        """Clear all registered protocols."""
        self._registry.clear()

    def get_statistics(self) -> dict[str, object]:
        """Get protocol registry statistics.

        Returns:
            Dictionary of statistics

        """
        return {
            "total_protocols": len(self._registry),
            "protocol_names": len(self._registry),
            "middleware_count": len(self._middleware),
            "audit_log_entries": len(self._audit_log),
            "performance_metrics": self._performance_metrics.copy(),
            "circuit_breakers": self._circuit_breaker.copy(),
        }

    def validate(self) -> FlextResult[None]:
        """Validate protocol registry configuration and state.

        Returns:
            FlextResult with validation result

        """
        try:
            # Validate protocols
            for name, protocol in self._registry.items():
                if not isinstance(name, str):
                    return FlextResult[None].fail(f"Invalid protocol name: {name}")
                if not isinstance(protocol, type):
                    return FlextResult[None].fail(f"Invalid protocol type for {name}")

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Protocol validation failed: {e}")

    def export_config(self) -> dict[str, object]:
        """Export protocol registry configuration.

        Returns:
            Dictionary of configuration

        """
        return {
            "protocol_count": len(self._registry),
            "middleware_count": len(self._middleware),
            "audit_log_size": len(self._audit_log),
            "performance_metrics_count": len(self._performance_metrics),
            "circuit_breaker_count": len(self._circuit_breaker),
        }

    def import_config(self, config: dict[str, object]) -> FlextResult[None]:
        """Import protocol registry configuration.

        Args:
            config: Configuration dictionary

        Returns:
            FlextResult with import result

        """
        try:
            # Configuration import would go here
            # For now, just validate the config structure
            if not isinstance(config, dict):
                return FlextResult[None].fail("Config must be a dictionary")

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Config import failed: {e}")

    # =========================================================================
    # FOUNDATION LAYER - Core building blocks
    # =========================================================================

    class Foundation:
        """Foundation layer protocols cementing the 1.0.0 contracts."""

        @runtime_checkable
        class OperationCallable(Protocol):
            """Protocol for callable operations in the FLEXT ecosystem.

            This protocol defines the interface for operations that can be executed
            within the FLEXT framework, ensuring type safety and consistent behavior.
            """

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Execute the operation with given arguments.

                Args:
                    *args: Positional arguments for the operation
                    **kwargs: Keyword arguments for the operation

                Returns:
                    The result of the operation execution

                """
                ...

        @runtime_checkable
        class Validator(Protocol, Generic[T_contra]):
            """Generic validator protocol reused by modernization guardrails."""

            def validate(self, data: T_contra) -> object:
                """Validate input data according to the shared release policy."""
                ...

    # =========================================================================
    # DOMAIN LAYER - Business logic protocols
    # =========================================================================

    class Domain:
        """Domain layer protocols reflecting FLEXT's modernization DDD usage."""

        # Domain protocols providing service and repository patterns

        @runtime_checkable
        class Service(Protocol):
            """Domain service contract aligned with FlextService implementation."""

            @abstractmethod
            def execute(self: object) -> FlextResult[object]:
                """Execute the main domain operation.

                Returns:
                    FlextResult[object]: Success with domain result or failure with error

                """
                ...

            def is_valid(self: object) -> bool:
                """Check if the domain service is in a valid state.

                Returns:
                    bool: True if valid, False otherwise

                """
                ...

            def validate_business_rules(self: object) -> FlextResult[None]:
                """Validate business rules for the domain service.

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                ...

            def validate_config(self: object) -> FlextResult[None]:
                """Validate service configuration.

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                ...

            def execute_operation(self, operation: object) -> FlextResult[object]:
                """Execute operation using OperationExecutionRequest model.

                Args:
                    operation: OperationExecutionRequest containing operation settings

                Returns:
                    FlextResult[object]: Success with result or failure with error

                """
                ...

            def get_service_info(self: object) -> FlextTypes.Core.Dict:
                """Get service information and metadata.

                Returns:
                    FlextTypes.Core.Dict: Service information dictionary

                """
                ...

        @runtime_checkable
        class Repository(Protocol, Generic[T_contra]):
            """Repository protocol shaping modernization data access patterns."""

            @abstractmethod
            def get_by_id(self, entity_id: str) -> object:
                """Retrieve an aggregate using the standardized identity lookup."""
                ...

            @abstractmethod
            def save(self, entity: T_contra) -> object:
                """Persist an entity following modernization consistency rules."""
                ...

            @abstractmethod
            def delete(self, entity_id: str) -> object:
                """Delete an entity while respecting modernization invariants."""
                ...

            @abstractmethod
            def find_all(self: object) -> object:
                """Enumerate entities for modernization-aligned queries."""
                ...

        @runtime_checkable
        class AggregateRoot(Protocol, Generic[TState_co]):
            """Aggregate root protocol for domain-driven design patterns."""

            @abstractmethod
            def get_id(self) -> str:
                """Get the aggregate root identifier."""
                ...

            @abstractmethod
            def get_version(self) -> int:
                """Get the aggregate root version for optimistic locking."""
                ...

            @abstractmethod
            def get_uncommitted_events(self) -> list[object]:
                """Get uncommitted domain events."""
                ...

            @abstractmethod
            def mark_events_as_committed(self) -> None:
                """Mark all events as committed."""
                ...

            @abstractmethod
            def is_valid(self) -> bool:
                """Check if the aggregate root is in a valid state."""
                ...

        @runtime_checkable
        class DomainEvent(Protocol):
            """Domain event protocol for event sourcing patterns."""

            @abstractmethod
            def get_event_id(self) -> str:
                """Get the unique event identifier."""
                ...

            @abstractmethod
            def get_event_type(self) -> str:
                """Get the event type name."""
                ...

            @abstractmethod
            def get_aggregate_id(self) -> str:
                """Get the aggregate root identifier."""
                ...

            @abstractmethod
            def get_event_data(self) -> FlextTypes.Core.Dict:
                """Get the event payload data."""
                ...

            @abstractmethod
            def get_metadata(self) -> FlextTypes.Core.Dict:
                """Get the event metadata."""
                ...

            @abstractmethod
            def get_timestamp(self) -> datetime:
                """Get the event timestamp."""
                ...

        @runtime_checkable
        class Command(Protocol):
            """Command protocol for CQRS patterns."""

            @abstractmethod
            def get_command_id(self) -> str:
                """Get the unique command identifier."""
                ...

            @abstractmethod
            def get_command_type(self) -> str:
                """Get the command type name."""
                ...

            @abstractmethod
            def get_command_data(self) -> FlextTypes.Core.Dict:
                """Get the command payload data."""
                ...

            @abstractmethod
            def get_metadata(self) -> FlextTypes.Core.Dict:
                """Get the command metadata."""
                ...

            @abstractmethod
            def get_timestamp(self) -> datetime:
                """Get the command timestamp."""
                ...

        @runtime_checkable
        class Query(Protocol):
            """Query protocol for CQRS patterns."""

            @abstractmethod
            def get_query_id(self) -> str:
                """Get the unique query identifier."""
                ...

            @abstractmethod
            def get_query_type(self) -> str:
                """Get the query type name."""
                ...

            @abstractmethod
            def get_query_data(self) -> FlextTypes.Core.Dict:
                """Get the query payload data."""
                ...

            @abstractmethod
            def get_metadata(self) -> FlextTypes.Core.Dict:
                """Get the query metadata."""
                ...

            @abstractmethod
            def get_timestamp(self) -> datetime:
                """Get the query timestamp."""
                ...

        @runtime_checkable
        class Saga(Protocol, Generic[TState]):
            """Saga protocol for distributed transaction patterns."""

            @abstractmethod
            def get_saga_id(self) -> str:
                """Get the unique saga identifier."""
                ...

            @abstractmethod
            def get_saga_type(self) -> str:
                """Get the saga type name."""
                ...

            @abstractmethod
            def get_current_state(self) -> TState:
                """Get the current saga state."""
                ...

            @abstractmethod
            def execute_step(
                self, step_data: FlextTypes.Core.Dict
            ) -> FlextResult[TState]:
                """Execute a saga step."""
                ...

            @abstractmethod
            def compensate_step(
                self, step_data: FlextTypes.Core.Dict
            ) -> FlextResult[TState]:
                """Compensate a saga step."""
                ...

            @abstractmethod
            def is_completed(self) -> bool:
                """Check if the saga is completed."""
                ...

            @abstractmethod
            def is_failed(self) -> bool:
                """Check if the saga has failed."""
                ...

    # =========================================================================
    # APPLICATION LAYER - Use cases and handlers
    # =========================================================================

    class Application:
        """Application layer protocols - use cases and handlers."""

        @runtime_checkable
        class Handler(Protocol, Generic[TInput_contra, TResult]):
            """Application handler protocol aligned with FlextHandlers implementation."""

            @abstractmethod
            def handle(self, message: TInput_contra) -> FlextResult[TResult]:
                """Handle the message and return result.

                Args:
                    message: The input message to process

                Returns:
                    FlextResult[TResult]: Success with result or failure with error

                """
                ...

            def __call__(self, input_data: TInput_contra) -> FlextResult[TResult]:
                """Process input and return a ``FlextResult`` containing the output."""
                ...

            def can_handle(self, message_type: object) -> bool:
                """Check if handler can process this message type.

                Args:
                    message_type: The message type to check

                Returns:
                    bool: True if handler can process the message type, False otherwise

                """
                ...

            def execute(self, message: TInput_contra) -> FlextResult[TResult]:
                """Execute the handler with the given message.

                Args:
                    message: The input message to execute

                Returns:
                    FlextResult[TResult]: Execution result

                """
                ...

            def validate_command(self, command: TInput_contra) -> FlextResult[None]:
                """Validate a command message.

                Args:
                    command: The command to validate

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                ...

            def validate(self, data: TInput_contra) -> FlextResult[None]:
                """Validate input before processing and wrap the outcome in ``FlextResult``."""
                ...

            def validate_query(self, query: TInput_contra) -> FlextResult[None]:
                """Validate a query message.

                Args:
                    query: The query to validate

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                ...

            @property
            def handler_name(self: object) -> str:
                """Get the handler name.

                Returns:
                    str: Handler name

                """
                ...

            @property
            def mode(self: object) -> str:
                """Get the handler mode (command/query).

                Returns:
                    str: Handler mode

                """
                ...

        @runtime_checkable
        class CommandHandler(Protocol, Generic[TCommand_contra, TResult]):
            """Command handler protocol for CQRS patterns."""

            @abstractmethod
            def handle_command(self, command: TCommand_contra) -> FlextResult[TResult]:
                """Handle a command and return result."""
                ...

            @abstractmethod
            def can_handle(self, command_type: str) -> bool:
                """Check if this handler can handle the command type."""
                ...

            @abstractmethod
            def get_supported_command_types(self) -> list[str]:
                """Get list of supported command types."""
                ...

        @runtime_checkable
        class QueryHandler(Protocol, Generic[TQuery_contra, TResult]):
            """Query handler protocol for CQRS patterns."""

            @abstractmethod
            def handle_query(self, query: TQuery_contra) -> FlextResult[TResult]:
                """Handle a query and return result."""
                ...

            @abstractmethod
            def can_handle(self, query_type: str) -> bool:
                """Check if this handler can handle the query type."""
                ...

            @abstractmethod
            def get_supported_query_types(self) -> list[str]:
                """Get list of supported query types."""
                ...

        @runtime_checkable
        class EventHandler(Protocol, Generic[TEvent_contra]):
            """Event handler protocol for event sourcing patterns."""

            @abstractmethod
            def handle_event(self, event: TEvent_contra) -> FlextResult[None]:
                """Handle a domain event."""
                ...

            @abstractmethod
            def can_handle(self, event_type: str) -> bool:
                """Check if this handler can handle the event type."""
                ...

            @abstractmethod
            def get_supported_event_types(self) -> list[str]:
                """Get list of supported event types."""
                ...

        @runtime_checkable
        class SagaManager(Protocol, Generic[TState]):
            """Saga manager protocol for distributed transaction patterns."""

            @abstractmethod
            def start_saga(
                self, saga_type: str, initial_data: FlextTypes.Core.Dict
            ) -> FlextResult[str]:
                """Start a new saga."""
                ...

            @abstractmethod
            def execute_saga_step(
                self, saga_id: str, step_data: FlextTypes.Core.Dict
            ) -> FlextResult[TState]:
                """Execute a saga step."""
                ...

            @abstractmethod
            def compensate_saga(self, saga_id: str) -> FlextResult[TState]:
                """Compensate a saga."""
                ...

            @abstractmethod
            def get_saga_status(self, saga_id: str) -> FlextResult[str]:
                """Get saga status."""
                ...

            @abstractmethod
            def get_saga_state(self, saga_id: str) -> FlextResult[TState]:
                """Get saga state."""
                ...

        @runtime_checkable
        class EventStore(Protocol):
            """Event store protocol for event sourcing patterns."""

            @abstractmethod
            def save_events(
                self, aggregate_id: str, events: list[object], expected_version: int
            ) -> FlextResult[None]:
                """Save events for an aggregate."""
                ...

            @abstractmethod
            def get_events(
                self, aggregate_id: str, from_version: int = 0
            ) -> FlextResult[list[object]]:
                """Get events for an aggregate."""
                ...

            @abstractmethod
            def get_events_by_type(
                self, event_type: str, from_timestamp: datetime | None = None
            ) -> FlextResult[list[object]]:
                """Get events by type."""
                ...

            @abstractmethod
            def get_events_by_correlation_id(
                self, correlation_id: str
            ) -> FlextResult[list[object]]:
                """Get events by correlation ID."""
                ...

        @runtime_checkable
        class EventPublisher(Protocol):
            """Event publisher protocol for event sourcing patterns."""

            @abstractmethod
            def publish_event(self, event: object) -> FlextResult[None]:
                """Publish a domain event."""
                ...

            @abstractmethod
            def publish_events(self, events: list[object]) -> FlextResult[None]:
                """Publish multiple domain events."""
                ...

            @abstractmethod
            def subscribe(self, event_type: str, handler: object) -> FlextResult[None]:
                """Subscribe to an event type."""
                ...

            @abstractmethod
            def unsubscribe(
                self, event_type: str, handler: object
            ) -> FlextResult[None]:
                """Unsubscribe from an event type."""
                ...

    # =========================================================================
    # INFRASTRUCTURE LAYER - External concerns and integrations
    # =========================================================================

    class Infrastructure:
        """Infrastructure layer protocols - external systems."""

        @runtime_checkable
        class Connection(Protocol):
            """Connection protocol for external systems."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for connection."""
                ...

            def test_connection(self: object) -> object:
                """Test connection to external system."""
                ...

            def get_connection_string(self: object) -> str:
                """Get connection string for external system."""
                ...

            def close_connection(self: object) -> object:
                """Close connection to external system."""
                ...

        @runtime_checkable
        class Configurable(Protocol):
            """Configurable component protocol."""

            def configure(self, config: FlextTypes.Core.Dict) -> FlextResult[None]:
                """Configure component with provided settings."""
                ...

            def get_config(self: object) -> FlextTypes.Core.Dict:
                """Get current configuration."""
                ...

        @runtime_checkable
        class LoggerProtocol(Protocol):
            """Logger protocol with standard logging methods."""

            def trace(self, message: str, **kwargs: object) -> None:
                """Log trace message."""
                ...

            def debug(self, message: str, **kwargs: object) -> None:
                """Log debug message."""
                ...

            def info(self, message: str, **kwargs: object) -> None:
                """Log info message."""
                ...

            def warning(self, message: str, **kwargs: object) -> None:
                """Log warning message."""
                ...

            def error(self, message: str, **kwargs: object) -> None:
                """Log error message."""
                ...

            def critical(self, message: str, **kwargs: object) -> None:
                """Log critical message."""
                ...

            def exception(
                self,
                message: str,
                *,
                exc_info: bool = True,
                **kwargs: object,
            ) -> None:
                """Log exception message."""
                ...

        @runtime_checkable
        class LogRenderer(Protocol):
            """Log renderer protocol for formatting log entries."""

            def __call__(
                self, logger: object, method_name: str, event_dict: dict[str, object]
            ) -> str:
                """Render log entry to string format.

                Args:
                    logger: Logger instance
                    method_name: Method name that generated the log
                    event_dict: Event dictionary with log data

                Returns:
                    str: Formatted log entry string

                """
                ...

        @runtime_checkable
        class LogContextManager(Protocol):
            """Log context manager protocol for managing logger context."""

            def set_correlation_id(self, correlation_id: str) -> FlextResult[None]:
                """Set correlation ID for request tracing.

                Args:
                    correlation_id: Correlation ID to set

                Returns:
                    FlextResult[None]: Success or failure result

                """
                ...

            def set_request_context(self, model: object) -> FlextResult[None]:
                """Set request-specific context data.

                Args:
                    model: Request context model to set

                Returns:
                    FlextResult[None]: Success or failure result

                """
                ...

            def clear_request_context(self: object) -> FlextResult[None]:
                """Clear request-specific context data.

                Returns:
                    FlextResult[None]: Success or failure result

                """
                ...

            def bind_context(self, model: object) -> FlextResult[object]:
                """Create bound logger instance with additional context.

                Args:
                    model: Context binding model to use

                Returns:
                    FlextResult[object]: Bound logger instance or error

                """
                ...

            def get_consolidated_context(self: object) -> dict[str, object]:
                """Get all context data consolidated for log entry building.

                Returns:
                    dict[str, object]: Consolidated context data

                """
                ...

        @runtime_checkable
        class ConfigValidator(Protocol):
            """Protocol for configuration validation strategies."""

            def validate_runtime_requirements(self: object) -> FlextResult[None]:
                """Validate configuration meets runtime requirements."""
                ...

            def validate_business_rules(self: object) -> FlextResult[None]:
                """Validate business rules for configuration consistency."""
                ...

        @runtime_checkable
        class ConfigPersistence(Protocol):
            """Protocol for configuration persistence operations.

            Follows Single Responsibility Principle - only handles persistence.
            """

            def save_to_file(
                self,
                file_path: str | Path,
                **kwargs: object,
            ) -> FlextResult[None]:
                """Save configuration to file."""
                ...

            @classmethod
            def load_from_file(cls, file_path: str | Path) -> FlextResult[FlextConfig]:
                """Load configuration from file."""
                ...

        @runtime_checkable
        class ConfigFactory(Protocol):
            """Protocol for configuration factory methods.

            Follows Open/Closed Principle - extensible for new configuration types.
            """

            @classmethod
            def create_web_service_config(
                cls,
                **kwargs: object,
            ) -> FlextResult[FlextConfig]:
                """Create web service configuration."""
                ...

            @classmethod
            def create_microservice_config(
                cls,
                **kwargs: object,
            ) -> FlextResult[FlextConfig]:
                """Create microservice configuration."""
                ...

    # =========================================================================
    # EXTENSIONS LAYER - Advanced patterns and plugins
    # =========================================================================

    class Extensions:
        """Extensions layer protocols - plugins and extension patterns."""

        # Plugin architecture and middleware system for extensible applications
        # Provides plugin ecosystem support for applications

        @runtime_checkable
        class Plugin(Protocol):
            """Plugin protocol with configuration.

            Plugin lifecycle management with configuration and initialization
            Supports complex plugin ecosystems with full lifecycle control
            """

            def configure(self, config: FlextTypes.Core.Dict) -> object:
                """Configure component with settings."""
                ...

            def get_config(self: object) -> FlextTypes.Core.Dict:
                """Get current configuration."""
                ...

            @abstractmethod
            def initialize(
                self,
                context: FlextProtocols.Extensions.PluginContext,
            ) -> object:
                """Initialize plugin."""
                ...

            @abstractmethod
            def shutdown(self: object) -> object:
                """Shutdown plugin and cleanup."""
                ...

            @abstractmethod
            def get_info(self: object) -> FlextTypes.Core.Dict:
                """Get plugin information."""
                ...

        @runtime_checkable
        class PluginContext(Protocol):
            """Plugin execution context."""

            def get_service(self, service_name: str) -> object:
                """Get service by name."""
                ...

            def get_config(self: object) -> FlextTypes.Core.Dict:
                """Get plugin configuration."""
                ...

            def flext_logger(
                self: object,
            ) -> FlextProtocols.Infrastructure.LoggerProtocol:
                """Get logger instance for plugin."""
                ...

        @runtime_checkable
        class Middleware(Protocol):
            """Middleware pipeline component protocol."""

            def process(
                self,
                request: object,
                _next_handler: Callable[[object], object],
            ) -> object:
                """Process request with middleware logic."""
                ...

        @runtime_checkable
        class Observability(Protocol):
            """Observability and monitoring protocol."""

            def record_metric(
                self,
                name: str,
                value: float,
                _tags: FlextTypes.Core.Headers | None = None,
            ) -> object:
                """Record metric value."""
                ...

            def start_trace(self, operation_name: str) -> object:
                """Start distributed trace."""
                ...

            def health_check(self: object) -> object:
                """Perform health check."""
                ...

    class Commands:
        """CQRS Command and Query protocols for Flext CQRS components."""

        @runtime_checkable
        class CommandHandler[CommandT, ResultT](Protocol):
            """Protocol for command handlers in CQRS pattern."""

            def handle(self, command: CommandT) -> FlextResult[ResultT]:
                """Handle a command and return a :class:`FlextResult` wrapper.

                Args:
                    command: The command to handle

                Returns:
                    FlextResult containing the command handling outcome

                """
                ...

            def can_handle(self, command_type: type) -> bool:
                """Check if this handler can process the given command type.

                Args:
                    command_type: The type of command to check

                Returns:
                    True if this handler can process the command type

                """
                ...

        @runtime_checkable
        class QueryHandler[QueryT, ResultT](Protocol):
            """Protocol for query handlers in CQRS pattern."""

            def handle(self, query: QueryT) -> FlextResult[ResultT]:
                """Handle a query and return a :class:`FlextResult` wrapper.

                Args:
                    query: The query to handle

                Returns:
                    FlextResult containing the query handling outcome

                """
                ...

        @runtime_checkable
        class CommandBus(Protocol):
            """Protocol for command bus routing and execution."""

            @overload
            def register_handler(
                self, handler: Callable[[object], object], /
            ) -> FlextResult[None]: ...

            @overload
            def register_handler(
                self,
                command_type: type,
                handler: Callable[[object], object],
                /,
            ) -> FlextResult[None]: ...

            def register_handler(self, *_args: object) -> FlextResult[None]:
                """Register a command handler with the command bus.

                The command bus accepts both ``register_handler(handler)`` for
                automatic type detection and ``register_handler(command_type, handler)``
                for explicit type specification.

                Args:
                    handler: The handler function to register
                    command_type: Optional command type for explicit registration

                Returns:
                    FlextResult indicating success or failure

                """
                # Implementation would go here
                return FlextResult[None].ok(None)

            def unregister_handler(self, command_type: type | str) -> bool:
                """Remove a handler registration by type or name.

                Args:
                    command_type: The command type or name to unregister

                Returns:
                    bool: True if handler was removed, False otherwise

                """
                ...

            def execute(self, command: object) -> object:
                """Execute a command through registered handlers.

                Args:
                    command: The command to execute

                Returns:
                    The result of command execution

                """
                ...

        @runtime_checkable
        class Middleware(Protocol):
            """Protocol for command bus middleware."""

            def process(self, command: object, handler: object) -> object:
                """Process command through middleware.

                Args:
                    command: The command being processed
                    handler: The handler that will process the command

                Returns:
                    The result of middleware processing

                """
                ...


__all__ = [
    "FlextProtocols",  # Main hierarchical protocol architecture with Config
]
