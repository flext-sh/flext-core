"""FlextHandlers - Simplified Handler Management System.

Comprehensive handler implementation following Clean Architecture and SOLID principles.
All handler functionality is organized under the FlextHandlers namespace.

Example Usage:
    >>> from flext_core import FlextHandlers
    >>> # Create handlers directly
    >>> handler = FlextHandlers.Handler()
    >>> validator = FlextHandlers.ValidatingHandler()
    >>> chain = FlextHandlers.HandlerChain()
    >>> registry = FlextHandlers.HandlerRegistry()
    >>> cmd_bus = FlextHandlers.CommandBus()
    >>> query_bus = FlextHandlers.QueryBus()
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final, Generic, Protocol, cast, override

from flext_core.commands import FlextAbstractCommandHandler, FlextCommands
from flext_core.protocols import FlextValidator
from flext_core.result import FlextResult
from flext_core.typings import (
    FlextTypes,
    TInput,
    TOutput,
    TQuery,
    TQueryResult,
)

# Use centralized types from FlextTypes where appropriate
HandlerName = FlextTypes.Handler.HandlerName


# Forward reference protocols to avoid circular issues
class QueryHandlerProtocol(Protocol):
    """Protocol for query handlers."""

    def handle_query(self, query: object) -> FlextResult[object]: ...


class ChainHandlerProtocol(Protocol):
    """Protocol for chain handlers."""

    @property
    def handler_name(self) -> str: ...
    def can_handle(self, message_type: object) -> bool: ...
    def handle(self, request: object) -> FlextResult[object]: ...


# =============================================================================
# FlextHandlers - Simplified Handler Management System
# =============================================================================


class FlextHandlers:
    """Simplified handler management system with direct class access.

    All handler classes are directly accessible without intermediate namespaces.
    """

    # Thread-safe handler management
    _handlers_lock: Final[threading.RLock] = threading.RLock()

    @staticmethod
    @contextmanager
    def thread_safe_operation() -> Iterator[None]:
        """Thread-safe context manager for handler operations."""
        with FlextHandlers._handlers_lock:
            yield

    @staticmethod
    def get_flext_commands_module() -> type:
        """Get FlextCommands module reference."""
        return FlextCommands

    # =========================================================================
    # Abstract Base Classes
    # =========================================================================

    class AbstractHandler(ABC, Generic[TInput, TOutput]):
        """Abstract handler base class."""

        @property
        @abstractmethod
        def handler_name(self) -> str:
            """Get handler name."""

        @abstractmethod
        def handle(self, request: TInput) -> FlextResult[TOutput]:
            """Handle request."""

        @abstractmethod
        def can_handle(self, message_type: object) -> bool:
            """Check capability."""

    # =========================================================================
    # Concrete Handler Implementations
    # =========================================================================

    class Handler:
        """Basic handler implementation."""

        def __init__(self, name: str | None = None) -> None:
            """Initialize handler with optional name."""
            self._handler_name: Final[str] = name or self.__class__.__name__
            self._metrics: FlextTypes.Handler.MetricsData = {
                "requests_processed": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_processing_time": 0.0,
                "total_processing_time": 0.0,
            }

        @property
        def handler_name(self) -> str:
            """Return handler name."""
            return self._handler_name

        def handle(self, request: object) -> FlextResult[object]:
            """Handle request with metrics collection."""
            start_time = time.time()

            try:
                self._metrics["requests_processed"] = (
                    cast("int", self._metrics["requests_processed"]) + 1
                )

                result = self._process_request(request)

                if result.success:
                    self._metrics["successful_requests"] = (
                        cast("int", self._metrics["successful_requests"]) + 1
                    )
                else:
                    self._metrics["failed_requests"] = (
                        cast("int", self._metrics["failed_requests"]) + 1
                    )

                return result

            finally:
                processing_time = time.time() - start_time
                self._metrics["total_processing_time"] = (
                    cast("float", self._metrics["total_processing_time"])
                    + processing_time
                )

                requests_count = cast("int", self._metrics["requests_processed"])
                if requests_count > 0:
                    total_time = self._metrics["total_processing_time"]
                    if isinstance(total_time, (int, float)):
                        self._metrics["average_processing_time"] = (
                            total_time / requests_count
                        )
                    else:
                        self._metrics["average_processing_time"] = 0.0

        def _process_request(self, request: object) -> FlextResult[object]:
            """Process request - Template method pattern."""
            return FlextResult[object].ok(request)

        def can_handle(self, message_type: object) -> bool:
            """Check if handler can process message type."""
            return message_type is not None

        def get_metrics(self) -> FlextTypes.Handler.MetricsData:
            """Get handler metrics."""
            return dict(self._metrics)

    class ValidatingHandler(Handler):
        """Handler with built-in validation capabilities."""

        def __init__(
            self,
            name: str | None = None,
            validators: list[FlextValidator[object]] | None = None,
        ) -> None:
            """Initialize with optional validators."""
            super().__init__(name)
            self._validators: Final[list[FlextValidator[object]]] = validators or []

        @override
        def handle(self, request: object) -> FlextResult[object]:
            """Handle with validation."""
            validation_result = self.validate(request)
            if validation_result.is_failure:
                return FlextResult[object].fail(
                    f"Validation failed: {validation_result.error}"
                )
            return super().handle(request)

        def validate(self, request: object) -> FlextResult[None]:
            """Validate request using configured validators."""
            for validator in self._validators:
                if hasattr(validator, "validate"):
                    result = validator.validate(request)
                    if hasattr(result, "is_failure") and result.is_failure:
                        return result
                elif callable(validator):
                    try:
                        if not validator(request):
                            return FlextResult[None].fail("Validation failed")
                    except Exception as e:
                        return FlextResult[None].fail(f"Validation error: {e}")

            return FlextResult[None].ok(None)

        def add_validator(self, validator: FlextValidator[object]) -> None:
            """Add validator to the chain."""
            self._validators.append(validator)

    class AuthorizingHandler(Handler):
        """Handler with authorization capabilities."""

        def __init__(
            self,
            name: str | None = None,
            authorization_check: Callable[[object], bool] | None = None,
        ) -> None:
            """Initialize with optional authorization check."""
            super().__init__(name)
            self._authorization_check = (
                authorization_check or self._default_authorization
            )

        @override
        def handle(self, request: object) -> FlextResult[object]:
            """Handle with authorization."""
            if not self._authorization_check(request):
                return FlextResult[object].fail("Authorization failed")
            return super().handle(request)

        def _default_authorization(self, _request: object) -> bool:
            """Default authorization - always allow."""
            return True

    class MetricsHandler(Handler):
        """Handler with enhanced metrics collection."""

        def __init__(self, name: str | None = None) -> None:
            """Initialize with enhanced metrics."""
            super().__init__(name)
            self._error_types: FlextTypes.Handler.ErrorCounterMap = {}
            self._request_sizes: FlextTypes.Handler.SizeList = []
            self._response_sizes: FlextTypes.Handler.SizeList = []
            self._peak_memory_usage: FlextTypes.Handler.CounterMetric = 0

        @override
        def handle(self, request: object) -> FlextResult[object]:
            """Handle with enhanced metrics collection."""
            request_size = len(str(request)) if request else 0
            self._request_sizes.append(request_size)

            result = super().handle(request)

            if result.is_failure:
                error_type = type(result.error).__name__
                self._error_types[error_type] = self._error_types.get(error_type, 0) + 1
            else:
                response_size = len(str(result.value)) if result.value else 0
                self._response_sizes.append(response_size)

            return result

        @override
        def get_metrics(self) -> FlextTypes.Handler.MetricsData:
            """Get enhanced metrics including base metrics."""
            base_metrics = super().get_metrics()
            enhanced_metrics: FlextTypes.Handler.MetricsData = {
                "error_types": self._error_types,
                "request_sizes": self._request_sizes,
                "response_sizes": self._response_sizes,
                "peak_memory_usage": self._peak_memory_usage,
            }

            if self._request_sizes:
                enhanced_metrics["average_request_size"] = sum(
                    self._request_sizes
                ) / len(self._request_sizes)

            if self._response_sizes:
                enhanced_metrics["average_response_size"] = sum(
                    self._response_sizes
                ) / len(self._response_sizes)

            return {**base_metrics, **enhanced_metrics}

    # =========================================================================
    # CQRS Handler Implementations
    # =========================================================================

    class CommandHandler(ABC, Generic[TInput, TOutput]):
        """Abstract command handler for CQRS pattern."""

        @abstractmethod
        def handle_command(self, command: TInput) -> FlextResult[TOutput]:
            """Handle command."""

        @abstractmethod
        def can_handle(self, message_type: object) -> bool:
            """Check command handling capability."""

    class QueryHandler(ABC, Generic[TQuery, TQueryResult]):
        """Abstract query handler for CQRS pattern."""

        @abstractmethod
        def handle_query(self, query: TQuery) -> FlextResult[TQueryResult]:
            """Handle query."""

    class EventHandler:
        """Event handler for domain events."""

        def __init__(self, name: str | None = None) -> None:
            """Initialize event handler."""
            self._name = name or "EventHandler"
            self._events_processed: FlextTypes.Handler.CounterMetric = 0
            self._event_types: FlextTypes.Handler.ErrorCounterMap = {}

        def handle_event(self, event: object) -> FlextResult[None]:
            """Handle domain event."""
            self._events_processed += 1
            event_type = type(event).__name__
            self._event_types[event_type] = self._event_types.get(event_type, 0) + 1
            return self._process_event(event)

        def _process_event(self, _event: object) -> FlextResult[None]:
            """Process event - Template method pattern."""
            return FlextResult[None].ok(None)

        def get_event_metrics(self) -> FlextTypes.Handler.MetricsData:
            """Get event-specific metrics."""
            return {
                "events_processed": self._events_processed,
                "event_types": self._event_types,
            }

    class CommandBus:
        """Command bus for routing commands to handlers."""

        def __init__(self) -> None:
            """Initialize command bus."""
            self._handlers: dict[type, FlextAbstractCommandHandler[object, object]] = {}
            self._commands_processed: FlextTypes.Handler.CounterMetric = 0
            self._successful_commands: FlextTypes.Handler.CounterMetric = 0
            self._failed_commands: FlextTypes.Handler.CounterMetric = 0

        def register(
            self,
            command_type: type,
            handler: FlextAbstractCommandHandler[object, object],
        ) -> FlextResult[None]:
            """Register command handler."""
            with FlextHandlers.thread_safe_operation():
                self._handlers[command_type] = handler
                return FlextResult[None].ok(None)

        def send(self, command: object) -> FlextResult[object]:
            """Send command to registered handler."""
            command_type = type(command)

            with FlextHandlers.thread_safe_operation():
                handler = self._handlers.get(command_type)

                if not handler:
                    return FlextResult[object].fail(
                        f"No handler registered for {command_type.__name__}"
                    )

                self._commands_processed += 1
                result = handler.handle(command)

                if result.success:
                    self._successful_commands += 1
                else:
                    self._failed_commands += 1

                return result

        def get_metrics(self) -> FlextTypes.Handler.MetricsData:
            """Get command bus metrics."""
            return {
                "commands_processed": self._commands_processed,
                "successful_commands": self._successful_commands,
                "failed_commands": self._failed_commands,
            }

    class QueryBus:
        """Query bus for routing queries to handlers."""

        def __init__(self) -> None:
            """Initialize query bus."""
            self._handlers: dict[type, QueryHandlerProtocol] = {}
            self._queries_processed: FlextTypes.Handler.CounterMetric = 0
            self._successful_queries: FlextTypes.Handler.CounterMetric = 0
            self._failed_queries: FlextTypes.Handler.CounterMetric = 0

        def register(
            self, query_type: type, handler: QueryHandlerProtocol
        ) -> FlextResult[None]:
            """Register query handler."""
            with FlextHandlers.thread_safe_operation():
                self._handlers[query_type] = handler
            return FlextResult[None].ok(None)

        def send(self, query: object) -> FlextResult[object]:
            """Send query to registered handler."""
            query_type = type(query)

            with FlextHandlers.thread_safe_operation():
                handler = self._handlers.get(query_type)

                if not handler:
                    return FlextResult[object].fail(
                        f"No handler registered for {query_type.__name__}"
                    )

                self._queries_processed += 1
                result = handler.handle_query(query)

                if result.success:
                    self._successful_queries += 1
                else:
                    self._failed_queries += 1

                return result

        def get_metrics(self) -> FlextTypes.Handler.MetricsData:
            """Get query bus metrics."""
            return {
                "queries_processed": self._queries_processed,
                "successful_queries": self._successful_queries,
                "failed_queries": self._failed_queries,
            }

    # =========================================================================
    # Pattern Implementations
    # =========================================================================

    class HandlerChain:
        """Chain of Responsibility pattern implementation."""

        def __init__(self, name: str | None = None) -> None:
            """Initialize handler chain."""
            self._name: str = name or "HandlerChain"
            self._handlers: list[ChainHandlerProtocol] = []
            self._chain_executions: FlextTypes.Handler.CounterMetric = 0
            self._successful_chains: FlextTypes.Handler.CounterMetric = 0
            self._handler_performance: FlextTypes.Handler.PerformanceMap = {}

        def add_handler(self, handler: ChainHandlerProtocol) -> FlextResult[None]:
            """Add handler to chain."""
            with FlextHandlers.thread_safe_operation():
                self._handlers.append(handler)
                return FlextResult[None].ok(None)

        def handle(self, request: object) -> FlextResult[object]:
            """Execute chain of handlers."""
            self._chain_executions += 1

            for handler in self._handlers:
                if handler.can_handle(request):
                    start_time = time.time()
                    result = handler.handle(request)
                    processing_time = time.time() - start_time

                    handler_name = handler.handler_name
                    if handler_name not in self._handler_performance:
                        self._handler_performance[handler_name] = {
                            "executions": 0,
                            "total_time": 0.0,
                            "average_time": 0.0,
                        }

                    perf = self._handler_performance[handler_name]
                    perf["executions"] += 1
                    perf["total_time"] += processing_time
                    perf["average_time"] = perf["total_time"] / perf["executions"]

                    if result.success:
                        self._successful_chains += 1

                    return result

            return FlextResult[object].fail("No handler in chain could process request")

        def get_chain_metrics(self) -> FlextTypes.Handler.MetricsData:
            """Get chain-specific metrics."""
            return {
                "chain_executions": self._chain_executions,
                "successful_chains": self._successful_chains,
                "handler_performance": self._handler_performance,
            }

    class HandlerRegistry:
        """Registry pattern implementation for handler management."""

        def __init__(self) -> None:
            """Initialize handler registry."""
            self._handlers: dict[str, object] = {}
            self._registrations: FlextTypes.Handler.CounterMetric = 0
            self._lookups: FlextTypes.Handler.CounterMetric = 0
            self._successful_lookups: FlextTypes.Handler.CounterMetric = 0

        def register(self, name: str, handler: object) -> FlextResult[object]:
            """Register handler with name."""
            with FlextHandlers.thread_safe_operation():
                self._handlers[name] = handler
                self._registrations += 1
                return FlextResult[object].ok(handler)

        def get_handler(self, name: str) -> FlextResult[object]:
            """Get handler by name."""
            with FlextHandlers.thread_safe_operation():
                self._lookups += 1
                handler = self._handlers.get(name)
                if handler:
                    self._successful_lookups += 1
                    return FlextResult[object].ok(handler)
                return FlextResult[object].fail(f"Handler '{name}' not found")

        def get_all_handlers(self) -> dict[str, object]:
            """Get all registered handlers."""
            with FlextHandlers.thread_safe_operation():
                return dict(self._handlers)

        def unregister(self, name: str) -> FlextResult[None]:
            """Unregister handler by name."""
            with FlextHandlers.thread_safe_operation():
                if name in self._handlers:
                    del self._handlers[name]
                    return FlextResult[None].ok(None)
                return FlextResult[None].fail(f"Handler '{name}' not found")

        def get_registry_metrics(self) -> FlextTypes.Handler.MetricsData:
            """Get registry-specific metrics."""
            return {
                "registrations": self._registrations,
                "lookups": self._lookups,
                "successful_lookups": self._successful_lookups,
            }

    class Pipeline:
        """Pipeline pattern for sequential processing."""

        def __init__(self, name: str | None = None) -> None:
            """Initialize processing pipeline."""
            self._name: Final[str] = name or "Pipeline"
            self._stages: list[Callable[[object], FlextResult[object]]] = []
            self._pipeline_executions: FlextTypes.Handler.CounterMetric = 0
            self._successful_pipelines: FlextTypes.Handler.CounterMetric = 0
            self._stage_performance: FlextTypes.Handler.PerformanceMap = {}

        def add_stage(
            self, stage: Callable[[object], FlextResult[object]]
        ) -> FlextResult[None]:
            """Add processing stage to pipeline."""
            with FlextHandlers.thread_safe_operation():
                self._stages.append(stage)
                return FlextResult[None].ok(None)

        def process(self, data: object) -> FlextResult[object]:
            """Process data through pipeline stages."""
            self._pipeline_executions += 1
            current_data = data

            for i, stage in enumerate(self._stages):
                stage_name = f"stage_{i}"
                start_time = time.time()

                result = stage(current_data)
                processing_time = time.time() - start_time

                if stage_name not in self._stage_performance:
                    self._stage_performance[stage_name] = {
                        "executions": 0,
                        "total_time": 0.0,
                        "average_time": 0.0,
                    }

                perf = self._stage_performance[stage_name]
                perf["executions"] += 1
                perf["total_time"] += processing_time
                perf["average_time"] = perf["total_time"] / perf["executions"]

                if result.is_failure:
                    return result

                current_data = result.value

            self._successful_pipelines += 1
            return FlextResult[object].ok(current_data)

        def get_pipeline_metrics(self) -> FlextTypes.Handler.MetricsData:
            """Get pipeline-specific metrics."""
            return {
                "pipeline_executions": self._pipeline_executions,
                "successful_pipelines": self._successful_pipelines,
                "stage_performance": self._stage_performance,
            }


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Maintain backward compatibility with existing code
FlextAbstractHandler = FlextHandlers.AbstractHandler
FlextBaseHandler = FlextHandlers.Handler
# Import the concrete implementations from FlextHandlers
FlextValidatingHandler = FlextHandlers.ValidatingHandler
FlextAuthorizingHandler = FlextHandlers.AuthorizingHandler
FlextEventHandler = FlextHandlers.EventHandler
FlextMetricsHandler = FlextHandlers.MetricsHandler
FlextHandlerRegistry = FlextHandlers.HandlerRegistry
FlextHandlerChain = FlextHandlers.HandlerChain
FlextCommandHandler = FlextHandlers.CommandHandler
FlextQueryHandler = FlextHandlers.QueryHandler


# Legacy facade for backward compatibility
class HandlersFacade:
    """Legacy facade for backward compatibility."""

    Handler = FlextHandlers.Handler
    ValidatingHandler = FlextHandlers.ValidatingHandler
    AuthorizingHandler = FlextHandlers.AuthorizingHandler


# =============================================================================
# Module-level Exports
# =============================================================================

__all__ = [
    "FlextAbstractHandler",
    "FlextAuthorizingHandler",
    "FlextBaseHandler",
    "FlextCommandHandler",
    "FlextEventHandler",
    "FlextHandlerChain",
    "FlextHandlerRegistry",
    "FlextHandlers",
    "FlextMetricsHandler",
    "FlextQueryHandler",
    "FlextValidatingHandler",
    "HandlersFacade",
]
