"""FlextHandlers - Hierarchical Handler Management System.

Comprehensive handler implementation following Clean Architecture and SOLID principles.
All handler functionality is organized under the FlextHandlers namespace for consistency
with other flext-core modules (context.py, constants.py, typings.py, protocols.py).

Architecture:
- FlextHandlers.Abstract: Base contracts and abstract classes
- FlextHandlers.Base: Concrete implementations
- FlextHandlers.CQRS: Command/Query/Event handlers
- FlextHandlers.Patterns: Chain of Responsibility, Registry, Pipeline
- FlextHandlers.Decorators: Cross-cutting concerns (validation, metrics)
- FlextHandlers.Utilities: Helper functions and factories

Follows SOLID principles:
- Single Responsibility: Each handler has one clear purpose
- Open/Closed: Extensible without modification
- Liskov Substitution: Implementations are substitutable
- Interface Segregation: Clean separation of concerns
- Dependency Inversion: Depends on abstractions

Example Usage:
    >>> from flext_core import FlextHandlers
    >>> # Create concrete handler
    >>> handler = FlextHandlers.Base.Handler()
    >>> # Use CQRS patterns
    >>> cmd_handler = FlextHandlers.CQRS.CommandHandler()
    >>> query_handler = FlextHandlers.CQRS.QueryHandler()
    >>> # Chain handlers
    >>> chain = FlextHandlers.Patterns.HandlerChain()
    >>> chain.add_handler(handler)
    >>> # Registry management
    >>> registry = FlextHandlers.Patterns.HandlerRegistry()
    >>> registry.register("my_handler", handler)
    >>> # Apply decorators
    >>> @FlextHandlers.Decorators.with_validation
    >>> @FlextHandlers.Decorators.with_metrics
    >>> def process_request(data):
    >>>     return FlextResult.ok(data)
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from os import environ
from typing import Any, Final, Generic, cast

from flext_core.constants import FlextConstants
from flext_core.protocols import FlextValidator
from flext_core.result import FlextResult
from flext_core.typings import (
    FlextTypes,
    P,
    T,
    TInput,
    TOutput,
    TQuery,
    TQueryResult,
)

# Use centralized types from FlextTypes where appropriate
HandlerName = FlextTypes.Handler.HandlerName

# =============================================================================
# FlextHandlers - Hierarchical Handler Management System
# =============================================================================


class FlextHandlers:
    """Hierarchical handler management system following Clean Architecture.

    Provides comprehensive handler functionality organized into logical domains:
    - Abstract: Base contracts and abstract classes
    - Base: Concrete implementations
    - CQRS: Command/Query/Event patterns
    - Patterns: Chain of Responsibility, Registry, Pipeline
    - Decorators: Cross-cutting concerns
    - Utilities: Helper functions and factories

    All handler functionality is centralized under this namespace for consistency
    with other flext-core modules and improved discoverability.

    Example:
        >>> # Basic handler usage
        >>> handler = FlextHandlers.Base.Handler()
        >>> result = handler.handle(request_data)
        >>> # CQRS pattern usage
        >>> command_handler = FlextHandlers.CQRS.CommandHandler()
        >>> query_handler = FlextHandlers.CQRS.QueryHandler()
        >>> # Pattern composition
        >>> chain = FlextHandlers.Patterns.HandlerChain()
        >>> registry = FlextHandlers.Patterns.HandlerRegistry()

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
        """Lazy import to avoid circular dependencies.

        SOLID: Dependency Inversion - depend on abstraction, not concrete implementation.
        """
        from flext_core.commands import FlextCommands  # noqa: PLC0415

        return FlextCommands

    # =========================================================================
    # Abstract Domain - Base Contracts and Abstract Classes
    # =========================================================================

    class Abstract:
        """Abstract base classes and contracts for handler implementations.

        Provides the foundational interfaces that all concrete handlers must implement,
        following Interface Segregation and Dependency Inversion principles.

        Key Components:
        - Handler: Basic handler contract
        - HandlerChain: Chain of Responsibility contract
        - HandlerRegistry: Registry pattern contract
        - MetricsHandler: Metrics collection contract
        - ValidatingHandler: Validation contract
        """

        class Handler(ABC, Generic[TInput, TOutput]):
            """Abstract handler base class following Interface Segregation Principle.

            SOLID PRINCIPLES APPLIED:
            - Single Responsibility: Defines only handler contract
            - Interface Segregation: Minimal, focused interface
            - Dependency Inversion: Abstract base for concrete implementations
            """

            @property
            @abstractmethod
            def handler_name(self) -> str:
                """Get handler name - Interface Segregation: focused method."""

            @abstractmethod
            def handle(self, request: TInput) -> FlextResult[TOutput]:
                """Handle request - Single Responsibility: core handler behavior."""

            @abstractmethod
            def can_handle(self, message_type: object) -> bool:
                """Check capability - Interface Segregation: capability query."""

        class HandlerChain(ABC, Generic[TInput, TOutput]):
            """Abstract handler chain following Chain of Responsibility pattern.

            SOLID PRINCIPLES APPLIED:
            - Single Responsibility: Chain management only
            - Open/Closed: Extensible through concrete implementations
            - Interface Segregation: Chain-specific operations only
            """

            @property
            def handler_name(self) -> str:
                """Return chain identifier - Interface Segregation principle."""
                return self.__class__.__name__

            @abstractmethod
            def handle(self, request: TInput) -> FlextResult[TOutput]:
                """Handle through chain - Single Responsibility: chain execution."""

        class HandlerRegistry(ABC, Generic[TInput]):
            """Abstract handler registry following Registry pattern.

            SOLID PRINCIPLES APPLIED:
            - Single Responsibility: Handler registration and lookup only
            - Interface Segregation: Registry-specific operations
            - Open/Closed: Extensible registration strategies
            """

            @abstractmethod
            def register(self, name: str, handler: TInput) -> FlextResult[TInput]:
                """Register handler - Single Responsibility: registration logic."""

            @abstractmethod
            def get_all_handlers(self) -> dict[str, TInput]:
                """Get all handlers - Interface Segregation: focused query."""

        class MetricsHandler(Handler[TInput, TOutput]):
            """Abstract metrics-collecting handler.

            Extends base handler with metrics collection capabilities while
            maintaining Interface Segregation.
            """

            @abstractmethod
            def get_metrics(self) -> FlextTypes.Handler.MetricsData:
                """Get handler metrics - Interface Segregation: metrics query."""

        class ValidatingHandler(Handler[TInput, TOutput]):
            """Abstract validating handler.

            Extends base handler with validation capabilities following
            Single Responsibility principle.
            """

            @abstractmethod
            def validate(self, request: TInput) -> FlextResult[None]:
                """Validate request - Single Responsibility: validation logic."""

    # =========================================================================
    # Base Domain - Concrete Handler Implementations
    # =========================================================================

    class Base:
        """Concrete base implementations of handler contracts.

        Provides standard implementations that can be used directly or extended
        by specific application handlers. Follows Open/Closed and Liskov
        Substitution principles.

        Key Components:
        - Handler: Standard handler implementation
        - ValidatingHandler: Handler with validation
        - AuthorizingHandler: Handler with authorization
        - MetricsHandler: Handler with metrics collection
        """

        class Handler:
            """Concrete base handler implementation.

            SOLID PRINCIPLES APPLIED:
            - Single Responsibility: Core handler logic only
            - Open/Closed: Extensible without modification
            - Liskov Substitution: Substitutable for abstract handler
            """

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
                """Handle request with metrics collection.

                SOLID: Single Responsibility - focused on request processing.
                """
                start_time = time.time()

                try:
                    self._metrics["requests_processed"] = (
                        cast("int", self._metrics["requests_processed"]) + 1
                    )

                    # Delegate to specific implementation
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
                    # Update timing metrics
                    processing_time = time.time() - start_time
                    self._metrics["total_processing_time"] = (
                        cast("float", self._metrics["total_processing_time"])
                        + processing_time
                    )

                    requests_count = cast("int", self._metrics["requests_processed"])
                    if requests_count > 0:
                        total_time = cast(
                            "float", self._metrics["total_processing_time"]
                        )
                        self._metrics["average_processing_time"] = (
                            total_time / requests_count
                        )

            def _process_request(self, request: object) -> FlextResult[object]:
                """Process request - Template method pattern.

                Override in subclasses for specific behavior.
                """
                return FlextResult[object].ok(request)

            def can_handle(self, message_type: object) -> bool:
                """Check if handler can process message type."""
                return message_type is not None

            def get_metrics(self) -> FlextTypes.Handler.MetricsData:
                """Get handler metrics."""
                return dict(self._metrics)

        class ValidatingHandler(Handler):
            """Handler with built-in validation capabilities.

            Combines base handler functionality with validation logic following
            Decorator pattern and Single Responsibility principle.
            """

            def __init__(
                self,
                name: str | None = None,
                validators: list[FlextValidator] | None = None,
            ) -> None:
                """Initialize with optional validators."""
                super().__init__(name)
                self._validators: Final[list[FlextValidator]] = validators or []

            def handle(self, request: object) -> FlextResult[object]:
                """Handle with validation - Template method pattern."""
                # First validate
                validation_result = self.validate(request)
                if validation_result.failure:
                    return FlextResult[object].fail(
                        f"Validation failed: {validation_result.error}"
                    )

                # Then process
                return super().handle(request)

            def validate(self, request: object) -> FlextResult[None]:
                """Validate request using configured validators."""
                for validator in self._validators:
                    if hasattr(validator, "validate"):
                        result = validator.validate(request)
                        if hasattr(result, "failure") and result.failure:
                            return result
                    elif callable(validator):
                        try:
                            if not validator(request):
                                return FlextResult[None].fail("Validation failed")
                        except Exception as e:
                            return FlextResult[None].fail(f"Validation error: {e}")

                return FlextResult[None].ok(None)

            def add_validator(self, validator: FlextValidator) -> None:
                """Add validator to the chain."""
                self._validators.append(validator)

        class AuthorizingHandler(Handler):
            """Handler with authorization capabilities.

            Extends base handler with authorization logic following
            Decorator pattern and separation of concerns.
            """

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

            def handle(self, request: object) -> FlextResult[object]:
                """Handle with authorization - Template method pattern."""
                # Check authorization first
                if not self._authorization_check(request):
                    return FlextResult[object].fail("Authorization failed")

                # Then process
                return super().handle(request)

            def _default_authorization(self, _request: object) -> bool:
                """Default authorization - always allow.

                Override in subclasses for specific authorization logic.
                """
                return True

        class MetricsHandler(Handler):
            """Handler with enhanced metrics collection.

            Provides detailed metrics beyond basic handler metrics,
            following Single Responsibility and Open/Closed principles.
            """

            def __init__(self, name: str | None = None) -> None:
                """Initialize with enhanced metrics."""
                super().__init__(name)
                # Use specific types for enhanced metrics
                self._error_types: FlextTypes.Handler.ErrorCounterMap = {}
                self._request_sizes: FlextTypes.Handler.SizeList = []
                self._response_sizes: FlextTypes.Handler.SizeList = []
                self._peak_memory_usage: FlextTypes.Handler.CounterMetric = 0

            def handle(self, request: object) -> FlextResult[object]:
                """Handle with enhanced metrics collection."""
                # Collect request metrics
                request_size = len(str(request)) if request else 0
                self._request_sizes.append(request_size)

                # Process request
                result = super().handle(request)

                # Collect response metrics
                if result.failure:
                    error_type = type(result.error).__name__
                    self._error_types[error_type] = (
                        self._error_types.get(error_type, 0) + 1
                    )
                else:
                    response_size = len(str(result.data)) if result.data else 0
                    self._response_sizes.append(response_size)

                return result

            def get_metrics(self) -> FlextTypes.Handler.MetricsData:
                """Get enhanced metrics including base metrics."""
                base_metrics = super().get_metrics()
                enhanced_metrics: FlextTypes.Handler.MetricsData = {
                    "error_types": self._error_types,
                    "request_sizes": self._request_sizes,
                    "response_sizes": self._response_sizes,
                    "peak_memory_usage": self._peak_memory_usage,
                }

                # Calculate additional metrics
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
    # CQRS Domain - Command/Query/Event Handler Implementations
    # =========================================================================

    class CQRS:
        """Command/Query/Event handler implementations following CQRS pattern.

        Provides specialized handlers for CQRS architecture separating command
        handling (write operations) from query handling (read operations) and
        event handling (domain events).

        Key Components:
        - CommandHandler: Handles commands (write operations)
        - QueryHandler: Handles queries (read operations)
        - EventHandler: Handles domain events
        - CommandBus: Routes commands to handlers
        - QueryBus: Routes queries to handlers
        """

        class CommandHandler(ABC, Generic[TInput, TOutput]):
            """Abstract command handler for CQRS pattern.

            SOLID PRINCIPLES APPLIED:
            - Single Responsibility: Command processing only
            - Interface Segregation: Command-specific interface
            - Dependency Inversion: Abstract base for implementations
            """

            @abstractmethod
            def handle_command(self, command: TInput) -> FlextResult[TOutput]:
                """Handle command - Single Responsibility: command processing."""

            @abstractmethod
            def can_handle(self, message_type: object) -> bool:
                """Check command handling capability."""

        class QueryHandler(ABC, Generic[TQuery, TQueryResult]):
            """Abstract query handler for CQRS pattern.

            SOLID PRINCIPLES APPLIED:
            - Single Responsibility: Query processing only
            - Interface Segregation: Query-specific interface
            - Dependency Inversion: Abstract base for implementations
            """

            @abstractmethod
            def handle_query(self, query: TQuery) -> FlextResult[TQueryResult]:
                """Handle query - Single Responsibility: query processing."""

        class EventHandler:
            """Event handler for domain events.

            Extends base handler with event-specific functionality following
            Open/Closed and Liskov Substitution principles.
            """

            def __init__(self, name: str | None = None) -> None:
                """Initialize event handler."""
                super().__init__(name or "EventHandler")
                # Use specific types for event metrics
                self._events_processed: FlextTypes.Handler.CounterMetric = 0
                self._event_types: FlextTypes.Handler.ErrorCounterMap = {}

            def handle_event(self, event: object) -> FlextResult[None]:
                """Handle domain event.

                SOLID: Single Responsibility - event processing only.
                """
                self._events_processed += 1

                event_type = type(event).__name__
                self._event_types[event_type] = (
                    self._event_types.get(event_type, 0) + 1
                )

                return self._process_event(event)

            def _process_event(self, _event: object) -> FlextResult[None]:
                """Process event - Template method pattern.

                Override in subclasses for specific event handling.
                """
                return FlextResult[None].ok(None)

            def get_event_metrics(self) -> FlextTypes.Handler.MetricsData:
                """Get event-specific metrics."""
                return {
                    "events_processed": self._events_processed,
                    "event_types": self._event_types,
                }

        class CommandBus:
            """Command bus for routing commands to handlers.

            Implements mediator pattern for command handling, providing
            centralized command routing and execution.
            """

            def __init__(self) -> None:
                """Initialize command bus."""
                self._handlers: dict[type, Any] = {}
                # Use specific types for command metrics
                self._commands_processed: FlextTypes.Handler.CounterMetric = 0
                self._successful_commands: FlextTypes.Handler.CounterMetric = 0
                self._failed_commands: FlextTypes.Handler.CounterMetric = 0

            def register(
                self, command_type: type, handler: object
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

                    # Execute command
                    result = handler.handle_command(command)

                    if result.success:
                        self._successful_commands += 1
                    else:
                        self._failed_commands += 1

                    return cast("FlextResult[object]", result)

            def get_metrics(self) -> FlextTypes.Handler.MetricsData:
                """Get command bus metrics."""
                return dict(self._metrics)

        class QueryBus:
            """Query bus for routing queries to handlers.

            Implements mediator pattern for query handling, providing
            centralized query routing and execution.
            """

            def __init__(self) -> None:
                """Initialize query bus."""
                self._handlers: dict[type, Any] = {}
                self._metrics: FlextTypes.Handler.MetricsData = {
                    "queries_processed": 0,
                    "successful_queries": 0,
                    "failed_queries": 0,
                }

            def register(self, query_type: type, handler: object) -> FlextResult[None]:
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

                    self._metrics["queries_processed"] += 1

                    # Execute query
                    result = handler.handle_query(query)

                    if result.success:
                        self._metrics["successful_queries"] += 1
                    else:
                        self._metrics["failed_queries"] += 1

                    return cast("FlextResult[object]", result)

            def get_metrics(self) -> FlextTypes.Handler.MetricsData:
                """Get query bus metrics."""
                return dict(self._metrics)

    # =========================================================================
    # Patterns Domain - Chain of Responsibility, Registry, Pipeline
    # =========================================================================

    class Patterns:
        """Handler patterns implementing common design patterns.

        Provides implementations of common handler patterns including
        Chain of Responsibility, Registry, and Pipeline patterns.

        Key Components:
        - HandlerChain: Chain of Responsibility implementation
        - HandlerRegistry: Registry pattern implementation
        - Pipeline: Pipeline pattern for sequential processing
        """

        class HandlerChain:
            """Chain of Responsibility pattern implementation.

            SOLID PRINCIPLES APPLIED:
            - Single Responsibility: Chain management and execution
            - Open/Closed: Extensible by adding new handlers
            - Liskov Substitution: Substitutable for base handler
            """

            def __init__(self, name: str | None = None) -> None:
                """Initialize handler chain."""
                super().__init__(name or "HandlerChain")
                self._handlers: list[object] = []
                self._chain_metrics: FlextTypes.Handler.MetricsData = {
                    "chain_executions": 0,
                    "successful_chains": 0,
                    "handler_performance": {},
                }

            def add_handler(self, handler: object) -> FlextResult[None]:
                """Add handler to chain.

                SOLID: Open/Closed - extend without modification.
                """
                with FlextHandlers.thread_safe_operation():
                    self._handlers.append(handler)
                    return FlextResult[None].ok(None)

            def handle(self, request: object) -> FlextResult[object]:
                """Execute chain of handlers.

                SOLID: Single Responsibility - chain execution logic.
                """
                self._chain_metrics["chain_executions"] += 1

                for handler in self._handlers:
                    if handler.can_handle(request):
                        start_time = time.time()
                        result = handler.handle(request)
                        processing_time = time.time() - start_time

                        # Update handler performance metrics
                        handler_name = handler.handler_name
                        if (
                            handler_name
                            not in self._chain_metrics["handler_performance"]
                        ):
                            self._chain_metrics["handler_performance"][handler_name] = {
                                "executions": 0,
                                "total_time": 0.0,
                                "average_time": 0.0,
                            }

                        perf = self._chain_metrics["handler_performance"][handler_name]
                        perf["executions"] += 1
                        perf["total_time"] += processing_time
                        perf["average_time"] = perf["total_time"] / perf["executions"]

                        if result.success:
                            self._chain_metrics["successful_chains"] += 1
                            return result

                return FlextResult[object].fail(
                    "No handler in chain could process request"
                )

            def get_chain_metrics(self) -> FlextTypes.Handler.MetricsData:
                """Get chain-specific metrics."""
                return dict(self._chain_metrics)

        class HandlerRegistry:
            """Registry pattern implementation for handler management.

            SOLID PRINCIPLES APPLIED:
            - Single Responsibility: Handler registration and lookup
            - Open/Closed: Extensible registration strategies
            - Interface Segregation: Registry-specific operations
            """

            def __init__(self) -> None:
                """Initialize handler registry."""
                self._handlers: FlextTypes.Handler.MetricsData = {}
                self._registry_metrics: FlextTypes.Handler.MetricsData = {
                    "registrations": 0,
                    "lookups": 0,
                    "successful_lookups": 0,
                }

            def register(self, name: str, handler: object) -> FlextResult[object]:
                """Register handler with name.

                SOLID: Single Responsibility - registration logic only.
                """
                with FlextHandlers.thread_safe_operation():
                    self._handlers[name] = handler
                    self._registry_metrics["registrations"] += 1
                    return FlextResult[Any].ok(handler)

            def get_handler(self, name: str) -> FlextResult[Any]:
                """Get handler by name."""
                with FlextHandlers.thread_safe_operation():
                    self._registry_metrics["lookups"] += 1

                    handler = self._handlers.get(name)
                    if handler:
                        self._registry_metrics["successful_lookups"] += 1
                        return FlextResult[Any].ok(handler)

                    return FlextResult[Any].fail(f"Handler '{name}' not found")

            def get_all_handlers(self) -> FlextTypes.Handler.MetricsData:
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
                return dict(self._registry_metrics)

        class Pipeline:
            """Pipeline pattern for sequential processing.

            Implements pipeline pattern where data flows through a series
            of processing stages, each potentially transforming the data.
            """

            def __init__(self, name: str | None = None) -> None:
                """Initialize processing pipeline."""
                self._name: Final[str] = name or "Pipeline"
                self._stages: list[Callable[[Any], FlextResult[Any]]] = []
                self._pipeline_metrics: FlextTypes.Handler.MetricsData = {
                    "pipeline_executions": 0,
                    "successful_pipelines": 0,
                    "stage_performance": {},
                }

            def add_stage(
                self, stage: Callable[[Any], FlextResult[Any]]
            ) -> FlextResult[None]:
                """Add processing stage to pipeline."""
                with FlextHandlers.thread_safe_operation():
                    self._stages.append(stage)
                    return FlextResult[None].ok(None)

            def process(self, data: object) -> FlextResult[object]:
                """Process data through pipeline stages."""
                self._pipeline_metrics["pipeline_executions"] += 1

                current_data = data

                for i, stage in enumerate(self._stages):
                    stage_name = f"stage_{i}"
                    start_time = time.time()

                    result = stage(current_data)
                    processing_time = time.time() - start_time

                    # Update stage performance metrics
                    if stage_name not in self._pipeline_metrics["stage_performance"]:
                        self._pipeline_metrics["stage_performance"][stage_name] = {
                            "executions": 0,
                            "total_time": 0.0,
                            "average_time": 0.0,
                        }

                    perf = self._pipeline_metrics["stage_performance"][stage_name]
                    perf["executions"] += 1
                    perf["total_time"] += processing_time
                    perf["average_time"] = perf["total_time"] / perf["executions"]

                    if result.failure:
                        return result

                    current_data = result.data

                self._pipeline_metrics["successful_pipelines"] += 1
                return FlextResult[Any].ok(current_data)

            def get_pipeline_metrics(self) -> FlextTypes.Handler.MetricsData:
                """Get pipeline-specific metrics."""
                return dict(self._pipeline_metrics)

    # =========================================================================
    # Decorators Domain - Cross-cutting Concerns
    # =========================================================================

    class Decorators:
        """Decorator patterns for cross-cutting concerns.

        Provides decorators for common cross-cutting concerns like validation,
        metrics collection, authorization, caching, and error handling.

        Key Components:
        - with_validation: Add validation to functions
        - with_metrics: Add metrics collection
        - with_authorization: Add authorization checks
        - with_caching: Add caching capabilities
        - with_error_handling: Add error handling
        """

        @staticmethod
        def with_validation(
            validator: FlextValidator | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Decorator for adding validation to handler methods.

            SOLID: Single Responsibility - validation concern only.
            """

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    # Extract request from arguments (assume first arg after self)
                    request = args[1] if len(args) > 1 else kwargs.get("request")

                    if validator and request:
                        if hasattr(validator, "validate"):
                            validation_result = validator.validate(request)
                            if (
                                hasattr(validation_result, "failure")
                                and validation_result.failure
                            ):
                                return FlextResult[Any].fail(
                                    f"Validation failed: {validation_result.error}"
                                )
                        elif callable(validator):
                            try:
                                if not validator(request):
                                    return FlextResult[Any].fail("Validation failed")
                            except Exception as e:
                                return FlextResult[Any].fail(f"Validation error: {e}")

                    return func(*args, **kwargs)

                return wrapper

            return decorator

        @staticmethod
        def with_metrics(
            metrics_collector: dict | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Decorator for adding metrics collection to handler methods.

            SOLID: Single Responsibility - metrics concern only.
            """

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    collector = {} if metrics_collector is None else metrics_collector

                    start_time = time.time()
                    collector.setdefault("executions", 0)
                    collector.setdefault("total_time", 0.0)
                    collector.setdefault("successes", 0)
                    collector.setdefault("failures", 0)

                    collector["executions"] += 1

                    try:
                        result = func(*args, **kwargs)
                        processing_time = time.time() - start_time
                        collector["total_time"] += processing_time
                        collector["average_time"] = (
                            collector["total_time"] / collector["executions"]
                        )

                        if hasattr(result, "success") and result.success:
                            collector["successes"] += 1
                        else:
                            collector["failures"] += 1

                        return result
                    except Exception as e:
                        processing_time = time.time() - start_time
                        collector["total_time"] += processing_time
                        collector["failures"] += 1
                        return FlextResult[Any].fail(f"Execution failed: {e}")

                return wrapper

            return decorator

        @staticmethod
        def with_authorization(
            auth_check: Callable | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Decorator for adding authorization to handler methods.

            SOLID: Single Responsibility - authorization concern only.
            """

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    # Extract request from arguments
                    request = args[1] if len(args) > 1 else kwargs.get("request")

                    if auth_check and request:
                        try:
                            if not auth_check(request):
                                return FlextResult[Any].fail("Authorization failed")
                        except Exception as e:
                            return FlextResult[Any].fail(f"Authorization error: {e}")

                    return func(*args, **kwargs)

                return wrapper

            return decorator

        @staticmethod
        def with_caching(
            cache: dict | None = None, ttl: int = 300
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Decorator for adding caching to handler methods.

            SOLID: Single Responsibility - caching concern only.
            """

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    if cache is None:
                        return func(*args, **kwargs)

                    # Create cache key from arguments
                    cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
                    current_time = time.time()

                    # Check cache
                    if cache_key in cache:
                        cached_result, timestamp = cache[cache_key]
                        if current_time - timestamp < ttl:
                            return cached_result

                    # Execute and cache result
                    result = func(*args, **kwargs)
                    cache[cache_key] = (result, current_time)

                    return result

                return wrapper

            return decorator

        @staticmethod
        def with_error_handling(
            error_handler: Callable | None = None,
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            """Decorator for adding error handling to handler methods.

            SOLID: Single Responsibility - error handling concern only.
            """

            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if error_handler:
                            return error_handler(e)
                        return FlextResult[Any].fail(f"Handler error: {e}")

                return wrapper

            return decorator

    # =========================================================================
    # Utilities Domain - Helper Functions and Factories
    # =========================================================================

    class Utilities:
        """Utility functions and factories for handler management.

        Provides helper functions for common handler operations,
        factory methods for creating handlers, and utility functions
        for handler composition and management.

        Key Components:
        - HandlerFactory: Factory for creating handlers
        - HandlerComposer: Utility for composing handlers
        - HandlerAnalyzer: Analysis utilities
        """

        class HandlerFactory:
            """Factory for creating handler instances.

            SOLID: Single Responsibility - handler creation only.
            """

            @staticmethod
            def create_basic_handler(name: str | None = None) -> object:
                """Create basic handler instance."""
                return FlextHandlers.Base.Handler(name)

            @staticmethod
            def create_validating_handler(
                name: str | None = None,
                validators: list[FlextValidator] | None = None,
            ) -> object:
                """Create validating handler instance."""
                return FlextHandlers.Base.ValidatingHandler(name, validators)

            @staticmethod
            def create_authorizing_handler(
                name: str | None = None,
                auth_check: Callable | None = None,
            ) -> object:
                """Create authorizing handler instance."""
                return FlextHandlers.Base.AuthorizingHandler(name, auth_check)

            @staticmethod
            def create_metrics_handler(name: str | None = None) -> object:
                """Create metrics handler instance."""
                return FlextHandlers.Base.MetricsHandler(name)

            @staticmethod
            def create_handler_chain(name: str | None = None) -> object:
                """Create handler chain instance."""
                return FlextHandlers.Patterns.HandlerChain(name)

            @staticmethod
            def create_handler_registry() -> object:
                """Create handler registry instance."""
                return FlextHandlers.Patterns.HandlerRegistry()

            @staticmethod
            def create_pipeline(name: str | None = None) -> object:
                """Create pipeline instance."""
                return FlextHandlers.Patterns.Pipeline(name)

            @staticmethod
            def create_command_bus() -> object:
                """Create command bus instance."""
                return FlextHandlers.CQRS.CommandBus()

            @staticmethod
            def create_query_bus() -> object:
                """Create query bus instance."""
                return FlextHandlers.CQRS.QueryBus()

        class HandlerComposer:
            """Utility for composing and chaining handlers.

            SOLID: Single Responsibility - handler composition only.
            """

            @staticmethod
            def compose_handlers(*handlers: object) -> object:
                """Compose multiple handlers into a chain."""
                chain = FlextHandlers.Patterns.HandlerChain("ComposedChain")
                for handler in handlers:
                    chain.add_handler(handler)
                return chain

            @staticmethod
            def create_decorated_handler(
                base_handler: object,
                validators: list[FlextValidator] | None = None,
                auth_check: Callable | None = None,
                *,
                enable_metrics: bool = True,
            ) -> object:
                """Create a decorated handler with cross-cutting concerns."""
                # This is a simplified example - in practice, you'd use proper decorator pattern
                if isinstance(base_handler, FlextHandlers.Base.Handler):
                    decorated = base_handler

                    # Add validation if specified
                    if validators:
                        # In practice, wrap with validation decorator
                        pass

                    # Add authorization if specified
                    if auth_check:
                        # In practice, wrap with authorization decorator
                        pass

                    # Metrics enabled by default
                    if enable_metrics:
                        pass

                    return decorated

                return base_handler

        class HandlerAnalyzer:
            """Analysis utilities for handler performance and behavior.

            SOLID: Single Responsibility - analysis and reporting only.
            """

            @staticmethod
            def analyze_handler_performance(
                handler: object,
            ) -> FlextTypes.Handler.MetricsData:
                """Analyze handler performance metrics."""
                analysis = {
                    "handler_name": handler.handler_name,
                    "handler_type": type(handler).__name__,
                    "analysis_timestamp": time.time(),
                }

                # Get metrics if available
                if hasattr(handler, "get_metrics"):
                    metrics = handler.get_metrics()
                    analysis["metrics"] = metrics

                    # Calculate performance indicators
                    if (
                        "requests_processed" in metrics
                        and metrics["requests_processed"] > 0
                    ):
                        success_rate = (
                            metrics.get("successful_requests", 0)
                            / metrics["requests_processed"]
                            * 100
                        )
                        analysis["success_rate_percent"] = round(success_rate, 2)

                    if "average_processing_time" in metrics:
                        analysis["performance_rating"] = (
                            "excellent"
                            if metrics["average_processing_time"]
                            < FlextConstants.Performance.HANDLER_EXCELLENT_THRESHOLD
                            else "good"
                            if metrics["average_processing_time"]
                            < FlextConstants.Performance.HANDLER_GOOD_THRESHOLD
                            else "fair"
                            if metrics["average_processing_time"]
                            < FlextConstants.Performance.HANDLER_FAIR_THRESHOLD
                            else "poor"
                        )

                return analysis

            @staticmethod
            def generate_handler_report(
                handlers: list[object],
            ) -> FlextTypes.Handler.MetricsData:
                """Generate comprehensive report for multiple handlers."""
                report = {
                    "report_timestamp": time.time(),
                    "total_handlers": len(handlers),
                    "handler_analyses": [],
                    "summary": {
                        "total_requests": 0,
                        "total_successes": 0,
                        "total_failures": 0,
                        "average_success_rate": 0.0,
                    },
                }

                total_success_rates = []

                for handler in handlers:
                    analysis = FlextHandlers.Utilities.HandlerAnalyzer.analyze_handler_performance(
                        handler
                    )
                    report["handler_analyses"].append(analysis)

                    # Aggregate summary data
                    if "metrics" in analysis:
                        metrics = analysis["metrics"]
                        report["summary"]["total_requests"] += metrics.get(
                            "requests_processed", 0
                        )
                        report["summary"]["total_successes"] += metrics.get(
                            "successful_requests", 0
                        )
                        report["summary"]["total_failures"] += metrics.get(
                            "failed_requests", 0
                        )

                    if "success_rate_percent" in analysis:
                        total_success_rates.append(analysis["success_rate_percent"])

                # Calculate average success rate
                if total_success_rates:
                    report["summary"]["average_success_rate"] = round(
                        sum(total_success_rates) / len(total_success_rates), 2
                    )

                return report

        @staticmethod
        def get_environment_config() -> FlextTypes.Handler.MetricsData:
            """Get handler configuration from environment variables."""
            return {
                "debug_mode": environ.get("FLEXT_HANDLERS_DEBUG", "false").lower()
                == "true",
                "metrics_enabled": environ.get("FLEXT_HANDLERS_METRICS", "true").lower()
                == "true",
                "thread_safe": environ.get("FLEXT_HANDLERS_THREAD_SAFE", "true").lower()
                == "true",
                "max_chain_depth": int(
                    environ.get("FLEXT_HANDLERS_MAX_CHAIN_DEPTH", "10")
                ),
                "cache_ttl": int(environ.get("FLEXT_HANDLERS_CACHE_TTL", "300")),
            }

        @staticmethod
        def validate_handler_configuration(
            config: FlextTypes.Handler.MetricsData,
        ) -> FlextResult[None]:
            """Validate handler configuration."""
            required_keys = ["debug_mode", "metrics_enabled", "thread_safe"]

            for key in required_keys:
                if key not in config:
                    return FlextResult[None].fail(f"Missing required config key: {key}")

            if (
                not isinstance(config.get("max_chain_depth", 0), int)
                or config.get("max_chain_depth", 0) <= 0
            ):
                return FlextResult[None].fail(
                    "max_chain_depth must be a positive integer"
                )

            return FlextResult[None].ok(None)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Maintain backward compatibility with existing code
FlextAbstractHandler = FlextHandlers.Abstract.Handler
FlextAbstractHandlerChain = FlextHandlers.Abstract.HandlerChain
FlextAbstractHandlerRegistry = FlextHandlers.Abstract.HandlerRegistry
FlextAbstractMetricsHandler = FlextHandlers.Abstract.MetricsHandler
FlextAbstractValidatingHandler = FlextHandlers.Abstract.ValidatingHandler

FlextBaseHandler = FlextHandlers.Base.Handler
# FlextValidatingHandler and FlextAuthorizingHandler moved to typings.py to avoid conflicts
FlextEventHandler = FlextHandlers.CQRS.EventHandler
FlextMetricsHandler = FlextHandlers.Base.MetricsHandler
FlextHandlerRegistry = FlextHandlers.Patterns.HandlerRegistry
FlextHandlerChain = FlextHandlers.Patterns.HandlerChain

FlextCommandHandler = FlextHandlers.CQRS.CommandHandler
FlextQueryHandler = FlextHandlers.CQRS.QueryHandler


# Legacy facade for backward compatibility
class HandlersFacade:
    """Legacy facade for backward compatibility.

    Provides the same interface as the old handlers module while
    delegating to the new hierarchical structure.
    """

    # Delegate to new structure
    Handler = FlextHandlers.Base.Handler
    ValidatingHandler = FlextHandlers.Base.ValidatingHandler
    AuthorizingHandler = FlextHandlers.Base.AuthorizingHandler
    MetricsHandler = FlextHandlers.Base.MetricsHandler
    EventHandler = FlextHandlers.CQRS.EventHandler
    HandlerChain = FlextHandlers.Patterns.HandlerChain
    HandlerRegistry = FlextHandlers.Patterns.HandlerRegistry
    CommandHandler = FlextHandlers.CQRS.CommandHandler
    QueryHandler = FlextHandlers.CQRS.QueryHandler
    CommandBus = FlextHandlers.CQRS.CommandBus
    QueryBus = FlextHandlers.CQRS.QueryBus


# =============================================================================
# Module-level Exports and Compatibility
# =============================================================================

__all__ = [
    "FlextAbstractHandler",
    "FlextAbstractHandlerChain",
    "FlextAbstractHandlerRegistry",
    "FlextAbstractMetricsHandler",
    "FlextAbstractValidatingHandler",
    # "FlextAuthorizingHandler", - moved to typings.py
    "FlextBaseHandler",
    "FlextCommandHandler",
    "FlextEventHandler",
    "FlextHandlerChain",
    "FlextHandlerRegistry",
    "FlextHandlers",
    "FlextMetricsHandler",
    "FlextQueryHandler",
    # "FlextValidatingHandler", - moved to typings.py
    "HandlersFacade",
]

# Legacy compatibility aliases for utility functions
_get_flext_commands_module = FlextHandlers.get_flext_commands_module
_thread_safe_operation = FlextHandlers.thread_safe_operation
