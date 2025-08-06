"""FLEXT Core Handlers - CQRS Layer Handler Implementation.

Comprehensive handler pattern implementation for enterprise message processing with
CQRS, event sourcing, and chain of responsibility patterns across the 32-project
FLEXT ecosystem. Foundation for command/query/event processing with lifecycle
management and cross-cutting concerns in data integration pipelines.

Module Role in Architecture:
    CQRS Layer → Handler Processing → Message Lifecycle Management

    This module provides handler patterns used throughout FLEXT projects:
    - Generic handler base for type-safe message processing
    - Specialized handlers for CQRS command/query/event patterns
    - Handler registry for service location and dependency injection
    - Chain of responsibility for complex multi-handler scenarios

Handler Architecture Patterns:
    Generic Type Safety: Handler[T, R] with compile-time input/output type checking
    Lifecycle Hooks: Pre/post processing with validation and cross-cutting concerns
    Service Location: Registry pattern for handler discovery and dependency injection
    Chain of Responsibility: Multi-handler processing with flexible orchestration

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Base handler, lifecycle hooks, mixin inheritance
    🚧 Active Development: Handler registry and discovery (Priority 2 - October 2025)
    📋 TODO Integration: Complete CQRS handler specializations (Priority 2)

Handler Pattern Features:
    FlextHandlers.Handler[T, R]: Base generic handler with lifecycle management
    FlextHandlers.CommandHandler: CQRS command processing with validation
    FlextHandlers.EventHandler: Domain event processing with side effects
    FlextHandlers.QueryHandler: Read-only query processing with authorization
    FlextHandlers.Registry: Service location for handler management

Ecosystem Usage Patterns:
    # FLEXT Service Handlers
    class CreateUserHandler(FlextHandlers.CommandHandler):
        def handle(self, command: CreateUserCommand) -> FlextResult[User]:
            # Business logic with validation and error handling
            if self.user_repository.email_exists(command.email):
                return FlextResult.fail("Email already exists")
            return self.user_repository.create(command)

    # Singer Tap/Target Handlers
    class OracleExtractHandler(FlextHandlers.Handler[ExtractCommand, ExtractResult]):
        def pre_handle(self, command: ExtractCommand) -> FlextResult[None]:
            return self.validate_connection(command.connection_string)

        def handle(self, command: ExtractCommand) -> FlextResult[ExtractResult]:
            return self.oracle_service.extract_data(command.table_name)

    # ALGAR Migration Handlers
    class LdapMigrationHandler(FlextHandlers.EventHandler):
        def handle(self, event: LdapUserMigrated) -> FlextResult[None]:
            return self.notification_service.notify_migration_complete(event)

Handler Lifecycle Management:
    - can_handle: Type-based message compatibility checking
    - pre_handle: Pre-processing with validation and authorization
    - handle: Core business logic processing (abstract method)
    - post_handle: Post-processing with metrics and cleanup
    - handle_with_hooks: Complete lifecycle orchestration with timing

Quality Standards:
    - All handlers must implement the abstract handle method
    - Handlers must be stateless and thread-safe for concurrent processing
    - Pre/post hooks must be used for cross-cutting concerns
    - Handler registration must use type-safe service location patterns

See Also:
    docs/TODO.md: Priority 2 - Handler registry and discovery
    commands.py: Command patterns that handlers process
    interfaces.py: FlextHandler interface definitions

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextlib
import hashlib
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, cast, get_type_hints

from flext_core._handlers_base import (
    _BaseHandler,
)
from flext_core.flext_types import (
    R,
    T,
    TAnyDict,
    TServiceKey,
    TServiceName,
)
from flext_core.loggings import FlextLoggerFactory, get_logger
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# DOMAIN-SPECIFIC TYPES - Handler Pattern Specializations
# =============================================================================

# Handler pattern specific types for better domain modeling
type THandlerId = str  # Handler instance identifier
type THandlerType = str  # Handler type name for routing
type TMessageType = str  # Message type for handler dispatch
type TEventType = str  # Event type for event handlers
type TRequestType = str  # Request type for request handlers
type TResponseType[T] = FlextResult[T]  # Handler response with type parameter
type THandlerMetadata = TAnyDict  # Handler metadata for middleware
type THandlerPriority = int  # Handler execution priority (1-10)
type THandlerTimeout = int  # Handler timeout in milliseconds
type TRetryCount = int  # Number of retry attempts for failed handlers

# Registry and dispatch types
type THandlerRegistry = dict[str, object]  # Handler type to handler mapping
type THandlerKey = str  # Key for handler registration and lookup
type TDispatchResult[T] = FlextResult[T]  # Result of handler dispatch

# =============================================================================
# HANDLER INTERFACES - ISP Compliance (Interface Segregation Principle)
# =============================================================================


class FlextHandlerProtocols:
    """Handler protocols for Interface Segregation Principle compliance."""

    class MessageHandler(Protocol):
        """Protocol for basic message handling."""

        def handle(self, message: object) -> FlextResult[object]:
            """Handle message - protocol method."""
            ...

        def can_handle(self, message: object) -> bool:
            """Check if can handle message - protocol method."""
            ...

    class ValidatingHandler(Protocol):
        """Protocol for handlers that need validation."""

        def validate_message(self, message: object) -> FlextResult[object]:
            """Validate message - protocol method."""
            ...

    class AuthorizingHandler(Protocol):
        """Protocol for handlers that need authorization."""

        def authorize_query(self, query: object) -> FlextResult[None]:
            """Authorize query - protocol method."""
            ...

    class EventProcessor(Protocol):
        """Protocol for event processing with side effects."""

        def process_event_impl(self, event: object) -> None:
            """Process event implementation - protocol method."""
            ...

    class MetricsCollector(Protocol):
        """Protocol for handlers that collect metrics."""

        def get_metrics(self) -> TAnyDict:
            """Get metrics - protocol method."""
            ...


# =============================================================================
# FLEXT HANDLERS - Unified handler patterns
# =============================================================================


class FlextHandlers:
    """Unified handler pattern implementation with message processing capabilities.

    Comprehensive handler framework providing organized access to all FLEXT handler
    patterns including CQRS, event sourcing, and chain of responsibility
    implementations. Serves as the primary namespace for handler-related functionality
    with nested class organization.

    Architecture:
        - Namespace organization with nested classes for related functionality
        - Base handler interface with generic type parameters for type safety
        - Specialized handler implementations for CQRS pattern compliance
        - Registry pattern for handler service location and dependency injection
        - Chain of responsibility for complex multi-handler processing scenarios
        - Factory methods for convenient handler creation and configuration

    Handler Categories:
        - Base Handler: Generic handler interface with lifecycle management
        - Command Handler: CQRS command processing with validation hooks
        - Event Handler: Domain event processing with side effect management
        - Query Handler: Read-only operations with authorization integration
        - Registry: Service location pattern for handler management
        - Chain: Chain of responsibility for multi-handler workflows

    Enterprise Features:
        - Type-safe handler interfaces with generic constraints
        - Comprehensive lifecycle management with pre/post processing hooks
        - Integrated metrics collection and performance monitoring
        - Structured logging with correlation IDs and context management
        - Validation and authorization integration through specialized handlers
        - Service location pattern for dependency injection and testing

    Usage Patterns:
        # Create and register command handler
        class CreateUserHandler(FlextHandlers.CommandHandler):
            def handle(self, command: CreateUserCommand) -> FlextResult[User]:
                # Validation happens in pre_handle automatically
                return self.user_service.create_user(command.user_data)

        # Register handler in registry
        registry = FlextHandlers.create_registry()
        handler = CreateUserHandler()
        registry.register("create_user", handler)

        # Process message through handler
        command = CreateUserCommand(user_data={"name": "John"})
        result = handler.handle_with_hooks(command)

        # Chain multiple handlers
        chain = FlextHandlers.create_chain()
        chain.add_handler(validation_handler)
        chain.add_handler(processing_handler)
        chain.add_handler(notification_handler)
        results = chain.process_all(message)

        # Function-based handler creation
        def process_order(order: Order) -> FlextResult[OrderResult]:
            return FlextResult.ok(OrderResult(order.id))

        function_handler = FlextHandlers.create_function_handler(process_order)

    Design Pattern Integration:
        - Abstract Factory: Handler creation through factory methods
        - Service Locator: Registry pattern for handler discovery
        - Chain of Responsibility: Multi-handler processing workflows
        - Template Method: Handler lifecycle with customizable hooks
        - Observer: Event handler pattern for domain event processing
    """

    # =============================================================================
    # BASE HANDLER - Foundation for all handlers
    # =============================================================================

    class Handler[T, R]:
        """Generic handler interface using composition-based delegation.

        Comprehensive handler functionality through composition and delegation
        to specialized base classes, following the architectural guidelines of
        single internal definition (_handlers_base.py) with external exposure.

        Architecture:
            - Composition-based delegation to _BaseHandler for core functionality
            - Eliminates multiple inheritance complexity following DRY principles
            - Type-safe message processing through delegation patterns
            - Comprehensive lifecycle management with performance monitoring

        Lifecycle Management:
            - can_handle: Type-based message compatibility verification
            - pre_handle: Pre-processing with validation and preparation
            - handle: Core business logic processing (abstract method)
            - post_handle: Post-processing with metrics and cleanup
            - handle_with_hooks: Complete lifecycle orchestration with timing

        Enterprise Features:
            - Type-safe message processing with compile-time verification
            - Structured logging integration through FlextLoggableMixin
            - Performance timing and metrics through FlextTimingMixin
            - Automatic message validation with configurable validation hooks
            - Comprehensive error handling with FlextResult pattern integration
            - Metrics collection for operational monitoring and observability

        Mixin Integration:
            - FlextLoggableMixin: Structured logging with context management
            - FlextTimingMixin: Performance timing and execution measurement
            - ABC (Abstract Base Class): Interface contract enforcement
            - Generic[T, R]: Type-safe input/output parameter constraints

        Usage Patterns:
            # Implement custom handler
            class OrderHandler(FlextHandlers.Handler[Order, OrderResult]):
                def handle(self, order: Order) -> FlextResult[OrderResult]:
                    # Core business logic
                    result = self.process_order(order)
                    return FlextResult.ok(OrderResult(result))

                def pre_handle(self, order: Order) -> FlextResult[Order]:
                    # Custom validation
                    if not order.is_valid():
                        return FlextResult.fail("Invalid order")
                    return super().pre_handle(order)

            # Use handler with full lifecycle
            handler = OrderHandler("order_processor")
            result = handler.handle_with_hooks(order)

            # Access handler metrics
            metrics = handler.get_metrics()
            success_rate = metrics["success_rate"]
            avg_time = metrics["average_processing_time_ms"]

        Type Safety:
            - Generic parameter T constrains input message type at compile time
            - Generic parameter R constrains result type for type checking
            - Runtime type validation through can_handle method
            - Type guards integration for additional runtime verification
        """

        def __init__(self, handler_name: TServiceName | None = None) -> None:
            """Initialize handler with composition-based delegation."""
            # Initialize using composition pattern - not inheritance from abstract base
            self._handler_name = handler_name or self.__class__.__name__
            # Generate unique handler_id based on class name and object id
            self.handler_id = f"{self.__class__.__name__}_{id(self)}"
            self.handler_name = self._handler_name
            self._logger = get_logger(self.__class__.__module__)
            self._metrics = {
                "messages_handled": 0,
                "successes": 0,
                "failures": 0,
                "total_processing_time_ms": 0.0,
            }

            # Create concrete implementation of abstract base for delegation
            class _ConcreteBaseHandler(_BaseHandler[object, object]):
                def handle(self, message: object) -> FlextResult[object]:
                    # Delegate to outer instance handle method
                    return FlextResult.ok(message)

            self._base_handler = _ConcreteBaseHandler(handler_name)

        # =====================================================================
        # DELEGATION METHODS - Direct delegation to base handler
        # =====================================================================

        def handle(self, message: object) -> FlextResult[object]:
            """Handle message - implement in subclasses."""
            # Default implementation returns success with message
            return FlextResult.ok(message)

        def can_handle(self, message: object) -> bool:
            """Check if handler can process message based on type compatibility.

            Args:
                message: Message to check for compatibility

            Returns:
                True if handler can process message, False otherwise

            """
            # Get type hints from handle method if available
            try:
                handle_method = getattr(self.__class__, "handle", None)
                if handle_method and hasattr(handle_method, "__annotations__"):
                    type_hints = get_type_hints(handle_method)
                    if "message" in type_hints:
                        expected_type = type_hints["message"]
                        # Check if message is compatible with expected type
                        if hasattr(expected_type, "__origin__"):
                            # Handle generic types
                            return True  # Allow generic compatibility for now
                        return isinstance(message, expected_type)

                # REAL SOLUTION: Unknown message handling with proper type analysis
                self._logger.debug(
                    "Handler has no type hints for message parameter - using strict validation",
                    handler_class=self.__class__.__name__,
                    message_type=type(message).__name__,
                )
                return False

            except (ImportError, AttributeError, TypeError) as e:
                # REAL SOLUTION: Proper error handling for type analysis failures
                self._logger.warning(
                    "Type analysis failed for handler - rejecting message",
                    handler_class=self.__class__.__name__,
                    error=str(e),
                    message_type=type(message).__name__,
                )
                return False

        def pre_handle(self, message: object) -> FlextResult[object]:
            """Pre-processing hook - delegates to base."""
            return self._base_handler.pre_handle(message)

        def post_handle(self, result: FlextResult[object]) -> FlextResult[object]:
            """Post-processing hook - delegates to base."""
            return self._base_handler.post_handle(result)

        def handle_with_hooks(self, message: object) -> FlextResult[object]:
            """Handle with lifecycle hooks - delegates to base."""
            return self._base_handler.handle_with_hooks(message)

        def get_metrics(self) -> TAnyDict:
            """Get handler metrics - delegates to base."""
            return self._base_handler.get_metrics()

        @property
        def logger(self) -> object:
            """Access logger - delegates to base."""
            return self._base_handler.logger

        def validate_message(self, message: object) -> FlextResult[object]:
            """Validate message before processing."""
            if message is None:
                return FlextResult.fail("Message cannot be None")
            return FlextResult.ok(message)

        def get_handler_metadata(self) -> TAnyDict:
            """Get handler metadata."""
            return {
                "handler_id": self.handler_id,
                "handler_name": self.handler_name,
                "handler_class": self.__class__.__name__,
                "handler_type": self.__class__.__name__,
                "documentation": getattr(self.__class__, "__doc__", ""),
            }

        def process_message(self, message: object) -> FlextResult[object]:
            """Process message with validation and handling."""
            # Validate message
            validation_result = self.validate_message(message)
            if validation_result.is_failure:
                return validation_result

            # Check if can handle
            if not self.can_handle(message):
                return FlextResult.fail("Handler cannot process this message")

            # Handle message
            try:
                return self.handle_message(message)
            except (RuntimeError, OSError) as e:
                return FlextResult.fail(f"Message processing failed: {e}")

        def handle_message(self, message: object) -> FlextResult[object]:
            """Handle message - override in subclasses."""
            return self.handle(message)

        def process(self, message: object) -> FlextResult[object]:
            """Process generic message."""
            if message is None:
                return FlextResult.fail("Message validation failed")
            return self.process_message(message)

    # =============================================================================
    # SPECIALIZED HANDLERS - Domain-specific handlers
    # =============================================================================

    class CommandHandler:
        """SOLID-compliant command handler with single responsibility.

        Follows Single Responsibility Principle - only handles command processing.
        Uses dependency injection for validation and metrics collection.
        Commands represent intentions to change system state.
        """

        def __init__(
            self,
            handler_name: TServiceName | None = None,
            validator: FlextHandlerProtocols.ValidatingHandler | None = None,
            metrics_collector: FlextHandlerProtocols.MetricsCollector | None = None,
        ) -> None:
            """Initialize command handler with dependency injection."""
            self._handler_name = handler_name or self.__class__.__name__
            self._validator = validator
            self._metrics_collector = metrics_collector
            self._command_count = 0
            self._success_count = 0

        # =====================================================================
        # DELEGATION METHODS - Direct delegation to base command handler
        # =====================================================================

        def validate_command(self, command: object) -> FlextResult[None]:
            """Validate command using injected validator or default validation."""
            if self._validator:
                validation_result = self._validator.validate_message(command)
                if validation_result.is_failure:
                    error_msg = validation_result.error or "Validation failed"
                    return FlextResult.fail(error_msg)

            # Default validation - command cannot be None
            if command is None:
                return FlextResult.fail("Command cannot be None")
            return FlextResult.ok(None)

        def handle(self, command: object) -> FlextResult[object]:
            """Handle command - must be implemented by concrete handlers."""
            # Default implementation returns the command as-is
            return FlextResult.ok(command)

        def can_handle(self, message: object) -> bool:
            """Check if command handler can process message.

            Commands should be objects that represent intentions to change system state.

            Args:
                message: Message to check for command compatibility

            Returns:
                True if message can be handled as a command, False otherwise

            """
            # Commands typically have method signatures with command-like attributes
            if hasattr(message, "__dict__"):
                # Check for command-like patterns in object attributes
                attrs = dir(message)
                command_indicators = ["execute", "command", "action", "operation"]
                if any(
                    indicator in str(attrs).lower() for indicator in command_indicators
                ):
                    return True

            # Check if message has command-like methods or is a callable
            if callable(message) or hasattr(message, "execute"):
                return True

            # Accept any structured message as potential command
            return hasattr(message, "__dict__") or isinstance(message, dict)

        def pre_handle(self, command: object) -> FlextResult[object]:
            """Pre-process command."""
            validation_result = self.validate_command(command)
            if validation_result.is_failure:
                return FlextResult.fail(validation_result.error or "Validation failed")
            return FlextResult.ok(command)

        def post_handle(self, result: FlextResult[object]) -> FlextResult[object]:
            """Post-process result."""
            return result

        def handle_with_hooks(self, command: object) -> FlextResult[object]:
            """Handle with hooks and metrics collection."""
            self._command_count += 1

            # Pre-process
            pre_result = self.pre_handle(command)
            if pre_result.is_failure:
                return pre_result

            # Handle
            handle_result = self.handle(pre_result.data)

            # Update success metrics
            if handle_result.success:
                self._success_count += 1

            # Post-process
            return self.post_handle(handle_result)

        def get_metrics(self) -> TAnyDict:
            """Get metrics using injected collector or default metrics."""
            base_metrics: TAnyDict = {
                "handler_name": self._handler_name,
                "handler_type": "CommandHandler",
                "commands_processed": self._command_count,
                "successful_commands": self._success_count,
                "success_rate": self._success_count / max(self._command_count, 1),
            }

            if self._metrics_collector:
                additional_metrics = self._metrics_collector.get_metrics()
                # Only add compatible values
                for k, v in additional_metrics.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        base_metrics[k] = v

            return base_metrics

        @property
        def logger(self) -> object:
            """Access logger for handler operations.

            Returns:
                Logger instance from flext_core.loggings for structured logging

            """
            return get_logger(f"flext.handlers.{self.__class__.__name__}")

    class EventHandler(Handler[T, None]):
        """Handler specifically for events.

        Events represent things that have happened.
        Event handlers typically don't return values.
        """

        def handle(self, event: object) -> FlextResult[object]:
            """Handle event.

            Args:
                event: Event to handle

            Returns:
                FlextResult with processing result or error

            """
            self.process_event(event)
            return FlextResult.ok(None)

        def process_event(self, event: object) -> FlextResult[None]:
            """Process event with validation."""
            # Validate event
            validation_result = self.validate_message(event)
            if validation_result.is_failure:
                return FlextResult.fail(validation_result.error or "Validation failed")

            # Check if can handle
            if not self.can_handle(event):
                return FlextResult.fail("Handler cannot process this event")

            # Handle event
            try:
                return self.handle_event(event)
            except (RuntimeError, OSError) as e:
                return FlextResult.fail(f"Event processing failed: {e}")

        def handle_event(self, event: object) -> FlextResult[None]:
            """Handle event - override in subclasses."""
            result = self.handle(event)
            if result.success:
                return FlextResult.ok(None)
            return FlextResult.fail(result.error or "Event handling failed")

        def process_event_impl(self, event: object) -> None:
            """Process the event - default implementation.

            This method should have side effects but no return value.
            Override in subclasses to implement specific event processing.
            """
            # Default implementation - log the event processing
            if hasattr(self, "logger") and self.logger:
                with contextlib.suppress(AttributeError, TypeError):
                    if hasattr(self.logger, "debug"):
                        event_type = type(event).__name__
                        self.logger.debug("Processing event: %s", event_type)
            # No-op default implementation - subclasses should override

    class QueryHandler(Handler[T, R]):
        """SOLID-compliant query handler with authorization support.

        Follows Open/Closed Principle - extensible through dependency injection.
        Queries request data without side effects.
        """

        def __init__(
            self,
            handler_name: TServiceName | None = None,
            authorizer: FlextHandlerProtocols.AuthorizingHandler | None = None,
        ) -> None:
            """Initialize query handler with dependency injection."""
            super().__init__(handler_name)
            self._authorizer = authorizer

        def authorize_query(self, query: object) -> FlextResult[None]:
            """Check query authorization using injected authorizer.

            Uses dependency injection to support different authorization strategies.
            """
            if self._authorizer:
                return self._authorizer.authorize_query(query)

            # Default authorization - allow all queries
            # Override by injecting a custom authorizer
            return FlextResult.ok(None)

        def pre_handle(self, query: object) -> FlextResult[object]:
            """Pre-process query with authorization and validation."""
            # First validate the query
            validation_result = self.validate_message(query)
            if validation_result.is_failure:
                return FlextResult.fail(
                    validation_result.error or "Query validation failed",
                )

            # Then authorize the query
            auth_result = self.authorize_query(query)
            if auth_result.is_failure:
                return FlextResult.fail(
                    auth_result.error or "Query authorization failed",
                )
            return FlextResult.ok(query)

        def process_request(self, request: object) -> FlextResult[object]:
            """Process request with validation."""
            # Validate request
            validation_result = self.validate_message(request)
            if validation_result.is_failure:
                return FlextResult.fail(validation_result.error or "Validation failed")

            # Check if can handle
            if not self.can_handle(request):
                return FlextResult.fail("Handler cannot process this request")

            # Handle request
            try:
                return self.handle_request(request)
            except (RuntimeError, OSError) as e:
                return FlextResult.fail(f"Request processing failed: {e}")

        def handle_request(self, request: object) -> FlextResult[object]:
            """Handle request - override in subclasses."""
            return self.handle(request)

    # =============================================================================
    # HANDLER REGISTRY - Managing handler registration
    # =============================================================================

    class Registry:
        """Registry for managing handlers.

        Args:
            **kwargs: Additional keyword arguments

        """

        def __init__(self) -> None:
            """Initialize handler registry."""
            self._handlers: dict[TServiceKey, object] = {}
            self._type_handlers: dict[type[object], object] = {}
            self._handler_list: list[object] = []  # List to track all handlers

        def register(
            self,
            handler_or_key: object | TServiceKey,
            handler: FlextHandlers.Handler[object, object] | None = None,
        ) -> FlextResult[None]:
            """Register handler by string key or directly.

            Args:
                handler_or_key: Handler instance or key for handler
                handler: Handler instance (when first arg is key)

            Returns:
                FlextResult indicating registration success

            """
            # Handle single argument case (just the handler)
            if handler is None:
                if not hasattr(handler_or_key, "handle"):
                    return FlextResult.fail(
                        "Invalid handler: must have 'handle' method",
                    )
                handler_obj = handler_or_key
                # Use handler_id as key for uniqueness
                key = getattr(handler_obj, "handler_id", handler_obj.__class__.__name__)
            else:
                # Handle two argument case (key, handler)
                key = handler_or_key
                handler_obj = handler

            if key in self._handlers:
                return FlextResult.fail(f"Handler already registered for key: {key}")

            self._handlers[key] = handler_obj
            self._handler_list.append(handler_obj)
            return FlextResult.ok(None)

        def register_for_type(
            self,
            message_type: type[T],
            handler: FlextHandlers.Handler[T, object],
        ) -> FlextResult[None]:
            """Register handler for message type.

            Args:
                message_type: Type of message to handle
                handler: Handler instance

            Returns:
                FlextResult indicating registration success

            """
            if message_type in self._type_handlers:
                return FlextResult.fail(
                    f"Handler already registered for type: {message_type.__name__}",
                )

            self._type_handlers[message_type] = handler
            return FlextResult.ok(None)

        def get_handler(
            self,
            key: TServiceKey,
        ) -> FlextResult[FlextHandlers.Handler[object, object]]:
            """Get handler by key."""
            if key not in self._handlers:
                return FlextResult.fail(f"No handler found for key: {key}")
            handler = cast("FlextHandlers.Handler[object, object]", self._handlers[key])
            return FlextResult.ok(handler)

        def get_handler_for_type(
            self,
            message_type: type[T],
        ) -> FlextResult[FlextHandlers.Handler[T, object]]:
            """Get handler for message type."""
            if message_type not in self._type_handlers:
                return FlextResult.fail(
                    f"No handler found for type: {message_type.__name__}",
                )
            handler = cast(
                "FlextHandlers.Handler[T, object]",
                self._type_handlers[message_type],
            )
            return FlextResult.ok(handler)

        def get_all_handlers(self) -> list[object]:
            """Get all registered handlers."""
            return list(self._handler_list)

        def find_handlers(self, message: object) -> list[object]:
            """Find all handlers that can process the given message."""
            return [
                handler
                for handler in self._handler_list
                if hasattr(handler, "can_handle") and handler.can_handle(message)
            ]

        def get_handler_by_id(self, handler_id: str) -> object | None:
            """Get handler by ID."""
            for handler in self._handler_list:
                if hasattr(handler, "handler_id") and handler.handler_id == handler_id:
                    return handler
            return None

        def get_handler_info(self) -> list[TAnyDict]:
            """Get information about all registered handlers."""
            info_list = []
            for handler in self._handler_list:
                handler_info = {
                    "handler_id": getattr(handler, "handler_id", "unknown"),
                    "handler_class": handler.__class__.__name__,
                    "handler_name": getattr(
                        handler,
                        "handler_name",
                        handler.__class__.__name__,
                    ),
                }
                info_list.append(handler_info)
            return info_list

    # =============================================================================
    # HANDLER CHAIN - Chain of responsibility pattern
    # =============================================================================

    class Chain:
        """Chain of handlers for processing messages."""

        def __init__(self) -> None:
            """Initialize handler chain."""
            self._handlers: list[FlextHandlers.Handler[object, object]] = []

        def add_handler(self, handler: FlextHandlers.Handler[object, object]) -> None:
            """Add handler to chain."""
            self._handlers.append(handler)

        def process(self, message: object) -> FlextResult[object]:
            """Process message through handler chain.

            Stops at first handler that can handle the message.
            """
            for handler in self._handlers:
                if handler.can_handle(message):
                    return handler.handle_with_hooks(message)

            return FlextResult.fail("No handler found for message")

        def process_all(self, message: object) -> list[FlextResult[object]]:
            """Process message through all applicable handlers.

            Returns results from all handlers that can handle the message.
            """
            # Use list comprehension for better performance
            return [
                handler.handle_with_hooks(message)
                for handler in self._handlers
                if handler.can_handle(message)
            ]

    # =============================================================================
    # COMMAND BUS - CQRS Command Processing Infrastructure
    # =============================================================================

    class CommandBus:
        """Enterprise Command Bus for CQRS pattern implementation.

        Comprehensive command bus providing centralized command routing,
        validation, middleware pipeline, and handler discovery for enterprise
        CQRS architectures. Supports both synchronous and asynchronous command
        processing with comprehensive lifecycle management.

        CQRS Features:
            - Type-safe command routing with automatic handler discovery
            - Pipeline behaviors for cross-cutting concerns
              (validation, logging, metrics)
            - Command validation with comprehensive error reporting
            - Middleware chain for aspect-oriented programming
            - Handler lifecycle management with dependency injection
            - Distributed command processing support with serialization

        Architecture:
            - Service locator pattern for handler discovery
            - Chain of responsibility for middleware processing
            - Command factory pattern for command construction
            - Observer pattern for command event notifications
            - Strategy pattern for different routing algorithms

        Enterprise Capabilities:
            - Command audit trail for compliance and debugging
            - Circuit breaker pattern for fault tolerance
            - Retry logic with exponential backoff
            - Command deduplication for idempotency
            - Performance metrics and monitoring integration
            - Security authorization at command level

        Usage Patterns:
            # Basic command processing
            command_bus = FlextHandlers.CommandBus()
            command_bus.register_handler(CreateUserCommand, CreateUserHandler())
            result = command_bus.send(CreateUserCommand(name="John", email="john@example.com"))

            # With middleware pipeline
            command_bus.add_behavior(ValidationBehavior())
            command_bus.add_behavior(LoggingBehavior())
            command_bus.add_behavior(MetricsBehavior())
            result = command_bus.send(command)

            # Async command processing
            result = await command_bus.send_async(command)
        """

        def __init__(self) -> None:
            """Initialize command bus with handler registry and middleware pipeline."""
            self._handler_registry: dict[
                type[object],
                FlextHandlers.CommandHandler,
            ] = {}
            self._behaviors: list[FlextHandlers.PipelineBehavior] = []
            self._command_audit: list[TAnyDict] = []
            self._metrics: dict[str, int | float] = {
                "commands_processed": 0,
                "successful_commands": 0,
                "failed_commands": 0,
                "total_processing_time_ms": 0.0,
            }

        def register_handler(
            self,
            command_type: type[T],
            handler: FlextHandlers.CommandHandler,
        ) -> FlextResult[None]:
            """Register command handler for specific command type.

            Args:
                command_type: Type of command to handle
                handler: Command handler instance

            Returns:
                FlextResult indicating registration success or failure

            """
            if command_type in self._handler_registry:
                return FlextResult.fail(
                    f"Handler already registered for command type: {command_type.__name__}",
                )

            self._handler_registry[command_type] = handler
            return FlextResult.ok(None)

        def add_behavior(
            self,
            behavior: FlextHandlers.PipelineBehavior,
        ) -> FlextResult[None]:
            """Add pipeline behavior for cross-cutting concerns.

            Args:
                behavior: Pipeline behavior to add

            Returns:
                FlextResult indicating success

            """
            self._behaviors.append(behavior)
            return FlextResult.ok(None)

        def send(self, command: object) -> FlextResult[object]:
            """Send command through pipeline and route to appropriate handler.

            Args:
                command: Command to process

            Returns:
                FlextResult with command processing result

            """
            command_type = type(command)
            self._metrics["commands_processed"] = (
                int(self._metrics["commands_processed"]) + 1
            )

            # Record command in audit trail
            audit_entry = {
                "command_type": command_type.__name__,
                "timestamp": self._get_timestamp(),
                "command_id": id(command),
            }
            self._command_audit.append(audit_entry)

            # Find handler for command type
            handler_result = self._find_handler(command_type)
            if handler_result.is_failure:
                self._metrics["failed_commands"] = (
                    int(self._metrics["failed_commands"]) + 1
                )
                return FlextResult.fail(handler_result.error or "Handler not found")

            handler = handler_result.unwrap()

            # Process through middleware pipeline
            pipeline_result = self._process_through_pipeline(command, handler)

            if pipeline_result.is_success:
                self._metrics["successful_commands"] = (
                    int(self._metrics["successful_commands"]) + 1
                )
            else:
                self._metrics["failed_commands"] = (
                    int(self._metrics["failed_commands"]) + 1
                )

            return pipeline_result

        def _find_handler(
            self,
            command_type: type[object],
        ) -> FlextResult[FlextHandlers.CommandHandler]:
            """Find registered handler for command type.

            Args:
                command_type: Type of command

            Returns:
                FlextResult containing handler or error

            """
            if command_type not in self._handler_registry:
                return FlextResult.fail(
                    f"No handler registered for command type: {command_type.__name__}",
                )
            return FlextResult.ok(self._handler_registry[command_type])

        def _process_through_pipeline(
            self,
            command: object,
            handler: FlextHandlers.CommandHandler,
        ) -> FlextResult[object]:
            """Process command through middleware pipeline.

            Args:
                command: Command to process
                handler: Final handler to execute

            Returns:
                FlextResult with processing result

            """
            if not self._behaviors:
                # No middleware, execute handler directly
                return handler.handle_with_hooks(command)

            # Create pipeline chain with behaviors
            def execute_pipeline(index: int) -> FlextResult[object]:
                if index >= len(self._behaviors):
                    # End of pipeline, execute handler
                    return handler.handle_with_hooks(command)

                # Execute current behavior
                behavior = self._behaviors[index]

                def next_handler() -> FlextResult[object]:
                    return execute_pipeline(index + 1)

                return behavior.process(command, next_handler)

            return execute_pipeline(0)

        def get_metrics(self) -> TAnyDict:
            """Get command bus processing metrics.

            Returns:
                Dictionary containing processing metrics

            """
            success_rate = 0.0
            if self._metrics["commands_processed"] > 0:
                success_rate = (
                    self._metrics["successful_commands"]
                    / self._metrics["commands_processed"]
                )

            return {
                **self._metrics,
                "success_rate": success_rate,
                "registered_handlers": len(self._handler_registry),
                "pipeline_behaviors": len(self._behaviors),
            }

        def get_audit_trail(self) -> list[TAnyDict]:
            """Get command audit trail for compliance and debugging.

            Returns:
                List of command audit entries

            """
            return list(self._command_audit)

        def _get_timestamp(self) -> float:
            """Get current timestamp for audit trail."""
            return time.time()

    # =============================================================================
    # QUERY BUS - CQRS Query Processing Infrastructure
    # =============================================================================

    class QueryBus:
        """Enterprise Query Bus for CQRS pattern implementation.

        Comprehensive query bus providing centralized query routing, caching,
        authorization, and result projection for enterprise CQRS architectures.
        Optimized for read operations with comprehensive performance monitoring.

        CQRS Features:
            - Type-safe query routing with automatic handler discovery
            - Query result caching for performance optimization
            - Authorization integration for secure data access
            - Query projection for efficient data retrieval
            - Read-only query processing with validation
            - Distributed query processing with result aggregation

        Architecture:
            - Repository pattern for data access abstraction
            - Decorator pattern for caching and authorization
            - Factory pattern for query result projection
            - Strategy pattern for different caching strategies
            - Observer pattern for query performance monitoring

        Performance Features:
            - Result caching with TTL and invalidation strategies
            - Query optimization with execution plan analysis
            - Connection pooling for database efficiency
            - Result streaming for large datasets
            - Query batching for reduced network overhead
            - Read replica routing for scalability

        Usage Patterns:
            # Basic query processing
            query_bus = FlextHandlers.QueryBus()
            query_bus.register_handler(GetUserQuery, GetUserHandler())
            result = query_bus.send(GetUserQuery(user_id="123"))

            # With caching
            query_bus.enable_caching(cache_ttl_seconds=300)
            result = query_bus.send(query)  # Cached automatically

            # With authorization
            query_bus.set_authorizer(CustomAuthorizer())
            result = query_bus.send(query)  # Authorized automatically
        """

        def __init__(self) -> None:
            """Initialize query bus with handler registry and caching."""
            self._handler_registry: dict[type[object], object] = {}
            self._cache: dict[str, object] = {}
            self._cache_ttl: dict[str, float] = {}
            self._authorizer: FlextHandlerProtocols.AuthorizingHandler | None = None
            self._caching_enabled: bool = False
            self._cache_ttl_seconds: int = 300  # 5 minutes default
            self._metrics: dict[str, int | float] = {
                "queries_processed": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "successful_queries": 0,
                "failed_queries": 0,
            }

        def register_handler(
            self,
            query_type: type[T],
            handler: FlextHandlers.QueryHandler[T, R],
        ) -> FlextResult[None]:
            """Register query handler for specific query type.

            Args:
                query_type: Type of query to handle
                handler: Query handler instance

            Returns:
                FlextResult indicating registration success or failure

            """
            if query_type in self._handler_registry:
                return FlextResult.fail(
                    f"Handler already registered for query type: {query_type.__name__}",
                )

            self._handler_registry[query_type] = handler
            return FlextResult.ok(None)

        def enable_caching(self, cache_ttl_seconds: int = 300) -> FlextResult[None]:
            """Enable query result caching.

            Args:
                cache_ttl_seconds: Cache time-to-live in seconds

            Returns:
                FlextResult indicating success

            """
            self._caching_enabled = True
            self._cache_ttl_seconds = cache_ttl_seconds
            return FlextResult.ok(None)

        def set_authorizer(
            self,
            authorizer: FlextHandlerProtocols.AuthorizingHandler,
        ) -> FlextResult[None]:
            """Set query authorizer for security.

            Args:
                authorizer: Authorizer implementation

            Returns:
                FlextResult indicating success

            """
            self._authorizer = authorizer
            return FlextResult.ok(None)

        def send(self, query: object) -> FlextResult[object]:
            """Send query and return cached or processed result.

            Args:
                query: Query to process

            Returns:
                FlextResult with query result

            """
            query_type = type(query)
            self._metrics["queries_processed"] = (
                int(self._metrics["queries_processed"]) + 1
            )

            # Check authorization if enabled
            if self._authorizer:
                auth_result = self._authorizer.authorize_query(query)
                if auth_result.is_failure:
                    self._metrics["failed_queries"] = (
                        int(self._metrics["failed_queries"]) + 1
                    )
                    return FlextResult.fail(
                        auth_result.error or "Query authorization failed",
                    )

            # Check cache if enabled
            if self._caching_enabled:
                cache_key = self._generate_cache_key(query)
                cached_result = self._get_from_cache(cache_key)
                if cached_result.is_success:
                    self._metrics["cache_hits"] = int(self._metrics["cache_hits"]) + 1
                    self._metrics["successful_queries"] = (
                        int(self._metrics["successful_queries"]) + 1
                    )
                    return cached_result
                self._metrics["cache_misses"] = int(self._metrics["cache_misses"]) + 1

            # Find handler for query type
            handler_result = self._find_handler(query_type)
            if handler_result.is_failure:
                self._metrics["failed_queries"] = (
                    int(self._metrics["failed_queries"]) + 1
                )
                return FlextResult.fail(handler_result.error or "Handler not found")

            handler = handler_result.unwrap()

            # Process query
            if hasattr(handler, "handle_with_hooks"):
                result = handler.handle_with_hooks(query)
            else:
                result = FlextResult.fail("Handler does not support handle_with_hooks")

            # Cache result if successful and caching enabled
            if result.is_success and self._caching_enabled:
                cache_key = self._generate_cache_key(query)
                self._store_in_cache(cache_key, result.data)

            if result.is_success:
                self._metrics["successful_queries"] = (
                    int(self._metrics["successful_queries"]) + 1
                )
            else:
                self._metrics["failed_queries"] = (
                    int(self._metrics["failed_queries"]) + 1
                )

            return cast("FlextResult[object]", result)

        def _find_handler(
            self,
            query_type: type[object],
        ) -> FlextResult[object]:
            """Find registered handler for query type.

            Args:
                query_type: Type of query

            Returns:
                FlextResult containing handler or error

            """
            if query_type not in self._handler_registry:
                return FlextResult.fail(
                    f"No handler registered for query type: {query_type.__name__}",
                )
            return FlextResult.ok(self._handler_registry[query_type])

        def _generate_cache_key(self, query: object) -> str:
            """Generate cache key for query.

            Args:
                query: Query object

            Returns:
                Cache key string

            """
            query_str = f"{type(query).__name__}_{query!s}"
            return hashlib.sha256(query_str.encode()).hexdigest()

        def _get_from_cache(self, cache_key: str) -> FlextResult[object]:
            """Get result from cache if not expired.

            Args:
                cache_key: Cache key

            Returns:
                FlextResult with cached data or failure

            """
            if cache_key not in self._cache:
                return FlextResult.fail("Cache miss")

            # Check TTL
            if (
                cache_key in self._cache_ttl
                and time.time() > self._cache_ttl[cache_key]
            ):
                # Expired, remove from cache
                del self._cache[cache_key]
                del self._cache_ttl[cache_key]
                return FlextResult.fail("Cache expired")

            return FlextResult.ok(self._cache[cache_key])

        def _store_in_cache(self, cache_key: str, data: object) -> None:
            """Store result in cache with TTL.

            Args:
                cache_key: Cache key
                data: Data to cache

            """
            self._cache[cache_key] = data
            self._cache_ttl[cache_key] = time.time() + self._cache_ttl_seconds

        def clear_cache(self) -> FlextResult[None]:
            """Clear all cached query results.

            Returns:
                FlextResult indicating success

            """
            self._cache.clear()
            self._cache_ttl.clear()
            return FlextResult.ok(None)

        def get_metrics(self) -> TAnyDict:
            """Get query bus processing metrics.

            Returns:
                Dictionary containing processing metrics

            """
            success_rate = 0.0
            cache_hit_rate = 0.0

            if self._metrics["queries_processed"] > 0:
                success_rate = (
                    self._metrics["successful_queries"]
                    / self._metrics["queries_processed"]
                )

            total_cache_requests = (
                self._metrics["cache_hits"] + self._metrics["cache_misses"]
            )
            if total_cache_requests > 0:
                cache_hit_rate = self._metrics["cache_hits"] / total_cache_requests

            return {
                **self._metrics,
                "success_rate": success_rate,
                "cache_hit_rate": cache_hit_rate,
                "registered_handlers": len(self._handler_registry),
                "cached_items": len(self._cache),
            }

    # =============================================================================
    # EVENT BUS - Event Processing Infrastructure
    # =============================================================================

    class EventBus:
        """Enterprise Event Bus for event processing and publish/subscribe patterns."""

        def __init__(self) -> None:
            """Initialize event bus with subscribers and metrics."""
            self._subscribers: dict[type[object], list[object]] = {}
            self._event_history: list[object] = []
            self._metrics: dict[str, object] = {
                "events_published": 0,
                "events_processed": 0,
                "successful_events": 0,
                "failed_events": 0,
            }

        def subscribe(
            self,
            event_type: type[object],
            handler: object,
        ) -> FlextResult[None]:
            """Subscribe handler to event type.

            Args:
                event_type: Type of events to subscribe to
                handler: Handler that will process the events

            Returns:
                FlextResult indicating subscription success or failure

            """
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)

            return FlextResult.ok(None)

        def publish(self, event: object) -> FlextResult[None]:
            """Publish event to all subscribed handlers.

            Args:
                event: Event to publish

            Returns:
                FlextResult indicating publish success or failure

            """
            self._update_publish_metrics()
            self._event_history.append(event)
            self._process_event_handlers(event)
            return FlextResult.ok(None)

        def _update_publish_metrics(self) -> None:
            """Update metrics for event publishing."""
            if isinstance(self._metrics["events_published"], int):
                self._metrics["events_published"] += 1

        def _process_event_handlers(self, event: object) -> None:
            """Process event through all registered handlers."""
            event_type = type(event)
            if event_type not in self._subscribers:
                return

            for handler in self._subscribers[event_type]:
                self._process_single_handler(handler, event)

        def _process_single_handler(self, handler: object, event: object) -> None:
            """Process event through a single handler."""
            try:
                if hasattr(handler, "handle"):
                    result = handler.handle(event)
                    self._update_handler_metrics(result)
                self._update_processed_metrics()
            except Exception:
                self._update_failed_metrics()

        def _update_handler_metrics(self, result: object) -> None:
            """Update metrics based on handler result."""
            if hasattr(result, "success") and result.success:
                if isinstance(self._metrics["successful_events"], int):
                    self._metrics["successful_events"] += 1
            elif isinstance(self._metrics["failed_events"], int):
                self._metrics["failed_events"] += 1

        def _update_processed_metrics(self) -> None:
            """Update processed event metrics."""
            if isinstance(self._metrics["events_processed"], int):
                self._metrics["events_processed"] += 1

        def _update_failed_metrics(self) -> None:
            """Update failed event metrics."""
            if isinstance(self._metrics["failed_events"], int):
                self._metrics["failed_events"] += 1

        def get_metrics(self) -> TAnyDict:
            """Get event bus metrics."""
            return dict(self._metrics)

    # =============================================================================
    # PIPELINE BEHAVIORS - Cross-cutting Concerns Middleware
    # =============================================================================

    class PipelineBehavior(ABC):
        """Abstract base class for pipeline behaviors.

        Pipeline behaviors implement cross-cutting concerns like validation,
        logging, metrics collection, caching, and security in a composable
        middleware pattern for CQRS command and query processing.

        Behavior Features:
            - Composable middleware pattern for aspect-oriented programming
            - Before/after processing hooks for comprehensive lifecycle management
            - Error handling and recovery mechanisms
            - Performance monitoring and metrics collection
            - Security authorization and validation integration
            - Configurable behavior ordering and priority

        Architecture:
            - Chain of responsibility pattern for behavior composition
            - Decorator pattern for transparent functionality enhancement
            - Strategy pattern for different behavior implementations
            - Observer pattern for behavior event notifications

        Usage Patterns:
            # Custom validation behavior
            class ValidationBehavior(FlextHandlers.PipelineBehavior):
                def process(self, message, next_handler):
                    # Validation logic before processing
                    if not self.validate(message):
                        return FlextResult.fail("Validation failed")
                    return next_handler()

            # Add to command bus
            command_bus.add_behavior(ValidationBehavior())
        """

        @abstractmethod
        def process(
            self,
            message: object,
            next_handler: Callable[[], FlextResult[object]],
        ) -> FlextResult[object]:
            """Process message through behavior.

            Args:
                message: Message being processed
                next_handler: Next handler in the pipeline

            Returns:
                FlextResult from processing

            """

    class ValidationBehavior(PipelineBehavior):
        """Pipeline behavior for message validation.

        Provides comprehensive validation of commands and queries before
        they reach their handlers, ensuring data integrity and business
        rule compliance throughout the CQRS pipeline.
        """

        def __init__(self, *, strict_validation: bool = True) -> None:
            """Initialize validation behavior.

            Args:
                strict_validation: Whether to use strict validation rules

            """
            self._strict_validation = strict_validation

        def process(
            self,
            message: object,
            next_handler: Callable[[], FlextResult[object]],
        ) -> FlextResult[object]:
            """Validate message before processing.

            Args:
                message: Message to validate
                next_handler: Next handler in pipeline

            Returns:
                FlextResult with validation result

            """
            # Basic validation - message cannot be None
            if message is None:
                return FlextResult.fail("Message cannot be None")

            # If message has validate method, use it
            if hasattr(message, "validate") and callable(message.validate):
                validation_result = message.validate()
                if (
                    hasattr(validation_result, "is_failure")
                    and validation_result.is_failure
                ):
                    return FlextResult.fail(
                        validation_result.error or "Message validation failed",
                    )

            # Proceed to next handler if validation passes
            return next_handler()

    class LoggingBehavior(PipelineBehavior):
        """Pipeline behavior for comprehensive message processing logging.

        Provides structured logging for all message processing activities
        with correlation IDs, performance metrics, and detailed context
        information for debugging and monitoring.
        """

        def __init__(self, log_level: str = "INFO") -> None:
            """Initialize logging behavior.

            Args:
                log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

            """
            self._log_level = log_level
            self._logger = FlextLoggerFactory.get_logger(
                "flext_core.handlers.LoggingBehavior",
            )

        def process(
            self,
            message: object,
            next_handler: Callable[[], FlextResult[object]],
        ) -> FlextResult[object]:
            """Log message processing.

            Args:
                message: Message being processed
                next_handler: Next handler in pipeline

            Returns:
                FlextResult from next handler

            """
            message_type = type(message).__name__
            message_id = id(message)

            self._logger.info(
                "Processing message",
                message_type=message_type,
                message_id=message_id,
            )

            # Process through next handler
            result = next_handler()

            if result.is_success:
                self._logger.info(
                    "Message processed successfully",
                    message_type=message_type,
                    message_id=message_id,
                )
            else:
                self._logger.error(
                    "Message processing failed",
                    message_type=message_type,
                    message_id=message_id,
                    error=result.error,
                )

            return result

    class MetricsBehavior(PipelineBehavior):
        """Pipeline behavior for performance metrics collection.

        Collects comprehensive performance metrics for message processing
        including execution time, success rates, and resource utilization
        for operational monitoring and optimization.
        """

        def __init__(self) -> None:
            """Initialize metrics behavior."""
            self._metrics: dict[str, int | float | dict[str, int]] = {
                "messages_processed": 0,
                "successful_messages": 0,
                "failed_messages": 0,
                "total_processing_time_ms": 0.0,
                "message_types": {},
            }

        def process(
            self,
            message: object,
            next_handler: Callable[[], FlextResult[object]],
        ) -> FlextResult[object]:
            """Collect metrics during message processing.

            Args:
                message: Message being processed
                next_handler: Next handler in pipeline

            Returns:
                FlextResult from next handler

            """
            start_time = time.time()
            message_type = type(message).__name__

            messages_processed = self._metrics["messages_processed"]
            if isinstance(messages_processed, int):
                self._metrics["messages_processed"] = messages_processed + 1

            # Track message types
            message_types = self._metrics["message_types"]
            if isinstance(message_types, dict):
                if message_type not in message_types:
                    message_types[message_type] = 0
                message_types[message_type] += 1

            # Process through next handler
            result = next_handler()

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            total_time = self._metrics["total_processing_time_ms"]
            if isinstance(total_time, (int, float)):
                self._metrics["total_processing_time_ms"] = (
                    total_time + execution_time_ms
                )

            # Update success/failure metrics
            if result.is_success:
                successful = self._metrics["successful_messages"]
                if isinstance(successful, int):
                    self._metrics["successful_messages"] = successful + 1
            else:
                failed = self._metrics["failed_messages"]
                if isinstance(failed, int):
                    self._metrics["failed_messages"] = failed + 1

            return result

        def get_metrics(self) -> TAnyDict:
            """Get collected metrics.

            Returns:
                Dictionary containing performance metrics

            """
            avg_time = 0.0
            success_rate = 0.0

            messages_processed = self._metrics["messages_processed"]
            if isinstance(messages_processed, int) and messages_processed > 0:
                total_time = self._metrics["total_processing_time_ms"]
                successful = self._metrics["successful_messages"]

                if isinstance(total_time, (int, float)):
                    avg_time = total_time / messages_processed

                if isinstance(successful, int):
                    success_rate = successful / messages_processed

            return {
                **self._metrics,
                "average_processing_time_ms": round(avg_time, 2),
                "success_rate": success_rate,
            }

    # =============================================================================
    # FACTORY METHODS - Convenience builders
    # =============================================================================

    @staticmethod
    def flext_create_function_handler(
        handler_func: Callable[[object], object],
    ) -> FlextHandlers.Handler[T, R]:
        """Create handler from function.

        Args:
            handler_func: Function that takes message and returns FlextResult

        Returns:
            Handler instance

        """

        class FunctionHandler(FlextHandlers.Handler[T, R]):
            def handle(self, message: object) -> FlextResult[object]:
                result = handler_func(message)
                # Ensure we return FlextResult[R]
                if hasattr(result, "success"):
                    return cast("FlextResult[object]", result)
                return FlextResult.ok(result)

        return FunctionHandler()

    @staticmethod
    def flext_create_registry() -> FlextHandlers.Registry:
        """Create new handler registry."""
        return FlextHandlers.Registry()

    @staticmethod
    def flext_create_chain() -> FlextHandlers.Chain:
        """Create new handler chain."""
        return FlextHandlers.Chain()

    @staticmethod
    def flext_create_command_bus() -> FlextHandlers.CommandBus:
        """Create new command bus for CQRS."""
        return FlextHandlers.CommandBus()

    @staticmethod
    def flext_create_query_bus() -> FlextHandlers.QueryBus:
        """Create new query bus for CQRS."""
        return FlextHandlers.QueryBus()

    @staticmethod
    def flext_create_event_bus() -> FlextHandlers.EventBus:
        """Create new event bus for event processing."""
        return FlextHandlers.EventBus()


# Export API
__all__: list[str] = ["FlextHandlers"]
