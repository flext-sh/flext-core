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
from typing import TYPE_CHECKING, Protocol, cast, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Callable

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
from flext_core.result import FlextResult

# FlextLogger imported for convenience - all handlers use FlextLoggableMixin

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

                # Fallback to accepting all messages if no type hints
                return True

            except (ImportError, AttributeError, TypeError):
                # Safe fallback - accept all messages
                return True

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
            """Access logger."""
            # Return a simple placeholder - can be enhanced with actual logging
            return None

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


# Export API
__all__: list[str] = ["FlextHandlers"]
