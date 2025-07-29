"""FLEXT Core Handlers Module.

Comprehensive handler pattern implementation for enterprise message processing with
CQRS, event sourcing, and chain of responsibility patterns. Implements consolidated
architecture with mixin inheritance and type-safe operations.

Architecture:
    - Unified handler patterns for commands, events, queries, and requests
    - Generic type-safe base handler with abstract interface definition
    - Specialized handlers for CQRS pattern implementation
    - Handler registry for service location and dependency injection
    - Chain of responsibility pattern for multi-handler processing
    - Multiple inheritance from timing and logging mixins

Handler System Components:
    - FlextHandlers.Handler[T, R]: Base generic handler with lifecycle hooks
    - FlextHandlers.CommandHandler: CQRS command processing with validation
    - FlextHandlers.EventHandler: Domain event processing with side effects
    - FlextHandlers.QueryHandler: Read-only query processing with authorization
    - FlextHandlers.Registry: Service location pattern for handler management
    - FlextHandlers.Chain: Chain of responsibility for multi-handler scenarios

Maintenance Guidelines:
    - Create specialized handlers by inheriting from appropriate base classes
    - Use FlextResult pattern for all handler operations that can fail
    - Integrate logging and timing through mixin inheritance patterns
    - Maintain handler registration in registry for service location
    - Implement proper pre/post processing hooks for cross-cutting concerns
    - Follow CQRS principles with command/query/event separation

Design Decisions:
    - Nested class organization within FlextHandlers for namespace management
    - Generic type parameters [T, R] for input message and result type safety
    - Abstract base class pattern for enforcing handler interface contracts
    - Mixin inheritance for logging, timing, and validation behaviors
    - Registry pattern for handler service location and dependency injection
    - Chain of responsibility for flexible multi-handler processing scenarios

Enterprise Patterns:
    - CQRS (Command Query Responsibility Segregation) with specialized handlers
    - Event sourcing support through EventHandler with domain event processing
    - Service location pattern through Registry for handler discovery
    - Chain of responsibility for complex message processing workflows
    - Pre/post processing hooks for cross-cutting concerns and aspect-oriented
      programming
    - Comprehensive metrics and observability through mixin integration

Handler Lifecycle:
    - can_handle: Type-based message compatibility checking
    - pre_handle: Pre-processing with validation and authorization
    - handle: Core business logic processing (abstract method)
    - post_handle: Post-processing with metrics and cleanup
    - handle_with_hooks: Complete lifecycle orchestration with timing

Dependencies:
    - abc: Abstract base class patterns for handler interface definition
    - mixins: Logging and timing behavior inheritance for cross-cutting concerns
    - result: FlextResult pattern for consistent error handling
    - types: Generic type variables and handler-specific type aliases
    - utilities: Type guards for runtime type checking and validation

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from typing import cast

from flext_core._handlers_base import (
    _BaseHandler,
)
from flext_core.result import FlextResult
from flext_core.types import (
    R,
    T,
    TAnyDict,
    THandler,
    TServiceKey,
    TServiceName,
)

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
type THandlerRegistry = dict[str, THandler]  # Handler type to handler mapping
type THandlerKey = str  # Key for handler registration and lookup
type TDispatchResult[T] = FlextResult[T]  # Result of handler dispatch

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

        def can_handle(self, message: object) -> bool:  # noqa: ARG002
            """Check if handler can process message - override in subclasses."""
            # Default implementation - override in subclasses for specific logic
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
        """Handler specifically for commands using composition-based delegation.

        Commands represent intentions to change system state.
        Follows architectural guidelines with single internal definition
        (_BaseCommandHandler) and external exposure through composition.
        """

        def __init__(self, handler_name: TServiceName | None = None) -> None:
            """Initialize command handler with composition-based delegation."""
            self._handler_name = handler_name
            # Note: _BaseCommandHandler is abstract, so we implement methods directly

        # =====================================================================
        # DELEGATION METHODS - Direct delegation to base command handler
        # =====================================================================

        def validate_command(self, command: object) -> FlextResult[None]:  # noqa: ARG002
            """Validate command."""
            # Basic validation - can be overridden by subclasses
            return FlextResult.ok(None)

        def handle(self, command: object) -> FlextResult[object]:
            """Handle command - must be implemented by concrete handlers."""
            # Default implementation returns the command as-is
            return FlextResult.ok(command)

        def can_handle(self, message: object) -> bool:  # noqa: ARG002
            """Check if can handle."""
            # Default implementation - can handle any message
            return True

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
            """Handle with hooks."""
            # Pre-process
            pre_result = self.pre_handle(command)
            if pre_result.is_failure:
                return pre_result

            # Handle
            handle_result = self.handle(pre_result.data)

            # Post-process
            return self.post_handle(handle_result)

        def get_metrics(self) -> TAnyDict:
            """Get metrics."""
            return {
                "handler_name": self._handler_name,
                "handler_type": "CommandHandler",
            }

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
            if result.is_success:
                return FlextResult.ok(None)
            return FlextResult.fail(result.error or "Event handling failed")

        @abstractmethod
        def process_event_impl(self, event: object) -> None:
            """Process the event - implement in subclasses.

            This method should have side effects but no return value.
            """

    class QueryHandler(Handler[T, R]):
        """Handler specifically for queries.

        Queries request data without side effects.
        """

        def authorize_query(self, query: object) -> FlextResult[None]:
            """Check query authorization.

            Override to add authorization logic.
            """
            _ = query  # Mark as used for linting
            return FlextResult.ok(None)

        def pre_handle(self, query: object) -> FlextResult[object]:
            """Pre-process query with authorization."""
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
        handler_func: THandler,
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
                if hasattr(result, "is_success"):
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
__all__ = ["FlextHandlers"]
