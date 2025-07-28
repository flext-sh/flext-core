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
from flext_core.flext_types import (
    R,
    T,
    TAnyDict,
    THandler,
    TServiceKey,
    TServiceName,
)
from flext_core.result import FlextResult

# FlextLogger imported for convenience - all handlers use FlextLoggableMixin


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

    class Handler(Generic[T, R]):
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
            # Create base handler instance for delegation
            self._base_handler = _BaseHandler[object, object](handler_name)

        # =====================================================================
        # DELEGATION METHODS - Direct delegation to base handler
        # =====================================================================

        def handle(self, message: object) -> FlextResult[object]:
            """Handle message - delegates to base handler."""
            return self._base_handler.handle(message)

        def can_handle(self, message: object) -> bool:
            """Check if handler can process message - delegates to base."""
            return self._base_handler.can_handle(message)

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


        def can_handle(self, message: object) -> bool:
            """Check if handler can process this message.

            Uses FlextUtilities type guards for validation.
            """
            self.logger.debug(
                "Checking if handler can process message",
                message_type=type(message).__name__,
            )

            # Get expected message type from Generic parameter
            if hasattr(self, "__orig_bases__"):
                for base in self.__orig_bases__:
                    if hasattr(base, "__args__") and len(base.__args__) >= 1:
                        expected_type = base.__args__[0]
                        # Use FlextTypeGuards - MAXIMIZA base usage com DRY
                        can_handle = FlextTypeGuards.is_instance_of(
                            message,
                            expected_type,
                        )

                        self.logger.debug(
                            "Handler check result",
                            can_handle=can_handle,
                            expected_type=getattr(
                                expected_type,
                                "__name__",
                                str(expected_type),
                            ),
                        )
                        return can_handle

            # Default to True if we can't determine type
            return True

        def pre_handle(self, message: T) -> FlextResult[T]:
            """Pre-processing hook with logging.

            Called before main handle method.
            Default implementation validates message.
            """
            self.logger.debug(
                "Pre-processing message",
                message_type=type(message).__name__,
            )

            # Validate message if it has validation method
            if hasattr(message, "validate"):
                validation_result = message.validate()
                if (
                    hasattr(validation_result, "is_failure")
                    and validation_result.is_failure
                ):
                    self.logger.warning(
                        "Message validation failed in pre-processing",
                        error=validation_result.error,
                    )
                    return FlextResult.fail(
                        validation_result.error or "Validation failed",
                    )

            return FlextResult.ok(message)

        def post_handle(self, result: FlextResult[R]) -> FlextResult[R]:
            """Post-processing hook with metrics.

            Called after main handle method.
            Updates metrics and logs results.
            """
            # Update metrics
            if result.is_success:
                self._metrics["successes"] += 1
                self.logger.debug(
                    "Post-processing successful result",
                    handler=self._handler_name,
                )
            else:
                self._metrics["failures"] += 1
                self.logger.warning(
                    "Post-processing failed result",
                    handler=self._handler_name,
                    error=result.error,
                )

            return result

        def handle_with_hooks(self, message: T) -> FlextResult[R]:
            """Handle with pre/post processing hooks and full logging."""
            start_time = self._start_timing()

            self.logger.info(
                "Starting message handling",
                handler=self._handler_name,
                message_type=type(message).__name__,
            )

            # Check if can handle
            if not self.can_handle(message):
                error_msg = (
                    f"{self._handler_name} cannot handle {type(message).__name__}"
                )
                self.logger.error(error_msg)
                return FlextResult.fail(error_msg)

            # Pre-process
            pre_result = self.pre_handle(message)
            if pre_result.is_failure:
                self.logger.error(
                    "Pre-processing failed",
                    error=pre_result.error,
                )
                return FlextResult.fail(pre_result.error or "Pre-processing failed")

            # Main handling
            processed_message = pre_result.unwrap()
            main_result = self.handle(processed_message)

            # Post-process
            final_result = self.post_handle(main_result)

            # Update metrics using timing mixin
            execution_time_ms = self._get_execution_time_ms(start_time)
            self._metrics["messages_handled"] += 1
            self._metrics["total_processing_time_ms"] += execution_time_ms

            self.logger.info(
                "Message handling completed",
                handler=self._handler_name,
                success=final_result.is_success,
                execution_time_ms=self._get_execution_time_ms_rounded(start_time),
                total_handled=self._metrics["messages_handled"],
            )

            return final_result

        def get_metrics(self) -> TAnyDict:
            """Get handler metrics.

            Returns:
                Dictionary of handler metrics

            """
            avg_time = 0.0
            if self._metrics["messages_handled"] > 0:
                avg_time = (
                    self._metrics["total_processing_time_ms"]
                    / self._metrics["messages_handled"]
                )

            return {
                **self._metrics,
                "average_processing_time_ms": round(avg_time, 2),
                "success_rate": self._metrics["successes"]
                / max(1, self._metrics["messages_handled"]),
            }

    # =============================================================================
    # SPECIALIZED HANDLERS - Domain-specific handlers
    # =============================================================================

    class CommandHandler(Handler[T, R]):
        """Handler specifically for commands.

        Commands represent intentions to change system state.
        """

        def validate_command(self, command: T) -> FlextResult[None]:
            """Validate command before processing.

            Override to add command-specific validation.
            """
            _ = command  # Mark as used for linting
            return FlextResult.ok(None)

        def pre_handle(self, command: T) -> FlextResult[T]:
            """Pre-process command with validation.

            Args:
                command: Command to pre-process

            Returns:
                FlextResult with pre-processed command

            """
            validation_result = self.validate_command(command)
            if validation_result.is_failure:
                return FlextResult.fail(
                    validation_result.error or "Command validation failed",
                )
            return FlextResult.ok(command)

    class EventHandler(Handler[T, None]):
        """Handler specifically for events.

        Events represent things that have happened.
        Event handlers typically don't return values.
        """

        def handle(self, event: T) -> FlextResult[None]:
            """Handle event.

            Args:
                event: Event to handle

            Returns:
                FlextResult with processing result or error

            """
            self.process_event(event)
            return FlextResult.ok(None)

        @abstractmethod
        def process_event(self, event: T) -> None:
            """Process the event.

            This method should have side effects but no return value.
            """

    class QueryHandler(Handler[T, R]):
        """Handler specifically for queries.

        Queries request data without side effects.
        """

        def authorize_query(self, query: T) -> FlextResult[None]:
            """Check query authorization.

            Override to add authorization logic.
            """
            _ = query  # Mark as used for linting
            return FlextResult.ok(None)

        def pre_handle(self, query: T) -> FlextResult[T]:
            """Pre-process query with authorization."""
            auth_result = self.authorize_query(query)
            if auth_result.is_failure:
                return FlextResult.fail(
                    auth_result.error or "Query authorization failed",
                )
            return FlextResult.ok(query)

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

        def register(
            self,
            key: TServiceKey,
            handler: FlextHandlers.Handler[object, object],
        ) -> FlextResult[None]:
            """Register handler by string key.

            Args:
                key: Unique key for handler
                handler: Handler instance

            Returns:
                FlextResult indicating registration success

            """
            if key in self._handlers:
                return FlextResult.fail(f"Handler already registered for key: {key}")

            self._handlers[key] = handler
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
            def handle(self, message: T) -> FlextResult[R]:
                result = handler_func(message)
                # Ensure we return FlextResult[R]
                if hasattr(result, "is_success"):
                    return cast("FlextResult[R]", result)
                return FlextResult.ok(cast("R", result))

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
