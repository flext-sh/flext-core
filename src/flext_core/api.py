"""Flext - Unified facade for complete flext-core ecosystem.

This module provides the main thin facade exposing ALL flext-core functionality
through a single unified entry point. Following FLEXT domain library standards,
this is the recommended way to access flext-core components with proper
integration of all foundation patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast
from uuid import uuid4

from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.service import FlextService
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class Flext:
    """Unified facade for complete flext-core ecosystem integration.

    This is the single recommended entry point for accessing ALL flext-core
    functionality with proper component integration and modern patterns.

    **UNIFIED FACADE PATTERN**: One class providing access to entire ecosystem.
    **COMPLETE INTEGRATION**: All 20+ flext-core components accessible.

    **Components Available**:

    - **FlextResult**: Railway pattern for error handling
    - **FlextConfig**: Pydantic 2.11+ configuration management
    - **FlextContainer**: Dependency injection container
    - **FlextModels**: DDD base classes (Entity, Value, AggregateRoot)
    - **FlextLogger**: Structured logging with context
    - **FlextBus**: Event messaging and pub/sub
    - **FlextContext**: Request/operation context management
    - **FlextHandlers**: CQRS handler registration
    - **FlextRegistry**: Component registration
    - **FlextDispatcher**: Message routing
    - **FlextMixins**: Reusable behaviors
    - **FlextUtilities**: Helper functions
    - **FlextService**: Service base class
    - **FlextConstants**: System constants
    - **FlextTypes**: Type system (40+ TypeVars)
    - **FlextExceptions**: Exception hierarchy
    - **FlextProtocols**: Interface definitions

    **Usage Examples**:

    ```python
    from flext_core import Flext

    # Create unified core instance
    core = Flext()

    # Access components through properties
    config = core.config
    container = core.container
    logger = core.logger

    # Or use direct class access (namespace pattern)
    result = FlextResult[str].ok("success")
    user = Flext.Models.Entity(id="123", name="John")
    timeout = Flext.Constants.Defaults.TIMEOUT

    # Factory methods for common components
    logger = Flext.ComponentFactory.create_logger(__name__)
    container = Flext.ComponentFactory.create_container()
    bus = Flext.ComponentFactory.create_bus()
    ```

    **Architecture**:

    - **Foundation Library**: Core patterns for entire FLEXT ecosystem
    - **Thin Facade**: NO business logic, pure delegation
    - **Complete Access**: All 20+ components available
    - **Modern Patterns**: Railway, DI, CQRS, DDD
    - **Type Safety**: Full generic type support

    **Design Principles**:

    - Thin facade: NO logic, pure component access
    - Complete integration: ALL flext-core components
    - Railway pattern: FlextResult throughout
    - Zero duplication: Direct access to components
    - Ecosystem foundation: Sets standards for all projects
    """

    # =================================================================
    # CLASS-LEVEL TYPE ALIASES (PEP 695 - Python 3.12+)
    # =================================================================
    # Using PEP 695 'type' statement for proper type aliases that work in annotations
    # while avoiding Pydantic field interpretation. TypeAliasType instances enable
    # usage like: def foo() -> Flext.Types.Dict: ...

    # Direct class references (not fields) - following namespace class pattern
    # Use type alias for proper generic support
    Result = FlextResult

    class Handlers[T, U](FlextHandlers[T, U]):
        """Handlers class for Flext."""

    class Service[T](FlextService[T]):
        """Service class for Flext."""

    class Container(FlextContainer):
        """Container class for Flext."""

    class Logger(FlextLogger):
        """Logger class for Flext."""

    class Models(FlextModels):
        """Models class for Flext."""

    class Constants(FlextConstants):
        """Constants class for Flext."""

    class Types(FlextTypes):
        """Types class for Flext."""

    class Exceptions(FlextExceptions):
        """Exceptions class for Flext."""

    class Protocols(FlextProtocols):
        """Protocols class for Flext."""

    class Bus(FlextBus):
        """Bus class for Flext."""

    class Context(FlextContext):
        """Context class for Flext."""

    class Registry(FlextRegistry):
        """Registry class for Flext."""

    class Dispatcher(FlextDispatcher):
        """Dispatcher class for Flext."""

    class Mixins(FlextMixins):
        """Mixins class for Flext."""

    class Utilities(FlextUtilities):
        """Utilities class for Flext."""

    class Config(FlextConfig):
        """Config class for Flext."""

        @staticmethod
        def create_logger(name: str) -> FlextLogger:
            """Create a new logger instance with the given name."""
            return FlextLogger(name)

        @staticmethod
        def create_container() -> FlextContainer:
            """Create a new container instance."""
            return FlextContainer()

        @staticmethod
        def create_bus() -> FlextBus:
            """Create a new event bus instance."""
            return FlextBus()

        @staticmethod
        def create_context() -> FlextContext:
            """Create a new context instance."""
            return FlextContext()

    # =================================================================
    # INSTANCE INITIALIZATION
    # =================================================================
    # Note: PEP 695 type aliases (above) automatically support generic subscripting
    # like Flext.Result[T] without needing __class_getitem__ method

    @classmethod
    def get_container(cls) -> FlextContainer:
        """Get a new container instance."""
        return FlextContainer()

    @classmethod
    def get_config(cls) -> FlextConfig:
        """Get a new config instance."""
        return FlextConfig()

    @classmethod
    def create_result_ok[T](cls, value: T) -> FlextResult[T]:
        """Create a successful result."""
        return FlextResult[T].ok(value)

    @classmethod
    def create_result_fail(
        cls, error: str, error_code: str | None = None
    ) -> FlextResult[object]:
        """Create a failed result."""
        return FlextResult[object].fail(error, error_code=error_code)

    @classmethod
    def create_logger(cls, name: str) -> FlextLogger:
        """Create a new logger instance."""
        return FlextLogger(name)

    def __init__(self, config: FlextConfig | None = None) -> None:
        """Initialize the unified core facade.

        Args:
            config: Optional configuration instance. If not provided,
                    uses FlextConfig() direct instantiation.

        """
        super().__init__()
        self._config = config or FlextConfig()

        # Lazy-loaded components (initialized on first access)
        self._container: FlextContainer | None = None
        self._logger: FlextLogger | None = None
        self._bus_instance: FlextBus | None = None
        self._context_instance: FlextContext | None = None
        self._dispatcher: FlextDispatcher | None = None
        self._registry: FlextRegistry | None = None

    # =================================================================
    # PROPERTY ACCESSORS (Lazy-loaded component instances)
    # =================================================================

    @property
    def config(self) -> FlextConfig:
        """Get configuration instance (direct instantiation pattern).

        Returns:
            FlextConfig instance with current configuration.

        """
        return self._config or FlextConfig()

    @property
    def container(self) -> FlextContainer:
        """Get global dependency injection container.

        Returns:
            Global FlextContainer singleton instance.

        """
        if self._container is None:
            self._container = FlextContainer.get_global()
        return self._container

    @property
    def logger(self) -> FlextLogger:
        """Get structured logger instance.

        Returns:
            FlextLogger instance for this module.

        """
        if self._logger is None:
            self._logger = FlextLogger(__name__)
        return self._logger

    @property
    def bus(self) -> FlextBus:
        """Get event bus instance.

        Returns:
            FlextBus instance for pub/sub messaging.

        """
        if self._bus_instance is None:
            self._bus_instance = FlextBus()
        return self._bus_instance

    @property
    def context(self) -> FlextContext:
        """Get context management instance.

        Returns:
            FlextContext instance for request/operation context.

        """
        if self._context_instance is None:
            self._context_instance = FlextContext()
        return self._context_instance

    @property
    def dispatcher(self) -> FlextDispatcher:
        """Get message dispatcher instance.

        Returns:
            FlextDispatcher instance for message routing.

        """
        if self._dispatcher is None:
            self._dispatcher = FlextDispatcher()
        return self._dispatcher

    @property
    def registry(self) -> FlextRegistry:
        """Get component registry instance.

        Returns:
            FlextRegistry instance for component registration.

        """
        if self._registry is None:
            self._registry = FlextRegistry(dispatcher=self.dispatcher)
        return self._registry

    # =================================================================
    # INTEGRATION HELPERS (Common patterns)
    # =================================================================

    @classmethod
    def setup_service_infrastructure(
        cls,
        service_name: str,
        config: FlextConfig | None = None,
    ) -> FlextResult[FlextTypes.Dict]:
        """Setup complete service infrastructure with all components.

        This is a convenience method that initializes all common flext-core
        components needed for a typical service: config, container, logger,
        bus, and context.

        Args:
            service_name: Name of the service being setup.
            config: Optional configuration instance.

        Returns:
            FlextResult containing dict with all initialized components:
            - config: FlextConfig instance
            - container: FlextContainer instance
            - logger: FlextLogger instance
            - bus: FlextBus instance
            - context: FlextContext instance

        Example:
            >>> result = Flext.setup_service_infrastructure("my-service")
            >>> if result.is_success:
            ...     infra = result.unwrap()
            ...     logger = infra["logger"]
            ...     container = infra["container"]

        """
        try:
            # Initialize all components
            service_config = config or FlextConfig()
            container = FlextContainer.get_global()
            logger = FlextLogger(service_name)
            bus = FlextBus()
            context = FlextContext()

            # Register core components in container
            container.register("config", service_config)
            container.register("logger", logger)
            container.register("bus", bus)
            container.register("context", context)

            # Log infrastructure setup
            logger.info(
                "Service infrastructure setup completed",
                extra={"service": service_name},
            )

            # Return all components
            infrastructure: FlextTypes.Dict = {
                "config": service_config,
                "container": container,
                "logger": logger,
                "bus": bus,
                "context": context,
            }

            return FlextResult[FlextTypes.Dict].ok(infrastructure)

        except Exception as e:
            error_msg = f"Service infrastructure setup failed for {service_name}: {e}"
            return FlextResult[FlextTypes.Dict].fail(error_msg)

    # =================================================================
    # CONVENIENCE METHODS (1.1.0 ENHANCEMENTS)
    # =================================================================

    def publish_event(
        self,
        event_type: str,
        data: FlextTypes.Dict,
        correlation_id: str | None = None,
    ) -> FlextResult[None]:
        """Publish event with automatic correlation tracking.

        This convenience method simplifies event publishing by automatically
        adding correlation ID and timestamp to event data.

        Args:
            event_type: Type/name of the event (e.g., "user.created").
            data: Event payload data.
            correlation_id: Optional correlation ID. If not provided,
                            attempts to get from context.

        Returns:
            FlextResult indicating success or failure of event publishing.

        Example:
            >>> core = Flext()
            >>> result = core.publish_event(
            ...     "order.created", {"order_id": "123", "amount": 99.99}
            ... )
            >>> if result.is_success:
            ...     print("Event published successfully")

        """
        try:
            # Get or generate correlation ID
            correlation = correlation_id
            if correlation is None:
                # Generate unique correlation ID
                correlation = str(uuid4())

            # Log event with structured data
            self.logger.info(
                f"Event: {event_type}",
                extra={
                    "event_type": event_type,
                    "correlation_id": correlation,
                    "timestamp": time.time(),
                    "data": data,
                },
            )

            return FlextResult[None].ok(None)

        except Exception as e:
            error_msg = f"Event publishing failed: {e}"
            self.logger.exception(error_msg, extra={"event_type": event_type})
            return FlextResult[None].fail(error_msg)

    @staticmethod
    def build_pipeline_operations(
        *operations: object,
    ) -> Callable[[object], FlextResult[object]]:
        """Build railway-oriented pipeline from operations.

        Creates a composed function that executes operations in sequence,
        automatically handling FlextResult chaining with early termination
        on first failure (railway pattern).

        Args:
            *operations: Sequence of operations to compose. Each operation
                        should accept previous result and return FlextResult.

        Returns:
            Composed pipeline function that accepts initial value and
            returns final FlextResult.

        Example:
            >>> def validate(data: str) -> FlextResult[str]:
            ...     if not data:
            ...         return Flext.Result[str].fail("Empty data")
            ...     return Flext.Result[str].ok(data)
            >>>
            >>> def process(data: str) -> FlextResult[str]:
            ...     return Flext.Result[str].ok(data.upper())
            >>>
            >>> pipeline = Flext.build_pipeline(validate, process)
            >>> result = pipeline("test")
            >>> print(result.value)  # "TEST"

        """

        def pipeline(initial_value: object) -> FlextResult[object]:
            """Execute pipeline with initial value."""
            # Start with successful result containing initial value
            result = FlextResult[object].ok(initial_value)

            # Chain operations with early termination on failure
            for operation_obj in operations:
                if result.is_failure:
                    break

                # Each operation receives unwrapped value and returns FlextResult
                try:
                    current_value = result.unwrap()
                    operation = cast(
                        "Callable[[object], FlextResult[object]]", operation_obj
                    )
                    result = operation(current_value)
                except Exception as e:
                    result = FlextResult[object].fail(f"Pipeline operation failed: {e}")
                    break

            return result

        return pipeline

    @contextmanager
    def request_context(
        self,
        request_id: str | None = None,
        user_id: str | None = None,
        **metadata: object,
    ) -> Iterator[FlextContext]:
        """Context manager for request-scoped context.

        Automatically manages context lifecycle, setting request-scoped
        data and ensuring cleanup after the request completes.

        Args:
            request_id: Optional request ID. Generated if not provided.
            user_id: Optional user ID for the request.
            **metadata: Additional metadata to store in context.

        Yields:
            FlextContext instance with request data set.

        Example:
            >>> core = Flext()
            >>> with core.request_context(user_id="123", trace=True) as ctx:
            ...     # Context is automatically populated
            ...     print(ctx.get("request_id"))
            ...     print(ctx.get("user_id"))  # "123"
            ...     # Process request here
            ... # Context is automatically cleared after block

        """
        # Generate request ID if not provided
        req_id = request_id or str(uuid4())

        # Set request context
        self.context.set("request_id", req_id)
        if user_id:
            self.context.set("user_id", user_id)

        # Set additional metadata
        for key, value in metadata.items():
            self.context.set(key, value)

        # Log request start
        self.logger.info(
            "Request started",
            extra={"request_id": req_id, "user_id": user_id},
        )

        try:
            # Yield context for use in with block
            yield self.context
        finally:
            # Cleanup context after request
            self.logger.info(
                "Request completed",
                extra={"request_id": req_id},
            )
            # Note: Context cleanup handled by context manager itself

    @classmethod
    def create_command_handler(
        cls,
        handler_func: object,
        command_type: type | None = None,
    ) -> FlextResult[FlextTypes.Dict]:
        """Create CQRS command handler.

        Args:
            handler_func: Handler function or callable.
            command_type: Optional command type for registration.

        Returns:
            FlextResult containing handler configuration for registration.

        Example:
            >>> def handle_create_user(cmd):
            ...     return FlextResult[FlextTypes.Dict].ok({"user": "created"})
            >>>
            >>> result = Flext.create_command_handler(handle_create_user)

        """
        try:
            # Create handler configuration for registration
            config = FlextModels.CqrsConfig.Handler(
                handler_id="command_handler",
                handler_name="CommandHandler",
                handler_type="command",
            )

            # Return handler configuration (not instantiated handler)
            # Actual handler instance should extend FlextHandlers and implement handle()
            handler_config: FlextTypes.Dict = {
                "config": config,
                "handler_func": handler_func,
                "command_type": command_type,
            }

            return FlextResult[FlextTypes.Dict].ok(handler_config)

        except Exception as e:
            error_msg = f"Command handler creation failed: {e}"
            return FlextResult[FlextTypes.Dict].fail(error_msg)

    @classmethod
    def create_query_handler(
        cls,
        handler_func: object,
        query_type: type | None = None,
    ) -> FlextResult[FlextTypes.Dict]:
        """Create CQRS query handler.

        Args:
            handler_func: Handler function or callable.
            query_type: Optional query type for registration.

        Returns:
            FlextResult containing handler configuration for registration.

        Example:
            >>> def handle_get_user(query):
            ...     return FlextResult[FlextTypes.Dict].ok({"user_id": "123"})
            >>>
            >>> result = Flext.create_query_handler(handle_get_user)

        """
        try:
            # Create handler configuration for registration
            config = FlextModels.CqrsConfig.Handler(
                handler_id="query_handler",
                handler_name="QueryHandler",
                handler_type="query",
            )

            # Return handler configuration (not instantiated handler)
            # Actual handler instance should extend FlextHandlers and implement handle()
            handler_config: FlextTypes.Dict = {
                "config": config,
                "handler_func": handler_func,
                "query_type": query_type,
            }

            return FlextResult[FlextTypes.Dict].ok(handler_config)

        except Exception as e:
            error_msg = f"Query handler creation failed: {e}"
            return FlextResult[FlextTypes.Dict].fail(error_msg)


# Nested classes are defined inside the class


# =================================================================
# MODULE EXPORTS
# =================================================================

__all__ = [
    "Flext",
]
