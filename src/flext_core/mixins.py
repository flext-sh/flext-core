"""Shared mixins anchoring service infrastructure for FLEXT.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar

from flext_core.config import FlextConfig
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextMixins:
    """Complete service infrastructure for FLEXT ecosystem.

    Follows FLEXT quality standards with single class containing nested mixins,
    type-safe Pydantic signatures, and direct implementation leveraging existing
    FLEXT components.

    **Function**: Reusable behavior mixins for ecosystem
        - Dependency injection container integration via Container
        - Request context and correlation management via Context
        - Structured logging with automatic DI via Logging
        - Performance monitoring and metrics via Metrics
        - Complete service infrastructure composition

    **Uses**: Core FLEXT infrastructure for mixins
        - FlextContainer for dependency injection
        - FlextContext for request context management
        - FlextLogger for structured logging
        - FlextConfig for configuration management
        - FlextResult[T] for operation results
        - threading for thread-safe operations

    **How to use**: Nested mixin classes or direct inheritance
        ```python
        from flext_core import FlextMixins


        # Example 1: Complete service infrastructure (recommended)
        class MyService(FlextMixins):
            def __init__(self):
                self._init_service("MyService")

            def process(self, data: dict):
                # All infrastructure automatically available
                with self._track_operation("process"):
                    self._log_with_context("info", "Processing", size=len(data))
                    return FlextResult[dict].ok({"status": "processed"})


        # Example 2: Use specific mixins individually
        class MyHandler(FlextMixins):
            def handle(self, data: dict):
                # Context-aware logging available
                self._log_with_context("info", "Handling data", size=len(data))
                return {"handled": True}


        # Example 3: Use container mixin for DI
        class MyRepository(FlextMixins):
            def __init__(self):
                # Container automatically available
                db_result = self.container.get("database")
                if db_result.is_success:
                    self.db = db_result.unwrap()
        ```

    Note:
        All mixin classes use __init_subclass__ for automatic initialization.
        Provides thread-safe operations with proper locking.
        Integrates with FlextLogger for structured logging.
        Supports both individual mixins and complete composition.
        FlextMixins is an alias to FlextMixins for backward compatibility.

    Warning:
        Mixins modify class behavior through multiple inheritance.
        Always use FlextResult for consistent error handling.

    See Also:
        FlextUtilities: For utility functions.
        FlextModels: For domain model definitions.
        FlextLogger: For structured logging.
        FlextConfig: For configuration management.
        FlextContainer: For dependency injection.

    """

    # =========================================================================
    # CONTAINER INTEGRATION - Dependency Injection Infrastructure
    # =========================================================================

    class _Container:
        """Container integration mixin for dependency injection.

        **Function**: Automatic DI container access and service registration
            - Lazy container access via property
            - Automatic service registration via __init_subclass__
            - Type-safe service resolution
            - FlextResult-based error handling
            - ABI compatibility through descriptors

        **Uses**: Existing FlextCore infrastructure
            - FlextContainer.get_global() for singleton access
            - FlextResult[T] for operation results
            - FlextLogger for diagnostics

        **How to use**: Inherit to add container capabilities
            ```python
            class MyService(FlextMixins):
                def __init__(self):
                    # _container automatically available
                    db_result = self.container.get("database")
                    if db_result.is_success:
                        self.db = db_result.unwrap()
            ```

        **ABI Compatibility**: Uses __init_subclass__ for automatic initialization,
        ensuring existing code works without changes.

        """

        def __init_subclass__(cls, **kwargs: object) -> None:
            """Auto-initialize container for subclasses (ABI compatibility)."""
            super().__init_subclass__(**kwargs)
            # Container is lazily initialized on first access

        @property
        def container(self) -> FlextContainer:
            """Get global FlextContainer instance with lazy initialization."""
            return FlextContainer.get_global()

        def _register_in_container(self, service_name: str) -> FlextResult[None]:
            """Register self in global container for service discovery."""
            try:
                return self.container.register(service_name, self)
            except Exception as e:
                # If already registered, return success (for test compatibility)
                if "already registered" in str(e).lower():
                    return FlextResult[None].ok(None)
                return FlextResult[None].fail(f"Service registration failed: {e}")

    # =========================================================================
    # CONTEXT INTEGRATION - Request Context and Correlation
    # =========================================================================

    class _Context:
        """Simplified context integration using FlextContext directly.

        **Function**: Direct delegation to FlextContext for all context operations
            - Request context with correlation IDs via FlextContext.Request
            - Service identification via FlextContext.Service
            - Correlation management via FlextContext.Correlation
            - Performance tracking via FlextContext.Performance
            - Automatic context propagation through FlextContext

        **Uses**: Direct FlextContext integration
            - No custom context management or lazy initialization
            - All context operations delegate to FlextContext
            - Maintains ABI compatibility through property access

        **How to use**: Direct access to FlextContext functionality
            ```python
            class MyService(FlextMixins):
                def process(self, data: dict):
                    # Direct access to FlextContext
                    corr_id = FlextContext.Correlation.get_correlation_id()
                    FlextContext.Request.set_operation_name("process_data")
                    return {"correlation_id": corr_id}
            ```

        **ABI Compatibility**: Property provides access to global FlextContext,
        ensuring existing code works without changes.

        """

        @property
        def context(self) -> FlextContext:
            """Get FlextContext instance.

            Creates a new FlextContext instance for context operations.
            All context operations should use FlextContext directly.
            """
            return FlextContext()

        # Convenience methods that delegate to FlextContext for backward compatibility
        def _propagate_context(self, operation_name: str) -> None:
            """Propagate context for current operation using FlextContext."""
            FlextContext.Request.set_operation_name(operation_name)
            FlextContext.Utilities.ensure_correlation_id()

        def _get_correlation_id(self) -> str | None:
            """Get current correlation ID from FlextContext."""
            return FlextContext.Correlation.get_correlation_id()

        def _set_correlation_id(self, correlation_id: str) -> None:
            """Set correlation ID in FlextContext."""
            FlextContext.Correlation.set_correlation_id(correlation_id)

    # =========================================================================
    # LOGGING INTEGRATION - Context-Aware Structured Logging
    # =========================================================================

    class _Logging:
        """Enhanced Logging mixin with dependency injection for logger.

        Provides automatic logger injection from FlextContainer, eliminating
        the need for manual FlextLogger(__name__) instantiation in every class.

        This mixin uses lazy initialization and caching to minimize overhead
        while providing full DI integration with context-aware structured logging.

        **Function**: Structured logging with automatic context and DI
            - Context-aware log messages with correlation ID and operation tracking
            - Automatic logger injection from FlextContainer
            - Thread-safe caching to avoid repeated DI lookups
            - FlextLogger integration with structured logging
            - ABI compatibility through __init_subclass__

        **Uses**: Existing FlextCore infrastructure
            - FlextLogger for structured logging
            - FlextContainer for dependency injection
            - FlextContext for correlation tracking
            - threading for thread-safe operations
            - FlextTypes for type safety

        **How to use**: Inherit to add logging capabilities with DI
            ```python
            class MyService(FlextMixins):
                def process(self, data: dict):
                    # Logger automatically available via DI with caching
                    self._log_with_context("info", "Processing", size=len(data))
                    self.logger.debug("Details...")
            ```

        **ABI Compatibility**: Uses __init_subclass__ for automatic initialization,
        ensuring existing code works without changes.

        """

        # Class-level cache for loggers to avoid repeated DI lookups
        _logger_cache: ClassVar[dict[str, FlextLogger]] = {}
        _cache_lock: ClassVar[threading.Lock] = threading.Lock()

        @classmethod
        def _get_or_create_logger(cls) -> FlextLogger:
            """Get or create DI-injected logger for this class.

            Uses FlextContainer for dependency injection with fallback to
            direct creation if DI is not available.

            Returns:
                FlextLogger instance from DI or newly created

            """
            # Generate unique logger name based on module and class
            logger_name = f"{cls.__module__}.{cls.__name__}"

            # Check cache first (thread-safe)
            with cls._cache_lock:
                if logger_name in cls._logger_cache:
                    return cls._logger_cache[logger_name]

            # Try to get from DI container
            try:
                container = FlextContainer.get_global()
                logger_key = f"logger:{logger_name}"

                # Attempt to retrieve logger from container
                logger_result: FlextResult[FlextLogger] = container.get_typed(
                    logger_key, FlextLogger
                )

                if logger_result.is_success:
                    logger: FlextLogger = logger_result.unwrap()
                    # Cache the result
                    with cls._cache_lock:
                        cls._logger_cache[logger_name] = logger
                    return logger

                # Logger not in container - create and register
                logger = FlextLogger(logger_name)
                container.register(logger_key, logger)

                # Cache the result
                with cls._cache_lock:
                    cls._logger_cache[logger_name] = logger

                return logger

            except Exception:
                # Fallback: create logger without DI if container unavailable
                logger = FlextLogger(logger_name)
                with cls._cache_lock:
                    cls._logger_cache[logger_name] = logger
                return logger

        @property
        def logger(self) -> FlextLogger:
            """Access logger via property (DI-backed with caching).

            Returns:
                FlextLogger instance for this class

            """
            return self._get_or_create_logger()

        def _log_with_context(self, level: str, message: str, **extra: object) -> None:
            """Log message with automatic context data inclusion."""
            context_data: FlextTypes.Dict = {
                "correlation_id": FlextContext.Correlation.get_correlation_id(),
                "operation": FlextContext.Request.get_operation_name(),
                **extra,
            }

            log_method = getattr(self.logger, level, self.logger.info)
            log_method(message, extra=context_data)

    # =========================================================================
    # METRICS INTEGRATION - Performance Tracking
    # =========================================================================

    class _Metrics:
        """Metrics integration mixin for automatic performance tracking.

        **Function**: Performance monitoring and timing
            - Operation timing with context managers
            - Automatic metric collection
            - FlextContext.Performance integration
            - ABI compatibility through __init_subclass__

        **Uses**: Existing FlextCore infrastructure
            - FlextContext.Performance.timed_operation for timing
            - contextmanager for scope management
            - FlextTypes for type safety

        **How to use**: Inherit to add metrics capabilities
            ```python
            class MyService(FlextMixins):
                def process(self, data: dict):
                    with self._track_operation("process_data") as metrics:
                        result = self._do_processing(data)
                        return result
            ```

        **ABI Compatibility**: Uses __init_subclass__ for automatic initialization,
        ensuring existing code works without changes.

        """

        def __init_subclass__(cls, **kwargs: object) -> None:
            """Auto-initialize metrics for subclasses (ABI compatibility)."""
            super().__init_subclass__(**kwargs)
            # Metrics tracking is automatic via context managers

        @contextmanager
        def _track_operation(self, operation_name: str) -> Iterator[FlextTypes.Dict]:
            """Track operation performance with automatic context integration."""
            with FlextContext.Performance.timed_operation(operation_name) as metrics:
                yield metrics

    # =========================================================================
    # SERVICE METHODS - Complete Infrastructure (inherited by FlextMixins)
    # =========================================================================

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-initialize service infrastructure for subclasses."""
        super().__init_subclass__(**kwargs)
        # All mixin initialization handled by parent __init_subclass__ calls

    def _init_service(self, service_name: str | None = None) -> None:
        """Initialize service with automatic registration and setup.

        Args:
            service_name: Optional service name for registration

        """
        service_name = service_name or self.__class__.__name__

        register_result = self._register_in_container(service_name)  # type: ignore[attr-defined]

        if register_result.is_failure:
            self.logger.warning(  # type: ignore[attr-defined]
                f"Service registration failed: {register_result.error}",
                extra={"service_name": service_name},
            )

    @property
    def config(self) -> FlextConfig:
        """Get global FlextConfig instance.

        Provides convenient access to global configuration instance
        for service classes using FlextMixins.

        Returns:
            FlextConfig: Global configuration instance

        """
        return FlextConfig.get_global_instance()

    # =========================================================================
    # CONTEXT ENRICHMENT METHODS - Automatic Context Management
    # =========================================================================

    def _enrich_context(self, **context_data: object) -> None:
        """Log service information ONCE at initialization.

        Logs service-level information at initialization instead of binding
        it to all log messages. This provides service context visibility
        without cluttering every log entry.

        Args:
            **context_data: Additional context data to log

        Example:
            ```python
            class OrderService(FlextMixins):
                def __init__(self):
                    self._init_service("OrderService")
                    self._enrich_context(service_version="1.0.0", team="orders")

                def process_order(self, order_id: str):
                    # Service info was logged once at initialization
                    self._log_with_context(
                        "info", "Processing order", order_id=order_id
                    )
            ```

        """
        # Build service context for logging
        service_context: FlextTypes.Dict = {
            "service_name": self.__class__.__name__,
            "service_module": self.__class__.__module__,
            **context_data,
        }
        # Log service initialization ONCE instead of binding to all logs
        self.logger.info("Service initialized", **service_context)  # type: ignore[attr-defined]

    def _with_operation_context(
        self,
        operation_name: str,
        **operation_data: object,
    ) -> None:
        """Set operation context for the current operation.

        Binds operation-level information to the context for tracking
        and debugging specific operations.

        Args:
            operation_name: Name of the operation being performed
            **operation_data: Additional operation context data

        Example:
            ```python
            class InventoryService(FlextMixins):
                def reserve_items(self, order_id: str, items: list):
                    # Set operation context
                    self._with_operation_context(
                        "reserve_items", order_id=order_id, item_count=len(items)
                    )

                    # All logs include operation context
                    self._log_with_context("info", "Reserving items")
                    return self._do_reserve(items)
            ```

        """
        # Propagate context using inherited Context mixin method
        self._propagate_context(operation_name)  # type: ignore[attr-defined]

        # Bind additional operation data using structlog's contextvars
        if operation_data:
            FlextLogger.bind_global_context(**operation_data)

    def _clear_operation_context(self) -> None:
        """Clear operation-specific context data.

        Useful for cleanup after operation completion or for
        resetting context between operations.

        Example:
            ```python
            class BatchProcessor(FlextMixins):
                def process_batch(self, items: list):
                    for item in items:
                        try:
                            self._with_operation_context(
                                "process_item", item_id=item.id
                            )
                            self._process_single_item(item)
                        finally:
                            # Clean up context after each item
                            self._clear_operation_context()
            ```

        """
        # Clear structlog context using contextvars
        FlextLogger.clear_global_context()

        # Clear FlextContext operation name
        FlextContext.Request.set_operation_name("")


# Make FlextMixins inherit from all nested mixins for direct usage
# This allows: class MyService(FlextMixins) instead of class MyService(FlextMixins)
class _FlextMixinsComplete(
    FlextMixins._Container,  # noqa: SLF001
    FlextMixins._Context,  # noqa: SLF001
    FlextMixins._Logging,  # noqa: SLF001
    FlextMixins._Metrics,  # noqa: SLF001
    FlextMixins,
):
    """Internal helper to enable FlextMixins direct inheritance."""


# Replace FlextMixins with the complete version
FlextMixins = _FlextMixinsComplete  # type: ignore[misc,assignment]

__all__ = [
    "FlextMixins",
]
