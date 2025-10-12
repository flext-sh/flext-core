"""Reusable behavior mixins for service infrastructure.

This module provides FlextMixins, a collection of reusable mixin classes
that add common infrastructure capabilities to service classes throughout
the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""
# ruff: disable=SLF001

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import cast
from typing import ClassVar

from flext_core.config import FlextConfig
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextMixins:
    """Reusable behavior mixins for service infrastructure.

    Provides a collection of mixin classes that add common infrastructure
    capabilities to service classes throughout the FLEXT ecosystem.

    Includes:
    - Dependency injection container integration
    - Request context and correlation management
    - Structured logging with automatic context
    - Performance monitoring and metrics tracking
    - Configuration access and validation
    - Service lifecycle management

    Usage:
        >>> from flext_core.mixins import FlextMixins
        >>>
        >>> class MyService(FlextMixins):
        ...     def process(self, data: dict):
        ...         with self.track("process"):
        ...             self.logger.info("Processing", size=len(data))
        ...             return {"status": "processed"}
    """

    # Class-level cache for loggers to avoid repeated DI lookups
    _logger_cache: ClassVar[dict[str, FlextLogger]] = {}
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-initialize container for subclasses (ABI compatibility)."""
        super().__init_subclass__(**kwargs)
        # Container is lazily initialized on first access

    @property
    def container(self) -> FlextContainer:
        """Get global FlextContainer instance with lazy initialization."""
        return FlextContainer.get_global()

    @property
    def context(self) -> FlextContext:
        """Get FlextContext instance.

        Creates a new FlextContext instance for context operations.
        All context operations should use FlextContext directly.
        """
        return FlextContext()

    @property
    def logger(self) -> FlextLogger:
        """Access logger via property (DI-backed with caching).

        Returns:
            FlextLogger instance for this class

        """
        return self._get_or_create_logger()

    @contextmanager
    def track(self, operation_name: str) -> Iterator[FlextTypes.Dict]:
        """Track operation performance with automatic context integration."""
        with FlextContext.Performance.timed_operation(operation_name) as metrics:
            yield metrics

    @property
    def config(self) -> FlextConfig:
        """Get global FlextConfig instance.

        Provides convenient access to global configuration instance
        for service classes using FlextMixins.

        Returns:
            FlextConfig: Global configuration instance

        """
        return FlextConfig.get_global_instance()

    def _register_in_container(self, service_name: str) -> FlextResult[None]:
        """Register self in global container for service discovery."""
        try:
            return self.container.register(service_name, self)
        except Exception as e:
            # If already registered, return success (for test compatibility)
            if "already registered" in str(e).lower():
                return FlextResult[None].ok(None)
            return FlextResult[None].fail(f"Service registration failed: {e}")

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
            logger_result: FlextResult[FlextLogger] = cast(
                "FlextResult[FlextLogger]",
                container.get_typed(logger_key, FlextResult[FlextLogger]),
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
    # SERVICE METHODS - Complete Infrastructure (inherited by FlextMixins)
    # =========================================================================

    def _init_service(self, service_name: str | None = None) -> None:
        """Initialize service with automatic registration and setup.

        Args:
            service_name: Optional service name for registration

        """
        service_name = service_name or self.__class__.__name__

        register_result = self._register_in_container(service_name)

        if register_result.is_failure:
            self.logger.warning(
                f"Service registration failed: {register_result.error}",
                extra={"service_name": service_name},
            )

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
        self.logger.info("Service initialized", **service_context)

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
        self._propagate_context(operation_name)

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


__all__ = [
    "FlextMixins",
]
