"""Structured logging utilities enabling the context-first pillar for 1.0.0.

The module provides a thin wrapper around structlog with FLEXT-specific
context management and configuration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog
import structlog.contextvars

from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

if TYPE_CHECKING:
    from flext_core.config import FlextConfig

# =============================================================================
# STRUCTLOG CONFIGURATION
# =============================================================================


# =============================================================================
# FLEXT LOGGER - THIN WRAPPER AROUND STRUCTLOG
# =============================================================================


class FlextLogger:
    """Thin wrapper around structlog with FLEXT-specific context management.

    This class provides a minimal interface that leverages structlog's
    built-in capabilities for context management, configuration, and logging.
    """

    # =========================================================================
    # PRIVATE MEMBERS - Configuration cache and structlog setup
    # =========================================================================

    _config_cache: FlextConfig | None = None

    @staticmethod
    def _get_config() -> FlextConfig:
        """Lazy load configuration to avoid import cycles."""
        if FlextLogger._config_cache is None:
            # Lazy import to break circular dependency
            from flext_core.config import FlextConfig

            # Store in class cache
            FlextLogger._config_cache = FlextConfig()
        return FlextLogger._config_cache

    @staticmethod
    def _configure_structlog() -> None:
        """Configure structlog with FLEXT-specific processors and settings."""
        if structlog.is_configured():
            return

        # Get FLEXT configuration
        config = FlextLogger._get_config()

        # Configure structlog with FLEXT processors
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,  # Merge context variables
                structlog.processors.add_log_level,  # Add log level
                structlog.processors.TimeStamper(fmt="ISO"),  # Add timestamp
                structlog.processors.StackInfoRenderer(),  # Add stack info
                structlog.dev.ConsoleRenderer(colors=True)
                if config.console_enabled
                else structlog.processors.JSONRenderer(),  # JSON or console output
            ],
            wrapper_class=structlog.make_filtering_bound_logger(config.log_level),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def __init__(
        self,
        name: str,
        *,
        _level: str | None = None,
        _service_name: str | None = None,
        _service_version: str | None = None,
        _correlation_id: str | None = None,
        _force_new: bool = False,
    ) -> None:
        """Initialize FlextLogger with context.

        Args:
            name: Logger name (typically __name__ or module path)
            _level: Optional log level override
            _service_name: Optional service name override
            _service_version: Optional service version override
            _correlation_id: Optional correlation ID override
            _force_new: Force creation of new instance (for testing)

        """
        # Configure structlog if not already configured
        FlextLogger._configure_structlog()

        # Store logger name for later use
        self._name = name

        # Create bound logger with initial context
        self._logger = structlog.get_logger(name)

        # Bind initial context if provided
        context = {}
        if _service_name:
            context["service_name"] = _service_name
        if _service_version:
            context["service_version"] = _service_version
        if _correlation_id:
            context["correlation_id"] = _correlation_id

        if context:
            self._logger = self._logger.bind(**context)

    @property
    def name(self) -> str:
        """Logger name."""
        return self._name

    def bind(self, **context: object) -> FlextLogger:
        """Bind additional context to the logger."""
        # Create new instance with bound logger
        new_logger = FlextLogger.__new__(FlextLogger)
        new_logger._name = self._name  # noqa: SLF001 - internal reassignment
        new_logger._logger = self._logger.bind(**context)  # noqa: SLF001 - internal reassignment
        return new_logger

    # =============================================================================
    # LOGGING METHODS - DELEGATE TO STRUCTLOG
    # =============================================================================

    def trace(self, message: str, *args: object, **kwargs: object) -> FlextResult[None]:
        """Log trace message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"

            self._logger.debug(
                formatted_message, **kwargs
            )  # structlog doesn't have trace
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def debug(
        self, message: str, *args: object, **context: object
    ) -> FlextResult[None]:
        """Log debug message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self._logger.debug(formatted_message, **context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def info(self, message: str, *args: object, **context: object) -> FlextResult[None]:
        """Log info message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self._logger.info(formatted_message, **context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def warning(
        self, message: str, *args: object, **context: object
    ) -> FlextResult[None]:
        """Log warning message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self._logger.warning(formatted_message, **context)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def error(self, message: str, *args: object, **kwargs: object) -> FlextResult[None]:
        """Log error message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self._logger.error(formatted_message, **kwargs)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def critical(
        self, message: str, *args: object, **kwargs: object
    ) -> FlextResult[None]:
        """Log critical message - LoggerProtocol implementation."""
        try:
            try:
                formatted_message = message % args if args else message
            except (TypeError, ValueError):
                formatted_message = f"{message} | args={args!r}"
            self._logger.critical(formatted_message, **kwargs)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    def exception(
        self, message: str, *, exc_info: bool = True, **kwargs: object
    ) -> FlextResult[None]:
        """Log exception message with stack trace - LoggerProtocol implementation."""
        try:
            if exc_info:
                import traceback

                kwargs["stack_trace"] = traceback.format_exc()
            self._logger.error(message, **kwargs)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Logging failed: {e}")

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def get_logger_attributes(self) -> FlextTypes.Dict:
        """Get logger attributes for debugging."""
        return {
            "name": self._name,
            "context": dict(structlog.contextvars.get_contextvars()),
        }

    def start_trace(self, operation_name: str, **context: object) -> str:
        """Start a distributed trace."""
        trace_id = str(uuid.uuid4())[:8]
        result = self.debug(
            f"TRACE_START: {operation_name}",
            trace_id=trace_id,
            operation=operation_name,
            **context,
        )
        # Note: trace_id returned even if logging fails
        if result.is_failure:
            # Could log to stderr or raise, depending on requirements
            pass
        return trace_id

    def end_trace(self, trace_id: str, operation_name: str, **context: object) -> None:
        """End a distributed trace."""
        result = self.debug(
            f"TRACE_END: {operation_name}",
            trace_id=trace_id,
            operation=operation_name,
            **context,
        )
        # Note: failures are silently ignored
        if result.is_failure:
            # Could log to stderr or raise, depending on requirements
            pass


# MODULE EXPORTS
# =============================================================================


__all__: FlextTypes.StringList = [
    "FlextLogger",
]
