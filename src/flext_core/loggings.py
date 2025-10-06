"""Structured logging utilities enabling the context-first pillar for 1.0.0.

The module provides a thin wrapper around structlog with FLEXT-specific
context management and configuration.

Dependency Layer: 2 (Foundation Logging)
Dependencies: structlog, result, typings
Used by: All Flext modules requiring logging

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

structlog = FlextRuntime.structlog()

# =============================================================================
# FLEXT LOGGER - THIN WRAPPER AROUND STRUCTLOG
# =============================================================================


class FlextLogger:
    """Thin wrapper around structlog with FLEXT-specific context management.

    **Function**: Structured logging with context management
        - Direct structlog integration with hardcoded defaults
        - Provides FlextResult-wrapped logging methods
        - Supports context binding and distributed tracing
        - Clean Layer 2 dependency (no config imports)

    **Uses**: Foundation layer components only
        - structlog for structured logging
        - FlextResult for railway pattern error handling
        - FlextTypes for type definitions
        - Python stdlib uuid for trace ID generation

    This class provides a minimal interface that leverages structlog's
    built-in capabilities for context management, configuration, and logging.
    """

    # =========================================================================
    # PRIVATE MEMBERS - Structlog configuration
    # =========================================================================

    _structlog_configured: bool = False

    @staticmethod
    def _configure_structlog_if_needed(
        log_level: int | None = None,
        *,
        console_enabled: bool = True,
    ) -> None:
        """Configure structlog with provided settings.

        Args:
            log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
            console_enabled: Use console renderer vs JSON renderer

        Note:
            Can be called with FlextConfig values to configure logging:
            >>> from flext_core import FlextConfig
            >>> config = FlextConfig()
            >>> log_level_int = getattr(logging, config.log_level.upper())
            >>> FlextLogger._configure_structlog_if_needed(log_level=log_level_int)

        """
        if FlextLogger._structlog_configured:
            return

        FlextRuntime.configure_structlog(
            log_level=log_level,
            console_renderer=console_enabled,
        )

        FlextLogger._structlog_configured = True

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
            _level: Optional log level override (currently unused, for future)
            _service_name: Optional service name override
            _service_version: Optional service version override
            _correlation_id: Optional correlation ID override
            _force_new: Force creation of new instance (for testing)

        """
        # Configure structlog if not already configured (NO config dependency)
        FlextLogger._configure_structlog_if_needed()

        # Store logger name for later use
        self._name = name

        # Build initial context
        context = {}
        if _service_name:
            context["service_name"] = _service_name
        if _service_version:
            context["service_version"] = _service_version
        if _correlation_id:
            context["correlation_id"] = _correlation_id

        # Create bound logger with initial context
        self._logger = structlog.get_logger(name).bind(**context)

    @property
    def name(self) -> str:
        """Logger name."""
        return self._name

    def bind(self, **context: object) -> FlextLogger:
        """Bind additional context to the logger."""
        # Create new instance with bound logger
        new_logger = FlextLogger.__new__(FlextLogger)
        new_logger._name = self._name  # noqa: SLF001
        new_logger._logger = self._logger.bind(**context)  # noqa: SLF001
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


# MODULE EXPORTS
# =============================================================================


__all__: FlextTypes.StringList = [
    "FlextLogger",
]
