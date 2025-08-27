"""Structured logging system."""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from typing import ClassVar

import structlog
from structlog.typing import EventDict, Processor

from flext_core.constants import FlextConstants, FlextLogLevel
from flext_core.typings import FlextTypes

# Types imported from centralized FlextTypes - no local TypedDict definitions


# Global state moved to FlextLogger class - no module-level globals

# Trace level setup moved to FlextLogger class methods

# All configuration moved to FlextLogger class


# =============================================================================
# FLEXT LOGGER - Class-based factory interface
# =============================================================================


class FlextLogger:
    """Structured logger with context management and level filtering.

    Single consolidated class for all FLEXT logging functionality following
    FLEXT architectural patterns. All logging functionality is centralized here
    using FlextConstants, FlextTypes, and proper protocols.
    """

    # Class variables for configuration state
    _configured: ClassVar[bool] = False
    _loggers: ClassVar[FlextTypes.Core.Dict] = {}
    _log_store: ClassVar[list[FlextTypes.Logging.LogEntry]] = []

    def __init__(self, name: str, level: str = "INFO") -> None:
        """Initialize logger.

        Args:
            name: Logger name (typically __name__)
            level: Minimum log level

        """
        # Auto-configure structlog if not already configured
        if not self._configured:
            self.configure()

        self._name = name

        # Use environment-aware level if default level is requested
        if level == "INFO":
            env_level = self._get_env_log_level_string()
            if env_level != "INFO":
                level = env_level

        self._level = level.upper()
        numeric_levels = FlextLogLevel.get_numeric_levels()
        self._level_value = numeric_levels.get(self._level, numeric_levels["INFO"])
        self._context: FlextTypes.Logging.ContextDict = {}

        # Ensure stdlib logging level is permissive enough for structlog filtering
        # The filter_by_level processor uses stdlib logging levels
        current_stdlib_level = logging.getLogger().getEffectiveLevel()
        if self._level_value < current_stdlib_level:
            logging.getLogger().setLevel(self._level_value)

        # Create structlog logger with the name - this will use the global configuration
        # that includes _add_to_log_store processor
        self._structlog_logger = structlog.get_logger(name)

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level."""
        numeric_levels = FlextLogLevel.get_numeric_levels()
        # Handle both string and enum inputs safely
        try:
            # Try to get value attribute first (for enums)
            level_value = getattr(level, "value", None)
            if level_value is not None:
                level_str: str = str(level_value).upper()
            else:
                level_str = str(level).upper()
        except AttributeError:
            # Fallback to string conversion
            level_str = str(level).upper()
        level_numeric = numeric_levels.get(level_str, numeric_levels["INFO"])
        return level_numeric >= self._level_value

    def _log_with_structlog(
        self,
        level: str,
        message: FlextTypes.Core.LogMessage,
        context: FlextTypes.Logging.ContextDict | None = None,
    ) -> None:
        """Log using structlog with context merging."""
        if not self._should_log(level):
            return

        # Merge instance context with method context
        merged_context = {**self._context}
        if context:
            merged_context.update(context)

        # Add our standard fields
        log_data: dict[str, object] = {
            "logger": self._name,
            **merged_context,
        }

        # Map our levels to standard logging levels for structlog
        level_mapping = {
            "TRACE": "trace",
            "DEBUG": "debug",
            "INFO": "info",
            "WARNING": "warning",
            "ERROR": "error",
            "CRITICAL": "critical",
        }

        level_str = str(level).upper()
        structlog_level = level_mapping.get(level_str, "info")
        log_method = getattr(self._structlog_logger, structlog_level)

        log_method(str(message), **log_data)

    def set_level(self, level: str) -> None:
        """Set logger level."""
        self._level = level.upper()
        numeric_levels = FlextLogLevel.get_numeric_levels()
        self._level_value = numeric_levels.get(self._level, numeric_levels["INFO"])

    def get_context(self) -> FlextTypes.Logging.ContextDict:
        """Get current context."""
        return dict(self._context)

    def set_context(self, context: FlextTypes.Logging.ContextDict) -> None:
        """Set logger context."""
        self._context = dict(context)

    def clear_context(self) -> None:
        """Clear logger context."""
        self._context = {}

    def with_context(self, **context: object) -> FlextLogger:
        """Create logger with additional context."""
        new_logger = FlextLogger(self._name, self._level)
        # Use the same structlog instance but with merged context
        new_logger._structlog_logger = self._structlog_logger
        new_logger._context = {**self._context, **context}
        return new_logger

    def trace(
        self, message: FlextTypes.Core.LogMessage, *args: object, **context: object
    ) -> None:
        """Log trace message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("TRACE", formatted_message, context)

    def debug(
        self, message: FlextTypes.Core.LogMessage, *args: object, **context: object
    ) -> None:
        """Log debug message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("DEBUG", formatted_message, context)

    def info(
        self, message: FlextTypes.Core.LogMessage, *args: object, **context: object
    ) -> None:
        """Log info message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("INFO", formatted_message, context)

    def warning(
        self, message: FlextTypes.Core.LogMessage, *args: object, **context: object
    ) -> None:
        """Log warning message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("WARNING", formatted_message, context)

    def error(
        self, message: FlextTypes.Core.LogMessage, *args: object, **context: object
    ) -> None:
        """Log error message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("ERROR", formatted_message, context)

    def critical(
        self, message: FlextTypes.Core.LogMessage, *args: object, **context: object
    ) -> None:
        """Log critical message with optional %s formatting."""
        formatted_message = self._format_message(message, args)
        self._log_with_structlog("CRITICAL", formatted_message, context)

    def exception(
        self, message: FlextTypes.Core.LogMessage, *args: object, **context: object
    ) -> None:
        """Log exception with traceback context and optional %s formatting."""
        formatted_message = self._format_message(message, args)
        context["traceback"] = traceback.format_exc()
        self._log_with_structlog("ERROR", formatted_message, context)

    def _format_message(
        self, message: FlextTypes.Core.LogMessage, args: tuple[object, ...]
    ) -> str:
        """Format message with %s placeholders if args are provided."""
        if not args:
            return str(message)

        try:
            return str(message) % args
        except (TypeError, ValueError):
            # If % formatting fails, fall back to string formatting
            try:
                return str(message).format(*args)
            except (KeyError, ValueError):
                # If both fail, just concatenate
                return f"{message} {' '.join(str(arg) for arg in args)}"

    def bind(self, **context: object) -> FlextLogger:
        """Create new logger with bound context.

        Args:
            **context: Context data to bind

        Returns:
            New FlextLogger instance with bound context

        """
        new_logger = FlextLogger(self._name, self._level)
        # Use the same structlog instance but with merged context
        new_logger._structlog_logger = self._structlog_logger
        new_logger._context = {**self._context, **context}
        return new_logger

    @classmethod
    def configure(
        cls,
        *,
        log_level: FlextLogLevel | None = None,
        _log_level: FlextLogLevel | None = None,
        json_output: bool = False,
        add_timestamp: bool = True,
        add_caller: bool = False,
    ) -> None:
        """Configure the logging system.

        Args:
            log_level: Log level to configure (new preferred name)
            _log_level: Log level to configure (compatibility)
            json_output: Whether to use JSON output format
            add_timestamp: Whether to add timestamps to logs
            add_caller: Whether to add caller information to logs

        """
        # Handle both parameter names for compatibility
        # Note: final_log_level can be used for future enhancement
        # For now just validate parameters exist
        _ = log_level or _log_level or FlextLogLevel.INFO

        # Build processors list while preserving essential ones
        # Import Processor type from structlog

        processors: list[Processor] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
        ]

        # Add optional processors
        if add_timestamp:
            processors.append(structlog.processors.TimeStamper(fmt="unix"))
        if add_caller:
            processors.append(structlog.processors.CallsiteParameterAdder())

        # Always preserve our essential processors
        processors.append(cls._add_to_log_store)  # Always keep log store

        # Choose final renderer
        if json_output:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(cls._create_human_readable_renderer())

        # Reconfigure structlog with preserved essential functionality
        # Use type ignore for processor compatibility across structlog versions
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Set up basic logging configuration
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stderr,
            level=cls._get_logging_level_from_env(),
        )

        # Set up custom trace level
        cls._setup_custom_trace_level()

        cls._configured = True

    # ==========================================================================
    # PRIVATE METHODS - Configuration and utility methods
    # ==========================================================================

    @staticmethod
    def _get_env_log_level_string() -> str:
        """Get logging level from environment as string."""
        return (
            os.environ.get("ALGAR_LOG_LEVEL")
            or os.environ.get("FLEXT_LOG_LEVEL")
            or os.environ.get("LOG_LEVEL")
            or "INFO"
        ).upper()

    @staticmethod
    def _get_logging_level_from_env() -> int:
        """Get logging level from environment variables."""
        env_level = FlextLogger._get_env_log_level_string()

        # Map to numeric levels using FlextConstants
        level_mapping = {
            "CRITICAL": 50,
            "ERROR": 40,
            "WARNING": 30,
            "INFO": 20,
            "DEBUG": 10,
            "TRACE": FlextConstants.Observability.TRACE_LEVEL,
        }

        return level_mapping.get(env_level, 20)  # Default to INFO

    @staticmethod
    def _create_human_readable_renderer() -> structlog.dev.ConsoleRenderer:
        """Create human-readable console renderer following market standards."""
        # Check if we're in development or production
        env_value = os.environ.get("ENVIRONMENT", "development").lower()
        is_development = env_value in {"development", "dev", "local"}
        colors_enabled = os.environ.get("FLEXT_LOG_COLORS", "true").lower()
        enable_colors = colors_enabled == "true" and is_development

        return structlog.dev.ConsoleRenderer(
            colors=enable_colors,
            level_styles={
                "critical": "\033[91m",  # Bright red
                "error": "\033[91m",  # Red
                "warning": "\033[93m",  # Yellow
                "info": "\033[92m",  # Green
                "debug": "\033[94m",  # Blue
                "trace": "\033[95m",  # Magenta
            }
            if enable_colors
            else None,
        )

    @classmethod
    def _add_to_log_store(
        cls,
        logger: object,
        method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Processor to add log entries to the class log store."""
        logger_name = str(event_dict.get("logger", getattr(logger, "name", "unknown")))

        log_entry: FlextTypes.Logging.LogEntry = {
            "timestamp": event_dict.get("timestamp", time.time()),
            "level": str(event_dict.get("level", "INFO")).upper(),
            "logger": logger_name,
            "message": str(event_dict.get("event", "")),
            "method": method_name,
            "context": {
                k: v
                for k, v in event_dict.items()
                if k not in {"timestamp", "level", "logger", "event"}
            },
        }
        cls._log_store.append(log_entry)
        return event_dict

    @classmethod
    def _setup_custom_trace_level(cls) -> None:
        """Set up custom TRACE level for both stdlib logging and structlog."""
        trace_level = FlextConstants.Observability.TRACE_LEVEL

        # Register TRACE level in standard logging
        logging.addLevelName(trace_level, "TRACE")

        # Register TRACE level mappings in structlog if available
        name_to_level = getattr(structlog.stdlib, "_NAME_TO_LEVEL", None) or getattr(
            structlog.stdlib,
            "NAME_TO_LEVEL",
            None,
        )
        if isinstance(name_to_level, dict):
            name_to_level["trace"] = trace_level

        level_to_name = getattr(structlog.stdlib, "_LEVEL_TO_NAME", None) or getattr(
            structlog.stdlib,
            "LEVEL_TO_NAME",
            None,
        )
        if isinstance(level_to_name, dict):
            level_to_name[trace_level] = "trace"

        # Inject trace methods
        cls._inject_trace_methods(trace_level)

    @staticmethod
    def _inject_trace_methods(trace_level: int) -> None:
        """Inject trace methods into logging and structlog loggers."""

        def trace_method(self: logging.Logger, msg: str, *args: object) -> None:
            if self.isEnabledFor(trace_level):
                self._log(trace_level, msg, args)

        def bound_trace_method(
            self: structlog.stdlib.BoundLogger,
            event: str | None = None,
            **kwargs: object,
        ) -> object:
            if hasattr(self, "_proxy_to_logger"):
                proxy_method = getattr(self, "_proxy_to_logger", None)
                if callable(proxy_method):
                    return proxy_method("trace", event, **kwargs)
            return None

        # Inject methods safely
        try:
            if not hasattr(logging.Logger, "trace"):
                logging.Logger.trace = trace_method  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

        try:
            if not hasattr(structlog.stdlib.BoundLogger, "trace"):
                structlog.stdlib.BoundLogger.trace = bound_trace_method  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

    # ==========================================================================
    # CLASS UTILITY METHODS - Testing and observability
    # ==========================================================================

    @classmethod
    def get_log_store(cls) -> list[FlextTypes.Logging.LogEntry]:
        """Get log store for testing."""
        return cls._log_store.copy()

    @classmethod
    def clear_log_store(cls) -> None:
        """Clear log store for testing."""
        cls._log_store.clear()


# =============================================================================
# FLEXT LOGGER FACTORY - Consolidado eliminando _BaseLoggerFactory
# =============================================================================


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    "FlextLogger",  # ONLY main class exported
]
