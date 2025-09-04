"""Structured logging with correlation IDs, performance tracking, and security sanitization.

Provides FlextLogger with structured logging, JSON output, correlation ID
tracking, performance metrics, and sensitive data sanitization using structlog.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import time
import traceback
import uuid
from datetime import UTC, datetime
from typing import ClassVar, Self, cast

import structlog
from structlog.typing import EventDict, Processor

from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.typings import FlextTypes

# =============================================================================
# ADVANCED STRUCTURED LOGGING - Enterprise Grade Implementation
# =============================================================================


class FlextLogger:
    """Structured logger with correlation IDs, performance tracking, and data sanitization.

    Provides structured JSON logging with automatic correlation ID generation,
    request context tracking, operation performance metrics, and sensitive data
    sanitization. Uses structlog for advanced formatting and processors.

    """

    # Class-level configuration and shared state
    _configured: ClassVar[bool] = False
    _global_correlation_id: ClassVar[str | None] = None  # Global correlation ID
    _service_info: ClassVar[dict[str, object]] = {}
    _request_context: ClassVar[dict[str, object]] = {}
    _performance_tracking: ClassVar[dict[str, float]] = {}

    # Logger instance cache for singleton pattern
    _instances: ClassVar[dict[str, FlextLogger]] = {}

    # Thread-local storage for per-request context
    _local = threading.local()

    # Instance attributes type declarations
    _name: str
    _level: FlextTypes.Config.LogLevel
    _environment: FlextTypes.Config.Environment

    def __new__(cls, name: str, *_args: object, **kwargs: object) -> Self:
        """Create or return cached logger instance for singleton pattern."""
        # Check if this is a bind() call that needs a new instance
        force_new = kwargs.pop("_force_new", False)

        if force_new or name not in cls._instances:
            instance = super().__new__(cls)
            if not force_new:  # Only cache if not forced new
                cls._instances[name] = instance
            return instance

        return cast("Self", cls._instances[name])

    def __init__(
        self,
        name: str,
        level: FlextTypes.Config.LogLevel = "INFO",
        service_name: str | None = None,
        service_version: str | None = None,
        correlation_id: str | None = None,
        environment: FlextTypes.Config.Environment = "development",
        *,
        _force_new: bool = False,  # Accept but ignore this parameter
    ) -> None:
        """Initialize structured logger instance with FlextTypes.Config integration."""
        if not self._configured:
            self.configure()

        self._name = name
        # Validate and set level (LogLevel is already a str type)
        if isinstance(level, str):
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            upper_level = level.upper()
            if upper_level in valid_levels:
                # Cast to proper LogLevel type after validation
                self._level = cast("FlextTypes.Config.LogLevel", upper_level)
            else:
                self._level = "INFO"
        # LogLevel type is str-based, so this branch is not needed
        self._environment = environment

        # Initialize service context
        self._service_name = service_name or self._extract_service_name()
        self._service_version = service_version or self._get_version()

        # Set up performance tracking
        self._start_time = time.time()

        # Instance-level correlation ID (can override global)
        try:
            context_id = FlextContext.Correlation.get_correlation_id()
        except ImportError:
            context_id = None

        self._correlation_id = (
            correlation_id
            or self._global_correlation_id
            or context_id  # Check global context
            or self._generate_correlation_id()
        )

        # Set up structured logger with enriched context (import locally)

        self._structlog_logger = structlog.get_logger(name)

        # Initialize persistent context
        self._persistent_context: dict[str, object] = {
            "service": {
                "name": self._service_name,
                "version": self._service_version,
                "instance_id": self._get_instance_id(),
                "environment": self._get_environment(),
            },
            "system": {
                "hostname": platform.node(),
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "process_id": os.getpid(),
                "thread_id": threading.get_ident(),
            },
        }

    def _extract_service_name(self) -> str:
        """Extract service name from logger name or environment variables."""
        if service_name := os.environ.get("SERVICE_NAME"):
            return service_name

        # Extract from module name
        min_parts = 2
        parts = self._name.split(".")
        if len(parts) >= min_parts and parts[0].startswith("flext_"):
            return parts[0].replace("_", "-")

        return "flext-core"

    def _get_version(self) -> str:
        """Get service version from environment or constants."""
        return (
            os.environ.get("SERVICE_VERSION") or FlextConstants.Core.VERSION or "0.9.0"
        )

    def _get_environment(self) -> str:
        """Determine current environment."""
        return (
            os.environ.get("ENVIRONMENT") or os.environ.get("ENV") or "development"
        ).lower()

    def _get_instance_id(self) -> str:
        """Get unique instance identifier."""
        return os.environ.get("INSTANCE_ID") or f"{platform.node()}-{os.getpid()}"

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for request tracing."""
        return f"corr_{uuid.uuid4().hex[:16]}"

    def _get_current_timestamp(self) -> str:
        """Get current ISO 8601 timestamp with timezone."""
        return datetime.now(UTC).isoformat()

    def _sanitize_context(self, context: dict[str, object]) -> dict[str, object]:
        """Sanitize context by redacting sensitive information."""
        sensitive_keys = {
            "password",
            "passwd",
            "secret",
            "token",
            "key",
            "auth",
            "authorization",
            "credential",
            "private",
            "api_key",
            "access_token",
            "refresh_token",
            "session_id",
            "cookie",
        }

        sanitized: dict[str, object] = {}
        for key, value in context.items():
            key_lower = str(key).lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_context(value)
            else:
                sanitized[key] = value

        return sanitized

    def _build_log_entry(
        self,
        level: str,
        message: str,
        context: dict[str, object] | None = None,
        error: Exception | str | None = None,
        duration_ms: float | None = None,
    ) -> dict[str, object]:
        """Build efficient structured log entry."""
        # Start with timestamp and correlation
        entry: dict[str, object] = {
            "@timestamp": self._get_current_timestamp(),
            "level": level.upper(),
            "message": str(message),
            "logger": self._name,
            "correlation_id": self._correlation_id,
        }

        # Add service and system context
        entry.update(self._persistent_context)

        # Add request context if available
        request_context = getattr(self._local, "request_context", {})
        if request_context:
            entry["request"] = request_context

        # Add permanent context if available
        permanent_context = getattr(self, "_permanent_context", {})
        if permanent_context:
            entry["permanent"] = permanent_context

        # Add performance metrics
        if duration_ms is not None:
            entry["performance"] = {
                "duration_ms": round(duration_ms, 3),
                "timestamp": self._get_current_timestamp(),
            }

        # Add error details if present
        if error:
            if isinstance(error, Exception):
                # Handle Exception objects with full details
                entry["error"] = {
                    "type": error.__class__.__name__,
                    "message": str(error),
                    "stack_trace": traceback.format_exception(
                        type(error), error, error.__traceback__,
                    ),
                    "module": getattr(error, "__module__", "unknown"),
                }
            else:
                # Handle string error messages
                entry["error"] = {
                    "type": "StringError",
                    "message": str(error),
                    "stack_trace": None,
                    "module": "unknown",
                }

        # Add sanitized user context
        if context:
            sanitized_context = self._sanitize_context(context)
            entry["context"] = sanitized_context

        # Add execution context
        entry["execution"] = {
            "function": self._get_calling_function(),
            "line": self._get_calling_line(),
            "uptime_seconds": round(time.time() - self._start_time, 3),
        }

        return entry

    def _get_calling_function(self) -> str:
        """Get the name of the calling function."""
        try:
            frame = sys._getframe(4)  # Skip internal logging frames
            return frame.f_code.co_name
        except (AttributeError, ValueError):
            return "unknown"

    def _get_calling_line(self) -> int:
        """Get the line number of the calling code."""
        try:
            frame = sys._getframe(4)  # Skip internal logging frames
            return frame.f_lineno
        except (AttributeError, ValueError):
            return 0

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for request tracing (instance level)."""
        self._correlation_id = correlation_id  # Instance-level correlation ID

    def set_request_context(self, **context: object) -> None:
        """Set request-specific context for tracing."""
        if not hasattr(self._local, "request_context"):
            self._local.request_context = {}
        self._local.request_context.update(context)

    def clear_request_context(self) -> None:
        """Clear request-specific context."""
        if hasattr(self._local, "request_context"):
            self._local.request_context.clear()

    def bind(self, **context: object) -> FlextLogger:
        """Create a new logger instance with bound context data."""
        # Create a new logger instance with same configuration
        # Use _force_new=True to bypass singleton pattern for bind()
        bound_logger = FlextLogger(
            name=self._name,
            level=self._level,
            service_name=getattr(self, "_service_name", None),
            service_version=getattr(self, "_service_version", None),
            correlation_id=getattr(self, "_correlation_id", None),
            environment=getattr(self, "_environment", "development"),
            _force_new=True,
        )

        # Copy existing request context
        if hasattr(self._local, "request_context"):
            bound_logger.set_request_context(**self._local.request_context)

        # Copy existing permanent context
        if hasattr(self, "_permanent_context"):
            bound_logger._permanent_context = self._permanent_context.copy()

        # Add new bound context
        bound_logger.set_request_context(**context)

        return bound_logger

    def set_context(
        self, context_dict: dict[str, object] | None = None, **context: object,
    ) -> None:
        """Set permanent context data for this logger instance."""
        if not hasattr(self, "_permanent_context"):
            self._permanent_context: dict[str, object] = {}

        if context_dict is not None:
            # Replace existing context with new dict
            self._permanent_context = dict(context_dict)
            # Add any additional kwargs
            self._permanent_context.update(context)
        else:
            # Just update existing context with kwargs
            self._permanent_context.update(context)

    def with_context(self, **context: object) -> FlextLogger:
        """Create a new logger instance with additional bound context."""
        return self.bind(**context)

    def start_operation(self, operation_name: str, **context: object) -> str:
        """Start tracking an operation with performance metrics."""
        operation_id = f"op_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        # Store operation start time
        if not hasattr(self._local, "operations"):
            self._local.operations = {}
        self._local.operations[operation_id] = {
            "name": operation_name,
            "start_time": start_time,
            "context": context,
        }

        self.info(
            f"Operation started: {operation_name}",
            operation_id=operation_id,
            operation_name=operation_name,
            **context,
        )

        return operation_id

    def complete_operation(
        self, operation_id: str, *, success: bool = True, **context: object,
    ) -> None:
        """Complete operation tracking with performance metrics."""
        if not hasattr(self._local, "operations"):
            return

        operation_info = self._local.operations.get(operation_id)
        if not operation_info:
            return

        duration_ms = (time.time() - operation_info["start_time"]) * 1000

        log_context = {
            "operation_id": operation_id,
            "operation_name": operation_info["name"],
            "success": success,
            "duration_ms": round(duration_ms, 3),
            **operation_info["context"],
            **context,
        }

        if success:
            self.info(f"Operation completed: {operation_info['name']}", **log_context)
        else:
            self.error(f"Operation failed: {operation_info['name']}", **log_context)

        # Clean up
        del self._local.operations[operation_id]

    # Standard logging methods with enhanced context
    def trace(self, message: str, *args: object, **context: object) -> None:
        """Log trace message with full context (uses debug level internally).

        Args:
            message: Log message with optional printf-style formatting.
            *args: Arguments for message formatting (printf-style).
            **context: Additional context data for the log entry.

        """
        formatted_message = message % args if args else message
        entry = self._build_log_entry("TRACE", formatted_message, context)
        self._structlog_logger.debug(
            formatted_message, **entry,
        )  # Use debug since structlog doesn't have trace

    def debug(self, message: str, *args: object, **context: object) -> None:
        """Log debug message with full context.

        Args:
            message: Log message with optional printf-style formatting.
            *args: Arguments for message formatting (printf-style).
            **context: Additional context data for the log entry.

        """
        formatted_message = message % args if args else message
        entry = self._build_log_entry("DEBUG", formatted_message, context)
        self._structlog_logger.debug(formatted_message, **entry)

    def info(self, message: str, *args: object, **context: object) -> None:
        """Log info message with full context.

        Args:
            message: Log message with optional printf-style formatting.
            *args: Arguments for message formatting (printf-style).
            **context: Additional context data for the log entry.

        """
        formatted_message = message % args if args else message
        entry = self._build_log_entry("INFO", formatted_message, context)
        self._structlog_logger.info(formatted_message, **entry)

    def warning(self, message: str, *args: object, **context: object) -> None:
        """Log warning message with full context.

        Args:
            message: Log message with optional printf-style formatting.
            *args: Arguments for message formatting (printf-style).
            **context: Additional context data for the log entry.

        """
        formatted_message = message % args if args else message
        entry = self._build_log_entry("WARNING", formatted_message, context)
        self._structlog_logger.warning(formatted_message, **entry)

    def error(
        self,
        message: str,
        *args: object,
        error: Exception | str | None = None,
        **context: object,
    ) -> None:
        """Log error message with full context and error details.

        Args:
            message: Log message with optional printf-style formatting.
            *args: Arguments for message formatting (printf-style).
            error: Optional exception or error message string to include error details and stack trace.
            **context: Additional context data for the log entry.

        """
        formatted_message = message % args if args else message
        entry = self._build_log_entry("ERROR", formatted_message, context, error)
        self._structlog_logger.error(formatted_message, **entry)

    def critical(
        self,
        message: str,
        *args: object,
        error: Exception | str | None = None,
        **context: object,
    ) -> None:
        """Log critical message with full context and error details.

        Args:
            message: Log message with optional printf-style formatting.
            *args: Arguments for message formatting (printf-style).
            error: Optional exception or error message string to include error details and stack trace.
            **context: Additional context data for the log entry.

        """
        formatted_message = message % args if args else message
        entry = self._build_log_entry("CRITICAL", formatted_message, context, error)
        self._structlog_logger.critical(formatted_message, **entry)

    def exception(self, message: str, *args: object, **context: object) -> None:
        """Log exception with full stack trace and context."""
        formatted_message = message % args if args else message
        exc_info = sys.exc_info()
        error = exc_info[1] if isinstance(exc_info[1], Exception) else None
        entry = self._build_log_entry("ERROR", formatted_message, context, error)
        self._structlog_logger.error(formatted_message, **entry)

    @classmethod
    def configure(
        cls,
        *,
        log_level: str = "INFO",
        json_output: bool | None = None,
        include_source: bool = True,
        structured_output: bool = True,
    ) -> None:
        """Configure advanced structured logging system."""
        # Auto-detect output format if not specified
        if json_output is None:
            env = os.environ.get("ENVIRONMENT", "development").lower()
            json_output = env in {"production", "staging", "prod"}

        processors: list[Processor] = [
            # Essential processors
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
        ]

        # Add timestamp processor with ISO 8601 format
        processors.append(
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="@timestamp"),
        )

        # Add source information if requested
        if include_source:
            processors.append(
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                    ],
                ),
            )

        # Add structured processors
        if structured_output:
            processors.extend([
                cls._add_correlation_processor,
                cls._add_performance_processor,
                cls._sanitize_processor,
            ])

        # Choose output format
        if json_output:
            processors.append(
                structlog.processors.JSONRenderer(
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            )
        else:
            processors.append(FlextLogger._create_enhanced_console_renderer())

        # Configure structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Configure stdlib logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stderr,
            level=getattr(logging, log_level.upper(), logging.INFO),
        )

        cls._configured = True

    @staticmethod
    def _add_correlation_processor(
        _logger: logging.Logger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Add correlation ID to all log entries."""
        if FlextLogger._global_correlation_id:
            event_dict["correlation_id"] = FlextLogger._global_correlation_id
        return event_dict

    @staticmethod
    def _add_performance_processor(
        _logger: logging.Logger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Add performance metrics to log entries."""
        event_dict["@metadata"] = {
            "processor": "flext_logging",
            "version": FlextConstants.Core.VERSION,
            "processed_at": datetime.now(UTC).isoformat(),
        }
        return event_dict

    @staticmethod
    def _sanitize_processor(
        _logger: logging.Logger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Sanitize sensitive information from log entries."""
        sensitive_keys = {
            "password",
            "passwd",
            "secret",
            "token",
            "key",
            "auth",
            "authorization",
            "credential",
            "private",
            "api_key",
        }

        for key in list(event_dict.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                event_dict[key] = "[REDACTED]"

        return event_dict

    @staticmethod
    def _create_enhanced_console_renderer() -> structlog.dev.ConsoleRenderer:
        """Create enhanced console renderer for development."""
        return structlog.dev.ConsoleRenderer(
            colors=True,
            level_styles={
                "critical": "\033[91;1m",  # Bright red, bold
                "error": "\033[91m",  # Red
                "warning": "\033[93m",  # Yellow
                "info": "\033[92m",  # Green
                "debug": "\033[94m",  # Blue
                "trace": "\033[95m",  # Magenta
            },
            repr_native_str=False,
        )

    @classmethod
    def set_global_correlation_id(cls, correlation_id: str | None) -> None:
        """Set global correlation ID for request tracing."""
        cls._global_correlation_id = correlation_id

    @classmethod
    def get_global_correlation_id(cls) -> str | None:
        """Get current global correlation ID."""
        return cls._global_correlation_id


__all__: list[str] = [
    "FlextLogger",
]
