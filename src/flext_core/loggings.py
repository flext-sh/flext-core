"""Structured logging with correlation IDs, performance tracking, and security sanitization.

Provides FlextLogger class with structured JSON logging, automatic correlation ID
generation, request context tracking, operation performance metrics, and
sensitive data sanitization.

Features:
- ISO 8601 timestamps with timezone
- Automatic correlation ID generation for request tracing
- Performance tracking for operations with start/complete methods
- Automatic sensitive data redaction (passwords, tokens, keys)
- Thread-safe request context storage
- Service metadata injection (name, version, environment)
- JSON output for production, colored console for development
- Integration with structlog for advanced formatting

Usage:
    Basic logging:
        logger = FlextLogger(__name__)
        logger.info("Processing request", user_id=123)
        logger.error("Failed to process", error=exception)

    Operation tracking:
        op_id = logger.start_operation("user_creation", user_id=123)
        # ... do work ...
        logger.complete_operation(op_id, success=True)

    Global correlation ID:
        FlextLogger.set_global_correlation_id("req_abc123")
        # All loggers will include this correlation ID
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
from typing import ClassVar, cast

import structlog
from structlog.typing import EventDict, Processor

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# =============================================================================
# ADVANCED STRUCTURED LOGGING - Enterprise Grade Implementation
# =============================================================================


class FlextLogger:
    """Structured logger with correlation IDs, performance tracking, and data sanitization.

    Provides structured JSON logging with automatic correlation ID generation,
    request context tracking, operation performance metrics, and sensitive data
    sanitization. Uses structlog for advanced formatting and processors.

    Attributes:
        _configured: Class-level flag indicating if logging system is configured.
        _global_correlation_id: Global correlation ID shared across all instances.
        _service_info: Service metadata (name, version, environment).
        _request_context: Request-specific context data.
        _performance_tracking: Performance metrics storage.

    Example:
        Basic usage:
            logger = FlextLogger(__name__)
            logger.info("Processing user request", user_id=123, action="login")
            logger.error("Database connection failed", error=exception)

        Operation tracking:
            op_id = logger.start_operation("user_creation", user_id=123)
            # ... perform operation ...
            logger.complete_operation(op_id, success=True, created_id="user_456")

        Global correlation ID:
            FlextLogger.set_global_correlation_id("req_abc123")
            # All subsequent log entries include this correlation ID

    """

    # Class-level configuration and shared state
    _configured: ClassVar[bool] = False
    _global_correlation_id: ClassVar[str | None] = None  # Global correlation ID
    _service_info: ClassVar[dict[str, object]] = {}
    _request_context: ClassVar[dict[str, object]] = {}
    _performance_tracking: ClassVar[dict[str, float]] = {}

    # Thread-local storage for per-request context
    _local = threading.local()

    def __init__(
        self,
        name: str,
        level: FlextTypes.Config.LogLevel = "INFO",
        service_name: str | None = None,
        service_version: str | None = None,
        correlation_id: str | None = None,
        environment: FlextTypes.Config.Environment = "development",
    ) -> None:
        """Initialize structured logger instance with FlextTypes.Config integration.

        Args:
            name: Logger name, typically `__name__` of calling module.
            level: Log level using FlextTypes.Config.LogLevel for validation.
            service_name: Service identifier for distributed tracing. Defaults to
                extracted from module name or SERVICE_NAME env var.
            service_version: Service version for deployment tracking. Defaults to
                SERVICE_VERSION env var or FlextConstants.Core.VERSION.
            correlation_id: Correlation ID for request tracing. Defaults to
                global correlation ID or auto-generated UUID.
            environment: Environment using FlextTypes.Config.Environment for consistency.

        Note:
            Automatically configures structlog if not already configured.
            Creates persistent context with service and system metadata.
            Uses FlextConstants.Config StrEnums for type safety.

        """
        if not self._configured:
            self.configure()

        self._name = name
        # Validate level is proper LogLevel type
        if isinstance(level, str):
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level.upper() in valid_levels:
                self._level = level.upper()  # type: ignore[assignment]
            else:
                self._level = "INFO"  # type: ignore[assignment]
        else:
            self._level = level  # type: ignore[assignment,unreachable]
        self._environment = environment

        # Initialize service context
        self._service_name = service_name or self._extract_service_name()
        self._service_version = service_version or self._get_version()

        # Set up performance tracking
        self._start_time = time.time()

        # Instance-level correlation ID (can override global)
        self._correlation_id = (
            correlation_id
            or self._global_correlation_id
            or self._generate_correlation_id()
        )

        # Set up structured logger with enriched context
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
        """Extract service name from logger name or environment variables.

        Returns:
            Service name extracted from SERVICE_NAME env var, module name,
            or defaults to "flext-core".

        Note:
            Converts underscores to hyphens for service names.

        """
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
        """Sanitize context by redacting sensitive information.

        Args:
            context: Dictionary containing log context data.

        Returns:
            Sanitized dictionary with sensitive values replaced with "[REDACTED]".

        Note:
            Sensitive keys include: password, secret, token, key, auth, credential.
            Recursively sanitizes nested dictionaries.

        """
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
        error: Exception | None = None,
        duration_ms: float | None = None,
    ) -> dict[str, object]:
        """Build comprehensive structured log entry.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: Primary log message.
            context: Optional additional context data for the log entry.
            error: Optional exception to include error details.
            duration_ms: Optional operation duration in milliseconds.

        Returns:
            Complete log entry dictionary with timestamp, correlation ID,
            sanitized context, performance metrics, and error details.

        """
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

        # Add performance metrics
        if duration_ms is not None:
            entry["performance"] = {
                "duration_ms": round(duration_ms, 3),
                "timestamp": self._get_current_timestamp(),
            }

        # Add error details if present
        if error:
            entry["error"] = {
                "type": error.__class__.__name__,
                "message": str(error),
                "stack_trace": traceback.format_exception(
                    type(error), error, error.__traceback__
                ),
                "module": getattr(error, "__module__", "unknown"),
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
        """Create a new logger instance with bound context data.

        This method creates a new FlextLogger instance that inherits all the
        configuration and state of the current logger but includes additional
        bound context that will be automatically included in all log messages.

        Args:
            **context: Key-value pairs to bind to the new logger context.

        Returns:
            FlextLogger: New logger instance with bound context.

        Example:
            >>> logger = FlextLogger("main")
            >>> bound_logger = logger.bind(operation="login", user_id="123")
            >>> bound_logger.info(
            ...     "User logged in"
            ... )  # Will include operation and user_id

        """
        # Create a new logger instance with same configuration

        bound_logger = FlextLogger(
            name=self._name,
            level=cast("FlextTypes.Config.LogLevel", self._level),
            service_name=getattr(self, "_service_name", None),
            service_version=getattr(self, "_service_version", None),
            correlation_id=getattr(self, "_correlation_id", None),
            environment=getattr(self, "_environment", "development"),
        )

        # Copy existing context
        if hasattr(self._local, "request_context"):
            bound_logger.set_request_context(**self._local.request_context)

        # Add new bound context
        bound_logger.set_request_context(**context)

        return bound_logger

    def start_operation(self, operation_name: str, **context: object) -> str:
        """Start tracking an operation with performance metrics.

        Args:
            operation_name: Human-readable name for the operation.
            **context: Additional context data to include with the operation.

        Returns:
            Operation ID string that can be used with complete_operation().

        Note:
            Stores operation start time and context in thread-local storage.
            Automatically logs operation start event with provided context.

        """
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
        self, operation_id: str, *, success: bool = True, **context: object
    ) -> None:
        """Complete operation tracking with performance metrics.

        Args:
            operation_id: Operation ID returned from start_operation().
            success: Whether the operation completed successfully.
            **context: Additional context data to include with completion log.

        Note:
            Calculates operation duration and logs completion event.
            Cleans up operation data from thread-local storage.
            Logs as info on success, error on failure.

        """
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
            formatted_message, **entry
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
        error: Exception | None = None,
        **context: object,
    ) -> None:
        """Log error message with full context and error details.

        Args:
            message: Log message with optional printf-style formatting.
            *args: Arguments for message formatting (printf-style).
            error: Optional exception to include error details and stack trace.
            **context: Additional context data for the log entry.

        """
        formatted_message = message % args if args else message
        entry = self._build_log_entry("ERROR", formatted_message, context, error)
        self._structlog_logger.error(formatted_message, **entry)

    def critical(
        self,
        message: str,
        *args: object,
        error: Exception | None = None,
        **context: object,
    ) -> None:
        """Log critical message with full context and error details.

        Args:
            message: Log message with optional printf-style formatting.
            *args: Arguments for message formatting (printf-style).
            error: Optional exception to include error details and stack trace.
            **context: Additional context data for the log entry.

        """
        formatted_message = message % args if args else message
        entry = self._build_log_entry("CRITICAL", formatted_message, context, error)
        self._structlog_logger.critical(formatted_message, **entry)

    def exception(self, message: str, *args: object, **context: object) -> None:
        """Log exception with full stack trace and context.

        Args:
            message: Log message with optional printf-style formatting.
            *args: Arguments for message formatting (printf-style).
            **context: Additional context data for the log entry.

        Note:
            Automatically captures current exception information using sys.exc_info().
            Should be called from within an exception handler block.

        """
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
        """Configure advanced structured logging system.

        Args:
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            json_output: Force JSON output format. Auto-detects from ENVIRONMENT
                if None (production/staging use JSON, development uses console).
            include_source: Include source filename, line number, and function name
                in log entries using CallsiteParameterAdder processor.
            structured_output: Enable structured logging processors for correlation,
                performance metrics, and data sanitization.

        Note:
            Configures both structlog and stdlib logging. Sets class-level
            _configured flag to True after successful configuration.
            Uses ISO 8601 timestamps with UTC timezone.

        """
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
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="@timestamp")
        )

        # Add source information if requested
        if include_source:
            processors.append(
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                    ]
                )
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
                )
            )
        else:
            processors.append(cls._create_enhanced_console_renderer())

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
        """Add correlation ID to all log entries.

        Args:
            _logger: Standard library logger instance (unused).
            _method_name: Log method name (unused).
            event_dict: Event dictionary to modify.

        Returns:
            Modified event dictionary with correlation_id field added.

        """
        if FlextLogger._global_correlation_id:
            event_dict["correlation_id"] = FlextLogger._global_correlation_id
        return event_dict

    @staticmethod
    def _add_performance_processor(
        _logger: logging.Logger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Add performance metrics to log entries.

        Args:
            _logger: Standard library logger instance (unused).
            _method_name: Log method name (unused).
            event_dict: Event dictionary to modify.

        Returns:
            Modified event dictionary with @metadata performance information.

        """
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
        """Sanitize sensitive information from log entries.

        Args:
            _logger: Standard library logger instance (unused).
            _method_name: Log method name (unused).
            event_dict: Event dictionary to modify.

        Returns:
            Modified event dictionary with sensitive values redacted.

        Note:
            Searches for keys containing sensitive terms and replaces
            their values with "[REDACTED]" for security.

        """
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
        """Create enhanced console renderer for development.

        Returns:
            ConsoleRenderer instance with colored output and level styling
            for improved development experience.

        """
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
        """Set global correlation ID for request tracing.

        Args:
            correlation_id: Correlation ID to set globally across all logger
                instances. Pass None to clear the global correlation ID.

        """
        cls._global_correlation_id = correlation_id

    @classmethod
    def get_global_correlation_id(cls) -> str | None:
        """Get current global correlation ID.

        Returns:
            Current global correlation ID string, or None if not set.

        """
        return cls._global_correlation_id

    # =============================================================================
    # CONFIGURATION MANAGEMENT - FlextTypes.Config Integration
    # =============================================================================

    @classmethod
    def configure_logging_system(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure logging system using FlextTypes.Config.

        Configures logging behavior, output format, log levels, and performance
        tracking based on environment and operational requirements.

        Args:
            config: Configuration dictionary with logging settings

        Returns:
            FlextResult containing validated logging configuration or error

        """
        try:
            # Since config is typed as ConfigDict (dict subtype), validation is ensured
            validated_config: FlextTypes.Config.ConfigDict = {}

            # Validate environment (required for logging behavior)
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate log level (controls logging verbosity)
            if "log_level" in config:
                log_level = config["log_level"]
                valid_levels = [level.value for level in FlextConstants.Config.LogLevel]
                if log_level not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {valid_levels}"
                    )
                validated_config["log_level"] = log_level
            # Default based on environment
            elif validated_config["environment"] == "production":
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.WARNING.value
                )
            elif validated_config["environment"] == "development":
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.DEBUG.value
                )
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.INFO.value
                )

            # Add logging specific configuration
            validated_config["enable_console_output"] = config.get(
                "enable_console_output", validated_config["environment"] != "production"
            )
            validated_config["enable_json_logging"] = config.get(
                "enable_json_logging", validated_config["environment"] == "production"
            )
            validated_config["enable_correlation_tracking"] = config.get(
                "enable_correlation_tracking", True
            )
            validated_config["enable_performance_logging"] = config.get(
                "enable_performance_logging", True
            )
            validated_config["enable_sensitive_data_sanitization"] = config.get(
                "enable_sensitive_data_sanitization", True
            )
            validated_config["max_log_message_size"] = config.get(
                "max_log_message_size", 10000
            )
            validated_config["log_rotation_enabled"] = config.get(
                "log_rotation_enabled", validated_config["environment"] == "production"
            )
            validated_config["async_logging_enabled"] = config.get(
                "async_logging_enabled", False
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Logging configuration error: {e}"
            )

    @classmethod
    def get_logging_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current logging system configuration.

        Returns:
            FlextResult containing current logging system configuration

        """
        try:
            # Build current configuration from system state
            current_config: FlextTypes.Config.ConfigDict = {
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "enable_console_output": True,
                "enable_json_logging": False,
                "enable_correlation_tracking": True,
                "enable_performance_logging": True,
                "enable_sensitive_data_sanitization": True,
                "max_log_message_size": 10000,
                "log_rotation_enabled": False,
                "async_logging_enabled": False,
                "global_correlation_id": cls._global_correlation_id or "",
                "active_operations": 0,  # Operation context tracking not implemented yet
                "logging_processors_enabled": [
                    "timestamp_processor",
                    "correlation_processor",
                    "sanitization_processor",
                ],
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(current_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get logging config: {e}"
            )

    @classmethod
    def create_environment_logging_config(
        cls, environment: FlextTypes.Config.Environment
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific logging configuration.

        Args:
            environment: Target environment for logging configuration

        Returns:
            FlextResult containing environment-optimized logging configuration

        """
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Create environment-specific configuration
            if environment == "production":
                config: FlextTypes.Config.ConfigDict = {
                    "environment": environment,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_console_output": False,  # No console output in production
                    "enable_json_logging": True,  # Structured JSON for production
                    "enable_correlation_tracking": True,
                    "enable_performance_logging": True,
                    "enable_sensitive_data_sanitization": True,  # Critical for production
                    "max_log_message_size": 5000,  # Limit message size
                    "log_rotation_enabled": True,  # Enable rotation for production
                    "async_logging_enabled": True,  # Async for performance
                }
            elif environment == "development":
                config = {
                    "environment": environment,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_console_output": True,  # Console output for development
                    "enable_json_logging": False,  # Human-readable for development
                    "enable_correlation_tracking": True,
                    "enable_performance_logging": True,
                    "enable_sensitive_data_sanitization": True,
                    "max_log_message_size": 20000,  # More details for debugging
                    "log_rotation_enabled": False,  # No rotation needed in dev
                    "async_logging_enabled": False,  # Sync for immediate feedback
                }
            elif environment == "test":
                config = {
                    "environment": environment,
                    "log_level": FlextConstants.Config.LogLevel.ERROR.value,
                    "enable_console_output": False,  # Minimal output for tests
                    "enable_json_logging": False,  # Simple output for tests
                    "enable_correlation_tracking": False,  # No correlation in tests
                    "enable_performance_logging": False,  # No perf logging in tests
                    "enable_sensitive_data_sanitization": True,
                    "max_log_message_size": 1000,  # Small messages for tests
                    "log_rotation_enabled": False,  # No rotation in tests
                    "async_logging_enabled": False,  # Sync for test reliability
                }
            else:  # staging, local, etc.
                config = {
                    "environment": environment,
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "enable_console_output": True,
                    "enable_json_logging": True,
                    "enable_correlation_tracking": True,
                    "enable_performance_logging": True,
                    "enable_sensitive_data_sanitization": True,
                    "max_log_message_size": 10000,
                    "log_rotation_enabled": True,
                    "async_logging_enabled": False,
                }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Environment logging config failed: {e}"
            )

    @classmethod
    def optimize_logging_performance(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize logging performance based on configuration.

        Args:
            config: Performance optimization configuration dictionary

        Returns:
            FlextResult containing optimized logging configuration

        """
        try:
            # Default performance configuration
            optimized_config: FlextTypes.Config.ConfigDict = {
                "async_logging_enabled": config.get("async_logging_enabled", False),
                "buffer_size": config.get("buffer_size", 1000),
                "flush_interval_ms": config.get("flush_interval_ms", 5000),
                "max_concurrent_operations": config.get(
                    "max_concurrent_operations", 100
                ),
                "enable_log_compression": config.get("enable_log_compression", False),
                "log_level_caching": config.get("log_level_caching", True),
                "disable_trace_logging": config.get("disable_trace_logging", False),
                "batch_log_processing": config.get("batch_log_processing", False),
            }

            # Optimize based on performance level
            performance_level = config.get("performance_level", "standard")

            if performance_level == "high":
                optimized_config.update({
                    "async_logging_enabled": True,
                    "buffer_size": 5000,
                    "flush_interval_ms": 1000,
                    "max_concurrent_operations": 500,
                    "enable_log_compression": True,
                    "batch_log_processing": True,
                    "disable_trace_logging": True,
                })
            elif performance_level == "low":
                optimized_config.update({
                    "async_logging_enabled": False,
                    "buffer_size": 100,
                    "flush_interval_ms": 10000,
                    "max_concurrent_operations": 10,
                    "enable_log_compression": False,
                    "batch_log_processing": False,
                })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Logging performance optimization failed: {e}"
            )


# Note: Helper functions moved to legacy.py for proper architecture


# =============================================================================
# EXPORTS
# =============================================================================

__all__: list[str] = [
    "FlextLogger",
    # Note: Helper functions moved to legacy.py for proper architecture
]
