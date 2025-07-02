"""Structured logging system for FLEXT framework."""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Basic log data
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        reserved_fields = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "getMessage",
            "timestamp",
            "level",
            "logger",
            "message",
            "function",
            "line",
        }

        for key, value in record.__dict__.items():
            if key not in reserved_fields and not key.startswith("_"):
                # Rename conflicting fields
                if key == "module":
                    key = "module_name"
                log_data[key] = value

        return json.dumps(log_data, default=str, ensure_ascii=False)


class FlextLogger:
    """Structured logger for FLEXT framework."""

    def __init__(self, name: str, level: str = "INFO") -> None:
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup log handlers for console and file output."""
        # Console handler with structured output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        console_handler.setLevel(logging.INFO)

        # File handler for persistent logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_dir / f"{self.name}.log")
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.DEBUG)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with extra data."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with extra data."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with extra data."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with extra data."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with extra data."""
        self.logger.critical(message, extra=kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)

    def log_operation(
        self, operation: str, duration_ms: float, success: bool, **kwargs: Any
    ) -> None:
        """Log an operation with timing and success status."""
        self.info(
            f"Operation completed: {operation}",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **kwargs,
        )

    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log API request with standard fields."""
        self.info(
            f"API request: {method} {path}",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
            **kwargs,
        )

    def log_pipeline_event(
        self, pipeline_id: str, event_type: str, status: str, **kwargs: Any
    ) -> None:
        """Log pipeline event with standard fields."""
        self.info(
            f"Pipeline event: {event_type}",
            pipeline_id=pipeline_id,
            event_type=event_type,
            status=status,
            **kwargs,
        )

    def log_plugin_event(
        self, plugin_name: str, event_type: str, success: bool, **kwargs: Any
    ) -> None:
        """Log plugin event with standard fields."""
        level = "info" if success else "error"
        getattr(self, level)(
            f"Plugin event: {event_type}",
            plugin_name=plugin_name,
            event_type=event_type,
            success=success,
            **kwargs,
        )


class LoggingContextManager:
    """Context manager for operation logging."""

    def __init__(self, logger: FlextLogger, operation: str, **kwargs: Any) -> None:
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time: Optional[float] = None

    def __enter__(self) -> "LoggingContextManager":
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation}", **self.kwargs)
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: object) -> None:
        duration_ms = (time.time() - (self.start_time or 0)) * 1000
        success = exc_type is None

        if success:
            self.logger.log_operation(self.operation, duration_ms, True, **self.kwargs)
        else:
            self.logger.log_operation(
                self.operation,
                duration_ms,
                False,
                error_type=exc_type.__name__ if exc_type else None,
                error_message=str(exc_val) if exc_val else None,
                **self.kwargs,
            )


# Global logger instances for each module
_loggers: dict[str, FlextLogger] = {}


def get_logger(name: str, level: Optional[str] = None) -> FlextLogger:
    """Get or create a structured logger for a module."""
    if name not in _loggers:
        # Get log level from environment or default
        log_level = level or os.getenv("FLX_LOG_LEVEL", "INFO")
        _loggers[name] = FlextLogger(name, log_level or "INFO")

    return _loggers[name]


def log_operation(operation: str, **kwargs: Any) -> Any:
    """Decorator for logging function operations."""
    from typing import Callable

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **func_kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            with LoggingContextManager(logger, operation, **kwargs):
                return func(*args, **func_kwargs)

        return wrapper

    return decorator


def log_async_operation(operation: str, **kwargs: Any) -> Any:
    """Decorator for logging async function operations."""
    from collections.abc import Awaitable
    from typing import Callable

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args: Any, **func_kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            with LoggingContextManager(logger, operation, **kwargs):
                return await func(*args, **func_kwargs)

        return wrapper

    return decorator


# Module-specific loggers
core_logger = get_logger("flext.core")
auth_logger = get_logger("flext.auth")
api_logger = get_logger("flext.api")
plugin_logger = get_logger("flext.plugin")
grpc_logger = get_logger("flext.grpc")
web_logger = get_logger("flext.web")
cli_logger = get_logger("flext.cli")
meltano_logger = get_logger("flext.meltano")
observability_logger = get_logger("flext.observability")


# Convenience functions
def log_startup(component: str, version: str, **kwargs: Any) -> None:
    """Log component startup."""
    logger = get_logger(f"flext.{component}")
    logger.info(f"Starting {component}", component=component, version=version, **kwargs)


def log_shutdown(component: str, **kwargs: Any) -> None:
    """Log component shutdown."""
    logger = get_logger(f"flext.{component}")
    logger.info(f"Shutting down {component}", component=component, **kwargs)


def log_performance_metric(
    metric_name: str, value: float, unit: str = "ms", **kwargs: Any
) -> None:
    """Log performance metric."""
    observability_logger.info(
        f"Performance metric: {metric_name}",
        metric_name=metric_name,
        value=value,
        unit=unit,
        **kwargs,
    )


def log_security_event(
    event_type: str, severity: str, user_id: Optional[str] = None, **kwargs: Any
) -> None:
    """Log security-related event."""
    auth_logger.warning(
        f"Security event: {event_type}",
        event_type=event_type,
        severity=severity,
        user_id=user_id,
        **kwargs,
    )
