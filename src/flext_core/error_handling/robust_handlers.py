"""Robust error handling system for FLEXT framework."""

import functools
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

# Python < 3.11 compatibility for datetime.UTC
try:
    from datetime import UTC
except ImportError:
    UTC = UTC
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")
AsyncT = TypeVar("AsyncT")


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    CONFIGURATION = "configuration"
    DATABASE = "database"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    PLUGIN = "plugin"
    PERMISSION = "permission"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    component: str
    operation: str
    user_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    additional_data: dict[str, Any] | None = None


@dataclass
class ErrorReport:
    """Comprehensive error report."""

    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    traceback: str
    context: ErrorContext
    retry_count: int = 0
    resolved: bool = False


class FlextException(Exception):
    """Base exception for FLEXT framework."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.cause = cause
        self.timestamp = datetime.now(UTC)


class ConfigurationError(FlextException):
    """Configuration-related errors."""

    def __init__(
        self, message: str, config_key: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, ErrorCategory.CONFIGURATION, **kwargs)
        self.config_key = config_key


class DatabaseError(FlextException):
    """Database-related errors."""

    def __init__(
        self, message: str, query: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, ErrorCategory.DATABASE, **kwargs)
        self.query = query


class AuthenticationError(FlextException):
    """Authentication-related errors."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message, ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH, **kwargs
        )


class ValidationError(FlextException):
    """Validation-related errors."""

    def __init__(
        self, message: str, field: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, ErrorCategory.VALIDATION, **kwargs)
        self.field = field


class PluginError(FlextException):
    """Plugin-related errors."""

    def __init__(
        self, message: str, plugin_name: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, ErrorCategory.PLUGIN, **kwargs)
        self.plugin_name = plugin_name


class NetworkError(FlextException):
    """Network-related errors."""

    def __init__(self, message: str, url: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, ErrorCategory.NETWORK, **kwargs)
        self.url = url


class TimeoutError(FlextException):
    """Timeout-related errors."""

    def __init__(
        self, message: str, timeout_seconds: float | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, ErrorCategory.TIMEOUT, **kwargs)
        self.timeout_seconds = timeout_seconds


class ResourceError(FlextException):
    """Resource-related errors."""

    def __init__(
        self, message: str, resource_type: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, ErrorCategory.RESOURCE, **kwargs)
        self.resource_type = resource_type


class RobustErrorHandler:
    """Robust error handling system with retry logic and reporting."""

    def __init__(self) -> None:
        self.error_reports: dict[str, ErrorReport] = {}
        self.error_count = 0

    def generate_error_id(self) -> str:
        """Generate unique error ID."""
        self.error_count += 1
        timestamp = int(time.time())
        return f"FLX-{timestamp}-{self.error_count:04d}"

    def handle_error(
        self,
        exception: Exception,
        context: ErrorContext | None = None,
        severity: ErrorSeverity | None = None,
    ) -> ErrorReport:
        """Handle an error and create a report."""
        error_id = self.generate_error_id()

        # Determine category and severity
        if isinstance(exception, FlextException):
            category = exception.category
            severity = severity or exception.severity
            context = context or exception.context
        else:
            category = self._classify_exception(exception)
            severity = severity or ErrorSeverity.MEDIUM

        # Create error report
        report = ErrorReport(
            error_id=error_id,
            timestamp=datetime.now(UTC),
            severity=severity,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            context=context or ErrorContext("unknown", "unknown"),
        )

        self.error_reports[error_id] = report

        # Log the error
        self._log_error(report)

        return report

    def _classify_exception(self, exception: Exception) -> ErrorCategory:
        """Classify unknown exception into a category."""
        exception_name = type(exception).__name__.lower()

        if any(
            keyword in exception_name
            for keyword in ["connection", "network", "timeout"]
        ):
            return ErrorCategory.NETWORK
        if any(keyword in exception_name for keyword in ["database", "sql", "query"]):
            return ErrorCategory.DATABASE
        if any(
            keyword in exception_name
            for keyword in ["auth", "permission", "unauthorized"]
        ):
            return ErrorCategory.AUTHENTICATION
        if any(
            keyword in exception_name for keyword in ["validation", "value", "type"]
        ):
            return ErrorCategory.VALIDATION
        if "timeout" in exception_name:
            return ErrorCategory.TIMEOUT
        return ErrorCategory.UNKNOWN

    def _log_error(self, report: ErrorReport) -> None:
        """Log error report using structured logging."""
        try:
            from flext_core.logging.structured_logger import get_logger

            logger = get_logger("flext.error_handler")

            logger.error(
                "Error handled: %s",
                report.message,
                error_id=report.error_id,
                severity=report.severity.value,
                category=report.category.value,
                exception_type=report.exception_type,
                component=report.context.component,
                operation=report.context.operation,
                user_id=report.context.user_id,
                request_id=report.context.request_id,
            )
        except ImportError:
            # Fallback to standard logging if structured logging not available
            import logging

            logging.exception("Error %s: %s", report.error_id, report.message)

    def get_error_report(self, error_id: str) -> ErrorReport | None:
        """Get error report by ID."""
        return self.error_reports.get(error_id)

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error statistics."""
        if not self.error_reports:
            return {"total_errors": 0}

        reports = list(self.error_reports.values())

        # Count by category
        category_counts: dict[str, int] = {}
        for report in reports:
            category = report.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

        # Count by severity
        severity_counts: dict[str, int] = {}
        for report in reports:
            severity = report.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_errors": len(reports),
            "by_category": category_counts,
            "by_severity": severity_counts,
            "last_error": reports[-1].timestamp.isoformat() if reports else None,
        }


# Global error handler instance
_error_handler = RobustErrorHandler()


def get_error_handler() -> RobustErrorHandler:
    """Get the global error handler instance."""
    return _error_handler


def handle_exceptions(
    context: ErrorContext | None = None,
    severity: ErrorSeverity | None = None,
    reraise: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for handling exceptions in functions."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or ErrorContext(
                    component=func.__module__, operation=func.__name__
                )

                _error_handler.handle_error(e, error_context, severity)

                if reraise:
                    raise

                # Return None for functions that expect a return value
                return None  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    return decorator


def handle_async_exceptions(
    context: ErrorContext | None = None,
    severity: ErrorSeverity | None = None,
    reraise: bool = False,
) -> Callable[[Callable[..., Awaitable[AsyncT]]], Callable[..., Awaitable[AsyncT]]]:
    """Decorator for handling exceptions in async functions."""

    def decorator(
        func: Callable[..., Awaitable[AsyncT]],
    ) -> Callable[..., Awaitable[AsyncT]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> AsyncT:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = context or ErrorContext(
                    component=func.__module__, operation=func.__name__
                )

                _error_handler.handle_error(e, error_context, severity)

                if reraise:
                    raise

                return None  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]

    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions on failure."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Log retry attempt
                        try:
                            from flext_core.logging.structured_logger import get_logger

                            logger = get_logger("flext.retry")
                            logger.warning(
                                "Retry attempt %s/%s for %s",
                                attempt + 1,
                                max_retries,
                                func.__name__,
                                function=func.__name__,
                                attempt=attempt + 1,
                                max_retries=max_retries,
                                error=str(e),
                            )
                        except ImportError:
                            pass

                        time.sleep(delay * (backoff**attempt))
                    else:
                        # Final attempt failed, handle error
                        context = ErrorContext(
                            component=func.__module__,
                            operation=func.__name__,
                            additional_data={"retry_attempts": max_retries},
                        )
                        _error_handler.handle_error(last_exception, context)
                        raise last_exception

            return None  # type: ignore[return-value]  # Should never reach here

        return wrapper  # type: ignore[return-value]

    return decorator


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    default: T = None,  # type: ignore[assignment]
    context: ErrorContext | None = None,
    **kwargs: Any,
) -> T:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_context = context or ErrorContext(
            component=func.__module__ if hasattr(func, "__module__") else "unknown",
            operation=func.__name__ if hasattr(func, "__name__") else "unknown",
        )

        _error_handler.handle_error(e, error_context)
        return default


async def safe_execute_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    default: T = None,  # type: ignore[assignment]
    context: ErrorContext | None = None,
    **kwargs: Any,
) -> T:
    """Safely execute an async function with error handling."""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        error_context = context or ErrorContext(
            component=func.__module__ if hasattr(func, "__module__") else "unknown",
            operation=func.__name__ if hasattr(func, "__name__") else "unknown",
        )

        _error_handler.handle_error(e, error_context)
        return default


# Context manager for error handling
class ErrorHandlingContext:
    """Context manager for error handling."""

    def __init__(
        self,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        reraise: bool = False,
    ) -> None:
        self.context = context
        self.severity = severity
        self.reraise = reraise
        self.error_report: ErrorReport | None = None

    def __enter__(self) -> "ErrorHandlingContext":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool | None:
        if exc_type is not None and exc_val is not None:
            # Convert BaseException to Exception if needed
            if isinstance(exc_val, Exception):
                self.error_report = _error_handler.handle_error(
                    exc_val, self.context, self.severity
                )

            if not self.reraise:
                return True  # Suppress exception

        return False  # Let exception propagate


# Convenience functions
def create_error_context(component: str, operation: str, **kwargs: Any) -> ErrorContext:
    """Create an error context with common fields."""
    return ErrorContext(component=component, operation=operation, **kwargs)


def log_error_recovery(operation: str, error_id: str, recovery_action: str) -> None:
    """Log error recovery action."""
    try:
        from flext_core.logging.structured_logger import get_logger

        logger = get_logger("flext.recovery")
        logger.info(
            "Error recovery: %s",
            operation,
            operation=operation,
            error_id=error_id,
            recovery_action=recovery_action,
        )
    except ImportError:
        pass
