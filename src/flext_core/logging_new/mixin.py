"""Logging mixin (new): single-class module providing mixin API.

Implements `FlextLoggingMixin` that delegates to `FlextLoggingCore`.
"""

from __future__ import annotations

from flext_core.loggings import FlextLogger


class FlextLoggingMixin:
    """Mixin adding structured logging methods to classes."""

    @property
    def logger(self) -> FlextLogger:  # pragma: no cover - simple delegation
        from .core import FlextLoggingCore

        return FlextLoggingCore.get_logger(self)

    def log_operation(self, operation: str, **kwargs: object) -> None:
        from .core import FlextLoggingCore

        FlextLoggingCore.log_operation(self, operation, **kwargs)

    def log_error(self, error: str | Exception, **kwargs: object) -> None:
        from .core import FlextLoggingCore

        FlextLoggingCore.log_error(self, error, **kwargs)

    def log_info(self, message: str, **kwargs: object) -> None:
        from .core import FlextLoggingCore

        FlextLoggingCore.log_info(self, message, **kwargs)

    def log_debug(self, message: str, **kwargs: object) -> None:
        from .core import FlextLoggingCore

        FlextLoggingCore.log_debug(self, message, **kwargs)

    def log_warning(self, message: str, **kwargs: object) -> None:
        from .core import FlextLoggingCore

        FlextLoggingCore.log_warning(self, message, **kwargs)

    def log_critical(self, message: str, **kwargs: object) -> None:
        from .core import FlextLoggingCore

        FlextLoggingCore.log_critical(self, message, **kwargs)

    def log_exception(self, message: str, **kwargs: object) -> None:
        from .core import FlextLoggingCore

        FlextLoggingCore.log_exception(self, message, **kwargs)

    def with_logger_context(self, **context: object) -> FlextLogger:
        from .core import FlextLoggingCore

        return FlextLoggingCore.with_logger_context(self, **context)

    def start_operation(self, operation_name: str, **context: object) -> str:
        from .core import FlextLoggingCore

        return FlextLoggingCore.start_operation(self, operation_name, **context)

    def complete_operation(
        self, operation_id: str, *, success: bool = True, **context: object
    ) -> None:
        from .core import FlextLoggingCore

        FlextLoggingCore.complete_operation(
            self, operation_id, success=success, **context
        )
