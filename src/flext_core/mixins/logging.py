"""FLEXT Logging Mixin - Structured logging using centralized components.

This module provides logging mixins that leverage centralized FLEXT
ecosystem components for consistent logging patterns.
"""

from __future__ import annotations

from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.typings import FlextTypes


class FlextLogging:
    """Unified structured logging system using centralized FLEXT components."""

    @staticmethod
    def get_logger(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> FlextLogger:
        """Get FlextLogger instance for an object."""
        if not hasattr(obj, "_logger"):
            logger_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            logger = FlextLogger(logger_name)
            obj._logger = logger
            return logger

        logger_attr = obj._logger
        if not isinstance(logger_attr, FlextLogger):
            logger_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            logger = FlextLogger(logger_name)
            obj._logger = logger
            return logger

        return logger_attr

    @staticmethod
    def _normalize_context(**kwargs: object) -> FlextTypes.Core.JsonObject:
        """Normalize context values for structured logging."""
        normalized: FlextTypes.Core.JsonObject = {}

        for key, value in kwargs.items():
            match value:
                # Pydantic v2 BaseModel
                case model if hasattr(model, "model_dump"):
                    normalized[key] = model.model_dump()

                # Basic JSON-serializable types
                case str() | int() | float() | bool() | None:
                    normalized[key] = value

                # Lists (recursively normalize)
                case list():
                    normalized[key] = [
                        item.model_dump()
                        if isinstance(item, BaseModel)
                        else str(item)
                        if item is not None
                        else None
                        for item in value
                    ]

                # Dictionaries (recursively normalize)
                case dict():
                    normalized[key] = {
                        k: v.model_dump()
                        if isinstance(v, BaseModel)
                        else str(v)
                        if v is not None
                        else None
                        for k, v in value.items()
                    }

                # Fallback: convert to string
                case _:
                    normalized[key] = str(value) if value is not None else None

        return normalized

    @staticmethod
    def log_operation(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        operation: str,
        **kwargs: object,
    ) -> None:
        """Log an operation with context."""
        logger = FlextLogging.get_logger(obj)
        context = FlextLogging._normalize_context(**kwargs)

        operation_id = FlextUtilities.Generators.generate_correlation_id()
        full_context: FlextTypes.Core.JsonObject = {
            "operation": operation,
            "operation_id": operation_id,
            "object_type": obj.__class__.__name__,
            **context,
        }

        logger.info(f"Operation: {operation}", **full_context)

    @staticmethod
    def log_error(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        error: str | Exception,
        **kwargs: object,
    ) -> None:
        """Log an error with context."""
        logger = FlextLogging.get_logger(obj)
        context = FlextLogging._normalize_context(**kwargs)

        if isinstance(error, Exception):
            error_context: FlextTypes.Core.JsonObject = {
                "error_type": type(error).__name__,
                "error_details": str(error),
                "object_type": obj.__class__.__name__,
                **context,
            }
            logger.error(f"Error: {error}", error=error, **error_context)
        else:
            error_context = {
                "error_details": str(error),
                "object_type": obj.__class__.__name__,
                **context,
            }
            logger.error(f"Error: {error}", error=error, **error_context)

    @staticmethod
    def log_info(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log an info message with context."""
        logger = FlextLogging.get_logger(obj)
        context = FlextLogging._normalize_context(**kwargs)
        full_context: FlextTypes.Core.JsonObject = {
            "object_type": obj.__class__.__name__,
            **context,
        }
        logger.info(message, **full_context)

    @staticmethod
    def log_debug(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        """Log a debug message with context."""
        logger = FlextLogging.get_logger(obj)
        context = FlextLogging._normalize_context(**kwargs)
        full_context: FlextTypes.Core.JsonObject = {
            "object_type": obj.__class__.__name__,
            **context,
        }
        logger.debug(message, **full_context)

    class Loggable:
        """Mixin class providing structured logging capabilities."""

        def get_logger(self) -> FlextLogger:
            """Get logger for this object."""
            return FlextLogging.get_logger(self)

        def log_operation(self, operation: str, **kwargs: object) -> None:
            """Log an operation."""
            FlextLogging.log_operation(self, operation, **kwargs)

        def log_error(self, error: str | Exception, **kwargs: object) -> None:
            """Log an error."""
            FlextLogging.log_error(self, error, **kwargs)

        def log_info(self, message: str, **kwargs: object) -> None:
            """Log an info message."""
            FlextLogging.log_info(self, message, **kwargs)

        def log_debug(self, message: str, **kwargs: object) -> None:
            """Log a debug message."""
            FlextLogging.log_debug(self, message, **kwargs)
