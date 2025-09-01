"""Logging operations (new): single-class module performing structured logging.

Implements `FlextLoggingOps` which provides the concrete logging behavior.
"""

from __future__ import annotations

from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextLoggingOps:
    """Concrete structured logging operations."""

    @staticmethod
    def get_logger(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> FlextLogger:
        if not hasattr(obj, "_logger") or not isinstance(
            getattr(obj, "_logger", None), FlextLogger
        ):
            logger_name: str = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            obj._logger = FlextLogger(logger_name)
        from typing import cast as _cast

        return _cast("FlextLogger", obj._logger)

    @staticmethod
    def _normalize_context(**kwargs: object) -> FlextTypes.Core.JsonObject:
        normalized: FlextTypes.Core.JsonObject = {}
        for key, value in kwargs.items():
            # Convert Pydantic models and dict-like via centralized utility
            data = FlextUtilities.ProcessingUtils.extract_model_data(value)
            if data:
                from typing import cast as _cast

                normalized[key] = _cast("FlextTypes.Core.JsonValue", data)
            else:
                # Fallback: stringify non-serializable values
                normalized[key] = (
                    str(value)
                    if not isinstance(value, (str, int, float, bool, type(None)))
                    else value
                )
        return normalized

    @staticmethod
    def log_operation(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        operation: str,
        **kwargs: object,
    ) -> None:
        logger = FlextLoggingOps.get_logger(obj)
        context = FlextLoggingOps._normalize_context(**kwargs)
        operation_id: str = FlextUtilities.Generators.generate_correlation_id()
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
        logger = FlextLoggingOps.get_logger(obj)
        context = FlextLoggingOps._normalize_context(**kwargs)
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
        logger = FlextLoggingOps.get_logger(obj)
        context = FlextLoggingOps._normalize_context(**kwargs)
        logger.info(message, **context)

    @staticmethod
    def log_debug(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        logger = FlextLoggingOps.get_logger(obj)
        context = FlextLoggingOps._normalize_context(**kwargs)
        logger.debug(message, **context)

    @staticmethod
    def log_warning(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        logger = FlextLoggingOps.get_logger(obj)
        context = FlextLoggingOps._normalize_context(**kwargs)
        logger.warning(message, **context)

    @staticmethod
    def log_critical(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        logger = FlextLoggingOps.get_logger(obj)
        context = FlextLoggingOps._normalize_context(**kwargs)
        err = context.pop("error", None)
        if err is not None and not isinstance(err, (str, Exception)):
            err = str(err)
        logger.critical(message, error=err, **context)

    @staticmethod
    def log_exception(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        message: str,
        **kwargs: object,
    ) -> None:
        logger = FlextLoggingOps.get_logger(obj)
        context = FlextLoggingOps._normalize_context(**kwargs)
        logger.exception(message, **context)

    @staticmethod
    def with_logger_context(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes, **context: object
    ) -> FlextLogger:
        normalized = FlextLoggingOps._normalize_context(**context)
        return FlextLoggingOps.get_logger(obj).with_context(**normalized)

    @staticmethod
    def start_operation(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        operation_name: str,
        **context: object,
    ) -> str:
        normalized = FlextLoggingOps._normalize_context(**context)
        return FlextLoggingOps.get_logger(obj).start_operation(
            operation_name, **normalized
        )

    @staticmethod
    def complete_operation(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        operation_id: str,
        *,
        success: bool = True,
        **context: object,
    ) -> None:
        normalized = FlextLoggingOps._normalize_context(**context)
        FlextLoggingOps.get_logger(obj).complete_operation(
            operation_id, success=success, **normalized
        )
