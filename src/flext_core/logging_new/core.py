"""Logging facade (new): single-class module delegating to ops.

Defines `FlextLoggingCore` facade that forwards to `FlextLoggingOps`.
"""

from __future__ import annotations

from .ops import FlextLoggingOps


class FlextLoggingCore:
    """Facade for structured logging operations and mixin lifters."""

    # Forwarders to operations (keep signatures stable and concise)
    get_logger = staticmethod(FlextLoggingOps.get_logger)
    log_operation = staticmethod(FlextLoggingOps.log_operation)
    log_error = staticmethod(FlextLoggingOps.log_error)
    log_info = staticmethod(FlextLoggingOps.log_info)
    log_debug = staticmethod(FlextLoggingOps.log_debug)
    log_warning = staticmethod(FlextLoggingOps.log_warning)
    log_critical = staticmethod(FlextLoggingOps.log_critical)
    log_exception = staticmethod(FlextLoggingOps.log_exception)
    with_logger_context = staticmethod(FlextLoggingOps.with_logger_context)
    start_operation = staticmethod(FlextLoggingOps.start_operation)
    complete_operation = staticmethod(FlextLoggingOps.complete_operation)
